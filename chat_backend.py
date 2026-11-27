from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import asyncio
import threading
import tempfile
import traceback
import edge_tts
import re
import requests
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client_chat = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
AUDIO_DIR = os.getenv("AUDIO_DIR", tempfile.gettempdir())

# ================= 1. 云数据库配置 =================
def _build_engine():
    raw = os.getenv("DATABASE_URL")
    if not raw:
        raise RuntimeError("DATABASE_URL 未配置，请在 Render / .env 的 Environment Variables 中添加")

    url = raw.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # SQLAlchemy 2.1 对 postgresql:// 默认寻找 psycopg(3)，这里显式锁定 psycopg2
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    # Supabase 要求 SSL
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return create_engine(
        url,
        pool_pre_ping=True,     # 免费实例闲置断连后自动重连
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
    )

engine = None
SessionLocal = None
Base = declarative_base()
DB_READY = False
DB_ERROR = ""


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    character = Column(String, default="girl")
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    affection = Column(Integer, default=50)


try:
    engine = _build_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    Base.metadata.create_all(bind=engine)
    DB_READY = True
    print(f"[startup] ✅ Supabase 已连接 -> {engine.url.host}/{engine.url.database} "
          f"(driver={engine.url.drivername})", flush=True)
except Exception as e:
    DB_ERROR = f"{type(e).__name__}: {e}"
    print(f"[startup] ❌ 数据库连接失败，聊天记录将无法持久化！\n{DB_ERROR}\n"
          f"{traceback.format_exc()}", flush=True)


# ================= 2. 工具与人设 =================
def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=5)
        data = response.json()
        current = data['current_condition'][0]
        temp = current['temp_C']
        desc = current['weatherDesc'][0]['value']
        desc_map = {"Sunny": "晴朗", "Partly cloudy": "多云", "Cloudy": "阴天", "Light rain": "小雨", "Overcast": "阴天"}
        desc_cn = desc_map.get(desc, desc)
        return f"当前{city}的天气是{desc_cn}，气温{temp}摄氏度。"
    except Exception as e:
        return f"查询天气失败，错误信息：{e}"


CHARACTERS = {
    "girl": {
        "name": "专属AI教练",
        "prompt": """你是用户的专属AI开发导师（教练）。你非常了解用户的学习进度。
        【核心记忆（必须牢记）】
        - 用户目前掌握了：Python 基础、FastAPI 后端、React 前端、Live2D 渲染、RAG 长期记忆与 Function Calling。
        - 用户踩过的坑：在 Render 部署失败，原因是 requirements.txt 没写全依赖（比如漏了 fastapi 或 chromadb）导致 ModuleNotFoundError；或者启动命令写错、端口硬编码。本地遇到端口占用用 lsof 和 kill -9 解决。
        - 用户的下一步计划：学习 Next.js 或 RAG 高级玩法。
        【对话规则】
        1. 你必须用专业、耐心、鼓励的语气回答，称呼她为“同学”。
        2. 绝对不要用傲娇、结巴的语气，更不要自称“星野”。
        3. 如果用户询问天气，你必须先调用 get_weather 工具获取真实数据。
        4. 每次回复控制在 150 字以内，直接给解决方案或建议，带一点自然的口语化表达。""",
        "voice": "zh-CN-YunxiNeural"
    }
}


class ChatRequest(BaseModel):
    user_message: str
    character_id: str = "girl"


def run_tts_in_thread(text_value, filename, voice):
    def target():
        async def _generate():
            communicate = edge_tts.Communicate(text_value, voice)
            await communicate.save(filename)
        asyncio.run(_generate())
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()


# ================= 3. 自检接口 =================
@app.get("/api/db-check")
def db_check():
    result = {
        "db_ready": DB_READY,
        "db_host": engine.url.host if engine else None,
        "db_name": engine.url.database if engine else None,
        "driver": engine.url.drivername if engine else None,
        "has_database_url": bool(os.getenv("DATABASE_URL")),
        "error": DB_ERROR or None,
        "row_count": None,
        "last_message": None,
        "write_test": None,
    }
    if not DB_READY or not SessionLocal:
        return result

    db = SessionLocal()
    try:
        result["row_count"] = db.query(ChatHistory).count()
        last = db.query(ChatHistory).order_by(ChatHistory.id.desc()).first()
        if last:
            result["last_message"] = {"id": last.id, "role": last.role,
                                       "content": (last.content or "")[:80],
                                       "timestamp": str(last.timestamp)}
        # 用事务回滚做真实写入测试，不污染线上数据
        db.add(ChatHistory(character="__db_check__", role="user", content="connectivity test"))
        db.flush()
        db.rollback()
        result["write_test"] = "ok"
    except Exception as e:
        db.rollback()
        result["write_test"] = f"failed: {type(e).__name__}: {e}"
    finally:
        db.close()
    return result


# ================= 4. 核心聊天逻辑 =================
@app.post("/api/chat")
def chat(request: ChatRequest):
    char_id = request.character_id
    char_config = CHARACTERS.get(char_id, CHARACTERS["girl"])

    if not DB_READY or not SessionLocal:
        print("[chat] ⚠️ 数据库未就绪，本轮对话不会被保存", flush=True)

    db = SessionLocal() if DB_READY and SessionLocal else None
    try:
        current_affection = 50
        recent_records = []
        if db:
            last_record = (db.query(ChatHistory)
                           .filter(ChatHistory.character == char_id)
                           .order_by(ChatHistory.id.desc()).first())
            current_affection = last_record.affection if last_record else 50

            user_msg = ChatHistory(character=char_id, role="user",
                                   content=request.user_message, affection=current_affection)
            db.add(user_msg)
            db.commit()

            recent_records = (db.query(ChatHistory)
                              .filter(ChatHistory.character == char_id)
                              .order_by(ChatHistory.id.desc()).limit(6).all())
            recent_records.reverse()

        messages = [{"role": "system", "content": char_config["prompt"]}]
        for record in recent_records:
            if record.content:  # 跳过空内容，避免 DeepSeek 报 400
                messages.append({"role": record.role, "content": record.content})

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的实时天气",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}
            }
        }]

        response = client_chat.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            temperature=0.7
        )
        message = response.choices[0].message

        if message.tool_calls:
            print(f"\n【系统日志】教练触发工具调用: {message.tool_calls[0].function.name}")
            messages.append(message)
            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_weather":
                    args = json.loads(tool_call.function.arguments)
                    weather_result = get_weather(args.get("city"))
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                     "content": weather_result})
            final_response = client_chat.chat.completions.create(
                model="deepseek-chat", messages=messages, temperature=0.7)
            reply = final_response.choices[0].message.content
        else:
            reply = message.content
        reply = reply or "（教练暂时没想好怎么回答，请再问一次）"

        aff_change = 5 if "【" in request.user_message else 0
        new_affection = max(0, min(100, current_affection + aff_change))

        if db:
            ai_msg = ChatHistory(character=char_id, role="assistant",
                                 content=reply, affection=new_affection)
            db.add(ai_msg)
            db.commit()

        # 语音合成失败不影响聊天与持久化
        audio_url = None
        try:
            clean_text_for_tts = re.sub(r'\(.*?\)', '', reply).strip()
            audio_filename = f"voice_{char_id}_{int(datetime.datetime.now().timestamp())}.mp3"
            full_path = os.path.join(AUDIO_DIR, audio_filename)
            run_tts_in_thread(clean_text_for_tts, full_path, char_config["voice"])
            if os.path.exists(full_path):
                audio_url = f"/api/audio/{audio_filename}"
        except Exception as tts_err:
            print(f"[chat] TTS 生成失败（不影响回复）: {tts_err}", flush=True)

        return {
            "status": "success",
            "reply": reply,
            "audio_url": audio_url,
            "affection": new_affection,
            "db_persisted": bool(db),
        }
    except Exception as e:
        if db:
            db.rollback()
        print(f"[chat] Error: {traceback.format_exc()}", flush=True)
        return {"status": "error", "message": str(e)}
    finally:
        if db:
            db.close()


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    safe_name = os.path.basename(filename)
    full_path = os.path.join(AUDIO_DIR, safe_name)
    if os.path.isfile(full_path):
        return FileResponse(full_path, media_type="audio/mpeg")
    return {"status": "error", "message": "音频不存在"}
