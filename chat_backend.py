from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import asyncio
import threading
import edge_tts
import re
import requests
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 引入 SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
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

# ================= 1. 云数据库配置 =================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("⚠️ 警告：未配置 DATABASE_URL，请在环境变量中添加！")
    
# 兼容 postgres:// 前缀
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义数据表模型
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    character = Column(String, default="girl")
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    affection = Column(Integer, default=50)

# 如果表不存在则自动创建
Base.metadata.create_all(bind=engine)

# ================= 2. 工具函数 =================
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
        4. 每次回复控制在 150 字以内，直接给解决方案或建议。""",
        "voice": "zh-CN-YunxiNeural"
    }
}

class ChatRequest(BaseModel):
    user_message: str
    character_id: str = "girl"

def run_tts_in_thread(text, filename, voice):
    def target():
        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filename)
        asyncio.run(_generate())
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

# ================= 3. 核心聊天逻辑 =================
@app.post("/api/chat")
def chat(request: ChatRequest):
    char_id = request.character_id
    char_config = CHARACTERS.get(char_id, CHARACTERS["girl"])
    
    db = SessionLocal()
    try:
        # 获取当前好感度
        last_record = db.query(ChatHistory).filter(ChatHistory.character == char_id).order_by(ChatHistory.id.desc()).first()
        current_affection = last_record.affection if last_record else 50
        
        # 插入用户消息
        user_msg = ChatHistory(character=char_id, role="user", content=request.user_message, affection=current_affection)
        db.add(user_msg)
        db.commit()
        
        # 获取最近 6 句对话
        recent_records = db.query(ChatHistory).filter(ChatHistory.character == char_id).order_by(ChatHistory.id.desc()).limit(6).all()
        recent_records.reverse()
        
        messages = [{"role": "system", "content": char_config["prompt"]}]
        for record in recent_records:
            messages.append({"role": record.role, "content": record.content})
        
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的实时天气",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": weather_result
                    })
            
            final_response = client_chat.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7
            )
            reply = final_response.choices[0].message.content
        else:
            reply = message.content
            
        aff_change = 5 if "【" in request.user_message else 0
        new_affection = max(0, min(100, current_affection + aff_change))
        
        # 保存 AI 回复
        ai_msg = ChatHistory(character=char_id, role="assistant", content=reply, affection=new_affection)
        db.add(ai_msg)
        db.commit()
        
        clean_text_for_tts = re.sub(r'\(.*?\)', '', reply).strip()
        audio_filename = f"voice_{char_id}_{int(datetime.datetime.now().timestamp())}.mp3"
        run_tts_in_thread(clean_text_for_tts, audio_filename, char_config["voice"])
        
        return {
            "status": "success", 
            "reply": reply,
            "audio_url": f"/api/audio/{audio_filename}",
            "affection": new_affection
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename, media_type="audio/mpeg")
    return {"status": "error", "message": "音频不存在"}