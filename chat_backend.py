from fastapi import FastAPI, HTTPException, Request
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

# Stripe 为可选依赖：未安装或未配置密钥时，支付接口降级但不影响聊天
try:
    import stripe
    STRIPE_SDK_OK = True
except ImportError:
    stripe = None
    STRIPE_SDK_OK = False

load_dotenv()
app = FastAPI(title="AI Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client_chat = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
AUDIO_DIR = os.getenv("AUDIO_DIR", tempfile.gettempdir())
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ai-companion-ivory-two.vercel.app")

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


class UserMemory(Base):
    """结构化长期记忆：AI 记住的关于用户的事实"""
    __tablename__ = "user_memories"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, default="preference")
    content = Column(Text, nullable=False)
    source = Column(String, default="manual")   # seed / manual / auto
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ProSubscription(Base):
    """Stripe 订阅状态：以浏览器匿名 ID 作为身份标识"""
    __tablename__ = "pro_subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, index=True, unique=True)
    stripe_session_id = Column(String)
    stripe_customer_id = Column(String)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# 出厂示例记忆：仅当表为空时播种，用户可在侧边栏自行删改
SEED_MEMORIES = [
    ("learning", "已掌握 Python 基础、FastAPI 后端、React 前端、Live2D 渲染、RAG 与 Function Calling"),
    ("pitfall", "Render 部署踩过坑：requirements.txt 漏写依赖导致 ModuleNotFoundError"),
    ("goal", "下一步计划学习 Next.js 与 RAG 高级玩法"),
    ("preference", "称呼她为「同学」，偏好专业、耐心、直接给解决方案的沟通方式"),
    ("preference", "喜欢红茶"),
]


def seed_memories():
    if not DB_READY or not SessionLocal:
        return
    db = SessionLocal()
    try:
        if db.query(UserMemory).count() == 0:
            for category, content in SEED_MEMORIES:
                db.add(UserMemory(category=category, content=content, source="seed"))
            db.commit()
            print(f"[startup] 已播种 {len(SEED_MEMORIES)} 条初始记忆", flush=True)
    except Exception as e:
        db.rollback()
        print(f"[startup] 记忆播种失败（不影响聊天）: {e}", flush=True)
    finally:
        db.close()


try:
    engine = _build_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    # 只创建不存在的表，不会破坏已有的 chat_history
    Base.metadata.create_all(bind=engine)
    DB_READY = True
    print(f"[startup] ✅ Supabase 已连接 -> {engine.url.host}/{engine.url.database} "
          f"(driver={engine.url.drivername})", flush=True)
    seed_memories()
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
        desc_map = {"Sunny": "晴朗", "Partly cloudy": "多云", "Cloudy": "阴天",
                    "Light rain": "小雨", "Overcast": "阴天"}
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

CATEGORY_LABELS = {
    "preference": "喜好偏好",
    "learning": "学习进度",
    "goal": "目标计划",
    "pitfall": "踩过的坑",
    "other": "其他",
}
VALID_CATEGORIES = set(CATEGORY_LABELS.keys())


class ChatRequest(BaseModel):
    user_message: str
    character_id: str = "girl"
    client_id: str = ""


class MemoryCreate(BaseModel):
    category: str = "preference"
    content: str


class CheckoutRequest(BaseModel):
    client_id: str


def run_tts_in_thread(text_value, filename, voice):
    def target():
        async def _generate():
            communicate = edge_tts.Communicate(text_value, voice)
            await communicate.save(filename)
        asyncio.run(_generate())
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()


# ================= 3. 记忆服务 =================
FREE_MEMORY_LIMIT = int(os.getenv("FREE_MEMORY_LIMIT", "8"))
AUTO_EXTRACT = os.getenv("AUTO_EXTRACT_MEMORIES", "true").lower() == "true"


def get_all_memories():
    """返回按分类分组的记忆，供接口与 system prompt 共用"""
    if not DB_READY or not SessionLocal:
        return [], {}
    db = SessionLocal()
    try:
        rows = db.query(UserMemory).order_by(UserMemory.id.asc()).all()
        items = [{"id": r.id, "category": r.category, "content": r.content,
                  "source": r.source, "created_at": str(r.created_at)} for r in rows]
        groups = {}
        for item in items:
            key = item["category"] if item["category"] in CATEGORY_LABELS else "other"
            groups.setdefault(key, []).append(item)
        return items, groups
    finally:
        db.close()


def build_system_prompt():
    """把结构化记忆动态拼进人设，让 AI 真的「记住」"""
    base = CHARACTERS["girl"]["prompt"]
    items, groups = get_all_memories()
    if not items:
        return base
    lines = ["\n\n【关于她的长期记忆（务必牢记，并在合适时自然提及）】"]
    for key, label in CATEGORY_LABELS.items():
        if groups.get(key):
            lines.append(f"- {label}：" + "；".join(m["content"] for m in groups[key]))
    lines.append("【注意】不要生硬复述这些记忆，只在相关话题下自然引用。")
    return base + "\n".join(lines)


def is_pro(client_id: str) -> bool:
    if not (DB_READY and SessionLocal and client_id):
        return False
    db = SessionLocal()
    try:
        return db.query(ProSubscription).filter(
            ProSubscription.client_id == client_id,
            ProSubscription.status == "active").first() is not None
    finally:
        db.close()


def extract_memories_async(user_text: str, client_id: str = ""):
    """聊天后异步抽取新的长期事实；失败完全不影响主流程"""
    if not (DB_READY and SessionLocal and AUTO_EXTRACT):
        return
    try:
        extract_prompt = """你是记忆抽取器。从用户这句话里抽取关于「用户本人」的长期事实
（喜好、正在学的技术、目标计划、踩过的坑、身份背景等）。
只输出 JSON 数组，格式：[{"category":"preference|learning|goal|pitfall|other","content":"简短事实"}]
若没有值得长期记住的新事实，输出 []。不要输出任何解释文字或代码块标记。"""
        resp = client_chat.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": extract_prompt},
                      {"role": "user", "content": user_text[:500]}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # 容错：去掉可能存在的 ```json 包裹
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        facts = json.loads(raw)
        if not isinstance(facts, list) or not facts:
            return
        db = SessionLocal()
        try:
            existing = {r.content.strip() for r in db.query(UserMemory).all()}
            pro = is_pro(client_id)
            # 免费版与手动添加共用同一条数上限，Pro 解锁无限
            if not pro and len(existing) >= FREE_MEMORY_LIMIT:
                return
            added = 0
            for fact in facts[:5]:
                if not isinstance(fact, dict):
                    continue
                if not pro and len(existing) >= FREE_MEMORY_LIMIT:
                    break
                content = str(fact.get("content", "")).strip()
                category = fact.get("category", "other")
                if category not in VALID_CATEGORIES:
                    category = "other"
                if not content or len(content) > 120 or content in existing:
                    continue
                db.add(UserMemory(category=category, content=content, source="auto"))
                existing.add(content)
                added += 1
            db.commit()
            if added:
                print(f"[memory] 自动新增 {added} 条记忆", flush=True)
        finally:
            db.close()
    except Exception as e:
        print(f"[memory] 自动抽取跳过（不影响聊天）: {type(e).__name__}: {e}", flush=True)


# ================= 4. 自检接口 =================
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
        "memories_count": None,
        "pro_count": None,
        "stripe_configured": bool(STRIPE_SDK_OK and os.getenv("STRIPE_SECRET_KEY")
                                  and os.getenv("STRIPE_PRICE_ID")),
    }
    if not DB_READY or not SessionLocal:
        return result

    db = SessionLocal()
    try:
        result["row_count"] = db.query(ChatHistory).count()
        result["memories_count"] = db.query(UserMemory).count()
        result["pro_count"] = db.query(ProSubscription).count()
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


# ================= 5. 记忆管理接口 =================
@app.get("/api/memories")
def list_memories():
    if not DB_READY:
        return {"status": "error", "message": "数据库未就绪", "memories": [],
                "groups": {}, "labels": CATEGORY_LABELS, "total": 0}
    items, groups = get_all_memories()
    return {"status": "success", "memories": items, "groups": groups,
            "labels": CATEGORY_LABELS, "total": len(items),
            "limit": FREE_MEMORY_LIMIT}


@app.post("/api/memories")
def add_memory(req: MemoryCreate, client_id: str = ""):
    if not DB_READY or not SessionLocal:
        return {"status": "error", "message": "数据库未就绪，无法保存记忆"}
    content = (req.content or "").strip()
    if not content:
        return {"status": "error", "message": "记忆内容不能为空"}
    if len(content) > 120:
        return {"status": "error", "message": "记忆内容请控制在 120 字以内"}
    category = req.category if req.category in VALID_CATEGORIES else "other"
    db = SessionLocal()
    try:
        count = db.query(UserMemory).count()
        # Free 用户有记忆条数上限，Pro 解锁无限
        if not is_pro(client_id) and count >= FREE_MEMORY_LIMIT:
            return {"status": "error", "code": "LIMIT_REACHED",
                    "message": f"免费版最多保存 {FREE_MEMORY_LIMIT} 条记忆，升级 Pro 可解锁无限记忆",
                    "limit": FREE_MEMORY_LIMIT}
        if any(r.content.strip() == content for r in db.query(UserMemory).all()):
            return {"status": "error", "message": "这条记忆已经存在啦"}
        row = UserMemory(category=category, content=content, source="manual")
        db.add(row)
        db.commit()
        return {"status": "success",
                "memory": {"id": row.id, "category": row.category,
                           "content": row.content, "source": row.source,
                           "created_at": str(row.created_at)}}
    except Exception as e:
        db.rollback()
        print(f"[memory] 新增失败: {traceback.format_exc()}", flush=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int):
    if not DB_READY or not SessionLocal:
        return {"status": "error", "message": "数据库未就绪"}
    db = SessionLocal()
    try:
        row = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
        if not row:
            return {"status": "error", "message": "记忆不存在"}
        db.delete(row)
        db.commit()
        return {"status": "success", "deleted_id": memory_id}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


@app.get("/api/history")
def get_history(limit: int = 30):
    """前端启动时加载历史消息，直观呈现「永久记忆」与每日早安"""
    if not DB_READY or not SessionLocal:
        return {"status": "success", "messages": []}
    db = SessionLocal()
    try:
        rows = (db.query(ChatHistory)
                .filter(ChatHistory.character == "girl", ChatHistory.content.isnot(None))
                .order_by(ChatHistory.id.desc()).limit(min(max(limit, 1), 100)).all())
        rows.reverse()
        return {"status": "success", "messages": [
            {"id": r.id, "role": r.role, "content": r.content,
             "timestamp": str(r.timestamp),
             "is_greeting": bool(r.content.startswith("【早安】"))}
            for r in rows
        ]}
    finally:
        db.close()


# ================= 6. 主动陪伴：每日早安 =================
CRON_SECRET = os.getenv("CRON_SECRET", "")


def generate_greeting_text():
    """结合长期记忆生成个性化早安问候，失败时兜底固定文案"""
    facts = []
    items, groups = get_all_memories()
    for key in ("goal", "learning"):
        for m in groups.get(key, []):
            facts.append(m["content"])
    hint = "；".join(facts[:3]) if facts else "她的学习计划"
    try:
        prompt = (f"你是用户的专属AI教练，现在是早上8点。请写一句简短温暖的早安问候，"
                  f"40字以内，称呼她「同学」，自然地提到或关心她的学习：{hint}。"
                  f"语气专业耐心，不要用括号动作描写，不要堆砌emoji。只输出问候语本身。")
        resp = client_chat.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8, max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return "【早安】" + text.strip('"“”')
    except Exception as e:
        print(f"[cron] 问候生成降级为固定文案: {e}", flush=True)
    return "【早安】同学，新的一天开始了，你的 Next.js 学得怎么样了？"


@app.api_route("/api/cron/morning-greeting", methods=["GET", "POST"])
def morning_greeting(secret: str = ""):
    """由 Render Cron Job 或外部定时器调用；GET 便于手工/外部 cron 触发"""
    if CRON_SECRET and secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="无效的 CRON_SECRET")
    if not DB_READY or not SessionLocal:
        return {"status": "error", "message": "数据库未就绪，问候未写入"}
    greeting = generate_greeting_text()
    db = SessionLocal()
    try:
        db.add(ChatHistory(character="girl", role="assistant",
                           content=greeting, affection=50))
        db.commit()
        print(f"[cron] 早安问候已写入: {greeting}", flush=True)
        return {"status": "success", "greeting": greeting,
                "time_utc": datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        db.rollback()
        print(f"[cron] 写入失败: {traceback.format_exc()}", flush=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


# ================= 7. Stripe 订阅支付 =================
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
if STRIPE_SDK_OK and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def stripe_ready():
    return bool(STRIPE_SDK_OK and STRIPE_SECRET_KEY and STRIPE_PRICE_ID)


def set_pro_status(client_id: str, active: bool,
                   session_id: str = None, customer_id: str = None):
    if not (DB_READY and SessionLocal and client_id):
        return
    db = SessionLocal()
    try:
        row = (db.query(ProSubscription)
               .filter(ProSubscription.client_id == client_id).first())
        if not row:
            row = ProSubscription(client_id=client_id)
            db.add(row)
        row.status = "active" if active else "inactive"
        if session_id:
            row.stripe_session_id = session_id
        if customer_id:
            row.stripe_customer_id = customer_id
        db.commit()
        print(f"[stripe] client_id={client_id[:8]}... -> {row.status}", flush=True)
    except Exception as e:
        db.rollback()
        print(f"[stripe] 状态写入失败: {e}", flush=True)
    finally:
        db.close()


@app.post("/api/stripe/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    if not stripe_ready():
        return {"status": "error", "code": "STRIPE_NOT_CONFIGURED",
                "message": "Stripe 尚未配置：请在 Render 环境变量中填入 STRIPE_SECRET_KEY 与 STRIPE_PRICE_ID"}
    if not req.client_id:
        return {"status": "error", "message": "缺少 client_id"}
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/?upgrade=success&client_id={req.client_id}",
            cancel_url=f"{FRONTEND_URL}/?upgrade=cancel",
            client_reference_id=req.client_id,
            metadata={"client_id": req.client_id},
            subscription_data={"metadata": {"client_id": req.client_id}},
            allow_promotion_codes=True,
        )
        return {"status": "success", "url": session.url}
    except Exception as e:
        print(f"[stripe] 创建会话失败: {traceback.format_exc()}", flush=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """Stripe 事件回调：验签后激活 / 取消 Pro 身份"""
    if not STRIPE_WEBHOOK_SECRET:
        return {"status": "error", "message": "未配置 STRIPE_WEBHOOK_SECRET"}
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        print(f"[stripe] 验签失败: {e}", flush=True)
        raise HTTPException(status_code=400, detail=f"Webhook 验签失败: {e}")

    etype = event.get("type")
    obj = event.get("data", {}).get("object", {})
    if etype == "checkout.session.completed":
        cid = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("client_id")
        set_pro_status(cid, True, session_id=obj.get("id"),
                       customer_id=obj.get("customer"))
    elif etype in ("customer.subscription.deleted", "customer.subscription.paused"):
        cid = (obj.get("metadata") or {}).get("client_id")
        set_pro_status(cid, False)
    elif etype == "invoice.payment_succeeded":
        # 续费成功，确保状态仍为 active
        cid = (obj.get("subscription_details") or {}).get("metadata", {}).get("client_id")
        if cid:
            set_pro_status(cid, True)
    print(f"[stripe] 已处理事件: {etype}", flush=True)
    return {"received": True}


@app.get("/api/subscription/status")
def subscription_status(client_id: str = ""):
    pro = is_pro(client_id)
    return {"status": "success", "is_pro": pro,
            "plan": "pro" if pro else "free",
            "price": "$7.99/月",
            "stripe_configured": stripe_ready(),
            "memory_limit": None if pro else FREE_MEMORY_LIMIT}


# ================= 8. 核心聊天逻辑 =================
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
                              .order_by(ChatHistory.id.desc()).limit(8).all())
            recent_records.reverse()

        # 人设 + 结构化长期记忆动态拼装
        messages = [{"role": "system", "content": build_system_prompt()}]
        for record in recent_records:
            # 跳过空内容与每日早安（早安是主动陪伴消息，不当作对话上下文）
            if record.content and not record.content.startswith("【早安】"):
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

        # 异步抽取长期记忆，不阻塞响应
        try:
            threading.Thread(target=extract_memories_async,
                             args=(request.user_message, request.client_id),
                             daemon=True).start()
        except Exception:
            pass

        return {
            "status": "success",
            "reply": reply,
            "audio_url": audio_url,
            "affection": new_affection,
            "db_persisted": bool(db),
            "is_pro": is_pro(request.client_id),
        }
    except Exception as e:
        if db:
            db.rollback()
        print(f"[chat] Error: {traceback.format_exc()}", flush=True)
        return {"status": "error", "message": str(e)}
    finally:
        if db:
            db.close()


# ================= 9. 音频文件 =================
@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    safe_name = os.path.basename(filename)
    full_path = os.path.join(AUDIO_DIR, safe_name)
    if os.path.isfile(full_path):
        return FileResponse(full_path, media_type="audio/mpeg")
    return {"status": "error", "message": "音频不存在"}
