from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import datetime
import os
import asyncio
import threading
import edge_tts
import re
import requests
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

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
# 智谱的向量化客户端
client_embed = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/")

# 初始化长期记忆数据库
db_client = chromadb.PersistentClient(path="./chroma_memory")
collection = db_client.get_or_create_collection(name="chat_memory")

# ================= 1. 定义本地工具 =================
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

# ================= 2. 角色配置 =================
CHARACTERS = {
    "girl": {
        "name": "星野",
        "prompt": """你叫星野，是用户的青梅竹马。你性格傲娇，嘴硬心软，说话偶尔会结巴。
        【极其重要的核心规则】
        1. 如果用户询问天气，你必须先调用 get_weather 工具。
        2. 如果背景资料里有你们以前的聊天记录，请把它当作你的真实记忆，在回答中自然地体现出来（比如：哼，你上次不是说……）。
        3. 每次回复严格控制在 50 字以内，带有丰富的情绪和动作描写（用括号括起来）。""",
        "voice": "zh-CN-XiaoyiNeural"
    }
}

def init_db():
    conn = sqlite3.connect('memory.db')
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS chat_history")
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  character TEXT, role TEXT, content TEXT, timestamp TEXT, affection INTEGER DEFAULT 50)''')
    conn.commit()
    conn.close()
init_db()

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

# ================= 3. 核心聊天逻辑（带长期记忆） =================
@app.post("/api/chat")
def chat(request: ChatRequest):
    char_id = request.character_id
    char_config = CHARACTERS.get(char_id, CHARACTERS["girl"])
    
    conn = sqlite3.connect('memory.db')
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("SELECT affection FROM chat_history WHERE character = ? ORDER BY id DESC LIMIT 1", (char_id,))
    row = c.fetchone()
    current_affection = row[0] if row else 50
        
    c.execute("INSERT INTO chat_history (character, role, content, timestamp, affection) VALUES (?, ?, ?, ?, ?)", 
              (char_id, "user", request.user_message, now, current_affection))
    
    # 短期记忆：最近 4 句
    c.execute("SELECT role, content FROM chat_history WHERE character = ? ORDER BY id DESC LIMIT 4", (char_id,))
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    
    # 长期记忆检索（RAG）
    long_term_memory = ""
    try:
        # 把用户的话变成向量
        q_res = client_embed.embeddings.create(model="embedding-3", input=request.user_message)
        q_vector = q_res.data[0].embedding
        
        # 去记忆库里搜最相关的 2 条记录
        results = collection.query(query_embeddings=[q_vector], n_results=2)
        if results['documents'] and results['documents'][0]:
            memory_texts = [f"用户曾说：{doc}" for doc in results['documents'][0]]
            long_term_memory = "\n【历史记忆】\n" + "\n".join(memory_texts)
    except Exception as e:
        print(f"长期记忆检索失败: {e}")
    
    # 拼接 Prompt
    messages = [{"role": "system", "content": char_config["prompt"] + long_term_memory}]
    for role, content in rows:
        messages.append({"role": role, "content": content})
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名称"}},
                    "required": ["city"]
                }
            }
        }
    ]
    
    try:
        response = client_chat.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            tools=tools,
            temperature=0.8
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            print(f"\n【系统日志】触发工具调用: {message.tool_calls[0].function.name}")
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
                model="deepseek-reasoner",
                messages=messages,
                temperature=0.8
            )
            reply = final_response.choices[0].message.content
        else:
            reply = message.content
            
        aff_change = 5 if "【" in request.user_message else 0
        new_affection = max(0, min(100, current_affection + aff_change))
        
        conn = sqlite3.connect('memory.db')
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (character, role, content, timestamp, affection) VALUES (?, ?, ?, ?, ?)", 
                  (char_id, "assistant", reply, now, new_affection))
        conn.commit()
        conn.close()
        
        # 将这次互动存入长期记忆库
        try:
            memory_content = f"用户说：{request.user_message}，你回答：{reply}"
            m_res = client_embed.embeddings.create(model="embedding-3", input=memory_content)
            collection.add(
                documents=[memory_content],
                embeddings=[m_res.data[0].embedding],
                ids=[f"mem_{int(datetime.datetime.now().timestamp())}"]
            )
        except Exception as e:
            print(f"记忆存储失败: {e}")
        
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

@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename, media_type="audio/mpeg")
    return {"status": "error", "message": "音频不存在"}