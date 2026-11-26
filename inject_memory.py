import os
import datetime
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 连接智谱的向量化客户端
client_embed = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/")
db_client = chromadb.PersistentClient(path="./chroma_memory")
collection = db_client.get_or_create_collection(name="chat_memory")

# 我们要植入的记忆总结
memories = [
    "用户的学习进度报告：第一周掌握了 Python 基础语法、Git 提交、pandas 清理 Excel。第二周掌握了 FastAPI 构建后端、React 搭建前端、接入 DeepSeek 和智谱 API。",
    "踩坑记录：用户在本地跑项目时，经常遇到端口被占用（Address already in use）。解决方案是运行 lsof -i :端口号，然后 kill -9 进程。或者直接换一个新端口。",
    "踩坑记录：用户在部署到 Render 时，遇到了 ModuleNotFoundError: No module named 'fastapi'，原因是忘记在 requirements.txt 中写全依赖包。解决方案是检查 requirements.txt，或者在 Render 的 Build Command 中强制 pip install 所有包。",
    "踩坑记录：Live2D 模型加载失败（白屏），原因是模型路径写错，少写了一层目录，导致 Vite 返回了 index.html 导致 JSON 解析失败。",
    "未来的学习计划：用户接下来想学习 TypeScript、Next.js，或者深入研究 RAG 的多路召回与重排序。"
]

print("正在将学习记忆注入外挂大脑...")

for i, text in enumerate(memories):
    # 将文本变成向量
    res = client_embed.embeddings.create(model="embedding-3", input=text)
    vector = res.data[0].embedding
    
    # 存入 ChromaDB，设置一个特殊的 ID 前缀
    collection.add(
        documents=[text],
        embeddings=[vector],
        ids=[f"memory_{int(datetime.datetime.now().timestamp())}_{i}"]
    )

print(f"✅ 成功注入 {len(memories)} 条学习记忆！")