from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import urllib.request
import ssl
import asyncio
import threading
import edge_tts
import requests
import time
from dotenv import load_dotenv
from moviepy import VideoFileClip, AudioFileClip

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 1. 文生图 =================
@app.get("/api/draw")
def generate_image(prompt: str):
    try:
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key: return {"status": "error", "message": "缺少 ZHIPU_API_KEY"}
        url = "https://open.bigmodel.cn/api/paas/v4/images/generations"
        headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
        payload = {"model": "cogview-3-flash", "prompt": prompt.strip()}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req, context=ssl._create_unverified_context()) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {"status": "success", "image_url": result['data'][0]['url']}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ================= 2. 语音合成流 =================
def run_tts_in_thread(text, filename):
    def target():
        async def _generate():
            voice = "zh-CN-YunxiNeural"
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filename)
        asyncio.run(_generate())
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

@app.get("/api/tts/audio")
def generate_and_serve_audio(text: str):
    audio_file = f"temp_{int(time.time())}.mp3"
    try:
        run_tts_in_thread(text, audio_file)
        return FileResponse(audio_file, media_type="audio/mpeg")
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ================= 3. 图生视频 =================
@app.get("/api/video")
def generate_video(prompt: str, image_url: str):
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key: return {"status": "error", "message": "缺少 DASHSCOPE_API_KEY"}
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json", "X-DashScope-Async": "enable"}
        payload = {"model": "wan2.6-i2v-flash", "input": {"prompt": prompt, "img_url": image_url}, "parameters": {"resolution": "720P", "duration": 5}}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        return {"status": "success", "task_info": response.json()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/video/status/{task_id}")
def check_video_status(task_id: str):
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        response = requests.get(url, headers=headers)
        return {"status": "success", "task_info": response.json()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ================= 4. 音视频合成 =================
@app.get("/api/merge")
def merge_video_audio(video_url: str, text: str, background_tasks: BackgroundTasks):
    try:
        video_path = f"vid_{int(time.time())}.mp4"
        audio_path = f"aud_{int(time.time())}.mp3"
        output_path = f"final_{int(time.time())}.mp4"

        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        run_tts_in_thread(text, audio_path)

        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        if video.duration < audio.duration:
            video = video.loop(duration=audio.duration)
        else:
            video = video.subclipped(0, audio.duration)

        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

        video.close()
        audio.close()
        final_video.close()

        def cleanup():
            time.sleep(5) 
            for f in [video_path, audio_path, output_path]:
                try: os.remove(f)
                except: pass
        
        background_tasks.add_task(cleanup)
        return FileResponse(output_path, media_type="video/mp4")
    except Exception as e:
        print(f"【合成报错】: {e}")
        return {"status": "error", "message": str(e)}