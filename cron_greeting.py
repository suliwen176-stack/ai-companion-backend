"""每日早安任务：由 Render Cron Job 在 UTC 00:00（北京时间 08:00）执行一次。

只通过 HTTP 调用后端接口，不需要数据库驱动，部署最轻。
本地自测：  python cron_greeting.py
"""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "https://saki-ai-backend.onrender.com").rstrip("/")
CRON_SECRET = os.getenv("CRON_SECRET", "")


def main():
    url = f"{BACKEND_URL}/api/cron/morning-greeting"
    try:
        resp = requests.post(url, params={"secret": CRON_SECRET}, timeout=90)
        print(f"[cron] HTTP {resp.status_code} -> {resp.text[:300]}")
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            print(f"[cron] 后端返回失败: {body}")
            sys.exit(1)
        print(f"[cron] ✅ 早安已写入: {body.get('greeting')}")
        refl = body.get("reflection") or {}
        print(f"[cron] 每日反思: {refl}")
    except Exception as e:
        print(f"[cron] ❌ 执行失败: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
