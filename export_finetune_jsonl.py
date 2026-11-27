#!/usr/bin/env python3
"""导出 LoRA 微调训练集（JSONL, chat 格式）。

数据来源：
  1. style_samples.json —— 手工精选的风格样本（权重最高，全部收录）
  2. Supabase chat_history —— 真实聊天记录，可用 --min-rating 等条件筛选

用法：
  python export_finetune_jsonl.py                      # 导出到 stdout 预览
  python export_finetune_jsonl.py -o train.jsonl       # 导出文件
  python export_finetune_jsonl.py --days 30 -o recent.jsonl
  python export_finetune_jsonl.py --ids 004,017,023    # 只导指定 id

训练格式（兼容 SiliconFlow / 百炼 LoRA）：
  {"messages": [{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}]}
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = "你是用户的数字伙伴。语气温柔、克制、有分寸；先接住情绪，再（在对方需要时）给建议。"


def load_style_samples(only_ids=None):
    path = os.path.join(HERE, "style_samples.json")
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    out = []
    for s in samples:
        sid = str(s.get("id", ""))
        if only_ids and sid not in only_ids:
            continue
        out.append({
            "id": sid,
            "category": s.get("category", ""),
            "user": s.get("user_zh") or s.get("user", ""),
            "assistant": s.get("assistant_zh") or s.get("assistant", ""),
        })
    return out


def load_chat_history(days=None):
    """从 Supabase 读取真实对话（user/assistant 成对的连续记录）"""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, ".env"), override=True)
        sys.path.insert(0, HERE)
        import chat_backend as cb
    except Exception as e:
        print(f"[warn] 无法连接数据库，跳过聊天记录: {e}", file=sys.stderr)
        return []
    db = cb.SessionLocal()
    try:
        q = db.query(cb.ChatHistory).filter(
            cb.ChatHistory.role.in_(["user", "assistant"]))
        if days:
            cutoff = __import__("datetime").datetime.utcnow() - __import__(
                "datetime").timedelta(days=days)
            q = q.filter(cb.ChatHistory.timestamp >= cutoff)
        rows = q.order_by(cb.ChatHistory.id.asc()).all()
    finally:
        db.close()
    pairs, pending_user = [], None
    for r in rows:
        text = (r.content or "").strip()
        if not text or text.startswith("【早安】"):
            continue
        if r.role == "user":
            pending_user = text
        elif r.role == "assistant" and pending_user:
            pairs.append({"id": str(r.id), "category": "chat",
                          "user": pending_user, "assistant": text})
            pending_user = None
    return pairs


def to_jsonl(rows):
    lines = []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["assistant"]},
        ]
        lines.append(json.dumps({"messages": messages},
                                ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="导出 LoRA 微调 JSONL")
    ap.add_argument("-o", "--output", help="输出文件路径（默认打印到 stdout）")
    ap.add_argument("--days", type=int, help="只导最近 N 天的聊天记录")
    ap.add_argument("--ids", help="只导指定风格样本 id，逗号分隔，如 004,017")
    ap.add_argument("--no-chat", action="store_true",
                    help="不读取数据库聊天记录，只导 style_samples")
    args = ap.parse_args()

    only_ids = set(args.ids.split(",")) if args.ids else None
    rows = load_style_samples(only_ids)
    print(f"[info] 风格样本: {len(rows)} 条", file=sys.stderr)
    if not args.no_chat:
        chat_rows = load_chat_history(args.days)
        print(f"[info] 聊天记录对: {len(chat_rows)} 条", file=sys.stderr)
        rows += chat_rows

    if not rows:
        print("没有可导出的数据", file=sys.stderr)
        sys.exit(1)
    print(f"[info] 合计 {len(rows)} 条训练样本", file=sys.stderr)
    data = to_jsonl(rows)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"✅ 已写入 {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(data)


if __name__ == "__main__":
    main()
