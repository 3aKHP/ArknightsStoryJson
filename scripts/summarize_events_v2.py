#!/usr/bin/env python3
"""V2 event summarization — feeds full chapter dialogue text to the LLM.

Unlike V1 (which summarized from per-chapter summaries), this script
passes the complete dialogue of every chapter in an event to the model,
allowing it to capture cross-chapter narrative arcs, character development,
and thematic through-lines directly.

Usage:
  DEEPSEEK_API_KEY=sk-... python scripts/summarize_events_v2.py \
    --data-root zh_CN --output event_summaries.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_CONCURRENCY = 6  # lower — each call has large input
MAX_RETRIES = 3
RETRY_DELAY = 2.0

STORY_REVIEW_TABLE = "zh_CN/gamedata/excel/story_review_table.json"
STORY_DIR = "zh_CN/gamedata/story"

SYSTEM_PROMPT = (
    "你是明日方舟官方剧情编辑，擅长将长篇游戏对话提炼为精炼流畅的叙事梗概。"
)

EVENT_PROMPT = """请将以下明日方舟活动「{event_name}」的完整剧情对话总结为一段300~500字的中文梗概。

要求：
- 覆盖主线脉络和核心冲突
- 突出关键角色的动机转变和重要抉择
- 捕捉章节之间的因果联系和伏笔照应
- 写出结局的情感和主题落点
- 用自然流畅的叙事语言，不要机械罗列章节

{full_text}

「{event_name}」梗概："""

# ---------------------------------------------------------------------------
# Text extraction (reused from summarize.py)
# ---------------------------------------------------------------------------

_RICH_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    text = text.replace("{@nickname}", "博士")
    text = _RICH_TAG_RE.sub("", text)
    return text.strip()


def extract_chapter_text(raw: dict) -> str:
    lines: list[str] = []
    for item in raw.get("storyList") or []:
        prop = (item.get("prop") or "").lower()
        attrs = item.get("attributes") or {}

        if prop == "name":
            name = attrs.get("name") or ""
            content = attrs.get("content") or ""
            if content:
                role = _clean(str(name)) if name else "？？？"
                lines.append(f"{role}：{_clean(str(content))}")
        elif prop in ("sticker", "subtitle", "animtext"):
            content = attrs.get("content") or attrs.get("text") or ""
            if content:
                lines.append(f"*{_clean(str(content))}*")
        elif prop == "decision":
            options = attrs.get("options") or []
            for opt in options:
                text = opt if isinstance(opt, str) else (opt.get("text") or "" if isinstance(opt, dict) else "")
                if text:
                    lines.append(f"【选项】{_clean(str(text))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _call_api(messages: list[dict], max_tokens: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            last_error = exc
            body_text = exc.read().decode(errors="replace")
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  HTTP {exc.code}, retrying in {wait}s... ({body_text[:200]})")
                time.sleep(wait)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  {exc}, retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"API call failed: {last_error}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_events(data_root: Path) -> list[dict]:
    """Build a list of events with their full chapter texts."""
    review_path = data_root / STORY_REVIEW_TABLE
    review = _load_json(str(review_path))

    events: list[dict] = []
    for ev_id, entry in sorted(review.items()):
        ev_name = entry.get("name") or ev_id
        datas = sorted(
            entry.get("infoUnlockDatas") or [],
            key=lambda x: x.get("storySort", 0),
        )

        chapter_texts: list[str] = []
        total_chars = 0

        for d in datas:
            story_key = d.get("storyTxt")
            if not story_key:
                continue
            story_path = data_root / STORY_DIR / f"{story_key}.json"
            if not story_path.is_file():
                continue

            raw = _load_json(str(story_path))
            text = extract_chapter_text(raw)
            if not text.strip():
                continue

            code = d.get("storyCode", "")
            name = d.get("storyName", "")
            tag = f"[{d.get('avgTag')}] " if d.get("avgTag") else ""

            chapter_texts.append(f"--- {code} {tag}{name} ---\n{text}")
            total_chars += len(text)

        if chapter_texts:
            events.append({
                "event_id": ev_id,
                "event_name": ev_name,
                "chapter_count": len(chapter_texts),
                "full_text": "\n\n".join(chapter_texts),
                "total_chars": total_chars,
            })

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 event summarization from full dialogue")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to zh_CN/ data dir")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--event-id", type=str, help="Process only this event (for testing)")
    args = parser.parse_args()

    events = _build_events(args.data_root)
    print(f"Found {len(events)} events")

    if args.event_id:
        events = [e for e in events if e["event_id"] == args.event_id]
        if not events:
            print(f"Event {args.event_id!r} not found")
            sys.exit(1)

    results: dict[str, str] = {}
    done = 0
    total = len(events)

    # Sort by chapter count descending — process big events first while concurrency is fresh
    events.sort(key=lambda e: e["chapter_count"], reverse=True)

    def _process(ev: dict) -> tuple[str, str]:
        prompt = EVENT_PROMPT.format(
            event_name=ev["event_name"],
            full_text=ev["full_text"],
        )
        summary = _call_api(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
        )
        return ev["event_id"], summary

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(_process, ev): ev for ev in events}
        for future in as_completed(futures):
            ev = futures[future]
            try:
                ev_id, summary = future.result()
                results[ev_id] = summary
                done += 1
                print(f"  [{done}/{total}] {ev['event_name']} ({ev['chapter_count']} chapters, "
                      f"{ev['total_chars']:,} chars input → {len(summary)} chars output)")
            except Exception as exc:
                print(f"  [{done}/{total}] {ev['event_name']} FAILED: {exc}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten {len(results)} event summaries to {args.output}")


if __name__ == "__main__":
    main()
