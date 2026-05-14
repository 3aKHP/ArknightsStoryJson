#!/usr/bin/env python3
"""Generate LLM summaries for Arknights story chapters and events.

Reads story JSON files from zh_CN/gamedata/story/, calls the DeepSeek
V4 Flash API, and produces:

  zh_CN/summaries.json       — per-chapter summaries, 5~7:1 compression
  zh_CN/event_summaries.json — per-event summaries,  10:1 compression

Usage:
  python scripts/summarize.py --data-root zh_CN          # process directory
  python scripts/summarize.py --zip zh_CN.zip            # process zip in-place
  python scripts/summarize.py --zip zh_CN.zip --output-dir out/  # output to dir

Requires DEEPSEEK_API_KEY environment variable.
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
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_CONCURRENCY = 8
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds

STORY_REVIEW_TABLE = "zh_CN/gamedata/excel/story_review_table.json"
STORY_DIR = "zh_CN/gamedata/story"

SYSTEM_PROMPT = (
    "你是明日方舟官方剧情编辑，擅长将游戏对话提炼为精炼的叙事摘要。"
)

CHAPTER_PROMPT = """请将以下章节对话总结为一段连贯的中文摘要（5~7句话），保留关键情节转折、角色互动和情感变化。不要逐句翻译，要提炼核心叙事。

章节：{code} {name}
活动：{event_name}
标签：{tag}

{text}

摘要："""

EVENT_PROMPT = """请根据以下各章节摘要，为整个活动写一段200~300字的中文梗概，覆盖主线脉络、核心冲突、关键角色和结局。不要机械罗列，要形成自然流畅的叙事综述。

活动：{event_name}（共{total}章）

{chapter_summaries}

活动梗概："""

# ---------------------------------------------------------------------------
# Text extraction from raw story JSON
# ---------------------------------------------------------------------------

_RICH_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    text = text.replace("{@nickname}", "博士")
    text = _RICH_TAG_RE.sub("", text)
    return text.strip()


def extract_chapter_text(raw: dict) -> str:
    """Extract a readable text representation from a raw story chapter JSON."""
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
# API call
# ---------------------------------------------------------------------------

def _call_api(messages: list[dict], max_tokens: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")

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
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                return content.strip()
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

    raise RuntimeError(f"API call failed after {MAX_RETRIES + 1} attempts: {last_error}")


def summarize_chapter(code: str, name: str, event_name: str, tag: str, text: str) -> str:
    """Generate a per-chapter summary (5~7:1 compression)."""
    if not text.strip():
        return "（无对话内容）"

    prompt = CHAPTER_PROMPT.format(
        code=code,
        name=name,
        event_name=event_name,
        tag=tag or "无",
        text=text,
    )
    return _call_api(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
    )


def summarize_event(event_name: str, total: int, chapter_summaries: str) -> str:
    """Generate a per-event summary (10:1 compression)."""
    prompt = EVENT_PROMPT.format(
        event_name=event_name,
        total=total,
        chapter_summaries=chapter_summaries,
    )
    return _call_api(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_chapters(root: Path) -> list[dict]:
    """Discover all story chapters and return metadata list."""
    review_path = root / STORY_REVIEW_TABLE
    if not review_path.is_file():
        print(f"Warning: {STORY_REVIEW_TABLE} not found, scanning directory...")
        return _scan_chapters(root)

    review = _load_json(str(review_path))

    # Build event_id → event_name map
    event_names: dict[str, str] = {}
    for ev_id, entry in review.items():
        event_names[ev_id] = entry.get("name") or ev_id

    chapters: list[dict] = []
    for ev_id, entry in review.items():
        event_name = entry.get("name") or ev_id
        for d in sorted(
            entry.get("infoUnlockDatas") or [],
            key=lambda x: x.get("storySort", 0),
        ):
            story_key = d.get("storyTxt")
            if not story_key:
                continue
            story_path = root / STORY_DIR / f"{story_key}.json"
            if not story_path.is_file():
                continue
            chapters.append({
                "story_key": story_key,
                "story_path": str(story_path),
                "code": d.get("storyCode", ""),
                "name": d.get("storyName", ""),
                "tag": d.get("avgTag") or "",
                "event_id": ev_id,
                "event_name": event_name,
            })

    return chapters


def _scan_chapters(root: Path) -> list[dict]:
    """Fallback: scan story directory without review table."""
    story_root = root / STORY_DIR
    chapters: list[dict] = []
    for json_path in sorted(story_root.rglob("*.json")):
        rel = json_path.relative_to(story_root)
        story_key = str(rel.with_suffix("")).replace("\\", "/")
        raw = _load_json(str(json_path))
        chapters.append({
            "story_key": story_key,
            "story_path": str(json_path),
            "code": raw.get("storyCode", ""),
            "name": raw.get("storyName", ""),
            "tag": raw.get("avgTag") or "",
            "event_id": raw.get("eventid", ""),
            "event_name": raw.get("eventName", ""),
        })
    return chapters


def _run_chapter_summaries(chapters: list[dict]) -> dict[str, str]:
    """Generate per-chapter summaries with concurrent API calls."""
    summaries: dict[str, str] = {}
    total = len(chapters)
    done = 0

    def _process(ch: dict) -> tuple[str, str]:
        raw = _load_json(ch["story_path"])
        text = extract_chapter_text(raw)
        summary = summarize_chapter(
            ch["code"], ch["name"], ch["event_name"], ch["tag"], text,
        )
        return ch["story_key"], summary

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(_process, ch): ch for ch in chapters}
        for future in as_completed(futures):
            ch = futures[future]
            try:
                key, summary = future.result()
                summaries[key] = summary
                done += 1
                print(f"  [{done}/{total}] {ch['code']} {ch['name']} ({len(summary)} chars)")
            except Exception as exc:
                print(f"  [{done}/{total}] {ch['code']} {ch['name']} FAILED: {exc}", file=sys.stderr)

    return summaries


def _run_event_summaries(
    chapters: list[dict],
    chapter_summaries: dict[str, str],
) -> dict[str, str]:
    """Generate per-event summaries from chapter summaries."""
    # Group chapters by event
    events: dict[str, dict] = {}
    for ch in chapters:
        ev_id = ch["event_id"]
        if ev_id not in events:
            events[ev_id] = {
                "event_name": ch["event_name"],
                "chapters": [],
            }
        summary = chapter_summaries.get(ch["story_key"], "")
        events[ev_id]["chapters"].append({
            "code": ch["code"],
            "name": ch["name"],
            "summary": summary,
        })

    event_summaries: dict[str, str] = {}
    total = len(events)
    done = 0

    for ev_id, ev_data in sorted(events.items()):
        ch_list = ev_data["chapters"]
        if len(ch_list) <= 1:
            # Single-chapter events: use the chapter summary as event summary
            event_summaries[ev_id] = ch_list[0]["summary"] if ch_list else ""
            done += 1
            continue

        parts = []
        for ch_info in ch_list:
            parts.append(f"{ch_info['code']} {ch_info['name']}：{ch_info['summary']}")

        try:
            summary = summarize_event(
                ev_data["event_name"],
                len(ch_list),
                "\n\n".join(parts),
            )
            event_summaries[ev_id] = summary
            done += 1
            print(f"  [{done}/{total}] Event: {ev_data['event_name']} ({len(ch_list)} chapters, {len(summary)} chars)")
        except Exception as exc:
            print(f"  [{done}/{total}] Event: {ev_data['event_name']} FAILED: {exc}", file=sys.stderr)

    return event_summaries


def _inject_into_zip(zip_path: Path, summaries: dict, event_summaries: dict) -> None:
    """Add summaries.json and event_summaries.json into an existing zip."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    try:
        with zipfile.ZipFile(zip_path, "r") as zin:
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if item.filename in ("zh_CN/summaries.json", "zh_CN/event_summaries.json"):
                        continue
                    zout.writestr(item, zin.read(item.filename))
                zout.writestr(
                    "zh_CN/summaries.json",
                    json.dumps(summaries, ensure_ascii=False, indent=2),
                )
                zout.writestr(
                    "zh_CN/event_summaries.json",
                    json.dumps(event_summaries, ensure_ascii=False, indent=2),
                )
        Path(tmp.name).replace(zip_path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM story summaries")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-root", type=Path, help="Path to zh_CN/ data directory")
    group.add_argument("--zip", type=Path, help="Path to zh_CN.zip (modified in-place)")
    parser.add_argument("--output-dir", type=Path, help="Write output files here instead")
    parser.add_argument("--chapters-only", action="store_true", help="Skip event summaries")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    # Resolve data root
    if args.zip:
        if not args.output_dir:
            args.output_dir = args.zip.parent
        import tempfile
        extract_dir = tempfile.mkdtemp(prefix="storyjson_")
        print(f"Extracting {args.zip} to {extract_dir}...")
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(extract_dir)
        data_root = Path(extract_dir)
    else:
        data_root = args.data_root

    # Discover chapters
    chapters = _iter_chapters(data_root)
    print(f"Found {len(chapters)} chapters in {len(set(ch['event_id'] for ch in chapters))} events.")

    # Per-chapter summaries
    print("\n--- Chapter summaries ---")
    chapter_summaries = _run_chapter_summaries(chapters)

    # Write chapter summaries
    summaries_path = args.output_dir / "summaries.json"
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    summaries_path.write_text(
        json.dumps(chapter_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nChapter summaries written to {summaries_path}")

    # Per-event summaries
    event_summaries: dict[str, str] = {}
    if not args.chapters_only:
        print("\n--- Event summaries ---")
        event_summaries = _run_event_summaries(chapters, chapter_summaries)

        event_path = args.output_dir / "event_summaries.json"
        event_path.write_text(
            json.dumps(event_summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Event summaries written to {event_path}")

    # Inject into zip if requested
    if args.zip:
        print(f"\nInjecting summaries into {args.zip}...")
        _inject_into_zip(args.zip, chapter_summaries, event_summaries)
        print("Done.")
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
