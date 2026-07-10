#!/usr/bin/env python3
"""Generate LLM summaries for Arknights story chapters and events.

Reads story JSON files from zh_CN/gamedata/story/, calls the DeepSeek
V4 Flash API, and produces:

  zh_CN/summaries.json       — per-chapter summaries, 5~7:1 compression
  zh_CN/event_summaries.json — per-event summaries,  10:1 compression

By default runs **incrementally**: existing summaries from --cache-zip /
--cache-dir / already-present files are reused; only missing chapter or
event keys trigger API calls.

Usage:
  python scripts/summarize.py --data-root zh_CN
  python scripts/summarize.py --zip zh_CN.zip --cache-zip prev/zh_CN.zip
  python scripts/summarize.py --zip zh_CN.zip --force   # full regenerate

Requires DEEPSEEK_API_KEY when there is work left after applying the cache.
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


def summarize_event_v2(event_name: str, full_text: str) -> str:
    """Generate a per-event summary from full chapter dialogue (V2)."""
    prompt = EVENT_PROMPT.format(
        event_name=event_name,
        full_text=full_text,
    )
    return _call_api(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
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


def _load_summary_maps_from_zip(zip_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load summaries.json / event_summaries.json from a previous package."""
    chapter: dict[str, str] = {}
    event: dict[str, str] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            if "zh_CN/summaries.json" in names:
                chapter = json.loads(zf.read("zh_CN/summaries.json").decode("utf-8"))
            if "zh_CN/event_summaries.json" in names:
                event = json.loads(zf.read("zh_CN/event_summaries.json").decode("utf-8"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Warning: failed to load cache zip {zip_path}: {exc}", file=sys.stderr)
        return {}, {}
    return (
        {k: v for k, v in chapter.items() if isinstance(k, str) and isinstance(v, str) and v.strip()},
        {k: v for k, v in event.items() if isinstance(k, str) and isinstance(v, str) and v.strip()},
    )


def _load_summary_maps_from_dir(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    chapter: dict[str, str] = {}
    event: dict[str, str] = {}
    for rel, target in (
        ("zh_CN/summaries.json", "chapter"),
        ("summaries.json", "chapter"),
        ("zh_CN/event_summaries.json", "event"),
        ("event_summaries.json", "event"),
    ):
        path = root / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"Warning: failed to load {path}: {exc}", file=sys.stderr)
            continue
        cleaned = {
            k: v for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()
        }
        if target == "chapter":
            chapter.update(cleaned)
        else:
            event.update(cleaned)
    return chapter, event


def _extract_chapter_text(ch: dict) -> str:
    raw = _load_json(ch["story_path"])
    return extract_chapter_text(raw)


def _run_chapter_summaries(
    chapters: list[dict],
    *,
    existing: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str], int]:
    """Generate missing chapter summaries.

    Returns (summaries, full_texts_for_dirty_events_only, api_calls).
    Cached chapters keep their summary; their full text is only extracted
    later when an event needs regeneration.
    """
    summaries: dict[str, str] = dict(existing or {})
    current_keys = {ch["story_key"] for ch in chapters}
    # Drop stale keys no longer present in the package.
    summaries = {k: v for k, v in summaries.items() if k in current_keys}

    pending = [ch for ch in chapters if ch["story_key"] not in summaries]
    reused = len(chapters) - len(pending)
    print(f"Chapter cache: reuse={reused} generate={len(pending)} total={len(chapters)}")

    full_texts: dict[str, str] = {}
    api_calls = 0
    if not pending:
        return summaries, full_texts, api_calls

    total = len(pending)
    done = 0

    def _process(ch: dict) -> tuple[str, str, str]:
        text = _extract_chapter_text(ch)
        summary = summarize_chapter(
            ch["code"], ch["name"], ch["event_name"], ch["tag"], text,
        )
        return ch["story_key"], summary, text

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {pool.submit(_process, ch): ch for ch in pending}
        for future in as_completed(futures):
            ch = futures[future]
            try:
                key, summary, text = future.result()
                summaries[key] = summary
                full_texts[key] = text
                api_calls += 1
                done += 1
                print(f"  [{done}/{total}] {ch['code']} {ch['name']} ({len(summary)} chars)")
            except Exception as exc:
                done += 1
                print(f"  [{done}/{total}] {ch['code']} {ch['name']} FAILED: {exc}", file=sys.stderr)

    return summaries, full_texts, api_calls


def _run_event_summaries_v2(
    chapters: list[dict],
    chapter_texts: dict[str, str],
    *,
    existing: dict[str, str] | None = None,
    dirty_event_ids: set[str] | None = None,
) -> tuple[dict[str, str], int]:
    """Generate per-event summaries from full chapter dialogue (V2).

    Reuses cached event summaries unless the event is marked dirty
    (new/changed chapters or missing cache entry).
    """
    event_summaries: dict[str, str] = dict(existing or {})
    current_events = {ch["event_id"] for ch in chapters if ch["event_id"]}
    event_summaries = {k: v for k, v in event_summaries.items() if k in current_events}

    events: dict[str, dict] = {}
    for ch in chapters:
        ev_id = ch["event_id"]
        if not ev_id:
            continue
        if ev_id not in events:
            events[ev_id] = {
                "event_name": ch["event_name"],
                "chapters": [],
                "total_chars": 0,
                "story_keys": [],
            }
        events[ev_id]["story_keys"].append(ch["story_key"])
        text = chapter_texts.get(ch["story_key"], "")
        if text:
            header = f"--- {ch['code']}"
            if ch.get("tag"):
                header += f" [{ch['tag']}]"
            header += f" {ch['name']} ---"
            events[ev_id]["chapters"].append(header + "\n" + text)
            events[ev_id]["total_chars"] += len(text)

    if dirty_event_ids is None:
        dirty_event_ids = set(events)
    else:
        dirty_event_ids = {e for e in dirty_event_ids if e in events}

    # Missing cache entries are always dirty.
    for ev_id in events:
        if ev_id not in event_summaries:
            dirty_event_ids.add(ev_id)

    reused = len(events) - len(dirty_event_ids)
    print(
        f"Event cache: reuse={reused} generate={len(dirty_event_ids)} total={len(events)}"
    )

    # Ensure dialogue text is available for dirty events (extract without LLM).
    key_to_ch = {ch["story_key"]: ch for ch in chapters}
    for ev_id in dirty_event_ids:
        for story_key in events[ev_id]["story_keys"]:
            if story_key in chapter_texts:
                continue
            ch = key_to_ch.get(story_key)
            if not ch:
                continue
            try:
                chapter_texts[story_key] = _extract_chapter_text(ch)
            except Exception as exc:  # noqa: BLE001
                print(f"  extract failed {story_key}: {exc}", file=sys.stderr)

        # Rebuild concatenated chapter text after fills.
        events[ev_id]["chapters"] = []
        events[ev_id]["total_chars"] = 0
        for story_key in events[ev_id]["story_keys"]:
            ch = key_to_ch[story_key]
            text = chapter_texts.get(story_key, "")
            if not text:
                continue
            header = f"--- {ch['code']}"
            if ch.get("tag"):
                header += f" [{ch['tag']}]"
            header += f" {ch['name']} ---"
            events[ev_id]["chapters"].append(header + "\n" + text)
            events[ev_id]["total_chars"] += len(text)

    api_calls = 0
    if not dirty_event_ids:
        return event_summaries, api_calls

    sorted_events = sorted(
        ((ev_id, events[ev_id]) for ev_id in dirty_event_ids),
        key=lambda x: len(x[1]["chapters"]),
        reverse=True,
    )
    total = len(sorted_events)
    done = 0

    def _process(ev_id: str, ev_data: dict) -> tuple[str, str, bool]:
        if len(ev_data["chapters"]) <= 1:
            text = ev_data["chapters"][0].split("\n", 1)[-1] if ev_data["chapters"] else ""
            return ev_id, text[:800] if len(text) > 800 else text, False
        full_text = "\n\n".join(ev_data["chapters"])
        summary = summarize_event_v2(ev_data["event_name"], full_text)
        return ev_id, summary, True

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = {
            pool.submit(_process, ev_id, ev_data): ev_id
            for ev_id, ev_data in sorted_events
        }
        for future in as_completed(futures):
            ev_id = futures[future]
            ev_data = events[ev_id]
            try:
                result_id, summary, used_api = future.result()
                event_summaries[result_id] = summary
                if used_api:
                    api_calls += 1
                done += 1
                print(
                    f"  [{done}/{total}] Event: {ev_data['event_name']} "
                    f"({len(ev_data['chapters'])} chapters, "
                    f"{ev_data['total_chars']:,} chars input → {len(summary)} chars)"
                )
            except Exception as exc:
                done += 1
                print(
                    f"  [{done}/{total}] Event: {ev_data['event_name']} FAILED: {exc}",
                    file=sys.stderr,
                )

    return event_summaries, api_calls


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
    parser.add_argument(
        "--cache-zip",
        type=Path,
        help="Previous zh_CN.zip providing summaries.json / event_summaries.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory containing previous summaries.json / event_summaries.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and regenerate all summaries",
    )
    args = parser.parse_args()

    # Resolve data root
    extract_dir: str | None = None
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
        if not args.output_dir:
            args.output_dir = data_root

    # Discover chapters
    chapters = _iter_chapters(data_root)
    print(
        f"Found {len(chapters)} chapters in "
        f"{len(set(ch['event_id'] for ch in chapters))} events."
    )

    # Load cache (unless --force)
    cached_chapters: dict[str, str] = {}
    cached_events: dict[str, str] = {}
    if not args.force:
        if args.cache_zip and args.cache_zip.is_file():
            print(f"Loading summary cache from zip: {args.cache_zip}")
            cached_chapters, cached_events = _load_summary_maps_from_zip(args.cache_zip)
        elif args.cache_dir and args.cache_dir.exists():
            print(f"Loading summary cache from dir: {args.cache_dir}")
            cached_chapters, cached_events = _load_summary_maps_from_dir(args.cache_dir)
        else:
            # Fall back to summaries already present in the package being processed.
            if args.zip:
                print(f"Loading summary cache from target zip (if any): {args.zip}")
                cached_chapters, cached_events = _load_summary_maps_from_zip(args.zip)
            else:
                cached_chapters, cached_events = _load_summary_maps_from_dir(data_root)
        print(
            f"Cache loaded: chapters={len(cached_chapters)} events={len(cached_events)}"
        )
    else:
        print("Force mode: ignoring summary cache")

    pending_chapter_keys = {
        ch["story_key"] for ch in chapters if ch["story_key"] not in cached_chapters
    }
    dirty_event_ids = {
        ch["event_id"] for ch in chapters
        if ch["event_id"] and (
            ch["story_key"] in pending_chapter_keys or ch["event_id"] not in cached_events
        )
    }
    # Multi-chapter dirty events always need an API call (single-chapter events are free).
    events_needing_api = set()
    if not args.chapters_only:
        from collections import Counter
        counts = Counter(ch["event_id"] for ch in chapters if ch["event_id"])
        events_needing_api = {e for e in dirty_event_ids if counts.get(e, 0) > 1}

    needs_api = bool(pending_chapter_keys) or bool(events_needing_api)
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))

    if not needs_api:
        print("Cache covers all chapters/events — skipping LLM calls.")
    elif not has_key:
        print(
            "Warning: DEEPSEEK_API_KEY not set; shipping partial cache only "
            f"(missing chapters={len(pending_chapter_keys)}, "
            f"events needing API={len(events_needing_api)}).",
            file=sys.stderr,
        )

    # Per-chapter summaries
    print("\n--- Chapter summaries ---")
    if has_key or not pending_chapter_keys:
        chapter_summaries, chapter_texts, chapter_api = _run_chapter_summaries(
            chapters, existing=cached_chapters,
        )
    else:
        # Cache-only path: keep what we have, do not call API.
        current_keys = {ch["story_key"] for ch in chapters}
        chapter_summaries = {k: v for k, v in cached_chapters.items() if k in current_keys}
        chapter_texts = {}
        chapter_api = 0
        print(
            f"Chapter cache-only: kept={len(chapter_summaries)} "
            f"missing={len(pending_chapter_keys)}"
        )

    summaries_path = args.output_dir / "summaries.json"
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    summaries_path.write_text(
        json.dumps(chapter_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nChapter summaries written to {summaries_path}")

    # Per-event summaries (V2 — from full dialogue)
    event_summaries: dict[str, str] = {}
    event_api = 0
    if not args.chapters_only:
        print("\n--- Event summaries (V2: full dialogue) ---")
        if has_key:
            event_summaries, event_api = _run_event_summaries_v2(
                chapters,
                chapter_texts,
                existing=cached_events,
                dirty_event_ids=None if args.force else dirty_event_ids,
            )
        else:
            current_events = {ch["event_id"] for ch in chapters if ch["event_id"]}
            event_summaries = {
                k: v for k, v in cached_events.items() if k in current_events
            }
            print(
                f"Event cache-only: kept={len(event_summaries)} "
                f"dirty={len(dirty_event_ids)}"
            )

        event_path = args.output_dir / "event_summaries.json"
        event_path.write_text(
            json.dumps(event_summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Event summaries written to {event_path}")

    print(
        f"\nAPI usage: chapter_calls={chapter_api} event_calls={event_api} "
        f"total={chapter_api + event_api}"
    )

    # Inject into zip if requested
    if args.zip:
        print(f"\nInjecting summaries into {args.zip}...")
        _inject_into_zip(args.zip, chapter_summaries, event_summaries)
        print("Done.")
        import shutil
        if extract_dir:
            shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
