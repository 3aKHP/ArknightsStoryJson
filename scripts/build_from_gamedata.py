#!/usr/bin/env python3
"""Build zh_CN StoryJson zip from Kengxxiao GameData using ASTR-Script.

Layout expectations:
  --gamedata-root /path/to/ArknightsGameData   # contains zh_CN/gamedata/...
  --astr-root     /path/to/ASTR-Script         # contains jsonconvert.py, func.py

ASTR historically expects short lang codes (cn/en/...). This script creates a
temporary cn -> zh_CN symlink when needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


SRC_LANG = "zh_CN"
ASTR_LANG = "cn"


def _ensure_astr_lang_link(gamedata_root: Path) -> None:
    src = gamedata_root / SRC_LANG
    if not (src / "gamedata" / "excel" / "story_review_table.json").is_file():
        raise SystemExit(f"missing GameData under {src}")
    link = gamedata_root / ASTR_LANG
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(SRC_LANG)
    print(f"created symlink {link} -> {SRC_LANG}")


def _build_chardict(character_table: Path) -> dict[str, dict]:
    with character_table.open(encoding="utf-8") as f:
        character_data = json.load(f)
    char_dict: dict[str, dict] = {}
    for cid, cdata in character_data.items():
        parts = cid.split("_")
        if len(parts) >= 3 and parts[0] == "char":
            char_dict[parts[2]] = {"name": cdata.get("name", ""), "id": parts[1]}
    return char_dict


def build(
    *,
    gamedata_root: Path,
    astr_root: Path,
    output_dir: Path,
    output_zip: Path,
) -> dict:
    sys.path.insert(0, str(astr_root))
    import func  # noqa: WPS433
    from jsonconvert import reader  # noqa: WPS433

    _ensure_astr_lang_link(gamedata_root)
    src_root = gamedata_root / SRC_LANG
    base = gamedata_root / ASTR_LANG

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    src_excel = src_root / "gamedata" / "excel"
    dst_excel = output_dir / "gamedata" / "excel"
    shutil.copytree(src_excel, dst_excel)
    print(f"excel copied: {dst_excel}")

    events = func.getEvents(gamedata_root, ASTR_LANG)
    story_info: dict[str, str] = {}
    word_count: dict[str, dict[str, int]] = {}
    char_dict = _build_chardict(dst_excel / "character_table.json")
    ok = 0
    skipped = 0
    errors: list[str] = []

    for event in events:
        word_count.setdefault(event.eventid, {})
        for story in event:
            story_path = Path(story.storyTxt)
            if not story_path.is_file():
                skipped += 1
                continue
            rel_parent = story_path.relative_to(base).parent
            json_path = output_dir / rel_parent / f"{story_path.stem}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                story_json, counter = reader(story)
                story_info[str(story.f)] = story_json.get("storyInfo") or ""
                with json_path.open("w", encoding="utf-8") as jf:
                    json.dump(story_json, jf, ensure_ascii=False)
                word_count[event.eventid][str(story.f)] = counter
                ok += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{story.f}: {exc}")
                skipped += 1

    extra_info: dict = {"extra": []}
    try:
        for extra in func.getExtraAvg(gamedata_root, ASTR_LANG):
            story_path = Path(extra.storyTxt)
            if not story_path.is_file():
                continue
            rel_parent = story_path.relative_to(base).parent
            json_path = output_dir / rel_parent / f"{story_path.stem}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                story_json, _counter = reader(extra)
                story_info[str(extra.f)] = story_json.get("storyInfo") or ""
                with json_path.open("w", encoding="utf-8") as jf:
                    json.dump(story_json, jf, ensure_ascii=False)
                extra_info["extra"].append(
                    {"storyName": extra.storyName, "storyTxt": extra.f}
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"extra {extra.f}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"extra stories skipped: {exc}")

    (output_dir / "storyinfo.json").write_text(
        json.dumps(story_info, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "chardict.json").write_text(
        json.dumps(char_dict, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "wordcount.json").write_text(
        json.dumps(word_count, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "extrastory.json").write_text(
        json.dumps(extra_info, ensure_ascii=False), encoding="utf-8"
    )

    print(f"parsed ok={ok} skipped={skipped} errors={len(errors)}")
    if errors:
        print("sample errors:")
        for item in errors[:20]:
            print(" ", item)
        raise SystemExit(f"build failed with {len(errors)} parse errors")

    if output_zip.exists():
        output_zip.unlink()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                arc = Path("zh_CN") / path.relative_to(output_dir)
                zf.write(path, arcname=str(arc).replace("\\", "/"))
    print(f"zip written: {output_zip} ({output_zip.stat().st_size} bytes)")

    review = json.loads((dst_excel / "story_review_table.json").read_text(encoding="utf-8"))
    return {
        "chapters_ok": ok,
        "events": len(review),
        "has_act21mini": "act21mini" in review,
        "zip_bytes": output_zip.stat().st_size,
    }


def validate_zip(zip_path: Path) -> list[str]:
    """Minimal contract check aligned with PRTS-MCP story dataset needs."""
    required = (
        "zh_CN/gamedata/excel/story_review_table.json",
        "zh_CN/storyinfo.json",
    )
    missing: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for path in required:
            if path not in names:
                missing.append(path)
        if "zh_CN/gamedata/excel/story_review_table.json" in missing:
            return missing
        table = json.loads(zf.read("zh_CN/gamedata/excel/story_review_table.json"))
        for entry in table.values():
            for data in entry.get("infoUnlockDatas") or []:
                key = data.get("storyTxt")
                if not key:
                    continue
                path = f"zh_CN/gamedata/story/{key}.json"
                if path not in names:
                    missing.append(path)
    return missing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamedata-root", type=Path, required=True)
    parser.add_argument("--astr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write build manifest JSON",
    )
    parser.add_argument(
        "--gamedata-commit",
        default="",
        help="Upstream GameData commit recorded in manifest",
    )
    parser.add_argument(
        "--astr-ref",
        default="",
        help="ASTR-Script ref/commit recorded in manifest",
    )
    args = parser.parse_args()

    stats = build(
        gamedata_root=args.gamedata_root.resolve(),
        astr_root=args.astr_root.resolve(),
        output_dir=args.output_dir.resolve(),
        output_zip=args.output_zip.resolve(),
    )
    missing = validate_zip(args.output_zip.resolve())
    if missing:
        raise SystemExit(
            "validation failed, missing "
            f"{len(missing)} entries; sample={missing[:10]}"
        )
    print(f"validation ok ({stats['events']} events, {stats['chapters_ok']} chapters)")

    if args.manifest:
        review_path = (
            args.output_dir.resolve()
            / "gamedata"
            / "excel"
            / "story_review_table.json"
        )
        manifest = {
            "schema_version": 1,
            "source": "gamedata-astr",
            "gamedata_commit": args.gamedata_commit,
            "astr_ref": args.astr_ref,
            "story_review_sha256": sha256_file(review_path),
            "zip_sha256": sha256_file(args.output_zip.resolve()),
            "zip_bytes": stats["zip_bytes"],
            "events": stats["events"],
            "chapters": stats["chapters_ok"],
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"manifest written: {args.manifest}")


if __name__ == "__main__":
    main()
