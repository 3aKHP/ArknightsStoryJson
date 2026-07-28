#!/usr/bin/env python3
"""Validate StoryJson release assets and bind the final zip to its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile


REVIEW_TABLE = "zh_CN/gamedata/excel/story_review_table.json"
STORY_INFO = "zh_CN/storyinfo.json"
CONTRACT_VERSION = "prts-story-v1"
MAX_COUNT_DROP_FRACTION = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(data: bytes, name: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON entry {name}: {exc}") from exc


def inspect_story_zip(zip_path: Path) -> dict[str, int | str]:
    """Validate the consumer contract and return stable release metrics."""
    try:
        with ZipFile(zip_path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError("zip contains duplicate entries")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe zip entry: {name}")

            missing = [name for name in (REVIEW_TABLE, STORY_INFO) if name not in names]
            if missing:
                raise ValueError(f"missing required entries: {missing}")

            review_bytes = archive.read(REVIEW_TABLE)
            review = _load_json(review_bytes, REVIEW_TABLE)
            story_info = _load_json(archive.read(STORY_INFO), STORY_INFO)
            if not isinstance(review, dict) or not review:
                raise ValueError("story_review_table.json must be a non-empty object")
            if not isinstance(story_info, dict):
                raise ValueError("storyinfo.json must be an object")

            referenced: set[str] = set()
            for event_id, event in review.items():
                if not isinstance(event, dict):
                    raise ValueError(f"event {event_id!r} must be an object")
                unlocks = event.get("infoUnlockDatas") or []
                if not isinstance(unlocks, list):
                    raise ValueError(f"event {event_id!r} infoUnlockDatas must be a list")
                for unlock in unlocks:
                    if not isinstance(unlock, dict):
                        raise ValueError(f"event {event_id!r} contains an invalid unlock")
                    story_key = unlock.get("storyTxt")
                    if story_key:
                        referenced.add(f"zh_CN/gamedata/story/{story_key}.json")

            missing_stories = sorted(referenced.difference(names))
            if missing_stories:
                raise ValueError(
                    f"missing {len(missing_stories)} referenced stories; "
                    f"sample={missing_stories[:10]}"
                )

            json_files = 0
            for name in names:
                if name.endswith(".json"):
                    _load_json(archive.read(name), name)
                    json_files += 1
    except BadZipFile as exc:
        raise ValueError(f"invalid zip file: {exc}") from exc

    return {
        "zip_sha256": sha256_file(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "story_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "events": len(review),
        "chapters": len(referenced),
        "json_files": json_files,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def check_regression(metrics: dict[str, int | str], previous: dict[str, Any] | None) -> None:
    if not previous:
        return
    regressions: list[str] = []
    for field in ("events", "chapters", "json_files"):
        old = previous.get(field)
        new = metrics[field]
        if isinstance(old, int) and old > 0 and isinstance(new, int):
            minimum = int(old * (1 - MAX_COUNT_DROP_FRACTION))
            if new < minimum:
                regressions.append(
                    f"{field} regressed from {old} to {new} (minimum {minimum})"
                )
    if regressions:
        raise ValueError("regressions: " + "; ".join(regressions))


def finalize_manifest(
    zip_path: Path,
    manifest_path: Path,
    previous_manifest: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    previous = _load_manifest(previous_manifest) if previous_manifest and previous_manifest.is_file() else None
    metrics = inspect_story_zip(zip_path)
    check_regression(metrics, previous)
    manifest.update(metrics)
    manifest.update(
        {
            "schema_version": 2,
            "contract_version": CONTRACT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_manifest(
    zip_path: Path,
    manifest_path: Path,
    previous_manifest: Path | None = None,
) -> None:
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("manifest schema_version must be 2")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"manifest contract_version must be {CONTRACT_VERSION}")
    metrics = inspect_story_zip(zip_path)
    for field, actual in metrics.items():
        if manifest.get(field) != actual:
            raise ValueError(
                f"manifest {field} mismatch: expected {manifest.get(field)!r}, actual {actual!r}"
            )
    previous = _load_manifest(previous_manifest) if previous_manifest and previous_manifest.is_file() else None
    check_regression(metrics, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--zip", type=Path, required=True)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--previous-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "finalize":
        finalize_manifest(args.zip, args.manifest, args.previous_manifest)
    else:
        verify_manifest(args.zip, args.manifest, args.previous_manifest)


if __name__ == "__main__":
    main()
