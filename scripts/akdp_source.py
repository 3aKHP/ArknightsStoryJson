#!/usr/bin/env python3
"""Download-verify helper for consuming AKDP factory releases.

Shared by both compatibility repositories (ArknightsGameData and
ArknightsStoryJson).  Behaviour must stay identical across the two repos so
that the factory release consumption path is the same on both sides.

This module only handles factory provenance and byte-level verification.  The
compat repo's own release gate (``release_gate.py``) still owns metrics,
regression checks, and the final manifest schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile


FACTORY_REPO = "3aKHP/arknights-data-pipeline"
EXPECTED_CONTRACT_VERSION = "prts-mcp-data/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_factory_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load and structurally validate an AKDP ``manifest.json`` asset.

    Returns ``(manifest_dict, manifest_file_sha256)``.
    """
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"factory manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("factory manifest must be a JSON object")

    if manifest.get("contractVersion") != EXPECTED_CONTRACT_VERSION:
        raise ValueError(
            f"factory contractVersion is {manifest.get('contractVersion')!r}, "
            f"expected {EXPECTED_CONTRACT_VERSION!r}"
        )

    source = manifest.get("source")
    if not isinstance(source, dict) or not source.get("versionId"):
        raise ValueError("factory manifest missing source.versionId")

    pipeline = manifest.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline.get("commit"):
        raise ValueError("factory manifest missing pipeline.commit")

    assets = manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("factory manifest missing assets map")

    for name, record in assets.items():
        if not isinstance(record, dict):
            raise ValueError(f"factory manifest asset {name!r} must be an object")
        if not record.get("sha256") or not isinstance(record.get("size"), int):
            raise ValueError(
                f"factory manifest asset {name!r} must have sha256 and integer size"
            )

    return manifest, sha256_bytes(raw)


def verify_asset(path: Path, expected_sha256: str, expected_size: int) -> None:
    """Verify a downloaded asset matches the factory manifest hash and size."""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path.name} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"{path.name} sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )


def verify_zip(path: Path) -> None:
    """Run ``unzip -t`` equivalent plus path-safety and JSON-parse checks."""
    try:
        archive = ZipFile(path)
    except BadZipFile as exc:
        raise ValueError(f"invalid zip {path.name}: {exc}") from exc
    try:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError(f"{path.name} contains duplicate entries")
        for name in names:
            item = PurePosixPath(name)
            if item.is_absolute() or ".." in item.parts:
                raise ValueError(f"unsafe zip entry in {path.name}: {name}")
        for name in names:
            if name.endswith(".json"):
                data = archive.read(name)
                try:
                    json.loads(data)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSON entry {name} in {path.name}: {exc}"
                    ) from exc
    finally:
        archive.close()


def verify_assets(manifest: dict[str, Any], asset_map: dict[str, Path]) -> None:
    """Verify every provided asset against the factory manifest, then zip-check."""
    factory_assets = manifest["assets"]
    missing = set(asset_map) - set(factory_assets)
    if missing:
        raise ValueError(f"assets not declared in factory manifest: {sorted(missing)}")
    for name, path in asset_map.items():
        record = factory_assets[name]
        verify_asset(path, record["sha256"], record["size"])
    for path in asset_map.values():
        if path.suffix == ".zip":
            verify_zip(path)


def provenance_fields(
    manifest: dict[str, Any],
    manifest_sha256: str,
    compatibility_contract: str,
) -> dict[str, str]:
    """Build the factory-provenance block for the compat repo's own manifest."""
    version_id = manifest["source"]["versionId"]
    return {
        "source": FACTORY_REPO,
        "source_release": f"data-{version_id}",
        "source_version_id": version_id,
        "factory_manifest_sha256": manifest_sha256,
        "factory_pipeline_commit": manifest["pipeline"]["commit"],
        "compatibility_contract": compatibility_contract,
    }


def _parse_asset_spec(spec: str) -> tuple[str, Path]:
    name, sep, raw_path = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"--asset must be name=path, got {spec!r}"
        )
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="verify downloaded assets against a factory manifest"
    )
    verify_parser.add_argument("--factory-manifest", type=Path, required=True)
    verify_parser.add_argument(
        "--asset",
        action="append",
        required=True,
        type=_parse_asset_spec,
        help="name=path (e.g. zh_CN-excel.zip=/tmp/zh_CN-excel.zip)",
    )

    prov_parser = subparsers.add_parser(
        "provenance", help="emit factory-provenance fields as JSON"
    )
    prov_parser.add_argument("--factory-manifest", type=Path, required=True)
    prov_parser.add_argument("--contract", required=True)

    args = parser.parse_args()

    manifest, manifest_sha = load_factory_manifest(args.factory_manifest)

    if args.command == "verify":
        verify_assets(manifest, dict(args.asset))
        print(f"verified {len(args.asset)} asset(s) against factory manifest")
    elif args.command == "provenance":
        fields = provenance_fields(manifest, manifest_sha, args.contract)
        print(json.dumps(fields, indent=2))


if __name__ == "__main__":
    main()
