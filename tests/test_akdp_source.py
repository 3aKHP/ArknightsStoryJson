from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.akdp_source import (
    EXPECTED_CONTRACT_VERSION,
    load_factory_manifest,
    provenance_fields,
    sha256_file,
    verify_assets,
    verify_zip,
)


def _write_json_zip(path: Path, entries: dict[str, object]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, json.dumps(data))


def _make_factory_manifest(root: Path, assets: dict[str, Path]) -> bytes:
    """Build a minimal valid factory manifest matching the given asset files."""
    manifest = {
        "contractVersion": EXPECTED_CONTRACT_VERSION,
        "source": {"versionId": "test-version-001"},
        "pipeline": {"commit": "abc123def456"},
        "assets": {
            name: {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for name, path in assets.items()
        },
    }
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    return raw


class FactoryManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_manifest(self, manifest: object) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_load_valid_manifest_returns_dict_and_sha(self) -> None:
        asset = self.root / "data.zip"
        _write_json_zip(asset, {"zh_CN/excel/test.json": {}})
        raw = _make_factory_manifest(self.root, {"data.zip": asset})
        path = self.root / "manifest.json"
        manifest, sha = load_factory_manifest(path)
        self.assertEqual(manifest["source"]["versionId"], "test-version-001")
        self.assertEqual(sha, __import__("hashlib").sha256(raw).hexdigest())

    def test_wrong_contract_version_rejected(self) -> None:
        path = self._write_manifest(
            {
                "contractVersion": "wrong/v2",
                "source": {"versionId": "v"},
                "pipeline": {"commit": "c"},
                "assets": {"x": {"sha256": "a", "size": 1}},
            }
        )
        with self.assertRaisesRegex(ValueError, "contractVersion"):
            load_factory_manifest(path)

    def test_missing_version_id_rejected(self) -> None:
        path = self._write_manifest(
            {
                "contractVersion": EXPECTED_CONTRACT_VERSION,
                "source": {},
                "pipeline": {"commit": "c"},
                "assets": {"x": {"sha256": "a", "size": 1}},
            }
        )
        with self.assertRaisesRegex(ValueError, "versionId"):
            load_factory_manifest(path)

    def test_missing_pipeline_commit_rejected(self) -> None:
        path = self._write_manifest(
            {
                "contractVersion": EXPECTED_CONTRACT_VERSION,
                "source": {"versionId": "v"},
                "pipeline": {},
                "assets": {"x": {"sha256": "a", "size": 1}},
            }
        )
        with self.assertRaisesRegex(ValueError, "pipeline.commit"):
            load_factory_manifest(path)

    def test_empty_assets_rejected(self) -> None:
        path = self._write_manifest(
            {
                "contractVersion": EXPECTED_CONTRACT_VERSION,
                "source": {"versionId": "v"},
                "pipeline": {"commit": "c"},
                "assets": {},
            }
        )
        with self.assertRaisesRegex(ValueError, "assets"):
            load_factory_manifest(path)

    def test_invalid_json_rejected(self) -> None:
        path = self.root / "manifest.json"
        path.write_bytes(b"{not json")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            load_factory_manifest(path)


class VerifyAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_matching_assets_pass(self) -> None:
        zip_path = self.root / "data.zip"
        _write_json_zip(zip_path, {"zh_CN/excel/test.json": {"key": "value"}})
        _make_factory_manifest(self.root, {"data.zip": zip_path})
        manifest, _ = load_factory_manifest(self.root / "manifest.json")
        verify_assets(manifest, {"data.zip": zip_path})

    def test_sha256_mismatch_rejected(self) -> None:
        zip_path = self.root / "data.zip"
        _write_json_zip(zip_path, {"zh_CN/excel/test.json": {}})
        # Build manifest with correct size but wrong sha256.
        manifest = {
            "contractVersion": EXPECTED_CONTRACT_VERSION,
            "source": {"versionId": "v"},
            "pipeline": {"commit": "c"},
            "assets": {
                "data.zip": {
                    "sha256": "0" * 64,
                    "size": zip_path.stat().st_size,
                }
            },
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            verify_assets(manifest, {"data.zip": zip_path})

    def test_size_mismatch_rejected(self) -> None:
        zip_path = self.root / "data.zip"
        _write_json_zip(zip_path, {"zh_CN/excel/test.json": {}})
        manifest_path = self.root / "manifest.json"
        manifest = {
            "contractVersion": EXPECTED_CONTRACT_VERSION,
            "source": {"versionId": "v"},
            "pipeline": {"commit": "c"},
            "assets": {
                "data.zip": {
                    "sha256": sha256_file(zip_path),
                    "size": zip_path.stat().st_size + 999,
                }
            },
        }
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            verify_assets(manifest, {"data.zip": zip_path})

    def test_asset_not_in_manifest_rejected(self) -> None:
        zip_path = self.root / "data.zip"
        _write_json_zip(zip_path, {"zh_CN/excel/test.json": {}})
        _make_factory_manifest(self.root, {"data.zip": zip_path})
        manifest, _ = load_factory_manifest(self.root / "manifest.json")
        with self.assertRaisesRegex(ValueError, "not declared in factory manifest"):
            verify_assets(manifest, {"other.zip": zip_path})


class VerifyZipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_valid_zip_passes(self) -> None:
        path = self.root / "ok.zip"
        _write_json_zip(path, {"zh_CN/data.json": {"a": 1}})
        verify_zip(path)

    def test_corrupt_zip_rejected(self) -> None:
        path = self.root / "corrupt.zip"
        path.write_bytes(b"PK\x03\x04 not actually a zip")
        with self.assertRaisesRegex(ValueError, "invalid zip"):
            verify_zip(path)

    def test_unsafe_path_rejected(self) -> None:
        path = self.root / "unsafe.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("../escape.json", "{}")
        with self.assertRaisesRegex(ValueError, "unsafe zip entry"):
            verify_zip(path)

    def test_duplicate_entries_rejected(self) -> None:
        import warnings

        path = self.root / "dup.zip"
        _write_json_zip(path, {"zh_CN/a.json": {}})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with ZipFile(path, "a") as archive:
                archive.writestr("zh_CN/a.json", "{}")
        with self.assertRaisesRegex(ValueError, "duplicate entries"):
            verify_zip(path)

    def test_invalid_json_in_zip_rejected(self) -> None:
        path = self.root / "badjson.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("zh_CN/broken.json", "{not json")
        with self.assertRaisesRegex(ValueError, "invalid JSON entry"):
            verify_zip(path)


class ProvenanceFieldsTests(unittest.TestCase):
    def test_fields_match_spec(self) -> None:
        manifest = {
            "source": {"versionId": "26-08-03-23-34-20_a745fc"},
            "pipeline": {"commit": "f5fb4254ffcc6c6af8a55bdcc0be5403e9e9014e"},
        }
        fields = provenance_fields(manifest, "deadbeef" * 8, "prts-gamedata-v1")
        self.assertEqual(fields["source"], "3aKHP/arknights-data-pipeline")
        self.assertEqual(
            fields["source_release"], "data-26-08-03-23-34-20_a745fc"
        )
        self.assertEqual(
            fields["source_version_id"], "26-08-03-23-34-20_a745fc"
        )
        self.assertEqual(fields["factory_manifest_sha256"], "deadbeef" * 8)
        self.assertEqual(
            fields["factory_pipeline_commit"],
            "f5fb4254ffcc6c6af8a55bdcc0be5403e9e9014e",
        )
        self.assertEqual(fields["compatibility_contract"], "prts-gamedata-v1")


if __name__ == "__main__":
    unittest.main()
