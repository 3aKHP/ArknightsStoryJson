from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.release_gate import check_regression, finalize_manifest, verify_manifest


def write_story_zip(path: Path, *, chapters: int = 2, invalid_json: bool = False) -> None:
    review = {
        "event": {
            "infoUnlockDatas": [
                {"storyTxt": f"story_{index}"} for index in range(chapters)
            ]
        }
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "zh_CN/gamedata/excel/story_review_table.json",
            json.dumps(review),
        )
        archive.writestr("zh_CN/storyinfo.json", "{}")
        for index in range(chapters):
            content = "{" if invalid_json and index == 0 else "{}"
            archive.writestr(f"zh_CN/gamedata/story/story_{index}.json", content)


class ReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.zip_path = self.root / "zh_CN.zip"
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text('{"source":"primary"}\n', encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_finalize_binds_post_processed_zip(self) -> None:
        write_story_zip(self.zip_path)
        manifest = finalize_manifest(self.zip_path, self.manifest_path)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["chapters"], 2)
        verify_manifest(self.zip_path, self.manifest_path)

        tampered = dict(manifest)
        tampered["story_review_sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "story_review_sha256 mismatch"):
            verify_manifest(self.zip_path, self.manifest_path)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with ZipFile(self.zip_path, "a") as archive:
            archive.writestr("zh_CN/summaries.json", "{}")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.zip_path, self.manifest_path)

    def test_invalid_json_is_rejected(self) -> None:
        write_story_zip(self.zip_path, invalid_json=True)
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            finalize_manifest(self.zip_path, self.manifest_path)

    def test_large_chapter_regression_is_rejected(self) -> None:
        write_story_zip(self.zip_path, chapters=2)
        previous = self.root / "previous.json"
        previous.write_text(
            json.dumps({"events": 1, "chapters": 10, "json_files": 12}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "chapters regressed"):
            finalize_manifest(self.zip_path, self.manifest_path, previous)

    def test_each_release_metric_regression_is_rejected(self) -> None:
        for field in ("events", "chapters", "json_files"):
            with self.subTest(field=field):
                metrics = {"events": 10, "chapters": 10, "json_files": 10}
                previous = {"events": 10, "chapters": 10, "json_files": 10}
                previous[field] = 20
                with self.assertRaisesRegex(ValueError, rf"{field} regressed"):
                    check_regression(metrics, previous)

    def test_all_regressions_are_reported_together(self) -> None:
        metrics = {"events": 1, "chapters": 2, "json_files": 4}
        previous = {"events": 10, "chapters": 20, "json_files": 40}
        with self.assertRaises(ValueError) as raised:
            check_regression(metrics, previous)
        message = str(raised.exception)
        for field in metrics:
            self.assertIn(f"{field} regressed", message)


if __name__ == "__main__":
    unittest.main()
