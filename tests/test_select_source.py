from __future__ import annotations

import unittest

from scripts.select_source import select_source


class SelectSourceTests(unittest.TestCase):
    def test_changed_current_primary_wins(self) -> None:
        self.assertEqual(
            select_source(
                story_tree="new-tree",
                story_review="review-b",
                gamedata_review="review-b",
                release_tree="old-tree",
                release_review="review-a",
            ),
            "primary",
        )

    def test_stale_primary_goes_directly_to_secondary(self) -> None:
        self.assertEqual(
            select_source(
                story_tree="new-tree",
                story_review="review-a",
                gamedata_review="review-b",
                release_tree="old-tree",
                release_review="review-a",
            ),
            "secondary",
        )

    def test_stale_primary_does_not_replace_current_secondary(self) -> None:
        self.assertEqual(
            select_source(
                story_tree="new-tree",
                story_review="review-a",
                gamedata_review="review-b",
                release_tree="old-tree",
                release_review="review-b",
            ),
            "none",
        )

    def test_forced_stale_primary_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing forced primary"):
            select_source(
                story_tree="new-tree",
                story_review="review-a",
                gamedata_review="review-b",
                release_tree="old-tree",
                release_review="review-b",
                force="primary",
            )


if __name__ == "__main__":
    unittest.main()
