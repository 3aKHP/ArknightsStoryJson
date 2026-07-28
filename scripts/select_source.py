#!/usr/bin/env python3
"""Choose a safe StoryJson release source from upstream fingerprints."""

from __future__ import annotations

import argparse
from typing import Literal


Source = Literal["none", "primary", "secondary"]


def select_source(
    *,
    story_tree: str,
    story_review: str,
    gamedata_review: str,
    release_tree: str,
    release_review: str,
    force: str = "auto",
) -> Source:
    """Return the source that can advance without publishing stale metadata."""
    if force not in {"auto", "primary", "secondary"}:
        raise ValueError(f"unknown force source: {force}")
    if not gamedata_review:
        raise ValueError("GameData story_review fingerprint must not be empty")
    if force == "secondary":
        return "secondary"
    if not story_review:
        raise ValueError("StoryJson story_review fingerprint must not be empty")

    primary_current = story_review == gamedata_review
    primary_changed = bool(story_tree) and story_tree != release_tree

    if force == "primary":
        if not primary_current:
            raise ValueError(
                "refusing forced primary release: its story_review differs from GameData"
            )
        return "primary"
    if primary_changed and primary_current:
        return "primary"
    if gamedata_review != release_review:
        return "secondary"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story-tree", default="")
    parser.add_argument("--story-review", required=True)
    parser.add_argument("--gamedata-review", required=True)
    parser.add_argument("--release-tree", default="")
    parser.add_argument("--release-review", default="")
    parser.add_argument("--force", choices=("auto", "primary", "secondary"), default="auto")
    args = parser.parse_args()
    print(
        select_source(
            story_tree=args.story_tree,
            story_review=args.story_review,
            gamedata_review=args.gamedata_review,
            release_tree=args.release_tree,
            release_review=args.release_review,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
