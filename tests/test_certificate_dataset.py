from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from minimind_v_bound.certificate.dataset import (
    build_certificate_caption_records,
)


def make_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE examples (
                row_index INTEGER PRIMARY KEY,
                image_sha256 TEXT NOT NULL,
                captions_json TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO examples VALUES (?, ?, ?)",
            [
                (8, "b" * 64, json.dumps(["b-first"])),
                (5, "a" * 64, json.dumps(["a-first", "a-second"])),
                (2, "a" * 64, json.dumps(["a-zero"])),
            ],
        )


def test_certificate_records_are_cluster_then_row_then_caption_ordered(
    tmp_path: Path,
) -> None:
    database = tmp_path / "clusters.sqlite3"
    make_database(database)
    records = build_certificate_caption_records(
        database_path=database, selected_cluster_hashes=["b" * 64, "a" * 64]
    )

    assert [record.caption for record in records] == [
        "b-first",
        "a-zero",
        "a-first",
        "a-second",
    ]
    assert [record.cluster_position for record in records] == [0, 1, 1, 1]
    assert [record.representative_row_index for record in records] == [8, 2, 2, 2]
    assert [record.caption_position for record in records] == list(range(4))


def test_certificate_records_reject_missing_or_duplicate_selection(tmp_path: Path) -> None:
    database = tmp_path / "clusters.sqlite3"
    make_database(database)
    with pytest.raises(ValueError, match="unique"):
        build_certificate_caption_records(
            database_path=database,
            selected_cluster_hashes=["a" * 64, "a" * 64],
        )
    with pytest.raises(RuntimeError, match="missing"):
        build_certificate_caption_records(
            database_path=database, selected_cluster_hashes=["c" * 64]
        )
