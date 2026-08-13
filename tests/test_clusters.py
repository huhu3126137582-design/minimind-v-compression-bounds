from __future__ import annotations

import io
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from minimind_v_bound.data.clusters import (
    assistant_captions,
    build_cluster_database,
    deterministic_cluster_split,
    normalize_and_hash_image,
    require_single_image,
)


def image_bytes(format_name: str, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color=color).save(buffer, format=format_name)
    return buffer.getvalue()


def test_pixel_hash_ignores_lossless_container_encoding() -> None:
    png = image_bytes("PNG", (12, 34, 56))
    bmp = image_bytes("BMP", (12, 34, 56))
    assert normalize_and_hash_image(png).sha256 == normalize_and_hash_image(bmp).sha256
    assert normalize_and_hash_image(png).raw_sha256 != normalize_and_hash_image(bmp).raw_sha256


def test_different_pixels_have_different_hashes() -> None:
    first = image_bytes("PNG", (12, 34, 56))
    second = image_bytes("PNG", (12, 34, 57))
    assert normalize_and_hash_image(first).sha256 != normalize_and_hash_image(second).sha256


def test_caption_extraction_is_deterministic() -> None:
    conversations = [
        {"role": "user", "content": "ignored"},
        {"role": "assistant", "content": "  caption one  "},
        {"role": "assistant", "content": "caption two"},
    ]
    assert assistant_captions(conversations) == ["caption one", "caption two"]


def test_single_image_contract() -> None:
    value = image_bytes("PNG", (1, 2, 3))
    assert require_single_image(value) == value
    assert require_single_image([value]) == value


def test_split_is_reproducible_and_disjoint() -> None:
    hashes = [f"{number:064x}" for number in range(100)]
    first_train, first_test = deterministic_cluster_split(
        hashes, train_fraction=0.9, seed=20260814
    )
    second_train, second_test = deterministic_cluster_split(
        list(reversed(hashes)), train_fraction=0.9, seed=20260814
    )
    assert first_train == second_train
    assert first_test == second_test
    assert len(first_train) == 90
    assert len(first_test) == 10
    assert set(first_train).isdisjoint(first_test)
    assert set(first_train) | set(first_test) == set(hashes)


def test_parquet_to_cluster_database_groups_duplicate_pixels(tmp_path) -> None:
    first_png = image_bytes("PNG", (12, 34, 56))
    first_bmp = image_bytes("BMP", (12, 34, 56))
    second_png = image_bytes("PNG", (12, 34, 57))
    conversations = [
        json.dumps([
            {"role": "user", "content": "<image>"},
            {"role": "assistant", "content": f"caption {index}"},
        ])
        for index in range(3)
    ]
    parquet_path = tmp_path / "sample.parquet"
    database_path = tmp_path / "clusters.sqlite3"
    pq.write_table(
        pa.table({
            "image_bytes": [first_png, first_bmp, second_png],
            "conversations": conversations,
        }),
        parquet_path,
    )

    summary = build_cluster_database(parquet_path, database_path, batch_size=2)
    assert summary == {"accepted_rows": 3, "rejected_rows": 0, "clusters": 2}
    with sqlite3.connect(database_path) as connection:
        sizes = [row[0] for row in connection.execute(
            "SELECT COUNT(*) FROM examples GROUP BY image_sha256 ORDER BY COUNT(*)"
        )]
    assert sizes == [1, 2]
