from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from minimind_v_bound.data.caption_dataset import (
    CanonicalCaptionEncoder,
    LockedTrainCaptionDataset,
    load_training_example_index,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("upstream/minimind-v/model")


def test_canonical_prompt_and_label_mask(tokenizer) -> None:
    encoder = CanonicalCaptionEncoder(tokenizer, max_sequence_length=128)
    caption = "A red car on a road."
    encoded = encoder.encode(caption)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert encoded.input_ids.tolist().count(image_token_id) == 64
    labeled = encoded.labels[encoded.labels != -100]
    assert labeled[-1].item() == tokenizer.eos_token_id
    decoded = tokenizer.decode(labeled[:-1], skip_special_tokens=False)
    assert decoded == caption
    assert encoded.valid_label_count == len(labeled)
    assert torch.all(encoded.labels[encoded.attention_mask == 0] == -100)


def test_training_index_uses_only_manifest_clusters(tmp_path: Path) -> None:
    database = tmp_path / "clusters.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE images(image_sha256 TEXT PRIMARY KEY, width INTEGER, height INTEGER);
            CREATE TABLE examples(
                row_index INTEGER PRIMARY KEY,
                image_sha256 TEXT,
                raw_image_sha256 TEXT,
                captions_json TEXT
            );
            """
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, 1, 1)", [("a" * 64,), ("b" * 64,)]
        )
        connection.executemany(
            "INSERT INTO examples VALUES (?, ?, ?, ?)",
            [
                (0, "a" * 64, "0" * 64, json.dumps(["one", "two"])),
                (1, "b" * 64, "1" * 64, json.dumps(["locked test caption"])),
            ],
        )
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(
        json.dumps({"cluster_sha256": "a" * 64}) + "\n", encoding="utf-8"
    )
    rows, captions = load_training_example_index(database, train_manifest)
    assert rows.tolist() == [0, 0]
    assert captions.tolist() == [0, 1]


def test_training_dataset_rejects_locked_test_manifest(tokenizer) -> None:
    with pytest.raises(ValueError, match="locked-test"):
        LockedTrainCaptionDataset(
            parquet_path=Path("not-read.parquet"),
            cluster_database_path=Path("not-read.sqlite3"),
            train_manifest_path=Path("dataset/locked_test/test_clusters.jsonl"),
            tokenizer=tokenizer,
            image_processor=object(),
        )
