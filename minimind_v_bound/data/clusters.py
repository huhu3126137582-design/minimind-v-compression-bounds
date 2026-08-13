from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps


NORMALIZATION_VERSION = "exif-transpose_rgb_width-height-pixels_v1"


@dataclass(frozen=True)
class NormalizedImage:
    sha256: str
    raw_sha256: str
    width: int
    height: int


def normalize_and_hash_image(image_bytes: bytes) -> NormalizedImage:
    """Hash the preregistered decoded-pixel representation of one image."""
    raw_sha256 = hashlib.sha256(image_bytes).hexdigest()
    with Image.open(io.BytesIO(image_bytes)) as source:
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        normalized.load()
        width, height = normalized.size
        payload = struct.pack(">II", width, height) + normalized.tobytes()
    return NormalizedImage(
        sha256=hashlib.sha256(payload).hexdigest(),
        raw_sha256=raw_sha256,
        width=width,
        height=height,
    )


def parse_conversations(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("conversations must decode to a list")
    if not all(isinstance(turn, dict) for turn in value):
        raise ValueError("every conversation turn must be an object")
    return value


def assistant_captions(value: Any) -> list[str]:
    captions = []
    for turn in parse_conversations(value):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            captions.append(content.strip())
    if not captions:
        raise ValueError("row has no non-empty assistant caption")
    return captions


def require_single_image(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 1:
            raise ValueError(f"caption row must contain exactly one image, got {len(value)}")
        item = value[0]
        if isinstance(item, (bytes, bytearray, memoryview)):
            return bytes(item)
    raise ValueError("image_bytes must be bytes or a one-element byte sequence")


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            image_sha256 TEXT PRIMARY KEY,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS examples (
            row_index INTEGER PRIMARY KEY,
            image_sha256 TEXT NOT NULL REFERENCES images(image_sha256),
            raw_image_sha256 TEXT NOT NULL,
            captions_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS examples_image_sha256_idx
            ON examples(image_sha256);
        CREATE TABLE IF NOT EXISTS rejected_rows (
            row_index INTEGER PRIMARY KEY,
            reason TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("normalization_version", NORMALIZATION_VERSION),
    )
    return connection


def iter_parquet_rows(path: Path, batch_size: int) -> Iterable[tuple[int, Any, Any]]:
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["image_bytes", "conversations"],
        use_threads=True,
    ):
        image_values = batch.column("image_bytes").to_pylist()
        conversation_values = batch.column("conversations").to_pylist()
        for image_value, conversations in zip(image_values, conversation_values, strict=True):
            yield row_index, image_value, conversations
            row_index += 1


def build_cluster_database(
    parquet_path: Path,
    database_path: Path,
    *,
    batch_size: int = 512,
    reject_invalid: bool = False,
) -> dict[str, int]:
    connection = connect_database(database_path)
    accepted = 0
    rejected = 0
    try:
        for row_index, image_value, conversation_value in iter_parquet_rows(
            parquet_path, batch_size
        ):
            try:
                raw = require_single_image(image_value)
                normalized = normalize_and_hash_image(raw)
                captions = assistant_captions(conversation_value)
            except Exception as error:
                if not reject_invalid:
                    raise RuntimeError(f"invalid row {row_index}: {error}") from error
                connection.execute(
                    "INSERT OR REPLACE INTO rejected_rows(row_index, reason) VALUES (?, ?)",
                    (row_index, f"{type(error).__name__}: {error}"),
                )
                rejected += 1
            else:
                connection.execute(
                    "INSERT OR IGNORE INTO images(image_sha256, width, height) VALUES (?, ?, ?)",
                    (normalized.sha256, normalized.width, normalized.height),
                )
                connection.execute(
                    """
                    INSERT INTO examples(
                        row_index, image_sha256, raw_image_sha256, captions_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        row_index,
                        normalized.sha256,
                        normalized.raw_sha256,
                        json.dumps(captions, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
                accepted += 1
            if (accepted + rejected) % batch_size == 0:
                connection.commit()
        connection.commit()
        clusters = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    finally:
        connection.close()
    return {"accepted_rows": accepted, "rejected_rows": rejected, "clusters": clusters}


def deterministic_cluster_split(
    cluster_hashes: Sequence[str], *, train_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    ordered = sorted(cluster_hashes)
    if len(ordered) != len(set(ordered)):
        raise ValueError("cluster hashes must be unique")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(ordered))
    train_size = int(np.floor(train_fraction * len(ordered)))
    train = [ordered[index] for index in permutation[:train_size]]
    test = [ordered[index] for index in permutation[train_size:]]
    return train, test

