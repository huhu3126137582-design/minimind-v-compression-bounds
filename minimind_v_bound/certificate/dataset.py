from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from minimind_v_bound.data.caption_dataset import CanonicalCaptionEncoder
from minimind_v_bound.data.clusters import assistant_captions, require_single_image


CERTIFICATE_DATASET_VERSION = "selected-cluster-all-captions-representative-image-v1"


@dataclass(frozen=True)
class CertificateCaptionRecord:
    caption_position: int
    cluster_position: int
    cluster_sha256: str
    representative_row_index: int
    source_row_index: int
    source_caption_index: int
    caption: str


def build_certificate_caption_records(
    *, database_path: Path, selected_cluster_hashes: list[str]
) -> list[CertificateCaptionRecord]:
    if not selected_cluster_hashes or len(selected_cluster_hashes) != len(
        set(selected_cluster_hashes)
    ):
        raise ValueError("selected certificate clusters must be non-empty and unique")
    rows_by_cluster: list[list[tuple[int, list[str]]]] = [
        [] for _ in selected_cluster_hashes
    ]
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        connection.execute(
            "CREATE TEMP TABLE selected(position INTEGER PRIMARY KEY, hash TEXT UNIQUE NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO selected(position, hash) VALUES (?, ?)",
            enumerate(selected_cluster_hashes),
        )
        cursor = connection.execute(
            """
            SELECT selected.position, examples.row_index, examples.captions_json
            FROM selected
            INNER JOIN examples ON examples.image_sha256 = selected.hash
            ORDER BY selected.position, examples.row_index
            """
        )
        for cluster_position, row_index, captions_json in cursor:
            captions = json.loads(captions_json)
            if not isinstance(captions, list) or not captions:
                raise RuntimeError("cluster database contains an invalid caption list")
            rows_by_cluster[cluster_position].append((row_index, captions))

    records: list[CertificateCaptionRecord] = []
    for cluster_position, (cluster_hash, rows) in enumerate(
        zip(selected_cluster_hashes, rows_by_cluster, strict=True)
    ):
        if not rows:
            raise RuntimeError(f"selected cluster is missing from database: {cluster_hash}")
        representative_row = rows[0][0]
        for source_row, captions in rows:
            for caption_index, caption in enumerate(captions):
                if not isinstance(caption, str) or not caption.strip():
                    raise RuntimeError("cluster database contains an invalid caption")
                records.append(
                    CertificateCaptionRecord(
                        caption_position=len(records),
                        cluster_position=cluster_position,
                        cluster_sha256=cluster_hash,
                        representative_row_index=representative_row,
                        source_row_index=source_row,
                        source_caption_index=caption_index,
                        caption=caption.strip(),
                    )
                )
    return records


class CertificateCaptionDataset(Dataset):
    """Deterministic all-caption view of the frozen certificate clusters."""

    def __init__(
        self,
        *,
        parquet_path: Path,
        records: list[CertificateCaptionRecord],
        tokenizer: Any,
        image_processor: Any,
        prompt: str,
        image_token_length: int,
        max_sequence_length: int,
    ) -> None:
        if not records:
            raise ValueError("certificate caption records are empty")
        self.records = records
        self.image_processor = image_processor
        self.encoder = CanonicalCaptionEncoder(
            tokenizer,
            prompt=prompt,
            image_token_length=image_token_length,
            max_sequence_length=max_sequence_length,
        )
        self.dataset = HFDataset.from_parquet(str(parquet_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.caption_position != index:
            raise RuntimeError("caption record ordering is not canonical")
        representative = self.dataset[record.representative_row_index]
        if record.source_row_index == record.representative_row_index:
            source = representative
        else:
            source = self.dataset[record.source_row_index]
        source_captions = assistant_captions(source["conversations"])
        if (
            record.source_caption_index >= len(source_captions)
            or source_captions[record.source_caption_index] != record.caption
        ):
            raise RuntimeError("parquet caption disagrees with the frozen cluster database")

        image_bytes = require_single_image(representative["image_bytes"])
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        payload = struct.pack(">II", image.width, image.height) + image.tobytes()
        if hashlib.sha256(payload).hexdigest() != record.cluster_sha256:
            raise RuntimeError("representative parquet image disagrees with cluster hash")
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        encoded = self.encoder.encode(record.caption)
        return {
            "input_ids": encoded.input_ids,
            "labels": encoded.labels,
            "attention_mask": encoded.attention_mask,
            "pixel_values": image_inputs,
            "caption_position": record.caption_position,
            "cluster_position": record.cluster_position,
            "representative_row_index": record.representative_row_index,
            "source_row_index": record.source_row_index,
            "source_caption_index": record.source_caption_index,
            "valid_label_count": encoded.valid_label_count,
        }


def certificate_caption_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_keys = batch[0]["pixel_values"].keys()
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
        "pixel_values": {
            key: torch.stack([sample["pixel_values"][key] for sample in batch])
            for key in pixel_keys
        },
        "caption_position": torch.tensor(
            [sample["caption_position"] for sample in batch], dtype=torch.int64
        ),
        "cluster_position": torch.tensor(
            [sample["cluster_position"] for sample in batch], dtype=torch.int64
        ),
        "representative_row_index": torch.tensor(
            [sample["representative_row_index"] for sample in batch], dtype=torch.int64
        ),
        "source_row_index": torch.tensor(
            [sample["source_row_index"] for sample in batch], dtype=torch.int64
        ),
        "source_caption_index": torch.tensor(
            [sample["source_caption_index"] for sample in batch], dtype=torch.int64
        ),
        "valid_label_count": torch.tensor(
            [sample["valid_label_count"] for sample in batch], dtype=torch.int64
        ),
    }

