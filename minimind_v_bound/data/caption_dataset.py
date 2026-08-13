from __future__ import annotations

import io
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .clusters import assistant_captions, require_single_image


LABEL_MASK_RULE_VERSION = "caption_tokens_plus_im_end_overlap_offsets_v1"
PROMPT_FORMAT_VERSION = "image64_newline_canonical_user_empty_think_assistant_v1"


@dataclass(frozen=True)
class EncodedCaption:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    valid_label_count: int


class CanonicalCaptionEncoder:
    """Build the fixed prompt and label only caption tokens plus ``im_end``."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt: str = "Describe this image.",
        image_special_token: str = "<|image_pad|>",
        image_token_length: int = 64,
        max_sequence_length: int = 450,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.image_special_token = image_special_token
        self.image_token_length = image_token_length
        self.max_sequence_length = max_sequence_length
        if tokenizer.pad_token_id is None or tokenizer.eos_token is None:
            raise ValueError("tokenizer must define pad_token_id and eos_token")

        image_prefix = image_special_token * image_token_length
        user_message = {
            "role": "user",
            "content": f"{image_prefix}\n{prompt}",
        }
        self.prefix_text = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        prefix_ids = tokenizer(self.prefix_text, add_special_tokens=False).input_ids
        image_token_id = tokenizer.convert_tokens_to_ids(image_special_token)
        if prefix_ids.count(image_token_id) != image_token_length:
            raise RuntimeError("canonical prompt does not contain exactly 64 image tokens")

    def encode(self, caption: str) -> EncodedCaption:
        caption = caption.strip()
        if not caption:
            raise ValueError("caption must be non-empty")
        caption_start = len(self.prefix_text)
        caption_end = caption_start + len(caption)
        full_text = self.prefix_text + caption + self.tokenizer.eos_token + "\n"
        label_end = caption_end + len(self.tokenizer.eos_token)
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoding.input_ids[: self.max_sequence_length])
        offsets = list(encoding.offset_mapping[: self.max_sequence_length])
        labels = [-100] * len(input_ids)
        for index, (start, end) in enumerate(offsets):
            if end > caption_start and start < label_end:
                labels[index] = input_ids[index]
        valid_label_count = sum(label != -100 for label in labels)
        if valid_label_count == 0:
            raise ValueError("caption has no label token after truncation")

        attention_mask = [1] * len(input_ids)
        padding = self.max_sequence_length - len(input_ids)
        if padding > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding)
            labels.extend([-100] * padding)
            attention_mask.extend([0] * padding)
        return EncodedCaption(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            valid_label_count=valid_label_count,
        )


def _read_cluster_hashes(path: Path) -> list[str]:
    hashes = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            value = record.get("cluster_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid cluster hash at {path}:{line_number}")
            hashes.append(value)
    if len(hashes) != len(set(hashes)):
        raise ValueError("training cluster manifest contains duplicates")
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_example_index(
    database_path: Path,
    train_manifest_path: Path,
    *,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (row index, assistant-caption index) using only the train manifest."""
    allowed_hashes = _read_cluster_hashes(train_manifest_path)
    rows: list[int] = []
    caption_indices: list[int] = []
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        connection.execute(
            "CREATE TEMP TABLE allowed_clusters(image_sha256 TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO allowed_clusters(image_sha256) VALUES (?)",
            ((value,) for value in allowed_hashes),
        )
        cursor = connection.execute(
            """
            SELECT examples.row_index, examples.captions_json
            FROM examples
            INNER JOIN allowed_clusters USING(image_sha256)
            ORDER BY examples.row_index
            """
        )
        for row_index, captions_json in cursor:
            caption_count = len(json.loads(captions_json))
            for caption_index in range(caption_count):
                rows.append(row_index)
                caption_indices.append(caption_index)
                if limit is not None and len(rows) >= limit:
                    return np.asarray(rows, dtype=np.uint32), np.asarray(
                        caption_indices, dtype=np.uint16
                    )
    return np.asarray(rows, dtype=np.uint32), np.asarray(
        caption_indices, dtype=np.uint16
    )


def load_cached_training_example_index(
    index_directory: Path, train_manifest_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = index_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["train_manifest_sha256"] != _sha256_file(train_manifest_path):
        raise RuntimeError("training index cache does not match the train manifest")
    row_path = index_directory / "row_indices.npy"
    caption_path = index_directory / "caption_indices.npy"
    if metadata["row_indices_sha256"] != _sha256_file(row_path):
        raise RuntimeError("cached row-index hash mismatch")
    if metadata["caption_indices_sha256"] != _sha256_file(caption_path):
        raise RuntimeError("cached caption-index hash mismatch")
    rows = np.load(row_path, mmap_mode="r", allow_pickle=False)
    captions = np.load(caption_path, mmap_mode="r", allow_pickle=False)
    if len(rows) != len(captions) or len(rows) != metadata["caption_examples"]:
        raise RuntimeError("cached training-index lengths disagree")
    return rows, captions


class LockedTrainCaptionDataset(Dataset):
    """Caption samples restricted to the preregistered training clusters."""

    def __init__(
        self,
        *,
        parquet_path: Path,
        cluster_database_path: Path,
        train_manifest_path: Path,
        tokenizer: Any,
        image_processor: Any,
        prompt: str = "Describe this image.",
        image_token_length: int = 64,
        max_sequence_length: int = 450,
        index_directory: Path | None = None,
        smoke_limit: int | None = None,
    ) -> None:
        if "locked_test" in train_manifest_path.parts:
            raise ValueError("refusing to use a locked-test manifest for training")
        self.parquet_path = Path(parquet_path)
        self.image_processor = image_processor
        self.encoder = CanonicalCaptionEncoder(
            tokenizer,
            prompt=prompt,
            image_token_length=image_token_length,
            max_sequence_length=max_sequence_length,
        )
        if index_directory is None:
            self.row_indices, self.caption_indices = load_training_example_index(
                Path(cluster_database_path), Path(train_manifest_path), limit=smoke_limit
            )
        else:
            rows, captions = load_cached_training_example_index(
                Path(index_directory), Path(train_manifest_path)
            )
            selection = slice(None, smoke_limit)
            self.row_indices = rows[selection]
            self.caption_indices = captions[selection]
        if len(self.row_indices) == 0:
            raise ValueError("training manifest selected no caption examples")
        self.dataset = HFDataset.from_parquet(str(self.parquet_path))

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = int(self.row_indices[index])
        caption_index = int(self.caption_indices[index])
        row = self.dataset[row_index]
        captions = assistant_captions(row["conversations"])
        if caption_index >= len(captions):
            raise RuntimeError("caption index disagrees with frozen cluster database")
        encoded = self.encoder.encode(captions[caption_index])
        image_bytes = require_single_image(row["image_bytes"])
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image_inputs = self.image_processor(images=image, return_tensors="pt")
        return {
            "input_ids": encoded.input_ids,
            "labels": encoded.labels,
            "attention_mask": encoded.attention_mask,
            "pixel_values": image_inputs,
            "row_index": row_index,
            "caption_index": caption_index,
            "valid_label_count": encoded.valid_label_count,
        }


def locked_caption_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_keys = batch[0]["pixel_values"].keys()
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
        "pixel_values": {
            key: torch.stack([sample["pixel_values"][key] for sample in batch])
            for key in pixel_keys
        },
        "row_index": torch.tensor([sample["row_index"] for sample in batch]),
        "caption_index": torch.tensor([sample["caption_index"] for sample in batch]),
        "valid_label_count": torch.tensor(
            [sample["valid_label_count"] for sample in batch]
        ),
    }
