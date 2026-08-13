from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets")
)
sys.path.insert(0, str(ROOT / "upstream/minimind-v"))

from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minimind_v_bound.data.caption_dataset import (  # noqa: E402
    LockedTrainCaptionDataset,
    locked_caption_collate,
)
from minimind_v_bound.models.intrinsic_projector import (  # noqa: E402
    prepare_intrinsic_projector,
    set_projector_training_mode,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="One-example real S-4K gradient test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real smoke test")

    set_seed(20260813)
    config = VLMConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=450,
        use_moe=False,
        dropout=0.0,
    )
    model = MiniMindVLM(
        config, vision_model_path=str(ROOT / "model/siglip2-base-p32-256-ve")
    )
    weights = torch.load(
        ROOT / "model/llm_768.pth", map_location="cpu", weights_only=True
    )
    incompatible = model.load_state_dict(weights, strict=False)
    missing_other = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("vision_encoder.", "vision_proj."))
    ]
    if missing_other or incompatible.unexpected_keys:
        raise RuntimeError(
            f"base checkpoint mismatch: {missing_other}, {incompatible.unexpected_keys}"
        )
    wrapper = prepare_intrinsic_projector(
        model, intrinsic_dimension=4096, seed=137
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    dataset = LockedTrainCaptionDataset(
        parquet_path=ROOT / "dataset/raw/pretrain_i2t.parquet",
        cluster_database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        train_manifest_path=ROOT / "dataset/manifests/train_clusters.jsonl",
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt="Describe this image.",
        image_token_length=64,
        max_sequence_length=450,
        index_directory=ROOT / "dataset/manifests/train_index",
        smoke_limit=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=locked_caption_collate,
    )
    batch = next(iter(loader))

    device = torch.device(args.device)
    model.to(device)
    set_projector_training_mode(model)
    assert not model.vision_encoder.training
    frozen_projector_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in wrapper.base_projector.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [wrapper.subspace_params],
        lr=4e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = {
        key: value.to(device) for key, value in batch["pixel_values"].items()
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids=input_ids, labels=labels, pixel_values=pixel_values)
        loss = output.loss + output.aux_loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    loss.backward()
    gradient = wrapper.subspace_params.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
        raise RuntimeError("subspace gradient is missing, non-finite, or zero")
    frozen_gradients = [
        name
        for name, parameter in model.named_parameters()
        if name != "vision_proj.subspace_params" and parameter.grad is not None
    ]
    if frozen_gradients:
        raise RuntimeError(f"frozen parameters received gradients: {frozen_gradients}")
    optimizer.step()
    if torch.count_nonzero(wrapper.subspace_params).item() == 0:
        raise RuntimeError("optimizer did not update the intrinsic coordinate")
    for name, parameter in wrapper.base_projector.named_parameters():
        if not torch.equal(parameter.detach().cpu(), frozen_projector_before[name]):
            raise RuntimeError(f"frozen Projector parameter changed: {name}")

    print(json.dumps({
        "device": str(device),
        "frozen_parameter_gradients": 0,
        "image_tokens": int((input_ids == 12).sum().item()),
        "loss_nats_per_token": float(loss.detach().cpu()),
        "row_index": int(batch["row_index"][0]),
        "subspace_gradient_norm": float(gradient.norm().detach().cpu()),
        "subspace_nonzero_after_step": int(
            torch.count_nonzero(wrapper.subspace_params).item()
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "valid_label_tokens": int(batch["valid_label_count"][0]),
        "vision_encoder_eval": not model.vision_encoder.training,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
