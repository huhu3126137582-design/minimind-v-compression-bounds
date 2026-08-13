from __future__ import annotations

import math

import pytest
import torch

from minimind_v_bound.training.checkpoint import (
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
)
from minimind_v_bound.training.schedule import minimind_cosine_learning_rate


def test_minimind_schedule_endpoints_and_midpoint() -> None:
    initial = 4e-4
    assert minimind_cosine_learning_rate(0, 100, initial) == pytest.approx(initial)
    assert minimind_cosine_learning_rate(50, 100, initial) == pytest.approx(
        initial * 0.55
    )
    assert minimind_cosine_learning_rate(100, 100, initial) == pytest.approx(
        initial * 0.1
    )
    values = [minimind_cosine_learning_rate(step, 100, initial) for step in range(101)]
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_checkpoint_round_trip_restores_coordinate_optimizer_and_progress(tmp_path) -> None:
    coordinate = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer = torch.optim.AdamW([coordinate], lr=4e-4)
    coordinate.grad = torch.ones_like(coordinate)
    optimizer.step()
    expected_coordinate = coordinate.detach().clone()
    expected_exp_avg = optimizer.state[coordinate]["exp_avg"].clone()
    contract = {"fixed": True, "world_size": 2}
    payload = build_checkpoint(
        subspace_params=coordinate,
        optimizer=optimizer,
        epoch=0,
        next_batch_in_epoch=7,
        global_step=7,
        total_steps=100,
        contract=contract,
    )
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(payload, path)

    restored = torch.nn.Parameter(torch.zeros(4))
    restored_optimizer = torch.optim.AdamW([restored], lr=1.0)
    progress = load_checkpoint(
        path,
        subspace_params=restored,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        expected_contract=contract,
    )
    assert torch.equal(restored, expected_coordinate)
    assert torch.equal(restored_optimizer.state[restored]["exp_avg"], expected_exp_avg)
    assert progress == {
        "epoch": 0,
        "next_batch_in_epoch": 7,
        "global_step": 7,
        "total_steps": 100,
    }


def test_checkpoint_rejects_contract_change(tmp_path) -> None:
    coordinate = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.AdamW([coordinate], lr=4e-4)
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(
        build_checkpoint(
            subspace_params=coordinate,
            optimizer=optimizer,
            epoch=0,
            next_batch_in_epoch=0,
            global_step=0,
            total_steps=100,
            contract={"world_size": 2},
        ),
        path,
    )
    with pytest.raises(RuntimeError, match="contract"):
        load_checkpoint(
            path,
            subspace_params=coordinate,
            optimizer=optimizer,
            device=torch.device("cpu"),
            expected_contract={"world_size": 1},
        )
