"""Compare PPO's scalar metric weighting with and without a CUDA .item() sync."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def run_case(counts, values, keep_count_on_device: bool):
    accum = torch.zeros(values.shape[1], device="cuda", dtype=values.dtype)
    started = time.perf_counter()
    for row in range(values.shape[0]):
        count = counts[row] if keep_count_on_device else counts[row].item()
        for column in range(values.shape[1]):
            accum[column] += values[row, column] * count
    torch.cuda.synchronize()
    return time.perf_counter() - started, accum


def run_ratio_case(ratios, losses, threshold: float, tensor_mask: bool):
    total_loss = torch.zeros((), device="cuda", dtype=losses.dtype)
    started = time.perf_counter()
    for row in range(ratios.shape[0]):
        if tensor_mask:
            keep = (ratios[row] <= threshold).to(losses.dtype)
            total_loss += losses[row] * keep
        elif ratios[row].item() > threshold:
            total_loss += losses[row] * 0.0
        else:
            total_loss += losses[row]
    torch.cuda.synchronize()
    return time.perf_counter() - started, total_loss


def run_ratio_training_case(logits, advantages, threshold: float, tensor_mask: bool):
    logits.grad = None
    loss_chunks = []
    started = time.perf_counter()
    for row in range(logits.shape[0]):
        log_ratio = logits[row] * 0.2
        ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
        avg_ratio = ratio.mean()
        pg_loss = (-advantages[row] * ratio).mean()
        if tensor_mask:
            skipped = avg_ratio > threshold
            pg_loss = pg_loss * (~skipped).to(pg_loss.dtype)
        elif avg_ratio.item() > threshold:
            pg_loss = pg_loss * 0.0
        pg_loss.backward()
        loss_chunks.append(pg_loss.detach())
    torch.cuda.synchronize()
    return time.perf_counter() - started, torch.stack(loss_chunks), logits.grad.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device("cuda")
    updates, metric_count, repeats = 48, 15, 7
    counts = torch.randint(100, 2000, (updates,), device=device)
    values = torch.randn(updates, metric_count, device=device)
    torch.cuda.synchronize()

    results = {"sync_each_update": [], "count_tensor_on_device": []}
    outputs = {}
    for _ in range(repeats):
        elapsed, outputs["sync_each_update"] = run_case(counts, values, False)
        results["sync_each_update"].append(elapsed * 1000)
        elapsed, outputs["count_tensor_on_device"] = run_case(counts, values, True)
        results["count_tensor_on_device"].append(elapsed * 1000)

    ratio_threshold = 1.1
    ratios = torch.rand(updates, device=device) + 0.5
    losses = torch.rand(updates, device=device)
    torch.cuda.synchronize()
    ratio_results = {"sync_each_update": [], "tensor_mask_on_device": []}
    ratio_outputs = {}
    for _ in range(repeats):
        elapsed, ratio_outputs["sync_each_update"] = run_ratio_case(
            ratios, losses, ratio_threshold, False
        )
        ratio_results["sync_each_update"].append(elapsed * 1000)
        elapsed, ratio_outputs["tensor_mask_on_device"] = run_ratio_case(
            ratios, losses, ratio_threshold, True
        )
        ratio_results["tensor_mask_on_device"].append(elapsed * 1000)

    logits = torch.randn(updates, 2048, device=device, requires_grad=True)
    advantages = torch.randn(updates, 2048, device=device)
    torch.cuda.synchronize()
    training_ratio_results = {"sync_each_update": [], "tensor_mask_on_device": []}
    training_ratio_outputs = {}
    for _ in range(repeats):
        elapsed, values, gradients = run_ratio_training_case(
            logits, advantages, ratio_threshold, False
        )
        training_ratio_results["sync_each_update"].append(elapsed * 1000)
        training_ratio_outputs["sync_each_update"] = (values, gradients)
        elapsed, values, gradients = run_ratio_training_case(
            logits, advantages, ratio_threshold, True
        )
        training_ratio_results["tensor_mask_on_device"].append(elapsed * 1000)
        training_ratio_outputs["tensor_mask_on_device"] = (values, gradients)

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {"updates": updates, "weighted_metrics_per_update": metric_count},
        "median_ms": {name: statistics.median(times) for name, times in results.items()},
        "speedup_ratio_old_over_new": statistics.median(results["sync_each_update"])
        / statistics.median(results["count_tensor_on_device"]),
        "max_abs_result_delta": (
            outputs["sync_each_update"] - outputs["count_tensor_on_device"]
        ).abs().max().item(),
        "ratio_skip_branch": {
            "threshold": ratio_threshold,
            "median_ms": {
                name: statistics.median(times)
                for name, times in ratio_results.items()
            },
            "speedup_ratio_old_over_new": statistics.median(
                ratio_results["sync_each_update"]
            )
            / statistics.median(ratio_results["tensor_mask_on_device"]),
            "max_abs_loss_delta": (
                ratio_outputs["sync_each_update"]
                - ratio_outputs["tensor_mask_on_device"]
            ).abs().item(),
            "measurement": "synthetic PPO ratio-threshold branch; old path synchronizes via .item() for every update; new path applies the same threshold mask on-device",
        },
        "ratio_skip_with_backward": {
            "shape": {"updates": updates, "tokens_per_update": 2048},
            "median_ms": {
                name: statistics.median(times)
                for name, times in training_ratio_results.items()
            },
            "speedup_ratio_old_over_new": statistics.median(
                training_ratio_results["sync_each_update"]
            )
            / statistics.median(training_ratio_results["tensor_mask_on_device"]),
            "max_abs_loss_delta": (
                training_ratio_outputs["sync_each_update"][0]
                - training_ratio_outputs["tensor_mask_on_device"][0]
            ).abs().max().item(),
            "max_abs_gradient_delta": (
                training_ratio_outputs["sync_each_update"][1]
                - training_ratio_outputs["tensor_mask_on_device"][1]
            ).abs().max().item(),
            "measurement": "synthetic PPO ratio-threshold forward/backward loop with 2048 tokens/update; old path calls .item() per update; new path applies a tensor mask",
        },
        "measurement": "synthetic PPO metric-weighting loop; old path calls CUDA scalar .item() once per update; new path keeps token count on device; includes Python loop and final CUDA synchronization",
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
