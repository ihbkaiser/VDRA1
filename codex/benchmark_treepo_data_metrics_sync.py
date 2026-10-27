"""Benchmark TreePO's per-scalar data metric sync against one packed copy.

The two paths implement the formulas in ``verl/trainer/ppo/metric_utils.py``.
This isolates end-of-update metric calculation and excludes model compute.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def make_batch(device: torch.device, batch_size: int, prompt_len: int, response_len: int):
    generator = torch.Generator(device=device).manual_seed(741)
    attention_mask = torch.zeros((batch_size, prompt_len + response_len), device=device)
    prompt_sizes = torch.randint(max(1, prompt_len // 4), prompt_len + 1, (batch_size,), generator=generator, device=device)
    response_sizes = torch.randint(max(1, response_len // 4), response_len + 1, (batch_size,), generator=generator, device=device)
    positions = torch.arange(prompt_len + response_len, device=device)
    attention_mask[:, :prompt_len] = positions[:prompt_len] < prompt_sizes[:, None]
    attention_mask[:, prompt_len:] = positions[prompt_len:] - prompt_len < response_sizes[:, None]
    data = {
        "attention_mask": attention_mask,
        "responses": torch.zeros((batch_size, response_len), dtype=torch.long, device=device),
        "token_level_scores": torch.randn((batch_size, response_len), generator=generator, device=device),
        "token_level_rewards": torch.randn((batch_size, response_len), generator=generator, device=device),
        "advantages": torch.randn((batch_size, response_len), generator=generator, device=device),
        "returns": torch.randn((batch_size, response_len), generator=generator, device=device),
        "values": torch.randn((batch_size, response_len), generator=generator, device=device),
    }
    return data


def _metric_tensors(batch, use_critic: bool = True):
    data = batch
    response_width = data["responses"].shape[-1]
    prompt_mask = data["attention_mask"][:, :-response_width]
    response_mask = data["attention_mask"][:, -response_width:].bool()
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    sequence_score = data["token_level_scores"].sum(-1)
    sequence_reward = data["token_level_rewards"].sum(-1)
    valid_adv = torch.masked_select(data["advantages"], response_mask)
    valid_returns = torch.masked_select(data["returns"], response_mask)

    metrics = {
        "critic/score/mean": torch.mean(sequence_score).detach(),
        "critic/score/max": torch.max(sequence_score).detach(),
        "critic/score/min": torch.min(sequence_score).detach(),
        "critic/rewards/mean": torch.mean(sequence_reward).detach(),
        "critic/rewards/max": torch.max(sequence_reward).detach(),
        "critic/rewards/min": torch.min(sequence_reward).detach(),
        "critic/advantages/mean": torch.mean(valid_adv).detach(),
        "critic/advantages/max": torch.max(valid_adv).detach(),
        "critic/advantages/min": torch.min(valid_adv).detach(),
        "critic/returns/mean": torch.mean(valid_returns).detach(),
        "critic/returns/max": torch.max(valid_returns).detach(),
        "critic/returns/min": torch.min(valid_returns).detach(),
    }
    if use_critic:
        valid_values = torch.masked_select(data["values"], response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
        metrics.update(
            {
                "critic/values/mean": torch.mean(valid_values).detach(),
                "critic/values/max": torch.max(valid_values).detach(),
                "critic/values/min": torch.min(valid_values).detach(),
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach(),
            }
        )
    metrics.update(
        {
            "response_length/mean": torch.mean(response_length).detach(),
            "response_length/max": torch.max(response_length).detach(),
            "response_length/min": torch.min(response_length).detach(),
            "response_length/clip_ratio": torch.mean(torch.eq(response_length, response_width).float()).detach(),
            "prompt_length/mean": torch.mean(prompt_length).detach(),
            "prompt_length/max": torch.max(prompt_length).detach(),
            "prompt_length/min": torch.min(prompt_length).detach(),
            "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, prompt_mask.size(-1)).float()).detach(),
        }
    )
    return metrics


def old_per_scalar(batch):
    data = batch
    response_width = data["responses"].shape[-1]
    prompt_mask = data["attention_mask"][:, :-response_width].bool()
    response_mask = data["attention_mask"][:, -response_width:].bool()
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()
    sequence_score = data["token_level_scores"].sum(-1)
    sequence_reward = data["token_level_rewards"].sum(-1)
    valid_adv = torch.masked_select(data["advantages"], response_mask)
    valid_returns = torch.masked_select(data["returns"], response_mask)

    # Match metric_utils.py's interleaving: calculate a scalar and immediately
    # call .item(), which can synchronize before the next reduction is queued.
    metrics = {
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
    }
    valid_values = torch.masked_select(data["values"], response_mask)
    return_diff_var = torch.var(valid_returns - valid_values)
    return_var = torch.var(valid_returns)
    metrics.update(
        {
            "critic/values/mean": torch.mean(valid_values).detach().item(),
            "critic/values/max": torch.max(valid_values).detach().item(),
            "critic/values/min": torch.min(valid_values).detach().item(),
            "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        }
    )
    metrics.update(
        {
            "response_length/mean": torch.mean(response_length).detach().item(),
            "response_length/max": torch.max(response_length).detach().item(),
            "response_length/min": torch.min(response_length).detach().item(),
            "response_length/clip_ratio": torch.mean(torch.eq(response_length, response_width).float()).detach().item(),
            "prompt_length/mean": torch.mean(prompt_length).detach().item(),
            "prompt_length/max": torch.max(prompt_length).detach().item(),
            "prompt_length/min": torch.min(prompt_length).detach().item(),
            "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, prompt_mask.size(-1)).float()).detach().item(),
        }
    )
    return metrics


def packed_copy(batch):
    tensors = _metric_tensors(batch)
    keys = list(tensors)
    values = torch.stack([tensors[key].reshape(()) for key in keys]).cpu().tolist()
    return dict(zip(keys, values))


def measure(fn, batch, updates: int, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(updates):
            fn(batch)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    batch = make_batch(device, batch_size=32, prompt_len=512, response_len=1536)

    old = old_per_scalar(batch)
    packed = packed_copy(batch)
    if old.keys() != packed.keys():
        raise RuntimeError("metric keys differ")
    max_abs_delta = max(abs(old[key] - packed[key]) for key in old)
    if max_abs_delta != 0:
        raise RuntimeError(f"metric values differ: max_abs_delta={max_abs_delta}")

    updates, warmups, repeats = 48, 3, 9
    for _ in range(warmups):
        measure(old_per_scalar, batch, updates=updates, repeats=1)
        measure(packed_copy, batch, updates=updates, repeats=1)
    old_samples = measure(old_per_scalar, batch, updates=updates, repeats=repeats)
    packed_samples = measure(packed_copy, batch, updates=updates, repeats=repeats)
    old_median = statistics.median(old_samples)
    packed_median = statistics.median(packed_samples)
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {"batch_size": 32, "max_prompt_tokens": 512, "max_response_tokens": 1536, "updates_per_sample": updates, "repeats": repeats, "warmups": warmups},
        "metric_count": len(old),
        "old_per_scalar_median_ms": old_median,
        "packed_copy_median_ms": packed_median,
        "speedup_ratio_old_over_packed": old_median / packed_median,
        "max_abs_metric_delta": max_abs_delta,
        "old_samples_ms": old_samples,
        "packed_samples_ms": packed_samples,
        "measurement": "TreePO compute_data_metrics formulas only; excludes transformer/model compute and Ray communication.",
    }
    rendered = json.dumps(result, indent=2)
    output = Path(__file__).resolve().parent / "phase_b_results" / "local_treepo_data_metrics_sync_20261026.json"
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"saved={output.as_posix()}")


if __name__ == "__main__":
    main()
