"""Microbenchmark IGPO selected-token log-probability kernels on a CUDA GPU.

This measures only logits-to-token-logprob forward/backward, not the model.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--shapes", default="1x128,2x128,1x512")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    methods = {
        "full_log_softmax_gather": lambda x, y: F.log_softmax(x, dim=-1).gather(
            -1, y.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1) * (y >= 0),
        "selected_logsumexp": lambda x, y: (
            x.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            - torch.logsumexp(x, dim=-1)
        ) * (y >= 0),
        "cross_entropy_ignore_index": lambda x, y: -F.cross_entropy(
            x.reshape(-1, x.shape[-1]),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(y.shape),
    }
    dtype_results = []
    for dtype in (torch.float32, torch.bfloat16):
        shape_results = []
        for spec in args.shapes.split(","):
            batch, seq = (int(part) for part in spec.split("x"))
            x = torch.randn(
                batch, seq, args.vocab, dtype=dtype, device="cuda", requires_grad=True
            )
            y = torch.randint(0, args.vocab, (batch, seq), device="cuda")
            y.reshape(-1)[::5] = -100
            weights = (y >= 0).to(torch.float32)
            baseline_out = methods["full_log_softmax_gather"](x, y).detach()
            baseline_grad = None
            method_results = {}
            base_allocated = torch.cuda.memory_allocated()

            for name, fn in methods.items():
                times = []
                torch.cuda.reset_peak_memory_stats()
                for iteration in range(args.warmup + args.repeats):
                    x.grad = None
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = fn(x, y)
                    (output.float() * weights).sum().backward()
                    end.record()
                    torch.cuda.synchronize()
                    if iteration >= args.warmup:
                        times.append(start.elapsed_time(end))
                    grad = x.grad.detach()
                    if name == "full_log_softmax_gather":
                        baseline_grad = grad.clone()
                    del output, grad
                peak_extra = torch.cuda.max_memory_allocated() - base_allocated
                output = fn(x, y).detach()
                output_delta = (output.float() - baseline_out.float()).abs().max().item()
                grad_delta = (x.grad.float() - baseline_grad.float()).abs().max().item()
                method_results[name] = {
                    "median_ms_forward_backward": statistics.median(times),
                    "peak_extra_allocated_bytes": peak_extra,
                    "max_abs_output_delta_vs_full": output_delta,
                    "max_abs_gradient_delta_vs_full": grad_delta,
                }
                del output
            reference_ms = method_results["full_log_softmax_gather"][
                "median_ms_forward_backward"
            ]
            for result in method_results.values():
                result["relative_speed_vs_full"] = reference_ms / result[
                    "median_ms_forward_backward"
                ]
            shape_results.append({"shape": [batch, seq, args.vocab], "methods": method_results})
            del x, y, weights, baseline_out, baseline_grad
            torch.cuda.empty_cache()
        dtype_results.append({"dtype": str(dtype), "shapes": shape_results})

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "CUDA synthetic logits-to-selected-token logprob forward/backward; excludes model forward",
        "warmup_iterations": args.warmup,
        "measured_iterations": args.repeats,
        "results": dtype_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
