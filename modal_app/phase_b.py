"""Phase B: probe the shared B200 stack once for each framework."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from modal_app.images import DEEPSPEED_VERSION, framework_images
from modal_app.single_rank_collectives import install_single_rank_collective_identity
from modal_app.telemetry import append_jsonl, run_start_benchmark


APP_NAME = "vdra-experiments"
VOLUME_NAME = "vdra-work"
WORK_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROBE_TIMEOUT_SECONDS = 600

app = modal.App(APP_NAME)
IMAGES = framework_images()


def _training_step(framework: str) -> dict[str, float]:
    import torch
    from torch import nn

    torch.manual_seed(7)
    if framework in {"igpo", "treerl"}:
        import faulthandler
        import deepspeed

        print("[probe] DeepSpeed import succeeded", flush=True)
        # A one-GPU B200 function has no peer to contact; disable NCCL P2P
        # while initializing the world-size-one communicator.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        install_single_rank_collective_identity()
        print("[probe] installed world-size-one collective identities", flush=True)
        model = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        print("[probe] entering deepspeed.initialize", flush=True)
        faulthandler.dump_traceback_later(60, repeat=True)
        try:
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                config={
                    "train_batch_size": 1,
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 1,
                    "bf16": {"enabled": False},
                    "zero_optimization": {"stage": 0},
                },
                dist_init_required=True,
            )
        finally:
            faulthandler.cancel_dump_traceback_later()
        print("[probe] deepspeed.initialize returned", flush=True)
        inputs = torch.randn((1, 128), device="cuda", dtype=torch.float32)
        loss = engine(inputs).float().square().mean()
        print("[probe] DeepSpeed forward pass returned", flush=True)
        engine.backward(loss)
        print("[probe] DeepSpeed backward pass returned", flush=True)
        engine.step()
        print("[probe] DeepSpeed optimizer step returned", flush=True)
        loss_value = float(loss.detach().cpu())
        del engine, model, optimizer
    else:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29582")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)
        model = FSDP(
            nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8)).cuda(),
            device_id=torch.cuda.current_device(),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn((1, 128), device="cuda", dtype=torch.float32)
        loss = model(inputs).float().square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        del model, optimizer

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.synchronize()
    return {"loss": loss_value, "peak_memory_bytes": int(torch.cuda.max_memory_allocated())}


def _vllm_probe() -> dict[str, Any]:
    import socket
    import subprocess
    import sys
    import httpx
    from transformers import AutoTokenizer

    prompt = "Compute 2 + 2."
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    server_log_path = Path("/tmp/vllm-api-probe.log")
    server_log = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL_ID,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "128",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.20",
            "--enforce-eager",
            "--disable-custom-all-reduce",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    try:
        deadline = time.monotonic() + 240
        request_timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        with httpx.Client(timeout=request_timeout) as client:
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        "vLLM OpenAI server exited during startup:\n"
                        + server_log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                    )
                try:
                    health = client.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
                    if health.status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "vLLM OpenAI server did not become healthy:\n"
                    + server_log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                )

            # Match IGPO's VLLMLogprobClient request exactly: echo prompt,
            # generate zero tokens, and request one logprob per prompt token.
            scored = client.post(
                f"{base_url}/completions",
                json={
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "max_tokens": 0,
                    "logprobs": 1,
                    "echo": True,
                    "temperature": 0.0,
                },
            )
            scored.raise_for_status()
            scored_choice = scored.json()["choices"][0]
            prompt_logprobs = scored_choice.get("logprobs", {}).get("token_logprobs") or []
            if len(prompt_logprobs) != len(token_ids):
                raise RuntimeError(
                    "OpenAI completions API returned "
                    f"{len(prompt_logprobs)} prompt logprob positions for {len(token_ids)} tokenizer tokens"
                )
            if not any(value is not None for value in prompt_logprobs):
                raise RuntimeError("OpenAI completions API returned no prompt logprobs")

            generated = client.post(
                f"{base_url}/completions",
                json={
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "max_tokens": 2,
                    "logprobs": 1,
                    "echo": False,
                    "temperature": 0.0,
                },
            )
            generated.raise_for_status()
            generated_choice = generated.json()["choices"][0]
            completion_token_logprobs = generated_choice.get("logprobs", {}).get("token_logprobs") or []
            if not generated_choice.get("text") or not completion_token_logprobs:
                raise RuntimeError("OpenAI completions API did not return completion tokens/logprobs")

            return {
                "model": MODEL_ID,
                "prompt": prompt,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_name_or_path": tokenizer.name_or_path,
                "prompt_token_ids": token_ids,
                "prompt_logprobs_positions": len(prompt_logprobs),
                "prompt_logprobs_nonempty": sum(value is not None for value in prompt_logprobs),
                "prompt_echo_text": scored_choice["text"],
                "completion_text": generated_choice["text"],
                "completion_token_logprobs": completion_token_logprobs,
                "endpoint": "/v1/completions",
                "scoring_request": {
                    "max_tokens": 0,
                    "logprobs": 1,
                    "echo": True,
                    "temperature": 0.0,
                },
            }
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()


def _run_probe(framework: str) -> dict[str, Any]:
    started_at = time.time()
    run_meta_path = Path(f"/work/probes/{framework}/run_meta.jsonl")
    probe_result: dict[str, Any] = {
        "framework": framework,
        "started_at": started_at,
        "model": MODEL_ID,
        "stack": {
            "cuda_base": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            "torch": "2.7.0",
            "vllm": "0.9.2",
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "deepspeed": DEEPSPEED_VERSION if framework in {"igpo", "treerl"} else None,
            "single_rank_collective_identity": framework in {"igpo", "treerl"},
        },
    }
    try:
        append_jsonl(run_meta_path, run_start_benchmark())
        probe_result["training_step"] = _training_step(framework)
        probe_result["vllm"] = _vllm_probe()
        probe_result["elapsed_seconds"] = time.time() - started_at
        probe_result["approx_cost_usd"] = probe_result["elapsed_seconds"] * 0.001736
        probe_result["ok"] = True
    except Exception as exc:
        probe_result["elapsed_seconds"] = time.time() - started_at
        probe_result["approx_cost_usd"] = probe_result["elapsed_seconds"] * 0.001736
        probe_result["ok"] = False
        probe_result["error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(
            run_meta_path,
            {
                "event": "probe_end",
                "time": time.time(),
                "wall_seconds": probe_result["elapsed_seconds"],
                "approx_cost_usd": probe_result["approx_cost_usd"],
                "error": probe_result["error"],
            },
        )
        WORK_VOLUME.commit()
        raise
    out_path = Path(f"/work/probes/{framework}/result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(probe_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(
        run_meta_path,
        {
            "event": "probe_end",
            "time": time.time(),
            "wall_seconds": probe_result["elapsed_seconds"],
            "approx_cost_usd": probe_result["elapsed_seconds"] * 0.001736,
            "ok": True,
        },
    )
    WORK_VOLUME.commit()
    return probe_result


@app.function(
    image=IMAGES["igpo"],
    gpu="B200",
    timeout=PROBE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def probe_igpo() -> dict[str, Any]:
    return _run_probe("igpo")


@app.function(
    image=IMAGES["treepo"],
    gpu="B200",
    timeout=PROBE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def probe_treepo() -> dict[str, Any]:
    return _run_probe("treepo")


@app.function(
    image=IMAGES["treerl"],
    gpu="B200",
    timeout=PROBE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def probe_treerl() -> dict[str, Any]:
    return _run_probe("treerl")


@app.local_entrypoint()
def main(framework: str = "all") -> None:
    probes = {
        "igpo": probe_igpo,
        "treepo": probe_treepo,
        "treerl": probe_treerl,
    }
    selected = list(probes) if framework == "all" else [framework]
    if any(name not in probes for name in selected):
        raise ValueError("framework must be all, igpo, treepo, or treerl")
    for name in selected:
        print(f"Starting {name} B200 compatibility probe (timeout={PROBE_TIMEOUT_SECONDS}s)")
        print(json.dumps(probes[name].remote(), indent=2, sort_keys=True))
