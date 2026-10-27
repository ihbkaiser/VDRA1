"""Phase D verifier for TreeRL actor/vLLM weight synchronization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _logprob(value: Any) -> float:
    if hasattr(value, "logprob"):
        value = value.logprob
    elif isinstance(value, Mapping) and "logprob" in value:
        value = value["logprob"]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("vLLM returned a non-finite prompt logprob")
    return number


def compare_prompt_logprobs(
    actor_logprobs: Sequence[float], vllm_prompt_logprobs: Mapping[int, Any], *, atol: float = 0.05
) -> dict[str, Any]:
    """Compare vLLM's prompt candidates with the actor distribution at that position."""
    if not math.isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")
    if not vllm_prompt_logprobs:
        raise ValueError("vLLM returned no prompt logprobs to compare")

    differences = []
    for token_id, value in vllm_prompt_logprobs.items():
        token_id = int(token_id)
        if token_id < 0 or token_id >= len(actor_logprobs):
            raise ValueError(f"vLLM token id {token_id} is outside the actor vocabulary")
        actor_value = float(actor_logprobs[token_id])
        vllm_value = _logprob(value)
        if not math.isfinite(actor_value):
            raise ValueError("actor returned a non-finite logprob")
        differences.append(abs(actor_value - vllm_value))

    maximum = max(differences)
    return {
        "matched": maximum <= float(atol),
        "compared_tokens": len(differences),
        "max_abs_diff": maximum,
        "atol": float(atol),
    }


def verify_actor_vllm_sync(actor_model: Any, tokenizer: Any, vllm_engine: Any, *, atol: float = 0.05) -> dict[str, Any]:
    """Compare actor and vLLM prompt logprobs after the most recent weight sync."""
    import ray
    import torch
    import torch.nn.functional as F
    from vllm import SamplingParams

    prompt_ids = tokenizer.encode("A fixed prompt for actor and vLLM weight sync.", add_special_tokens=False)
    if len(prompt_ids) < 2:
        raise RuntimeError("fixed TreeRL sync prompt must contain at least two tokens")
    device = next(actor_model.parameters()).device
    input_ids = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=device)
    was_training = actor_model.training
    actor_model.eval()
    try:
        with torch.inference_mode():
            output = actor_model(input_ids=input_ids)
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            actor_logprobs = F.log_softmax(logits[0, -1].float(), dim=-1).cpu().tolist()
    finally:
        actor_model.train(was_training)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=20,
    )
    results = ray.get(
        vllm_engine.generate.remote(
            prompt_token_ids=[prompt_ids], sampling_params=sampling_params
        )
    )
    prompt_logprobs = getattr(results[0], "prompt_logprobs", None)
    if not prompt_logprobs or not prompt_logprobs[-1]:
        raise RuntimeError("vLLM returned no prompt logprobs for the fixed TreeRL sync prompt")
    report = compare_prompt_logprobs(actor_logprobs, prompt_logprobs[-1], atol=atol)
    report["prompt_token_count"] = len(prompt_ids)
    if not report["matched"]:
        raise RuntimeError(
            "TreeRL actor/vLLM prompt logprobs differ after weight sync: "
            f"max_abs_diff={report['max_abs_diff']:.6g}, atol={report['atol']:.6g}"
        )
    return report
