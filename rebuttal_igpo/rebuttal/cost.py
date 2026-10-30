"""Operational cost accounting for frozen-policy signals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cost:
    generated_tokens: int = 0
    scored_tokens: int = 0
    gpu_seconds: float = 0.0
    peak_vram_bytes: int | None = None

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "generated_tokens": int(self.generated_tokens),
            "scored_tokens": int(self.scored_tokens),
            "gpu_seconds": float(self.gpu_seconds),
            "gpu_hours": self.gpu_hours,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


def signal_cost(
    signal: str,
    *,
    pilot_tokens: int,
    second_phase_tokens: int = 0,
    scored_tokens: int = 0,
    gpu_seconds: float = 0.0,
    peak_vram_bytes: int | None = None,
) -> Cost:
    if signal == "uniform":
        return Cost(peak_vram_bytes=peak_vram_bytes)
    generated = int(pilot_tokens)
    if signal in {"raw_tv", "vdra", "vdra_lambda", "oracle"}:
        generated += int(second_phase_tokens)
        scored = int(scored_tokens)
    else:
        scored = 0
    return Cost(generated, scored, float(gpu_seconds), peak_vram_bytes)
