import pytest

from modal_app.telemetry import _validate_compute_capability


def test_telemetry_accepts_an_explicit_older_gpu_for_reference_probes():
    _validate_compute_capability((9, 0), expected=(9, 0))


def test_telemetry_still_rejects_non_b200_for_default_training_starts():
    with pytest.raises(RuntimeError, match="compute capability 10.0, saw 9.0"):
        _validate_compute_capability((9, 0))
