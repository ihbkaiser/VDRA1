from pathlib import Path

import modal_app.telemetry as telemetry


def test_host_cpu_prefers_a_cpu_model_from_proc_cpuinfo(monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, **kwargs: "processor : 0\nmodel name : Example CPU 9000\n",
    )
    monkeypatch.setattr(telemetry, "_command_output", lambda args: None)
    monkeypatch.setattr(telemetry.platform, "processor", lambda: "unknown")
    monkeypatch.setattr(telemetry.platform, "machine", lambda: "x86_64")

    assert telemetry.host_cpu() == "Example CPU 9000"


def test_host_cpu_uses_architecture_and_visible_cpus_when_model_is_hidden(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(telemetry, "_command_output", lambda args: None)
    monkeypatch.setattr(telemetry.platform, "processor", lambda: "unknown")
    monkeypatch.setattr(telemetry.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        telemetry.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )

    assert telemetry.host_cpu() == "x86_64 (8 visible logical CPUs; model unavailable)"


def test_host_cpu_reports_when_model_and_count_are_unavailable(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(telemetry, "_command_output", lambda args: None)
    monkeypatch.setattr(telemetry.platform, "processor", lambda: "unknown")
    monkeypatch.setattr(telemetry.platform, "machine", lambda: "unknown")
    monkeypatch.setattr(telemetry.os, "sched_getaffinity", lambda pid: set(), raising=False)
    monkeypatch.setattr(telemetry.os, "cpu_count", lambda: None)

    assert telemetry.host_cpu() == "unknown (CPU model and count unavailable)"
