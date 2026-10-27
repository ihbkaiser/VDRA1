import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    _git_bash_candidates = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/usr/bin/bash.exe",
        Path("C:/Program Files/Git/usr/bin/bash.exe"),
    ]
    BASH = next((str(path) for path in _git_bash_candidates if path.is_file()), shutil.which("bash"))
else:
    BASH = shutil.which("bash")


def _bash_path(path):
    value = str(path)
    if os.name == "nt":
        cygpath = str(Path(BASH).parent / "cygpath.exe")
        result = subprocess.run([cygpath, "-u", value],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    return value


def test_dry_run_queue_does_not_execute_raw_jobs(tmp_path):
    assert BASH, "bash is required for the shell-script tests"
    marker = tmp_path / "job-ran.txt"
    jobs = tmp_path / "jobs.txt"
    jobs.write_text(f"printf launched > '{_bash_path(marker)}'\n", encoding="utf-8")
    lib = _bash_path(ROOT / "scripts/lib.sh")
    run_dir = _bash_path(tmp_path)
    env = dict(os.environ, DRY_RUN="1", GPU_GROUPS="0", RUNS_DIR=run_dir,
               CODEX_TEST_BASH=_bash_path(BASH))
    if os.name == "nt":
        bash_dir = Path(BASH).parent
        env["PATH"] = str(bash_dir) + ";" + str(bash_dir.parent / "bin") + ";" + env.get("PATH", "")
    script = (
        f'source "{lib}"\n'
        'bash() { "$CODEX_TEST_BASH" "$@"; }\n'
        f'run_queue "{_bash_path(jobs)}"\n'
    )

    result = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "GPUS=0" in result.stderr
    assert not marker.exists()
