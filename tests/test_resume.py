"""Resume machinery: plan semantics, the TreeRL loop logic after a crash,
run status detection, crash-aware GPU-hours, the resilient wrapper (with a fake
trainer) and the resume-all job builder."""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from vdra_core.resume import ResumePlan, capture_rng, restore_rng  # noqa: E402
import gpu_hours  # noqa: E402
import run_status  # noqa: E402
from compute_matched_target import compute_target  # noqa: E402


def _bash_executable():
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/usr/bin/bash.exe",
            Path("C:/Program Files/Git/usr/bin/bash.exe"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return shutil.which("bash") or "bash"


def _prepare_bash_environment(env, tmp_path):
    if sys.platform == "win32":
        bash = Path(_bash_executable())
        env["PATH"] = str(bash.parent) + ";" + str(bash.parent.parent / "bin") + ";" + env.get("PATH", "")
        bash_env = tmp_path / "bash_env.sh"
        bash_env.write_text('python3() { py -3.12 "$@"; }\n')
        env["BASH_ENV"] = str(bash_env)
    return env


# --------------------------------------------------------------------------- plan

def test_plan_save_rules():
    p = ResumePlan.from_state(None, stop_update=22, resume_every=5, keep_every=10)
    assert [u for u in range(1, 23) if p.save_eval_weights(u)] == [10, 20, 22]
    assert [u for u in range(1, 23) if p.save_resume_state(u)] == [5, 10, 15, 20, 22]
    assert p.is_stop(22) and not p.is_stop(21) and not p.already_done
    q = ResumePlan.from_state({"update_step": 22, "consumed_micro_batches": 704}, stop_update=22)
    assert q.already_done and q.skip_micro_batches == 704


def test_rng_roundtrip():
    import random
    s = capture_rng()
    a = [random.random() for _ in range(3)]
    restore_rng(s)
    assert [random.random() for _ in range(3)] == a


def simulate_treerl(n_batches, update_timesteps, stop, resume_every, keep_every, crash_after=None, state=None):
    """Mirror of the patched ReinforceTrainer.fit / save_logs_and_checkpoints logic.
    Returns (trained micro-batch indices, saved resume states, eval saves)."""
    plan = ResumePlan.from_state(state, stop_update=stop, resume_every=resume_every, keep_every=keep_every)
    trained, saves, evals = [], [], []
    if plan.already_done:
        return trained, saves, evals
    past = plan.skip_micro_batches if state else 0
    global_step = 1
    for batch in range(n_batches):
        if past > 0:
            past -= 1
            global_step += 1
            continue
        trained.append(batch)
        if global_step % update_timesteps == 0:
            update = global_step // update_timesteps
            consumed = global_step
            if plan.save_eval_weights(update):
                evals.append(update)
            if plan.save_resume_state(update):
                saves.append({"update_step": update, "consumed_micro_batches": consumed})
            if plan.is_stop(update):
                return trained, saves, evals
            if crash_after is not None and update == crash_after:
                raise RuntimeError(("crash", trained, saves, evals))
        global_step += 1
    return trained, saves, evals


def test_treerl_loop_resumes_exactly_where_it_left_off():
    ref, ref_saves, ref_evals = simulate_treerl(400, 32 // 4, stop=22, resume_every=5, keep_every=10)
    with pytest.raises(RuntimeError) as e:
        simulate_treerl(400, 8, stop=22, resume_every=5, keep_every=10, crash_after=13)
    _, part, saves, _ = e.value.args[0]
    last = saves[-1]                                  # resume from update 10
    assert last["update_step"] == 10
    rest, _, evals = simulate_treerl(400, 8, stop=22, resume_every=5, keep_every=10, state=last)
    # batches 0..79 (updates 1-10) from the first launch, the rest recomputed after resume
    combined = part[: last["consumed_micro_batches"]] + rest
    assert combined == ref
    assert evals == [20, 22] and ref_evals == [10, 20, 22]
    # resuming a finished run does nothing
    assert simulate_treerl(400, 8, stop=22, resume_every=5, keep_every=10, state=ref_saves[-1]) == ([], [], [])


# --------------------------------------------------------------------------- status and GPU-hours

def _meta(d, *events):
    with open(d / "run_meta.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_run_status_all_frameworks(tmp_path):
    t = tmp_path / "tt"
    (t / "checkpoints" / "ckpt--iter_25--epoch_1--step_100" / "hf_pretrained").mkdir(parents=True)
    (t / "checkpoints" / "ckpt--iter_45--epoch_1--step_180").mkdir(parents=True)
    _meta(t, {"event": "start", "time": time.time() - 9000, "stop": 220, "framework": "treetune", "run": "R2", "seed": 42},
          {"event": "end", "time": time.time() - 8000, "rc": 1})
    assert run_status.status(str(t))["resumable_update"] == 45
    assert run_status.status(str(t))["status"] == "incomplete"

    v = tmp_path / "verl"
    (v / "ckpts").mkdir(parents=True)
    (v / "ckpts" / "latest_checkpointed_iteration.txt").write_text("220")
    _meta(v, {"event": "start", "time": time.time() - 100, "stop": 220, "framework": "verl"})
    assert run_status.status(str(v))["status"] == "done"

    o = tmp_path / "orl"
    (o / "ckpt" / "_actor_resume").mkdir(parents=True)
    (o / "ckpt" / "_actor_resume" / "latest").write_text("update35")
    _meta(o, {"event": "start", "time": time.time() - 30, "stop": 220, "framework": "openrlhf"},
          {"event": "beat", "time": time.time() - 5})
    s = run_status.status(str(o))
    assert s["resumable_update"] == 35 and s["status"] == "running"


def test_gpu_hours_exclude_crash_loss_and_downtime(tmp_path):
    d = tmp_path / "run"
    (d / "ckpts").mkdir(parents=True)
    t0 = 1_000_000.0
    # launch 1: 0-3600 s, checkpoint at 1800 s, crash (last heartbeat 3600 s); 2 h downtime;
    # launch 2: 10800-12600 s, checkpoint at 12600 s, clean exit
    _meta(d, {"event": "start", "time": t0}, {"event": "beat", "time": t0 + 3600},
          {"event": "start", "time": t0 + 10800}, {"event": "end", "time": t0 + 12600, "rc": 0})
    with open(d / "ckpt_times.jsonl", "w") as f:
        f.write(json.dumps({"step": 5, "mtime": t0 + 1800}) + "\n")
        f.write(json.dumps({"step": 10, "mtime": t0 + 12600}) + "\n")
    h = gpu_hours.ckpt_hours(str(d), gpus=2, pattern="ckpts/global_step_*", step_re=r"global_step_(\d+)")
    assert h[5] == pytest.approx(1.0)          # 0.5 h x 2 GPUs
    assert h[10] == pytest.approx(2.0)         # + 0.5 h productive in launch 2; crash loss and downtime excluded


def test_compute_matched_target_does_not_grow_on_rerun():
    hybrid = {0: 0.0, 220: 12.0}
    first = compute_target({0: 0.0, 220: 10.0}, hybrid, 220, 5)
    assert first == 265
    # Once the saved continuation already meets the target hours, rerunning the
    # continuation command keeps its current checkpoint as the target.
    assert compute_target({0: 0.0, 220: 10.0, first: 12.1}, hybrid, 220, 5) == first


# --------------------------------------------------------------------------- wrapper and resume-all

FAKE = r'''
mkdir -p "$RUN_DIR/checkpoints"
last=$(ls "$RUN_DIR/checkpoints" | sed -n 's/ckpt--iter_\([0-9]*\)--.*/\1/p' | sort -n | tail -1); last=${last:-0}
echo "start at $last" >> "$RUN_DIR/fake.log"
for ((u=last+1; u<=STOP; u++)); do
  if (( u % RESUME_EVERY == 0 )); then mkdir -p "$RUN_DIR/checkpoints/ckpt--iter_${u}--epoch_0--step_${u}"; fi
  if [[ ! -f "$RUN_DIR/crashed" && $u -eq 7 ]]; then touch "$RUN_DIR/crashed"; exit 3; fi
done
'''


def _wrapper_env(tmp_path, **kw):
    env = dict(os.environ, WORK=str(tmp_path), RUNS_DIR=str(tmp_path / "runs"), RESTART_DELAY="0",
               RESUME_EVERY="5", MAX_RESTARTS="2", VDRA_FAKE_TRAIN_CMD=FAKE, DRY_RUN="0",
               EVAL_EVERY_N="0", EVAL_AT_END="0")
    _prepare_bash_environment(env, tmp_path)
    env.update({k: str(v) for k, v in kw.items()})
    return env


def test_wrapper_restarts_after_crash_and_skips_when_done(tmp_path):
    env = _wrapper_env(tmp_path)
    cmd = [_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "20", "wrap"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    d = tmp_path / "runs" / "wrap"
    assert (d / "fake.log").read_text().split("\n")[:2] == ["start at 0", "start at 5"]
    assert run_status.resumable_update(str(d), "treetune") == 20
    events = [json.loads(l) for l in open(d / "run_meta.jsonl")]
    starts = [e for e in events if e["event"] == "start"]
    ends = [e for e in events if e["event"] == "end"]
    assert [s["resume_from"] for s in starts] == [0, 5] and [e["rc"] for e in ends] == [3, 0]
    # finished: a second call does nothing
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert "nothing to do" in r2.stderr and (d / "fake.log").read_text().count("start at") == 2
    # continuing to a larger STOP resumes from 20
    r3 = subprocess.run([_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "30", "wrap"],
                        env=env, capture_output=True, text=True)
    assert r3.returncode == 0 and "start at 20" in (d / "fake.log").read_text()


def test_wrapper_gives_up_after_max_restarts(tmp_path):
    env = _wrapper_env(tmp_path, MAX_RESTARTS=0, VDRA_FAKE_TRAIN_CMD="exit 9")
    r = subprocess.run([_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "20", "fail"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "after 1 attempts" in r.stderr


def test_resume_all_builds_jobs_from_last_launch(tmp_path):
    runs = tmp_path / "runs"
    d = runs / "R4_s123"
    (d / "ckpts").mkdir(parents=True)
    (d / "ckpts" / "latest_checkpointed_iteration.txt").write_text("60")
    _meta(d, {"event": "start", "time": time.time() - 9000, "run": "R4", "seed": 123, "stop": 220,
              "framework": "verl", "keep_every": 25, "resume_every": 5, "variant": "{\"weighting\": \"incoming\"}"},
          {"event": "end", "time": time.time() - 8000, "rc": 1})
    done = runs / "R3_s123"
    (done / "ckpts").mkdir(parents=True)
    (done / "ckpts" / "latest_checkpointed_iteration.txt").write_text("220")
    _meta(done, {"event": "start", "time": time.time() - 9000, "run": "R3", "seed": 123, "stop": 220, "framework": "verl"},
          {"event": "end", "time": time.time() - 8000, "rc": 0})
    env = dict(os.environ, WORK=str(tmp_path), RUNS_DIR=str(runs), DRY_RUN="1")
    _prepare_bash_environment(env, tmp_path)
    r = subprocess.run([_bash_executable(), str(ROOT / "scripts/22_resume_all.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "R4_s123" in r.stdout and "incomplete" in r.stdout and "done" in r.stdout
    jobs = sorted((runs / "_cfg").glob("resume_jobs_*.txt"))[-1].read_text().strip().splitlines()
    assert len(jobs) == 1
    assert "20_train.sh R4 123 220 R4_s123" in jobs[0] and "KEEP_EVERY=25" in jobs[0]
    assert "weighting" in jobs[0]


# --------------------------------------------------------------------------- treetune replay buffer

def _load_resumable_buffer_module():
    """Import the resumable generator against stub treetune modules; the stub
    OnPolicyReplayBuffer is a verbatim copy of SPO's class."""
    import importlib.util
    import random
    import types
    from collections import defaultdict

    class OnPolicyReplayBuffer:
        def __init__(self):
            self.edges = []

        def add_edges(self, new_edges, iteration):
            for edge in new_edges:
                edge["iteration"] = iteration
            self.edges.extend(new_edges)

        def get_edges(self, iteration):
            new_buffer, removed = [], 0
            for edge in self.edges:
                if iteration - edge["iteration"] >= 8:
                    removed += 1
                else:
                    new_buffer.append(edge)
            self.edges = new_buffer
            by_q = defaultdict(list)
            for edge in self.edges:
                by_q[edge["question_id"]].append(edge)
            sampled = []
            for _, lst in by_q.items():
                sampled.extend(random.sample(lst, 32) if len(lst) > 32 else lst)
            self.edges = [e for e in self.edges if e not in sampled]
            return sampled, removed

    mods = {}
    for name in ["treetune", "treetune.episode_generators", "treetune.episode_generators.base_episode_generator",
                 "treetune.episode_generators.hybrid_episode_generator", "treetune.logging_utils"]:
        mods[name] = sys.modules.get(name) or types.ModuleType(name)
        sys.modules[name] = mods[name]

    class EpisodeGenerator:
        @classmethod
        def register(cls, name, exist_ok=False):
            return lambda k: k

    class HybridEpisodeGenerator:
        def __init__(self, **kw):
            self.replay_buffer_type = "on_policy"
            self.replay_buffer = OnPolicyReplayBuffer()
            self.exp_root = kw["exp_root"]

    import logging
    mods["treetune.episode_generators.base_episode_generator"].EpisodeGenerator = EpisodeGenerator
    mods["treetune.episode_generators.hybrid_episode_generator"].HybridEpisodeGenerator = HybridEpisodeGenerator
    mods["treetune.episode_generators.hybrid_episode_generator"].OnPolicyReplayBuffer = OnPolicyReplayBuffer
    mods["treetune.logging_utils"].get_logger = logging.getLogger
    path = ROOT / "integrations/igpo/treetune/episode_generators/resumable_hybrid_episode_generator.py"
    spec = importlib.util.spec_from_file_location("resumable_gen_under_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_replay_buffer_survives_a_restart(tmp_path):
    m = _load_resumable_buffer_module()

    def edges(it):
        return [{"question_id": q, "k": f"{it}-{q}-{j}"} for q in range(4) for j in range(40)]

    gen = m.ResumableHybridEpisodeGenerator(exp_root=tmp_path)
    for it in range(4):
        gen.replay_buffer.add_edges(edges(it), it)
        gen.replay_buffer.get_edges(it)
    before = sorted(e["k"] for e in gen.replay_buffer.edges)
    assert before, "buffer should hold stale edges"
    # crash; a new process resumes at iteration 4
    gen2 = m.ResumableHybridEpisodeGenerator(exp_root=tmp_path)
    gen2.replay_buffer.add_edges(edges(4), 4)
    restored = sorted(e["k"] for e in gen2.replay_buffer.edges if not e["k"].startswith("4-"))
    assert restored == before
    assert len(list((tmp_path / "replay_buffer").glob("*.pkl"))) <= 3


# --------------------------------------------------------------------------- in-training evaluation

FAKE_KEEP = r"""
mkdir -p "$RUN_DIR/checkpoints"
last=$(ls "$RUN_DIR/checkpoints" | sed -n 's/ckpt--iter_\([0-9]*\)--.*/\1/p' | sort -n | tail -1); last=${last:-0}
echo "train $last->$STOP" >> "$FAKE_LOG"
for ((u=last+1; u<=STOP; u++)); do
  if (( u % RESUME_EVERY == 0 )); then
    c="$RUN_DIR/checkpoints/ckpt--iter_${u}--epoch_0--step_${u}"; mkdir -p "$c"
    if (( u % 10 == 0 )); then mkdir -p "$c/hf_pretrained"; fi
  fi
  if [[ -n "$CRASH_AT" && ! -f "$RUN_DIR/crashed" && $u -eq $CRASH_AT ]]; then touch "$RUN_DIR/crashed"; exit 3; fi
done
"""

FAKE_KEEP_PHASE_D = FAKE_KEEP.replace(
    "if (( u % 10 == 0 )); then mkdir -p \"$c/hf_pretrained\"; fi",
    "if (( u % RESUME_EVERY == 0 )); then mkdir -p \"$c/hf_pretrained\"; fi",
)

FAKE_EVAL = r"""
import argparse, json, os, sys
sys.path.insert(0, os.environ["VDRA_HOME"] + "/analysis")
from datasets_util import load_problems
from eval_checkpoints import evaluation_signature, limit_problems, result_path, write_result_metadata
from vdra_core import verifier
ap = argparse.ArgumentParser()
for k in ["--run_id", "--ckpt_glob", "--step", "--out_dir", "--max_model_len", "--sample_k",
          "--sample_datasets", "--sample_min_step", "--gpu_hours_json", "--tokenizer",
          "--temperature", "--seed"]:
    ap.add_argument(k)
ap.add_argument("--limit", type=int)
ap.add_argument("--steps")
ap.add_argument("--dataset", nargs="+")
a = ap.parse_args()
log_path = os.environ["FAKE_LOG"] + ".eval_gpu" + os.environ["CUDA_VISIBLE_DEVICES"]
with open(log_path, "a") as f:
    f.write(f"eval {a.step} gpu{os.environ['CUDA_VISIBLE_DEVICES']} {','.join(d.split('=')[0] for d in a.dataset)} limit={a.limit}\n")
gpu_hours = json.load(open(a.gpu_hours_json)) if a.gpu_hours_json and os.path.exists(a.gpu_hours_json) else {}
for d in a.dataset:
    benchmark, spec = d.split("=", 1)
    problems = limit_problems(load_problems(spec), a.limit)
    signature = evaluation_signature(
        run_id=a.run_id, step=int(a.step), checkpoint_path=a.ckpt_glob,
        benchmark=benchmark, dataset_spec=spec, problems=problems, mode="greedy",
        max_model_len=int(a.max_model_len), tokenizer=a.tokenizer or a.ckpt_glob,
        seed=int(a.seed), temperature=0.0, sample_k=1,
        verifier_backend=verifier.backend_name(), verifier_fingerprint=verifier.backend_fingerprint(),
        gpu_hours=gpu_hours.get(str(a.step)),
        limit=a.limit,
    )
    path = result_path(a.out_dir, a.run_id, int(a.step), benchmark, "greedy")
    with open(path, "w") as out:
        for problem in problems:
            out.write(json.dumps({
                "run_id": a.run_id, "step": int(a.step), "benchmark": benchmark,
                "mode": "greedy", "problem_id": problem["id"],
                "verifier_backend": verifier.backend_name(),
            }) + "\n")
    write_result_metadata(path, signature, row_count=len(problems))
"""


def _eval_env(tmp_path, **kw):
    (tmp_path / "fake_eval.py").write_text(FAKE_EVAL)
    eval_specs = []
    for name in ("a", "b", "c"):
        data_path = tmp_path / f"{name}_eval.jsonl"
        data_path.write_text(json.dumps({
            "id": name,
            "problem": "What is 1 + 1?",
            "answer": "2",
        }) + "\n")
        eval_specs.append(f"{name}={data_path.as_posix()}")
    env = _wrapper_env(tmp_path, VDRA_FAKE_TRAIN_CMD=FAKE_KEEP, FAKE_LOG=tmp_path / "order.log",
                       RESULTS_DIR=tmp_path / "results", GPUS="0,1", EVAL_EVERY_N=10, EVAL_AT_END=1,
                       EVAL_STEP0=1, SAMPLE_K=0, MODEL_PATH=tmp_path / "base", CRASH_AT="",
                       EVAL_SETS=" ".join(eval_specs), VDRA_HOME=ROOT,
                       VDRA_FAKE_EVAL_CMD=f"{Path(sys.executable).as_posix()} {(tmp_path / 'fake_eval.py').as_posix()}")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _fake_eval_log_lines(tmp_path):
    return [
        line
        for path in sorted(tmp_path.glob("order.log.eval_gpu*"))
        for line in path.read_text().splitlines()
    ]


def test_training_pauses_for_evaluation_every_n_updates(tmp_path):
    env = _eval_env(tmp_path, CRASH_AT=13)
    cmd = [_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "25", "ev"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    train_log = (tmp_path / "order.log").read_text().splitlines()
    eval_log = _fake_eval_log_lines(tmp_path)
    # base model, train to 10, eval 10, crash at 13 -> restart from 10, eval 20, train to 25 (STOP)
    assert [line for line in train_log if line.startswith("train ")] == [
        "train 0->10",
        "train 10->20",
        "train 10->20",
        "train 20->25",
    ]
    # Each checkpoint is evaluated on both GPUs. Per-GPU logs avoid concurrent
    # fake subprocesses overwriting one another on Windows.
    assert len(eval_log) == 6, eval_log
    for step in (0, 10, 20):
        assert {line.split()[2] for line in eval_log if line.split()[1] == str(step)} == {"gpu0", "gpu1"}
    metadata = [json.loads(line) for line in open(tmp_path / "runs" / "ev" / "run_meta.jsonl")]
    event_order = []
    for row in metadata:
        if row["event"] == "eval_start":
            event_order.append(("eval", row["step"]))
        elif row["event"] == "start":
            event_order.append(("train", row["chunk_stop"]))
    assert event_order == [
        ("eval", 0), ("train", 10), ("eval", 10), ("train", 20),
        ("train", 20), ("eval", 20), ("train", 25),
    ]
    out = tmp_path / "results" / "eval"
    for step in (0, 10, 20):
        for b in "abc":
            assert (out / f"ev.step{step}.{b}.greedy.jsonl").exists()
    # evaluation time is not training time: no eval inside a training segment
    d = tmp_path / "runs" / "ev"
    ev = metadata
    open_seg = False
    for e in ev:
        if e["event"] == "start":
            open_seg = True
        elif e["event"] == "end":
            open_seg = False
        elif e["event"] == "eval_start":
            assert not open_seg
    assert [s["chunk_stop"] for s in ev if s["event"] == "start"] == [10, 20, 20, 25]
    # rerunning a finished run evaluates nothing new
    n = len(train_log) + len(eval_log)
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True)
    new_log_count = len((tmp_path / "order.log").read_text().splitlines()) + len(_fake_eval_log_lines(tmp_path))
    assert r2.returncode == 0 and new_log_count == n


def test_mock_evaluation_limit_and_steps_reach_evaluator(tmp_path):
    env = _eval_env(
        tmp_path,
        EVAL_EVERY_N=2,
        RESUME_EVERY=2,
        VDRA_EVAL_LIMIT=16,
        VDRA_EVAL_STEPS="0,2,4",
        VDRA_FAKE_TRAIN_CMD=FAKE_KEEP_PHASE_D,
        GPUS="0",
    )
    bash_env = tmp_path / "bash_env.sh"
    bash_env.write_text('python3() { py -3.12 "$@"; }\n')
    env["BASH_ENV"] = str(bash_env)
    shell = _bash_executable()
    r = subprocess.run([shell, str(ROOT / "scripts/20_train.sh"), "R1", "42", "4", "mock_eval"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    evals = _fake_eval_log_lines(tmp_path)
    assert {line.split()[1] for line in evals if line.startswith("eval ")} == {"0", "2", "4"}
    assert all("limit=16" in line for line in evals if line.startswith("eval "))


def test_local_six_run_orchestration_uses_fake_trainers(tmp_path):
    work = tmp_path / "work"
    igpo, treepo, treerl = work / "igpo", work / "TreePO", work / "TreeRL"
    data = work / "data_vdra"
    for path in (igpo, treepo, treerl, data / "math-nodiag"):
        path.mkdir(parents=True, exist_ok=True)
    for path in (data / "train.parquet", data / "train_treerl.jsonl"):
        path.touch()

    fake = r'''
echo "$RUN_DIR" >> "$RUN_DIR/fake-trainer.log"
mkdir -p "$RUN_DIR/checkpoints/ckpt--iter_${STOP}--epoch_0--step_${STOP}"
mkdir -p "$RUN_DIR/ckpts/global_step_${STOP}/actor"
printf '%s\n' "$STOP" > "$RUN_DIR/ckpts/latest_checkpointed_iteration.txt"
mkdir -p "$RUN_DIR/ckpt/_actor_resume" "$RUN_DIR/ckpt/_actor_global_step${STOP}"
printf 'global_step_%s\n' "$STOP" > "$RUN_DIR/ckpt/_actor_resume/latest"
'''
    bash_env = tmp_path / "bash_env.sh"
    if sys.platform == "win32":
        bash_env.write_text('python3() { py -3.12 "$@"; }\n')
        git_candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe",
            Path("C:/Program Files/Git/bin/bash.exe"),
        ]
        git_bash = next((path for path in git_candidates if path.is_file()), None)
        shell = _bash_executable() if git_bash else (shutil.which("bash") or "bash")
    else:
        bash_env.write_text("python3() { command python3 \"$@\"; }\n")
        shell = _bash_executable()

    env = dict(
        os.environ,
        VDRA_HOME=str(ROOT),
        WORK=str(work),
        IGPO_DIR=str(igpo),
        TREEPO_DIR=str(treepo),
        TREERL_DIR=str(treerl),
        DATA_DIR=str(data),
        RUNS_DIR=str(work / "runs"),
        RESULTS_DIR=str(work / "results"),
        BANKS_DIR=str(work / "banks"),
        STOP_STEP="2",
        SEEDS="42",
        RESUME_EVERY="1",
        SAVE_EVERY="1",
        MAX_RESTARTS="0",
        EVAL_EVERY_N="0",
        EVAL_AT_END="0",
        EVAL_STEP0="0",
        COMPUTE_MATCHED="0",
        DRY_RUN="0",
        VDRA_FAKE_TRAIN_CMD=fake,
        BASH_ENV=str(bash_env),
    )
    _prepare_bash_environment(env, tmp_path)

    launchers = {
        "R1": "R1_spo.sh",
        "R2": "R2_spo_vdra.sh",
        "R3": "R3_treepo.sh",
        "R4": "R4_treepo_vdra.sh",
        "R5": "R5_treerl.sh",
        "R6": "R6_treerl_vdra.sh",
    }

    def launch(item):
        exp, launcher = item
        result = subprocess.run(
            [shell, str(ROOT / "scripts" / "exp" / launcher)],
            env=dict(env, GPU="0"),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{exp}: {result.stderr[-2000:]}"
        return exp

    with ThreadPoolExecutor(max_workers=6) as pool:
        completed = list(pool.map(launch, launchers.items()))
    assert set(completed) == set(launchers)

    framework = {"R1": "treetune", "R2": "treetune", "R3": "verl", "R4": "verl", "R5": "openrlhf", "R6": "openrlhf"}
    for exp in launchers:
        run_dir = work / "runs" / f"{exp}_s42"
        assert (run_dir / "fake-trainer.log").is_file()
        assert run_status.resumable_update(str(run_dir), framework[exp]) == 2
        starts = [json.loads(line) for line in (run_dir / "run_meta.jsonl").read_text().splitlines()]
        starts = [row for row in starts if row.get("event") == "start"]
        assert starts and all(row.get("gpus") == "0" for row in starts)


def test_active_modal_session_spend_is_allocated_once(tmp_path):
    from modal_app.status import _active_session_costs

    training = tmp_path / "training.json"
    training.write_text(json.dumps({
        "call_id": "train-1",
        "experiment": "R1",
        "seeds": ["42", "123"],
        "started_at": 100.0,
        "status": "running",
    }))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "call_id": "eval-1",
        "run_id": "R3_s42",
        "started_at": 150.0,
        "status": "running",
    }))

    count, by_run, call_ids, unallocated = _active_session_costs(
        [str(training), str(evaluation)], now=200.0
    )
    assert count == 2
    assert call_ids == {"train-1", "eval-1"}
    assert by_run["R1_s42"] == pytest.approx(100 * 0.001736 / 2)
    assert by_run["R1_s123"] == pytest.approx(100 * 0.001736 / 2)
    assert by_run["R3_s42"] == pytest.approx(50 * 0.001736)
    assert unallocated == 0.0


def test_eval_every_n_must_align_with_treetune_resume(tmp_path):
    env = _eval_env(tmp_path, EVAL_EVERY_N=7)
    r = subprocess.run([_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "20", "bad"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "multiple of RESUME_EVERY" in r.stderr


def test_eval_failure_does_not_stop_training(tmp_path):
    env = _eval_env(tmp_path, VDRA_FAKE_EVAL_CMD="false")
    r = subprocess.run([_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "20", "evfail"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert run_status.resumable_update(str(tmp_path / "runs" / "evfail"), "treetune") == 20
    assert "training continues" in r.stderr


def test_eval_pending_split_and_skip(tmp_path):
    import eval_pending
    plan = eval_pending.split_over_gpus(["s", "m", "l", "xl"], {"s": 30, "m": 40, "l": 500, "xl": 900}, ["0", "1"])
    assert plan == {"0": ["xl"], "1": ["l", "m", "s"]}
    assert eval_pending.split_over_gpus(["a"], {}, ["0", "1", "2"]) == {"0": ["a"]}

    class A:
        out_dir = str(tmp_path); sample_k = 16; sample_datasets = "aime24"; sample_min_step = 100
    from eval_checkpoints import result_path
    open(result_path(A.out_dir, "r", 100, "aime24", "greedy"), "w").close()
    open(result_path(A.out_dir, "r", 100, "math500", "greedy"), "w").close()
    # Filename-only legacy outputs cannot prove which data/config/checkpoint produced them.
    assert eval_pending.pending("r", 100, {"aime24": "", "math500": ""}, A) == [
        "aime24", "math500"
    ]
    for b in ("aime24", "math500"):
        open(result_path(A.out_dir, "r", 50, b, "greedy"), "w").close()
    assert eval_pending.pending("r", 50, {"aime24": "", "math500": ""}, A) == [
        "aime24", "math500"
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="process reaping check requires Linux /proc")
def test_leftover_trainer_processes_are_reaped_before_eval(tmp_path):
    pidfile = tmp_path / "orphan.pid"
    fake = FAKE_KEEP + f'\nnohup sleep 1000 >/dev/null 2>&1 & echo $! > {pidfile}\n'
    env = _eval_env(tmp_path, VDRA_FAKE_TRAIN_CMD=fake, EVAL_EVERY_N=0, EVAL_STEP0=0)
    r = subprocess.run([_bash_executable(), str(ROOT / "scripts/20_train.sh"), "R1", "42", "10", "reap"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    pid = int(pidfile.read_text())
    alive = os.path.exists(f"/proc/{pid}") and "Z" not in open(f"/proc/{pid}/stat").read().split()[2]
    assert not alive and "[reap]" in r.stdout + r.stderr
    assert "eval 10" in "\n".join(_fake_eval_log_lines(tmp_path))      # EVAL_AT_END


# --------------------------------------------------------------------------- TreeRL single-GPU weight sync

def _colocated_sync_fn():
    """Extract _sync_to_colocated_vllm from our TreeRL patch (no torch/ray needed)."""
    import textwrap
    lines, grab = [], False
    for l in open(ROOT / "integrations/treerl/vdra_treerl.patch"):
        if l.startswith("+    def _sync_to_colocated_vllm"):
            grab = True
        elif grab and (not l.startswith("+") or l.startswith("+    def ") or l.startswith("+    # def")):
            break
        if grab:
            lines.append(l[1:])
    assert lines, "function not found in patch"
    return textwrap.dedent("".join(lines))


def test_treerl_colocated_sync_sends_every_weight_once():
    import types
    src = _colocated_sync_fn()
    sent = []

    class P:
        def __init__(self, n): self.n = n; self.data = self
        def detach(self): return self
        def to(self, *a, **k): return self
        def numel(self): return self.n
        def element_size(self): return 2

    class Engine:
        class update_weights_from_cpu:
            @staticmethod
            def remote(ref): sent.append(ref); return ref

    for rank in (0, 1):
        sent.clear()
        ns = {
            "os": types.SimpleNamespace(environ={"VDRA_TREERL_SYNC_BUCKET_MB": "1"}),
            "torch": types.SimpleNamespace(distributed=types.SimpleNamespace(get_rank=lambda: rank),
                                           cuda=types.SimpleNamespace(empty_cache=lambda: None)),
            "deepspeed": types.SimpleNamespace(zero=types.SimpleNamespace(
                GatheredParameters=lambda *a, **k: __import__("contextlib").nullcontext())),
            "_z3_params_to_fetch": lambda ps: [],
            "ray": types.SimpleNamespace(put=lambda x: list(x), get=lambda x: x),
        }
        exec(src, ns)
        params = [(f"w{i}", P(300_000)) for i in range(7)]          # 0.6 MB each, bucket 1 MB
        self = types.SimpleNamespace(
            actor=types.SimpleNamespace(model=types.SimpleNamespace(module=types.SimpleNamespace(
                named_parameters=lambda: iter(params)))),
            strategy=types.SimpleNamespace(args=types.SimpleNamespace(zero_stage=2)),
            vllm_engines=[Engine()])
        ns["_sync_to_colocated_vllm"](self)
        names = [n for bucket in sent for n, _ in bucket]
        if rank == 0:
            assert names == [n for n, _ in params] and len(sent) == 4     # buckets of 2,2,2,1
        else:
            assert names == []


def test_one_gpu_experiment_script_runs_all_seeds_on_its_gpu(tmp_path):
    (tmp_path / "igpo").mkdir(); (tmp_path / "data" / "math-nodiag").mkdir(parents=True)
    env = _eval_env(tmp_path, IGPO_DIR=tmp_path / "igpo", DATA_DIR=tmp_path / "data", SEEDS="1 2",
                    STOP_STEP=20, COMPUTE_MATCHED=0, GPU=3, GPUS="")
    r = subprocess.run([_bash_executable(), str(ROOT / "scripts/exp/R1_spo.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    for seed in (1, 2):
        d = tmp_path / "runs" / f"R1_s{seed}"
        assert run_status.resumable_update(str(d), "treetune") == 20
        starts = [json.loads(l) for l in open(d / "run_meta.jsonl") if '"start"' in l]
        assert {s["gpus"] for s in starts} == {"3"}
    log = "\n".join(_fake_eval_log_lines(tmp_path))
    assert "eval 10 gpu3" in log and "eval 20 gpu3" in log
    assert (tmp_path / "runs" / "_logs" / "R1_gpu3.log").exists()
    bad = subprocess.run([_bash_executable(), str(ROOT / "scripts/exp/R1_spo.sh")], env=dict(env, GPU="0,1"),
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "one GPU id" in bad.stderr
