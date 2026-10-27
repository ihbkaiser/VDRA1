#!/usr/bin/env bash
# Find every interrupted training run under RUNS_DIR and resume it with the
# exact parameters of its last launch (RUN, seed, stop, KEEP_EVERY,
# RESUME_EVERY, VDRA variant).  Runs that are done or currently running
# (heartbeat < 10 min, or lock held) are skipped.
#
#   bash scripts/22_resume_all.sh                       # show status only
#   APPLY=1 FRAMEWORKS="treetune verl" GPU_GROUPS="0,1 2,3 4,5 6,7" bash scripts/22_resume_all.sh
#   APPLY=1 FRAMEWORKS="openrlhf"      GPU_GROUPS="0,1,2 3,4,5"     bash scripts/22_resume_all.sh
# INCLUDE="pilot_ null_" also resumes pilot / null-test runs (skipped by default).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
FRAMEWORKS="${FRAMEWORKS:-treetune verl openrlhf}"
mkdir -p "${RUNS_DIR}/_cfg"
jobs="${RUNS_DIR}/_cfg/resume_jobs_$(date +%Y%m%d-%H%M%S).txt"
python3 - "${RUNS_DIR}" "${VDRA_HOME}" "${FRAMEWORKS}" "${INCLUDE:-}" "${jobs}" <<'PY'
import json, os, shlex, sys
runs_dir, home, frameworks, include, jobs = sys.argv[1:6]
sys.path.insert(0, os.path.join(home, "analysis"))
from run_status import status
frameworks = frameworks.split()
include = include.split()
rows, lines = [], []
for name in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
    d = os.path.join(runs_dir, name)
    if name.startswith("_") or not os.path.isfile(os.path.join(d, "run_meta.jsonl")):
        continue
    if name.startswith(("pilot_", "null_")) and not any(name.startswith(p) for p in include):
        continue
    st = status(d)
    rows.append(st)
    meta = st["last_launch"] or {}
    if st["status"] != "incomplete" or st["framework"] not in frameworks:
        continue
    env = {
        "KEEP_EVERY": meta.get("keep_every"), "RESUME_EVERY": meta.get("resume_every"),
        "VDRA_VARIANT_JSON": meta.get("variant") or "{}",
    }
    prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v not in (None, ""))
    lines.append(f"{prefix} bash {home}/scripts/20_train.sh {meta['run']} {meta['seed']} {meta['stop']} {name}")
print(f"{'run':<22}{'framework':<10}{'progress':>10}{'stop':>6}  status")
for r in rows:
    print(f"{r['name']:<22}{str(r['framework']):<10}{r['resumable_update']:>10}{r['stop']:>6}  {r['status']}")
with open(jobs, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))
print(f"\n{len(lines)} run(s) to resume -> {jobs}")
PY
if [[ "${APPLY:-0}" == "1" ]]; then
  [[ -s "${jobs}" ]] && run_queue "${jobs}" || log "nothing to resume"
else
  log "status only; set APPLY=1 to resume the listed runs"
fi
