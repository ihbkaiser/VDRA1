#!/usr/bin/env bash
# Shared helpers.  Source after env.sh.

set -euo pipefail

log() { echo "[$(date '+%F %T')] $*" >&2; }

run() {
  # Print and execute a command; DRY_RUN=1 only prints it.
  log "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

use_env() {
  # use_env <conda env name or empty>
  local name="${1:-}"
  if [[ -n "${name}" && "${DRY_RUN}" != "1" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${name}"
  fi
}

meta() {
  # meta <run_dir> <event> [json fields]  -> appends to run_meta.jsonl (used for GPU-hours)
  local dir="$1" ev="$2" extra="${3:-}"
  mkdir -p "${dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  python3 - "$dir" "$ev" "$extra" <<'PY'
import json, sys, time, os, socket
d, ev, extra = sys.argv[1], sys.argv[2], sys.argv[3]
host = os.uname().nodename if hasattr(os, "uname") else socket.gethostname()
row = {"event": ev, "time": time.time(), "host": host,
       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
if extra:
    row.update(json.loads(extra))
open(os.path.join(d, "run_meta.jsonl"), "a").write(json.dumps(row) + "\n")
PY
}

record_provenance() {
  # record_provenance <run_dir> <code_dir>
  local dir="$1" code="$2"
  mkdir -p "${dir}"
  [[ "${DRY_RUN}" == "1" ]] && return 0
  {
    echo "code_dir=${code}"
    local code_commit home_commit
    code_commit="$(git -C "${code}" rev-parse HEAD 2>/dev/null || echo none)"
    home_commit="$(git -C "${VDRA_HOME}" rev-parse HEAD 2>/dev/null || echo none)"
    echo "commit=${code_commit}"
    git -C "${code}" status --porcelain 2>/dev/null | head -50 || true
    echo "vdra_home_commit=${home_commit}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  } > "${dir}/provenance.txt"
  pip freeze > "${dir}/pip_freeze.txt" 2>/dev/null || true
}

ngpus() { local g="$1"; awk -F, '{print NF}' <<<"${g}"; }

run_gpus() {
  # run_gpus <run_dir> <default>: GPU count of the run's last launch (from run_meta.jsonl)
  python3 - "$1" "$2" <<'PY' 2>/dev/null || echo "$2"
import json, os, sys
n = sys.argv[2]
p = os.path.join(sys.argv[1], "run_meta.jsonl")
if os.path.exists(p):
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "start" and r.get("gpus"):
            n = len([g for g in str(r["gpus"]).split(",") if g.strip()])
print(n)
PY
}

vdra_json_get() {
  # vdra_json_get <json> -> "k=v" lines (for hydra overrides)
  python3 -c 'import json,sys; [print(f"{k}={str(v).lower() if isinstance(v,bool) else v}") for k,v in json.loads(sys.argv[1]).items()]' "$1"
}

run_queue() {
  # run_queue <jobs file>: each line is a command that accepts GPUS=<group> in its env.
  # Runs at most one job per GPU group concurrently; waits for all.
  local jobs="$1"
  local -a groups=(${GPU_GROUPS})
  local -a pids=()
  local -a busy=()
  local i n=0
  mkdir -p "${RUNS_DIR}/_logs"
  for i in "${!groups[@]}"; do busy[$i]=0; done
  while IFS= read -r cmd || [[ -n "${cmd}" ]]; do
    [[ -z "${cmd// }" || "${cmd}" =~ ^# ]] && continue
    if [[ "${DRY_RUN}" == "1" ]]; then
      local group_index=$((n % ${#groups[@]}))
      log "[queue] GPUS=${groups[$group_index]} :: ${cmd}"
      log "[queue] dry run; job not executed"
      n=$((n + 1))
      continue
    fi
    while true; do
      for i in "${!groups[@]}"; do
        if [[ "${busy[$i]}" == "0" ]] || ! kill -0 "${busy[$i]}" 2>/dev/null; then
          log "[queue] GPUS=${groups[$i]} :: ${cmd}"
          n=$((n + 1))
          local lf="${RUNS_DIR}/_logs/$(date +%Y%m%d-%H%M%S)-job${n}.log"
          printf '%s\t%s\t%s\n' "${lf}" "${groups[$i]}" "${cmd}" >> "${RUNS_DIR}/_logs/index.tsv"
          GPUS="${groups[$i]}" bash -c "${cmd}" > "${lf}" 2>&1 &
          busy[$i]=$!
          pids+=($!)
          continue 3
        fi
      done
      sleep 20
    done
  done < "${jobs}"
  local rc=0
  for p in "${pids[@]:-}"; do [[ -n "${p}" ]] && { wait "${p}" || rc=1; }; done
  return ${rc}
}
