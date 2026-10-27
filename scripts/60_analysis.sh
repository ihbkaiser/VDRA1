#!/usr/bin/env bash
# Statistics, audits and figures for everything that has finished.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
manifest="${RESULTS_DIR}/runs.csv"
mkdir -p "${RESULTS_DIR}/stats" "${RESULTS_DIR}/audit"
python3 - "${RUNS_DIR}" "${manifest}" <<'PY'
import csv, os, re, sys
runs_dir, out = sys.argv[1], sys.argv[2]
pair = {"R1": ("spo", "baseline"), "R2": ("spo", "hybrid"), "R3": ("treepo", "baseline"),
        "R4": ("treepo", "hybrid"), "R5": ("treerl", "baseline"), "R6": ("treerl", "hybrid")}
rows = []
for d in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
    m = re.fullmatch(r"(R[1-6])_s(\d+)", d)
    if m:
        p, role = pair[m.group(1)]
        rows.append({"run_id": d, "pair": p, "role": role, "seed": m.group(2)})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run_id", "pair", "role", "seed"]); w.writeheader(); w.writerows(rows)
print(f"[manifest] {len(rows)} runs -> {out}")
PY
run python3 "${VDRA_HOME}/analysis/stats.py" --eval "${RESULTS_DIR}/eval/*.jsonl" --runs "${manifest}" \
  --out "${RESULTS_DIR}/stats" --step "${STOP_STEP}"
for d in "${RUNS_DIR}"/R2_s* "${RUNS_DIR}"/R4_s* "${RUNS_DIR}"/R6_s*; do
  [[ -d "${d}" ]] && run python3 "${VDRA_HOME}/analysis/budget_audit.py" --logs "${d}" --require-budget-evidence --out "${RESULTS_DIR}/audit/$(basename "${d}").json"
done
run python3 "${VDRA_HOME}/analysis/figures.py" --results "${RESULTS_DIR}" --out "${RESULTS_DIR}/figures"
