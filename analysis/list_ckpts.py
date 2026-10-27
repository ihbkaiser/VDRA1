#!/usr/bin/env python3
"""Print "<step>\t<hf_dir>" for every HF checkpoint of a run (any framework)."""
import glob
import os
import re
import sys

run_dir = sys.argv[1]
found = {}
for d in glob.glob(os.path.join(run_dir, "checkpoints", "ckpt--iter_*", "hf_pretrained")):   # treetune
    found[int(float(re.search(r"iter_([0-9.]+)", d).group(1)))] = d
for d in glob.glob(os.path.join(run_dir, "hf", "global_step_*")):                              # verl (merged)
    if os.path.exists(os.path.join(d, "config.json")):
        found[int(d.rsplit("_", 1)[1])] = d
for d in glob.glob(os.path.join(run_dir, "ckpt", "_actor_global_step*")):                     # OpenRLHF
    found[int(re.search(r"global_step(\d+)$", d).group(1))] = d
for step in sorted(found):
    print(f"{step}\t{found[step]}")
