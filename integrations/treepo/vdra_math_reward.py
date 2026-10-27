"""verl custom reward for TreePO runs: the same math verifier as every other run.

Enable with
  custom_reward_function.path=<this file> custom_reward_function.name=compute_score
"""

import os
import sys

sys.path.insert(0, os.environ.get("VDRA_HOME", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from vdra_core import verifier  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    problem = (extra_info or {}).get("problem", "") if isinstance(extra_info, dict) else ""
    score = float(verifier.score(solution_str, str(ground_truth), problem))
    return {"score": score, "acc": score, "extra_info": {}}
