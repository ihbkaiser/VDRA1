import gzip
import json

from rebuttal_igpo.rebuttal.node_bank import ResumableBank, collect_node_bank


def test_bank_resume_skips_completed_nodes_and_appends_new_ones(tmp_path):
    path = tmp_path / "bank.jsonl.gz"
    bank = ResumableBank(path)
    assert bank.completed_ids == set()
    bank.append({"node_id": "a", "prompt_id": "p1"})

    resumed = ResumableBank(path)
    assert resumed.completed_ids == {"a"}
    assert resumed.append({"node_id": "a", "prompt_id": "p1"}) is False
    assert resumed.append({"node_id": "b", "prompt_id": "p2"}) is True

    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert [row["node_id"] for row in rows] == ["a", "b"]


def test_bank_seed_assignment_is_resume_stable_and_child_uses_full_prefix(tmp_path):
    path = tmp_path / "bank.jsonl.gz"
    seen = []
    progress = []

    def continuation(parent, n, seed):
        seen.append((parent["node_id"], parent.get("prefix"), seed))
        return [{"full_text": str(parent.get("prefix", "")) + " child"} for _ in range(n)]

    collect_node_bank(
        [{
            "node_id": "prompt-a:depth-1:0", "prompt_id": "prompt-a", "depth": 1,
            "prefix": "root", "parent_generation_seed": 77,
            "candidates": [{"full_text": "root candidate"}],
        }],
        path,
        proxy_fn=lambda parent: {"sigma2": 0.2},
        continuation_fn=continuation,
        reward_fn=lambda parent, samples: [0.0] * len(samples),
        mc_per_half=2,
        child_mc=2,
        seed=42,
        progress_callback=progress.append,
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        row = json.loads(next(stream))
    assert row["R_a"]["seed"] == row["R_b"]["seed"] - 1
    assert row["parent_metadata"]["parent_generation_seed"] == 77
    assert progress[0]["completed_nodes"] == 1
    assert progress[0]["last_node_id"] == "prompt-a:depth-1:0"
    child_call = next(call for call in seen if ":candidate:" in call[0])
    assert child_call[1] == "root candidate"
    assert row["child_values"][0]["seed"] == child_call[2]
