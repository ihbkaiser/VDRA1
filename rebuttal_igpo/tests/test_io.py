import json

import pytest

from rebuttal_igpo.rebuttal.io import iter_trees, load_demos, walk


def _tree():
    return {
        "gear_segment_id": "root",
        "reward": 0.25,
        "gear_sigma2": 0.5,
        "gear_allocated_branch_factor": 2,
        "gear_tv_logp_matrix": [[-1.0]],
        "children": [
            {
                "gear_segment_id": "child",
                "reward": 1.0,
                "gear_budget_queue_id": 3,
                "children": [],
            }
        ],
    }


def test_walk_normalizes_aliases_and_drops_logp_matrix():
    rows = list(walk(_tree()))

    assert rows[0]["depth"] == 0
    assert rows[0]["n_children"] == 1
    assert rows[0]["reward_variance"] == 0.5
    assert rows[0]["allocated_branch_factor"] == 2
    assert "gear_tv_logp_matrix" not in rows[0]
    assert rows[1]["parent_id"] == "root"
    assert rows[1]["queue_id"] == 3


def test_iter_trees_reads_json_and_hf_saved_dataset(tmp_path):
    run = tmp_path / "run"
    json_dir = run / "temp_episodes" / "iteration__0002" / "infer_results" / "p0"
    json_dir.mkdir(parents=True)
    (json_dir / "tree.json").write_text(
        json.dumps({"_treetune__reasoning_tree": json.dumps(_tree())}),
        encoding="utf-8",
    )

    datasets = pytest.importorskip("datasets")
    ds_dir = run / "temp_episodes" / "iteration__0003" / "infer_results" / "p1" / "traj_results_ds"
    ds_dir.parent.mkdir(parents=True)
    datasets.Dataset.from_list(
        [{"_treetune__reasoning_tree": json.dumps(_tree())}]
    ).save_to_disk(str(ds_dir))

    rows = list(iter_trees(run))
    assert [(r[0], r[2]) for r in rows] == [(2, 0), (3, 0)]
    assert rows[0][3]["gear_segment_id"] == "root"
    assert rows[1][3]["gear_segment_id"] == "root"


def test_iter_trees_filters_iterations_and_limits_per_iteration(tmp_path):
    base = tmp_path / "run" / "temp_episodes" / "iteration__0004" / "infer_results"
    for idx in range(3):
        p = base / f"process_{idx}" / "tree.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"_treetune__reasoning_tree": _tree()}), encoding="utf-8")

    rows = list(iter_trees(tmp_path / "run", iters={4}, max_per_iter=2))
    assert len(rows) == 2


def test_load_demos_streams_jsonl(tmp_path):
    path = tmp_path / "demos.jsonl"
    path.write_text('{"prompt_id":"a","n":1}\n{"prompt_id":"b","n":2}\n', encoding="utf-8")
    assert [x["prompt_id"] for x in load_demos(path)] == ["a", "b"]
