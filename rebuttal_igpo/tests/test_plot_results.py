import gzip
import json

from rebuttal_igpo.rebuttal.plot_results import _load_a3_proxy_variance


def test_a3_plot_source_hashes_prompt_ids_and_keeps_numeric_fields(tmp_path):
    bank = tmp_path / "bank.jsonl.gz"
    rows = [
        {"prompt_id": "private-prompt-alpha", "depth": 0, "proxy": {"sigma2": 0.1}, "R_b": {"rewards": [0, 1, 0]}},
        {"prompt_id": "private-prompt-alpha", "depth": 1, "proxy": {"sigma2": 0.2}, "R_b": {"rewards": [0, 0, 0]}},
        {"prompt_id": "private-prompt-beta", "depth": 0, "proxy": {"sigma2": 0.3}, "R_b": {"rewards": [1, 1, 0]}},
        {"prompt_id": "private-prompt-beta", "depth": 1, "proxy": {"sigma2": 0.4}, "R_b": {"rewards": [1, 0, 1]}},
    ]
    with gzip.open(bank, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    data, summary = _load_a3_proxy_variance(bank)

    assert len(data) == 4
    assert all(set(row) == {"prompt_hash", "depth", "proxy_sigma2", "rb_sample_variance"} for row in data)
    assert all(len(row["prompt_hash"]) == 64 for row in data)
    assert "private-prompt" not in json.dumps(data)
    assert summary["n_parents"] == 4
    assert summary["n_prompt_clusters"] == 2
