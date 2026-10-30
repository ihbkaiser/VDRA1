"""Single entry point for tree inspection and synthetic identity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import alloc_stats, io
from .bank_analysis import run_a2, run_a4, run_a5
from .eval_bootstrap import run as run_eval_bootstrap
from .sim_eq5_eq7 import main as sim_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rebuttal_igpo.rebuttal.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    walk_cmd = sub.add_parser("walk", help="summarize trees from a run directory")
    walk_cmd.add_argument("run_dir")
    walk_cmd.add_argument("--iteration", type=int, action="append")
    walk_cmd.add_argument("--max-per-iteration", type=int)
    alloc_cmd = sub.add_parser("alloc-stats", help="write allocation CSVs from stored trees")
    alloc_cmd.add_argument("run_dir")
    alloc_cmd.add_argument("output_dir")
    alloc_cmd.add_argument("--max-per-iteration", type=int)
    a2_cmd = sub.add_parser("bank-a2", help="run A2 proxy-noise sensitivity on a completed bank")
    a2_cmd.add_argument("bank")
    a2_cmd.add_argument("output_dir")
    a2_cmd.add_argument("--seed", type=int, default=42)
    a4_cmd = sub.add_parser("bank-a4", help="run A4 TV/value-gap calibration on a completed bank")
    a4_cmd.add_argument("bank")
    a4_cmd.add_argument("output_dir")
    a5_cmd = sub.add_parser("bank-a5", help="run A5 allocation/error diagnostics on a completed bank")
    a5_cmd.add_argument("bank")
    a5_cmd.add_argument("output_dir")
    sub.add_parser("sim-eq5-eq7", help="run seeded synthetic identity checks")
    eval_cmd = sub.add_parser("eval-bootstrap", help="paired prompt bootstrap for saved A7 results")
    eval_cmd.add_argument("--spo-root", required=True)
    eval_cmd.add_argument("--vdra-root", required=True)
    eval_cmd.add_argument("--evaluator-root", required=True)
    eval_cmd.add_argument("--output-dir", required=True)
    eval_cmd.add_argument("--datasets", default="aime24,aime25,amc23")
    eval_cmd.add_argument("--samples", type=int, default=16)
    eval_cmd.add_argument("--replicates", type=int, default=1000)
    eval_cmd.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "sim-eq5-eq7":
        return sim_main()
    if args.command == "eval-bootstrap":
        result = run_eval_bootstrap(
            spo_root=args.spo_root,
            vdra_root=args.vdra_root,
            evaluator_root=args.evaluator_root,
            output_dir=args.output_dir,
            datasets=args.datasets.split(","),
            samples=args.samples,
            replicates=args.replicates,
            seed=args.seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "bank-a2":
        print(json.dumps(run_a2(args.bank, args.output_dir, seed=args.seed), sort_keys=True))
        return 0
    if args.command == "bank-a4":
        print(json.dumps(run_a4(args.bank, args.output_dir), sort_keys=True))
        return 0
    if args.command == "bank-a5":
        print(json.dumps(run_a5(args.bank, args.output_dir), sort_keys=True))
        return 0
    if args.command == "walk":
        for iteration, shard, index, tree in io.iter_trees(
            args.run_dir, iters=args.iteration, max_per_iter=args.max_per_iteration
        ):
            print(json.dumps({"iteration": iteration, "shard": shard, "index": index,
                              "nodes": list(io.walk(tree))}, ensure_ascii=False))
        return 0
    if args.command == "alloc-stats":
        nodes = []
        for iteration, shard, index, tree in io.iter_trees(
            args.run_dir, max_per_iter=args.max_per_iteration
        ):
            prompt_id = f"{iteration}:{shard}:{index}"
            for row in io.walk(tree):
                nodes.append({**row, "iteration": iteration, "prompt_id": prompt_id})
        summaries, queues = alloc_stats.analyze_records(nodes)
        out = Path(args.output_dir)
        alloc_stats.write_csv(out / "allocation_statistics.csv", summaries)
        alloc_stats.write_csv(out / "queue_level.csv", queues)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
