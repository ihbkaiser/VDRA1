"""Single entry point for tree inspection and synthetic identity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import alloc_stats, io
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
    sub.add_parser("sim-eq5-eq7", help="run seeded synthetic identity checks")
    args = parser.parse_args(argv)
    if args.command == "sim-eq5-eq7":
        return sim_main()
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
