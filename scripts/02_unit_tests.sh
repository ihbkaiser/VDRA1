#!/usr/bin/env bash
# CPU-only tests: solver optimality, TV estimator, queues, tree builder,
# Exp 1/2 against synthetic banks with known answers, statistics pipeline.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "${VDRA_HOME}" && run python3 -m pytest -q tests
