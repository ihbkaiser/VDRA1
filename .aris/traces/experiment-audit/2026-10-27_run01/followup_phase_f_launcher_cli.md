# Phase F launcher CLI follow-up

Date: 2026-10-27

Both budgeted R3-R6 entry scripts lacked argument parsing, so `--help` fell through to the budget check and Modal deployment path. They now use `argparse` before any billing query or deployment call. A parameterized regression test replaces both side-effect functions with failures and confirms `main(["--help"])` exits successfully first. Direct help invocations for both scripts also printed usage and exited 0.

The focused regression passed 2/2. Full project suite: 183 passed, 1 skipped, with 2 protobuf deprecation warnings. No GPU task started in this check.
