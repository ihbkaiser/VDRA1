# Phase F TreeRL retry review - 2026-10-27

- Read the retained Phase F TreeRL retry log: vLLM sampler `torch.log_softmax(logits)` requested 36.98 GiB while 9.93 GiB was free; the run produced no update and timed out. `scripts/20_train.sh` records the earlier problematic prompt-logprob batch as 65,536 tokens.
- Current co-located TreeRL settings in `scripts/20_train.sh` and `modal_app/phase_c.py` use vLLM memory utilization 0.25, `max_num_batched_tokens=16384`, and engine max length 4096. The R1-R6 dry-run log explicitly shows 0.25/16,384 for R6. This is a bounded-memory mitigation in the current launch config, not proof the B200 OOM is fixed.
- Fresh read-only Modal checks: active profile is `billionaireproject8`, metered cost is $87.10290662, and the app list has no training tasks. `modal billing --help` offers summary/report/rates; `modal workspace settings set --help` permits only image-builder-version/default-environment. The workspace spend limit cannot be raised by these CLI commands. No launch was retried.
- Phase F GPU validation remains blocked by Modal's earlier workspace spend-limit rejection. Existing null-equality and R2-TV headroom gates remain unresolved.
