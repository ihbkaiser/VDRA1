from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_framework_trainers_have_phase_d_metric_capture_hook():
    files = {
        "treetune": ROOT / "work/igpo/treetune/trainers/restem_trainer.py",
        "verl": ROOT / "work/TreePO/verl/trainer/ppo/ray_trainer.py",
        "openrlhf": ROOT / "work/TreeRL/openrlhf/trainer/reinforce_trainer.py",
    }
    missing = [framework for framework, path in files.items() if "write_training_metrics" not in path.read_text(encoding="utf-8")]
    assert not missing, f"trainers lack Phase D metric capture hooks: {missing}"


def test_phase_d_metric_hooks_are_reproduced_by_setup_integrations():
    igpo_installer = (ROOT / "integrations/igpo/install.sh").read_text(encoding="utf-8")
    treepo_patch = (ROOT / "integrations/treepo/vdra_treepo.patch").read_text(encoding="utf-8")
    treerl_patch = (ROOT / "integrations/treerl/vdra_treerl.patch").read_text(encoding="utf-8")

    assert "restem_metrics.patch" in igpo_installer
    assert "write_training_metrics" in treepo_patch
    assert "write_training_metrics" in treerl_patch


def test_phase_d_enables_tree_rl_sync_evidence_and_metrics_paths():
    phase_d = (ROOT / "modal_app/phase_d.py").read_text(encoding="utf-8")
    assert "VDRA_TREERL_VERIFY_SYNC" in phase_d
    assert "VDRA_TREERL_SYNC_REPORT" in phase_d
    assert "VDRA_METRICS_JSONL" in phase_d
    assert "_checkpoint_updates(run_dir, FRAMEWORK_NAMES[framework])" in phase_d


def test_treetune_captures_the_final_training_update_when_logging_interval_does_not_align():
    trainer = (ROOT / "work/igpo/treetune/trainers/restem_trainer.py").read_text(encoding="utf-8")
    assert "globalstep_last_logged != self.state.global_step" in trainer


def test_phase_d_and_phase_zero_through_full_analysis_require_hybrid_budget_evidence():
    phase_d = (ROOT / "modal_app/phase_d.py").read_text(encoding="utf-8")
    assert phase_d.count('audit_command.append("--require-budget-evidence")') == 2

    null_tests = (ROOT / "scripts/10_null_tests.sh").read_text(encoding="utf-8")
    pilot = (ROOT / "scripts/11_pilot.sh").read_text(encoding="utf-8")
    full_analysis = (ROOT / "scripts/60_analysis.sh").read_text(encoding="utf-8")
    assert "--require-budget-evidence" in null_tests
    assert "budget_args=(--require-budget-evidence)" in pilot
    assert "--require-budget-evidence" in full_analysis


def test_treepo_and_treerl_queue_logs_record_cap_flags_for_budget_audits():
    treepo_source = (ROOT / "work/TreePO/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py").read_text(encoding="utf-8")
    treepo_patch = (ROOT / "integrations/treepo/vdra_treepo.patch").read_text(encoding="utf-8")
    treerl_source = (ROOT / "work/TreeRL/openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py").read_text(encoding="utf-8")
    treerl_patch = (ROOT / "integrations/treerl/vdra_treerl.patch").read_text(encoding="utf-8")

    treepo_flags = '"capped": [bool(it.capped) for it in items]'
    treerl_flags = '"capped": [bool(it.capped) for it in chunk]'
    assert treepo_flags in treepo_source
    assert treepo_flags in treepo_patch
    assert treerl_flags in treerl_source
    assert treerl_flags in treerl_patch


def test_treepo_text_model_does_not_disable_vllm_multimodal_preprocessor_cache():
    treepo_root = ROOT / "work/TreePO/verl/workers/rollout/vllm_rollout"
    sPMd = (treepo_root / "vllm_rollout_spmd.py").read_text(encoding="utf-8")
    async_server = (treepo_root / "vllm_async_server.py").read_text(encoding="utf-8")
    cache_patch = (ROOT / "integrations/treepo/vllm_text_mm_cache.patch").read_text(encoding="utf-8")
    treepo_installer = (ROOT / "integrations/treepo/install.sh").read_text(encoding="utf-8")

    for source in (sPMd, async_server):
        assert "disable_mm_preprocessor_cache=False" in source
        assert "disable_mm_preprocessor_cache=True" not in source
    assert cache_patch.count("+            disable_mm_preprocessor_cache=False,") == 2
    assert "vllm_text_mm_cache.patch" in treepo_installer
