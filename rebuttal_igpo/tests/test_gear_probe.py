from rebuttal_igpo.rebuttal.gear_probe import probe_protocol
from rebuttal_igpo.rebuttal.smoke_vllm import checkpoint_description


def test_probe_protocol_records_selected_checkpoint_not_default_iteration():
    checkpoint = "/workspace/src/vdra_iter_0050/checkpoints/ckpt--iter_0050--epoch_1.00--step_0200/hf_pretrained"

    protocol = probe_protocol(checkpoint_path=checkpoint)

    assert protocol["checkpoint"] == checkpoint
    assert protocol["checkpoint"] != probe_protocol()["checkpoint"]


def test_smoke_summary_records_checkpoint_and_served_model_alias():
    checkpoint = "/workspace/src/vdra_iter_0150/checkpoints/ckpt--iter_0150--epoch_1.00--step_0600/hf_pretrained"

    description = checkpoint_description(checkpoint_path=checkpoint, model="iter150")

    assert description == {"checkpoint": checkpoint, "served_model": "iter150"}
