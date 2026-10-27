// Stop after VDRA_STOP_STEP policy updates without changing num_iterations,
// so the WarmupDecayLR schedule (defined over num_iterations) is unchanged
// and a later continuation resumes on the same schedule.
{
  early_stop_iteration: std.parseInt(std.extVar('APP_VDRA_STOP_STEP')),
}
