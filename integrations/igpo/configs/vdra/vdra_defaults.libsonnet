// VDRA hyperparameters (Technical Supplement, Table 4).
// Override any field by chaining another overlay that sets `vdra+: {...}`.
{
  episode_generator+: {
    inference_strategy+: {
      vdra: {
        k0: 6,
        pilot_tokens: 96,
        r: 6,
        m: 32,
        eps_tail: 0.01,
        eps_trunc: 0.02,
        terminal_pair_mode: 'answer',
        n_min: 1,
        n_queues: 32,
        queue_capacity: 16,
        queue_timeout_s: 1.0,
        weighting: 'equal',       // 'incoming' = lambda-weighted variant (Exp 2 / revision)
        proxy: 'tv',              // 'constant' = null test, 'raw_tv' = no tail correction, no cap
        use_redundancy_cap: true,
        reuse_terminal_pilots: true,
        log_dir: '',
      },
    },
  },
}
