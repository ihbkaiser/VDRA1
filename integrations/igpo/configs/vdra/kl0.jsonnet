// The paper reports beta_KL = 0; the DeepSeek MATH base config imports a 1e-4
// KL loss.  Applied to every R1/R2 run so both sides of the pair match.
{
  trainer+: {
    params+: {
      init_kl_coef: 0.0,
    },
  },
}
