# Eq. (5)/(7) derivation and synthetic checks

## Scope

This note derives the identities requested in `PLAN.md` from the stated estimands. The paper and supplement were not present in the archive, so it does not claim that these formulas reproduce the paper's exact notation or every assumption. Synthetic results below come from `rebuttal/sim_eq5_eq7.py` and are saved in `/workspace/rebuttal_out/eq5_eq7_simulation.json`; they are not measurements from any checkpoint or training run.

## Eq. (7): allocation for fixed coefficients

For fixed nonnegative coefficients `C_v`, positive integer sample counts `n_v`, and fixed total budget `B`, consider

```text
minimize  J(n) = Σ_v C_v / n_v
subject to Σ_v n_v = B,  lower_v ≤ n_v ≤ upper_v.
```

Without active bounds, the continuous Lagrangian gives `n_v = sqrt(C_v / μ)`, hence `n_v ∝ sqrt(C_v)`. With integer counts, the marginal decrease from adding the `(n+1)`th sample is `C_v/n − C_v/(n+1)`. Repeatedly assigning the next sample to the largest feasible marginal decrease is an exact integer solution for this separable convex objective. Bounds are handled by excluding nodes at their limits; feasibility requires `Σ lower_v ≤ B ≤ Σ upper_v` (with an infinite upper bound allowed).

This is an exact optimizer **conditional on fixed coefficients and this objective**. It does not establish that a proxy coefficient equals the true held-out variance, that sampling costs are equal, or that a complete tree objective is `Σ C_v/n_v`. The replay's `eq7_exact` is a diagnostic reference, not a description of how the archived checkpoints were trained.

For contrast, when the source allocator uses weight `sqrt(σ⁴ − λ)` and floor rounding, it is not generally the optimizer above: the implied coefficient is transformed, positive `λ` changes relative weights, and floor rounding can leave budget unused. These settings are source defaults only; the run's actual resolved configuration is missing.

## Variance identities and their conditions

Let `X_u = μ_u + ε_u` be independent child estimates, with fixed means `μ_u`, zero-mean independent errors, and `Var(ε_u)=τ_u²`. If `S_p` is the population variance across the `n_p` realized `X_u`, then

```text
E[n_p S_p | μ] = ((n_p − 1)/n_p) Σ_u τ_u²
                 + Σ_u (μ_u − mean(μ))².
```

Therefore the requested identity

```text
E[n_p S_p | C(p)] = ((n_p − 1)/n_p) Σ_u τ_u²
```

holds when the conditioned child means are equal, or when `S_p` is computed from centered estimation errors rather than the raw heterogeneous child estimates. If candidate children have different true values and raw child estimates are used, the between-child term remains; dropping it is an assumption, not an algebraic identity. For `n_p=1`, the population variance and the factor `λ_p=(n_p−1)/n_p` are both zero.

For a child `u`, suppose each continuation is drawn independently with replacement from a fixed candidate population whose values have variance `σ_u²`, and the reward observation adds independent zero-mean noise with mean conditional variance `τ̄_u²`. The sample mean of `n_u` continuations then has

```text
τ_u² = (σ_u² + τ̄_u²) / n_u.
```

This law-of-total-variance identity assumes equal-weight IID sampling and equal/averaged within-candidate noise. Without-replacement sampling, unequal candidate probabilities, adaptive reuse, or candidate-dependent continuation noise require the corresponding finite-population or weighted variance terms.

Substituting the second identity into the first (under the equal-mean / centered-error condition) yields contributions proportional to `λ_p σ_u²/n_u`, where `λ_p=(n_p−1)/n_p`. Summing these terms over non-root nodes motivates the local proxy objective `Σ_{u≠root} λ_{p(u)} σ_u²/n_u`. The tree-level step is assumption-dependent: it omits the between-child term unless centered errors are intended, and it treats the parent allocation and child population as fixed. It is not, by itself, a proof about clipped-PPO gradient error or an adaptively selected tree.

## Evidence classification

| Claim | Classification | Evidence / limitation |
| --- | --- | --- |
| Marginal-gain greedy solves integer `Σ C_v/n_v` with fixed coefficients, bounds and feasible budget | Exact identity/optimizer | Follows from decreasing marginal gains of each convex term; covered by brute-force unit cases in `tests/test_alloc_replay.py`. |
| Continuous `n_v ∝ sqrt(C_v)` | Exact for the unbounded continuous relaxation | Lagrange multiplier solution; integer and bounds change the allocation. |
| `E[n_p S_p]` formula without between-child term | Exact only for equal means or centered residuals | General formula above has an additional between-child term. |
| `τ_u²=(σ_u²+τ̄_u²)/n_u` | Exact under IID equal-weight sampling assumptions | Follows from variance of a sample mean and total variance. |
| Tree sum `Σ λ σ²/n` | Local surrogate under the conditions above | Requires assumptions about centered/equal-mean errors, fixed coefficients and equal costs. |
| TV-based `σ²` predicts held-out value dispersion | Unproven for these archived runs | Run configs, trees, demos and MC references are absent. A4 would test this on new controlled probes. |
| Lower local objective implies lower PPO gradient error or better task accuracy | Needs experiment / not implied | Requires held-out gradient or task-level evidence; A5 and A7 address narrower proxies only. |

## Seeded synthetic checks

The program used 20,000 replications per stochastic check with seed 0.

| Check | Empirical | Analytic/reference | Interpretation |
| --- | ---: | ---: | --- |
| `E[n_p S_p]`, 4 independent zero-mean child errors | 2.310881 | 2.325000 | Monte Carlo agreement for the centered-error case. |
| Root case, `n_p=1` | 0 | 0 | No sibling-dispersion contribution. |
| Sample-mean MSE, 5 IID child draws | 0.213484 | 0.211520 | Monte Carlo agreement with `(σ²+τ̄²)/n`. |
| Top-logprob reuse; all 32 vs selected 8 reward means | 0.002079 | 0 by symmetry for all candidates | In this synthetic correlated-logprob setup the selected mean is 0.815269; this demonstrates possible selection bias, not its magnitude in any run. |
| Adaptive plug-in allocation objective | predicted 2.806379 | true-coefficient objective 3.426117 | Same noisy pilot estimates choose counts and score the predicted objective; the gap is selection optimism in this constructed example. |

The adaptive and top-logprob cases are deliberately illustrative stress tests. They do not establish that the archived run used those exact parameters or that its bias has these values.
