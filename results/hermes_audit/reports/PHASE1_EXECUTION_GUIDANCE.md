# PHASE1 Execution Guidance from Phase0 Audit

Generated: 2026-07-03T00:36:56.818658+00:00

## 0. Entry gate

Phase 0 audit result: **PASS**. Phase 1 may proceed, but only as **DGB-RLEF minimal implementation + unit tests + config/report generation**. Do not run Phase 2 1000-step candidate training in Phase 1.

## 1. Codebase-specific direction

The attachment suggests `src/a1lef/...`, but the actual project package is:

```text
src/rlef/
```

Therefore Phase 1 should implement under `src/rlef/`, not create a parallel `src/a1lef/` package.

Existing useful foundations:

- `src/rlef/models/rlef_former.py`
  - already has multiscale backbone;
  - already centers `E_raw` and adds `mu_E`;
  - already has domain-conditioned `head_bias`/`gate_bias`/`feature_affine+gate_bias`;
  - already has `domain_head_bias_l2` for P7b/P7c.
- `src/rlef/losses/total_loss.py`
  - already has rec/poisson/gauge/e_shape/gate/q/id/wtv;
  - already has real-domain head-bias anchor;
  - currently keeps E-shape loss as an internal helper.
- `src/rlef/metrics/exposure_field.py`
  - already has E_MAE/E_MAE_aligned/E_corr, LEE, NAI, q_ece, identity_drop.

Phase 1 should extract and formalize DGB-RLEF components without breaking these baselines.

## 2. Required Phase 1 deliverables

Create or update:

```text
src/rlef/models/exposure_gauge.py
src/rlef/losses/e_shape_loss.py
src/rlef/models/gauge_head.py
src/rlef/losses/gauge_scheduler.py
src/rlef/models/recoverability_router.py
src/rlef/metrics/domain_metrics.py
configs/dgb_rlef/dgb_rlef_minimal_seed3407.yml
results/hermes_audit/reports/PHASE1_DGB_RLEF_IMPLEMENTATION.md
results/hermes_audit/logs/unit_tests.log
```

Recommended test files:

```text
tests/test_dgb_exposure_gauge_contract.py
tests/test_dgb_e_shape_loss_contract.py
tests/test_dgb_gauge_head_scheduler_contract.py
tests/test_dgb_recoverability_router_contract.py
tests/test_dgb_domain_metrics_contract.py
```

## 3. Strict TDD task order

### Task 1 — Zero-mean exposure decomposition

**Test first:** `tests/test_dgb_exposure_gauge_contract.py`

Required behavior:

```text
S = E_raw - mean(E_raw)
mu is scalar per image
E = S + mu
mean(S) ~= 0
E ~= S + mu
```

Implementation target:

```text
src/rlef/models/exposure_gauge.py
```

Suggested API:

```python
decompose_exposure(E_raw, mu=None) -> dict with keys S, mu, E
```

Do not change `RLEFFormer` until this standalone contract passes.

### Task 2 — Public gauge-invariant E-shape loss

**Test first:** `tests/test_dgb_e_shape_loss_contract.py`

Required behavior:

```text
e_shape_loss(E, E_gt) == e_shape_loss(E + c, E_gt)
low-pass kernel defaults to 7
beta defaults to 0.1 for gradient term
```

Implementation target:

```text
src/rlef/losses/e_shape_loss.py
```

Then optionally make `total_loss.py` call this public function instead of the hidden `_lowpass_centered_corr_loss`, but only after tests pass.

### Task 3 — Warm gauge scheduler

**Test first:** `tests/test_dgb_gauge_head_scheduler_contract.py`

Required schedule:

```text
step < 300: 0
step 300..700: linear ramp 0 -> 0.005
step >= 700: full weight, with real-anchor equivalent <= 0.010
```

Implementation target:

```text
src/rlef/losses/gauge_scheduler.py
```

Hard guard:

```text
Do not expose default RA015/020/030; P7c already rejected larger scalar anchors.
```

### Task 4 — Image-statistics gauge head

**Test first:** same or separate gauge-head test.

Required input stats:

```text
mean(Y_l), std(Y_l), dark_ratio, sat_ratio, grad_mean, local_contrast
```

Implementation target:

```text
src/rlef/models/gauge_head.py
```

Suggested API:

```python
image_luminance_stats(low) -> tensor [B, 6]
ImageStatsGaugeHead(in_dim=6, hidden=32, mu_min=-1.0, mu_max=2.5)
```

This must be the deployable main path. Domain-id gauge may only be an oracle diagnostic later.

### Task 5 — Recoverability router

**Test first:** `tests/test_dgb_recoverability_router_contract.py`

Required endpoint behavior:

```text
A=1 -> final == I_rest
A=0 -> final == I_safe
I_safe = alpha * I_low + (1-alpha) * I_phys_clamped
safe_alpha default = 0.70
```

Implementation target:

```text
src/rlef/models/recoverability_router.py
```

Do not reuse P6c scalar `gate_identity` as the main route. It is a rejected scalar-knob diagnostic.

### Task 6 — Domain metrics ledger

**Test first:** `tests/test_dgb_domain_metrics_contract.py`

Required output keys:

```text
psnr, ssim, lee, nai,
E_MAE, E_MAE_aligned, E_corr,
Gauge_MAE, S_corr,
A_AUC, Q_ECE, identity_drop, unsafe_overenhance, route_entropy
```

Implementation target:

```text
src/rlef/metrics/domain_metrics.py
```

If a metric cannot be computed because optional GT is absent, return an explicit `None`/`nan` plus a note field; do not silently fabricate values.

### Task 7 — Minimal config scaffold

Create:

```text
configs/dgb_rlef/dgb_rlef_minimal_seed3407.yml
```

It should declare the intended minimal DGB route, but Phase 1 should not launch 1000-step training.

Important config defaults:

```yaml
seed: 3407
model:
  type: DGB_RLEF_MINIMAL
  trunk: multiscale
  trunk_blocks: 3
  e_shape_weight: 0.05
  gauge:
    mode: image_stats
    warmup_no_anchor_steps: 300
    ramp_start: 300
    full_start: 700
    real_anchor_init: 0.010
  route:
    safe_alpha: 0.70
    lambda_A: 0.02
    lambda_Q: 0.02
    lambda_id: 0.02
forbidden:
  teacher_distill: true
  rec_by_dataset: true
  real_anchor_above_0p010: true
```

The `forbidden` block is documentation only unless Phase 1 implements validators.

## 4. Phase 1 verification commands

Use the existing env:

```bash
cd /home/user/a1_rlef_project
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest   tests/test_dgb_exposure_gauge_contract.py   tests/test_dgb_e_shape_loss_contract.py   tests/test_dgb_gauge_head_scheduler_contract.py   tests/test_dgb_recoverability_router_contract.py   tests/test_dgb_domain_metrics_contract.py -q   2>&1 | tee results/hermes_audit/logs/unit_tests.log

/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m compileall -q src scripts
```

Expected gate:

```text
all new DGB tests pass
existing 72-test suite still passes or expands with new tests
compileall passes
```

## 5. Phase 1 stop conditions

Stop Phase 1 and report instead of continuing if:

1. A new module requires changing Phase 0 baseline config/checkpoint behavior.
2. P3c/P6/P7B eval contracts fail after implementation.
3. Gauge head outputs unstable defaults outside `[-1.0, 2.5]` in smoke tests.
4. Router endpoint tests fail.
5. E-shape loss is not invariant to `E+c`.
6. Any implementation attempts teacher distillation, `rec_by_dataset`, or real anchors > 0.010 as a main path.

## 6. Phase 1 report requirements

Write:

```text
results/hermes_audit/reports/PHASE1_DGB_RLEF_IMPLEMENTATION.md
```

Must include:

- modules created/modified;
- tests added;
- exact test output from `results/hermes_audit/logs/unit_tests.log`;
- explicit note that no Phase 2 training was run;
- readiness decision for Phase 2.

## 7. Guidance conclusion

Because Phase 0 passed cleanly, Phase 1 should focus on **interface-correct minimal DGB components**, not performance. The baseline evidence says:

- Use P6 multiscale trunk as structural base.
- Keep P3c `e_shape=0.05` as the frozen shape default.
- Treat P7B_DHEAD_RA010 as a near-miss diagnostic, not a solved method.
- Do not repeat P7c larger-anchor sweep.
- Implement image-stat gauge + recoverability route as the next meaningful innovation surface.
