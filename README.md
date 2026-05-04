# gauge-lib

Shared library for the NeurIPS 2026 paper *Gauge-Fixed Energy Control for
Proposal Composition*. Used by all experiment notebooks (synthetic, attention,
MoE, GNN) to keep controller variants, gauge interventions, and the storage
schema consistent across testbeds.

## Install

From a Colab notebook:

```python
!pip install -q git+https://github.com/<you>/gauge-lib.git
```

For local development:

```bash
git clone https://github.com/<you>/gauge-lib.git
cd gauge-lib && pip install -e .
python gauge_lib.py   # runs the self-tests
```

## What's in the box

| Component                       | Purpose                                                                 |
|---------------------------------|-------------------------------------------------------------------------|
| `Controller`                    | Six controller variants (i)–(vi) from Section 4 in one class.           |
| `apply_gauge_traversal`         | Apply `(a, z) -> (c⁻¹a, c·z)` for the gauge-traversal experiments.       |
| `delta_gauge`                   | Relative deviation of `e` after a gauge intervention.                    |
| `predict_e_lin`                 | Linearized prediction `-g/(2λ)` (Corollary 1).                           |
| `predict_e_quad_explicit`       | Curvature-corrected prediction with an explicit Hessian (Corollary 2).   |
| `predict_e_quad_hvp`            | Same prediction via Hessian-vector products + conjugate gradient.        |
| `predict_e_proj`                | Capacity-limited prediction with a user-supplied projector (Corollary 3).|
| `residual_norms`, `stationary_residual` | r_lin / r_quad / r_proj / r_stat from Section 5.3.              |
| `cos_d_neg_g`, `lambda_m_over_g`| Calibration diagnostics for Section 5.2.                                 |
| `task_grad_wrt_e`, `task_hvp_fn`, `task_hessian_explicit` | Task-loss derivatives w.r.t. realized displacement. |
| `RunLogger` / `RunMeta`         | Fixed-schema pickled storage for `analysis.ipynb`.                       |

## Convention

- Displacements are `(..., n)`; the last axis is the displacement vector.
- The scalar gate `s` is `(...)` and is broadcast into `z`. Its semantic role
  depends on the controller mode:
    - amplitude variants (`none`, `amp_only`, `agg_only`, `raw_disp`): `s ≡ a`,
      `e = s·z`.
    - `gauge_fixed`: `s ≡ m`, `e = s · z/‖z‖`.
    - `dir_norm`: `s` is ignored, `e = m_fixed · z/‖z‖`.
- Logged `m` always equals `‖e‖`, so it is comparable across modes.

## Example: a single controller step

```python
import torch
from gauge_lib import Controller

ctrl = Controller(mode="gauge_fixed", lam=1e-2)
z = torch.randn(32, 64)            # raw aggregate, batch=32, n=64
s = torch.rand(32) + 0.1           # positive scalar per example
e, info = ctrl(z, s)               # e: (32, 64); info has a, z_norm, e_norm, m, d, penalty
loss = task_loss(e) + info.penalty
loss.backward()
```

## Storage layout

```
<root>/<testbed>/runs/<variant>_lam<lam>_seed<seed>.pkl
```

Each pickle is a dict with keys `schema_version`, `meta`, `config`,
`training_log`, `final`. See the `RunLogger` docstring for the per-example
arrays used by `analysis.ipynb`.

## Self-test

```bash
python gauge_lib.py
```

Verifies: controller shapes, magnitude semantics, all six penalty formulas,
exact gauge invariance, ε-level `delta_gauge`, agreement of HVP+CG with
explicit-H solves, inverse-λ scaling of `2λm/‖g‖`, and a `RunLogger`
round-trip.
