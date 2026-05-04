"""
gauge_lib.py
============

Shared library for the NeurIPS 2026 "Gauge-Fixed Energy Control for Proposal
Composition" experiments.

This module is testbed-agnostic. Each notebook (synthetic, attention, MoE, GNN)
provides its own architecture, data, and gauge-intervention functions; this
library provides:

  1. ``Controller`` — a single class implementing the six controller variants
     compared in the paper (Section 4):
       (i)   ``none``         — no controller, e = a*z, no penalty
       (ii)  ``amp_only``     — penalty lambda * a^2
       (iii) ``agg_only``     — penalty lambda * ||z||^2
       (iv)  ``dir_norm``     — direction-only with fixed magnitude
       (v)   ``raw_disp``     — penalty lambda * ||a*z||^2
       (vi)  ``gauge_fixed``  — e = m*d, d = z/||z||, penalty lambda * m^2

  2. Gauge-intervention helpers (``apply_gauge_traversal``, ``delta_gauge``)
     for the (a,z) -> (c^{-1} a, c*z) transformation tested in Section 5.1.

  3. Residual predictions and norms:
       e_lin   = -g / (2 lambda)
       e_quad  = -(H + 2 lambda I)^{-1} g    (explicit-H or HVP+CG)
       e_proj  = Pi_C(-g / (2 lambda))       (user supplies projector)

  4. ``RunLogger`` — a fixed-schema logger that pickles results to a path
     compatible with the analysis notebook.

Conventions
-----------
- Displacements have shape ``(..., n)``. The last axis is the displacement
  vector; leading axes are batch/token/example.
- Scalars (``s``, ``a``, ``m``) have shape ``(...)`` matching the leading
  axes of ``z``. They are broadcast by unsqueezing the trailing dim.
- All penalties are returned as scalar tensors averaged over leading axes.
  The caller is responsible for summing across multiple controllers.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Constants & utilities
# ---------------------------------------------------------------------------

EPS = 1e-12  # for safe normalization of z

ALL_MODES = (
    "none",
    "amp_only",
    "agg_only",
    "dir_norm",
    "raw_disp",
    "gauge_fixed",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_normalize(z: Tensor, eps: float = EPS) -> Tuple[Tensor, Tensor]:
    """
    Normalize ``z`` along the last dimension. Returns ``(d, z_norm)`` where
    ``d = z / max(||z||, eps)`` and ``z_norm = ||z||`` (without the clamp,
    so callers see the true norm).
    """
    z_norm = z.norm(dim=-1, keepdim=True)
    d = z / z_norm.clamp_min(eps)
    return d, z_norm.squeeze(-1)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class ControllerInfo:
    """Per-call diagnostic info returned by ``Controller.forward``.

    Tensors track gradients (penalty does); detach before logging if needed.
    Shapes are the leading axes of the input ``z`` (the displacement axis is
    summarized to a norm).
    """
    a: Tensor          # amplitude factor used (1 for gauge_fixed, 0 for dir_norm)
    z_norm: Tensor     # ||z|| of the raw aggregate
    e_norm: Tensor     # ||e|| of the realized displacement (gauge-invariant)
    m: Tensor          # alias for e_norm — the calibrated magnitude
    d: Tensor          # unit direction z / ||z||  (shape (..., n))
    penalty: Tensor    # scalar mean penalty (already multiplied by lambda)


class Controller(nn.Module):
    """
    Wraps a scalar-gated proposal aggregate. Stateless w.r.t. parameters; the
    testbed owns ``a``/``m`` as part of its model. The controller's only job
    is to combine ``z`` and ``s`` into ``e`` and compute the right penalty.

    Parameters
    ----------
    mode : str
        One of ``ALL_MODES``.
    lam : float
        Regularization strength (lambda in the paper). Ignored for
        ``none`` and ``dir_norm``.
    fixed_magnitude : float
        The fixed ``m`` used in ``dir_norm`` mode. Ignored otherwise.
    eps : float
        Numerical floor for ``||z||`` in normalization.

    Notes
    -----
    The semantic role of ``s`` depends on ``mode``:

      - ``none``, ``amp_only``, ``agg_only``, ``raw_disp``: ``s`` is the
        amplitude ``a > 0``; ``e = s * z``.
      - ``dir_norm``: ``s`` is ignored; ``e = fixed_magnitude * z/||z||``.
      - ``gauge_fixed``: ``s`` is the magnitude ``m >= 0``; ``e = s * z/||z||``.

    The Controller never owns parameters. It is reusable across calls and
    across modes; you can swap modes by constructing a new instance with the
    same upstream model.
    """

    def __init__(
        self,
        mode: str,
        lam: float = 0.0,
        fixed_magnitude: float = 1.0,
        eps: float = EPS,
    ):
        super().__init__()
        if mode not in ALL_MODES:
            raise ValueError(f"mode must be one of {ALL_MODES}, got {mode!r}")
        self.mode = mode
        self.lam = float(lam)
        self.fixed_magnitude = float(fixed_magnitude)
        self.eps = eps

    # ------------------------------------------------------------------ utils

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, lam={self.lam}, fixed_magnitude={self.fixed_magnitude}"

    # ----------------------------------------------------------------- forward

    def forward(self, z: Tensor, s: Optional[Tensor] = None) -> Tuple[Tensor, ControllerInfo]:
        """
        Compute the realized displacement ``e`` and the penalty.

        Parameters
        ----------
        z : Tensor, shape ``(..., n)``
            Raw proposal aggregate.
        s : Tensor, shape ``(...)``, optional
            Positive scalar gate (amplitude or magnitude depending on mode).
            Required for all modes except ``dir_norm``. The caller is
            responsible for ensuring ``s >= 0``; the controller clamps for
            safety.

        Returns
        -------
        e : Tensor, shape ``(..., n)``
            Realized displacement.
        info : ControllerInfo
        """
        if z.dim() < 1:
            raise ValueError(f"z must have at least 1 dim, got shape {tuple(z.shape)}")

        d, z_norm = safe_normalize(z, eps=self.eps)  # d: (..., n), z_norm: (...)

        if self.mode == "dir_norm":
            # variant (iv): direction-only, s ignored
            m = torch.full_like(z_norm, self.fixed_magnitude)
            e = self.fixed_magnitude * d
            a = torch.zeros_like(z_norm)  # not meaningful in this mode
            penalty = z.new_zeros(())
            return e, ControllerInfo(a=a, z_norm=z_norm, e_norm=m, m=m, d=d, penalty=penalty)

        if s is None:
            raise ValueError(f"mode {self.mode!r} requires a scalar tensor s")
        if s.shape != z.shape[:-1]:
            raise ValueError(
                f"s must have shape {tuple(z.shape[:-1])}, got {tuple(s.shape)}"
            )

        s = s.clamp_min(0.0)
        s_b = s.unsqueeze(-1)  # (..., 1) for broadcasting against z

        if self.mode == "gauge_fixed":
            # variant (vi): s plays the role of m; e = m * d
            e = s_b * d
            a = torch.ones_like(s)
            m = s
            penalty = self.lam * (m.pow(2)).mean() if self.lam else z.new_zeros(())
            return e, ControllerInfo(a=a, z_norm=z_norm, e_norm=m, m=m, d=d, penalty=penalty)

        # Amplitude variants: e = s * z
        e = s_b * z
        e_norm = s * z_norm
        a = s
        m = e_norm  # logged magnitude is always ||e||

        if self.mode == "none":
            penalty = z.new_zeros(())
        elif self.mode == "amp_only":
            penalty = self.lam * (a.pow(2)).mean() if self.lam else z.new_zeros(())
        elif self.mode == "agg_only":
            penalty = self.lam * (z_norm.pow(2)).mean() if self.lam else z.new_zeros(())
        elif self.mode == "raw_disp":
            penalty = self.lam * (e_norm.pow(2)).mean() if self.lam else z.new_zeros(())
        else:  # pragma: no cover
            raise AssertionError(f"unreachable mode {self.mode!r}")

        return e, ControllerInfo(a=a, z_norm=z_norm, e_norm=e_norm, m=m, d=d, penalty=penalty)


# ---------------------------------------------------------------------------
# Gauge interventions
# ---------------------------------------------------------------------------


def apply_gauge_traversal(
    model: nn.Module,
    c: float,
    scale_amp_fn: Callable[[nn.Module, float], None],
    scale_agg_fn: Callable[[nn.Module, float], None],
) -> None:
    """
    Apply the gauge transformation ``(a, z) -> (c^{-1} a, c * z)`` in place.

    The two scaling functions are testbed-specific. For the exact-synthetic
    testbed they multiply the amplitude parameter and the aggregate weight
    matrix; for attention / MoE / GNN they encapsulate the chosen amplitude
    channel and the upstream aggregate scaling. See Section 2.1
    "Architectural scope" in the paper for what counts as a valid local
    gauge realization.

    The realized displacement should be invariant under this map for any
    architecture that satisfies Assumption 1. ``delta_gauge`` measures the
    residual deviation, which should be ~0 in exact gauges and small in
    locally realized ones.
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    scale_amp_fn(model, 1.0 / c)
    scale_agg_fn(model, c)


def delta_gauge(e_before: Tensor, e_after: Tensor) -> Tensor:
    """
    Relative L2 deviation of realized displacements after a gauge intervention.

    Returns ``||e_after - e_before|| / ||e_before||`` per leading-axis element,
    averaged over leading axes (scalar). Use elementwise variants if you need
    per-example values.
    """
    diff = (e_after - e_before).norm(dim=-1)
    base = e_before.norm(dim=-1).clamp_min(EPS)
    return (diff / base).mean()


# ---------------------------------------------------------------------------
# Predictions and residuals
# ---------------------------------------------------------------------------


def predict_e_lin(g: Tensor, lam: float) -> Tensor:
    """First-order prediction (Corollary 1): e_lin = -g / (2 lambda)."""
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    return -g / (2.0 * lam)


def predict_e_quad_explicit(H: Tensor, g: Tensor, lam: float) -> Tensor:
    """
    Curvature-corrected prediction (Corollary 2) using an explicit Hessian.

    Solves ``(H + 2 lambda I) e = -g`` with ``torch.linalg.solve`` (LU). Use
    when ``n`` is small enough to form ``H``. For larger problems use
    ``predict_e_quad_hvp``.

    Parameters
    ----------
    H : Tensor, shape ``(n, n)``
        Hessian of the task loss w.r.t. the displacement, evaluated at the
        reference point. Symmetry is not enforced; the solver works on the
        symmetrized matrix.
    g : Tensor, shape ``(n,)``
    lam : float

    Returns
    -------
    e_quad : Tensor, shape ``(n,)``
    """
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square 2D, got shape {tuple(H.shape)}")
    if g.dim() != 1 or g.shape[0] != H.shape[0]:
        raise ValueError(f"g must be 1D matching H, got shape {tuple(g.shape)}")
    n = H.shape[0]
    A = 0.5 * (H + H.transpose(-1, -2)) + 2.0 * lam * torch.eye(n, device=H.device, dtype=H.dtype)
    return torch.linalg.solve(A, -g)


def predict_e_quad_hvp(
    hvp_fn: Callable[[Tensor], Tensor],
    g: Tensor,
    lam: float,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Curvature-corrected prediction via Hessian-vector products and conjugate
    gradient. Solves ``(H + 2 lambda I) e = -g``.

    The HVP is assumed to give ``H @ v`` where ``H = grad^2 L_task`` at the
    reference point. The system is PD when ``H + 2 lambda I`` is, which holds
    for sufficiently large ``lam`` even when ``H`` has negative eigenvalues.

    Returns ``(e_quad, info)`` where ``info`` has CG convergence diagnostics.
    """
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")

    def Av(v: Tensor) -> Tensor:
        return hvp_fn(v) + 2.0 * lam * v

    b = -g
    x = torch.zeros_like(g)
    r = b - Av(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    b_norm = b.norm().clamp_min(EPS)
    iters = 0
    converged = False
    for i in range(max_iter):
        Ap = Av(p)
        pAp = torch.dot(p, Ap)
        if pAp.item() <= 0:
            # H + 2 lambda I is not PD on the current Krylov subspace.
            # Bail out; caller can retry with larger lam or use explicit solve.
            return x, {"converged": False, "iters": i, "reason": "non_pd", "residual": (r.norm() / b_norm).item()}
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        iters = i + 1
        if (r.norm() / b_norm).item() < tol:
            converged = True
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, {
        "converged": converged,
        "iters": iters,
        "reason": "tol" if converged else "max_iter",
        "residual": (r.norm() / b_norm).item(),
    }


def predict_e_proj(
    g: Tensor,
    lam: float,
    proj_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Capacity-limited prediction (Corollary 3 / eqn 32): project ``-g/(2 lam)``
    onto the testbed's feasible displacement set ``C_x`` via the user-supplied
    projector. The projector decides convex vs. local-projection semantics."""
    return proj_fn(predict_e_lin(g, lam))


def residual_norms(
    e_obs: Tensor,
    *,
    e_lin: Optional[Tensor] = None,
    e_quad: Optional[Tensor] = None,
    e_proj: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Compute residual norms ``||e_obs - e_*||`` for whichever predictions are
    supplied. Reduces over the displacement axis only; preserves leading axes.
    """
    out: Dict[str, Tensor] = {}
    if e_lin is not None:
        out["r_lin"] = (e_obs - e_lin).norm(dim=-1)
    if e_quad is not None:
        out["r_quad"] = (e_obs - e_quad).norm(dim=-1)
    if e_proj is not None:
        out["r_proj"] = (e_obs - e_proj).norm(dim=-1)
    return out


def stationary_residual(grad_at_eobs: Tensor, e_obs: Tensor, lam: float) -> Tensor:
    """
    Section 5.3, eqn 45: ``r_stat = ||grad L_task(e_obs) + 2 lambda e_obs||``.

    Note ``grad_at_eobs`` is the task gradient *evaluated at the realized
    displacement*, not at the reference point.
    """
    return (grad_at_eobs + 2.0 * lam * e_obs).norm(dim=-1)


def cos_d_neg_g(d: Tensor, g: Tensor) -> Tensor:
    """Cosine similarity between unit direction ``d`` and ``-g``."""
    g_norm = g.norm(dim=-1, keepdim=True).clamp_min(EPS)
    g_hat = -g / g_norm
    return (d * g_hat).sum(dim=-1)


def lambda_m_over_g(m: Tensor, g: Tensor, lam: float) -> Tensor:
    """The diagnostic ``2*lambda*m / ||g||``; should concentrate near 1 in the
    linearized regime (Corollary 1)."""
    g_norm = g.norm(dim=-1).clamp_min(EPS)
    return 2.0 * lam * m / g_norm


# ---------------------------------------------------------------------------
# Task-loss derivatives w.r.t. displacement
# ---------------------------------------------------------------------------


def task_grad_wrt_e(
    task_loss_fn: Callable[[Tensor], Tensor],
    e: Tensor,
) -> Tensor:
    """
    Compute ``g = grad_e L_task(e)`` for a scalar-valued ``task_loss_fn``.

    ``e`` is treated as a leaf for this gradient call. The output has the
    same shape as ``e``.
    """
    e = e.detach().clone().requires_grad_(True)
    loss = task_loss_fn(e)
    if loss.dim() != 0:
        raise ValueError(f"task_loss_fn must return a scalar, got shape {tuple(loss.shape)}")
    (g,) = torch.autograd.grad(loss, e, create_graph=False)
    return g.detach()


def task_hvp_fn(
    task_loss_fn: Callable[[Tensor], Tensor],
    e: Tensor,
) -> Callable[[Tensor], Tensor]:
    """
    Build a Hessian-vector product closure ``v -> H @ v`` where
    ``H = grad^2_e L_task(e)``.

    Note: the closure rebuilds the graph on each call. For repeated solves at
    a fixed reference, this is fine for the synthetic testbeds. For larger
    setups consider ``torch.func.hvp``.
    """
    e_ref = e.detach().clone()

    def hvp(v: Tensor) -> Tensor:
        x = e_ref.clone().requires_grad_(True)
        loss = task_loss_fn(x)
        (g,) = torch.autograd.grad(loss, x, create_graph=True)
        gv = (g * v).sum()
        (Hv,) = torch.autograd.grad(gv, x)
        return Hv.detach()

    return hvp


def task_hessian_explicit(
    task_loss_fn: Callable[[Tensor], Tensor],
    e: Tensor,
) -> Tensor:
    """Form the Hessian ``H = grad^2_e L_task(e)`` explicitly. Use only for
    small ``n`` (the synthetic testbeds and per-token attention/GNN heads)."""
    if e.dim() != 1:
        raise ValueError(f"task_hessian_explicit requires 1-D e, got shape {tuple(e.shape)}")
    n = e.shape[0]
    hvp = task_hvp_fn(task_loss_fn, e)
    H = torch.zeros(n, n, device=e.device, dtype=e.dtype)
    eye = torch.eye(n, device=e.device, dtype=e.dtype)
    for i in range(n):
        H[:, i] = hvp(eye[i])
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


@dataclass
class RunMeta:
    testbed: str
    variant: str
    lam: float
    seed: int
    completed_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class RunLogger:
    """
    Fixed-schema logger for a single training run.

    Storage layout
    --------------
    ``<root>/<testbed>/runs/<variant>_lam<lam>_seed<seed>.pkl``

    The pickle is a dict::

        {
            "meta":         dict (RunMeta),
            "config":       dict,                 # all hyperparameters
            "training_log": list[dict],           # per-step or per-epoch
            "final":        dict,                 # final eval / per-example arrays
        }

    Final-eval arrays (per held-out example) typically include::

        a, z_norm, e_norm, m, g_norm, lambda_m_over_g, cos_d_neg_g,
        r_lin, r_quad, r_proj, r_stat, task_metric (scalar),
        gauge_traverse: {c: {...}},
        rho_intervention: {rho: {...}}.

    The analysis notebook glob-loads everything and assumes this schema.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        meta: RunMeta,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.root = Path(root)
        self.meta = meta
        self.config = dict(config or {})
        self.training_log: List[Dict[str, Any]] = []
        self.final: Dict[str, Any] = {}

    # ------------------------------------------------------------------ paths

    @property
    def run_dir(self) -> Path:
        return self.root / self.meta.testbed / "runs"

    @property
    def filename(self) -> str:
        # Use a fixed-width-ish lambda string to keep glob ordering reasonable.
        lam_str = f"{self.meta.lam:.6g}"
        return f"{self.meta.variant}_lam{lam_str}_seed{self.meta.seed}.pkl"

    @property
    def path(self) -> Path:
        return self.run_dir / self.filename

    # ------------------------------------------------------------------ logging

    def log_step(self, **kwargs: Any) -> None:
        """Append a row to the training log. Tensors are detached and cast to float."""
        row: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                row[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().numpy()
            else:
                row[k] = v
        self.training_log.append(row)

    def set_final(self, key: str, value: Any) -> None:
        """Set a top-level final-eval entry (scalar, array, or nested dict)."""
        self.final[key] = _to_storable(value)

    def update_final(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.set_final(k, v)

    # ------------------------------------------------------------------ I/O

    def save(self) -> Path:
        self.meta.completed_at = datetime.now(timezone.utc).isoformat()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "meta": asdict(self.meta),
            "config": self.config,
            "training_log": self.training_log,
            "final": self.final,
        }
        with open(self.path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self.path

    @classmethod
    def load(cls, path: str | Path) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_testbed(cls, root: str | Path, testbed: str) -> List[Dict[str, Any]]:
        """Load all runs under a testbed. Returns a list of payload dicts."""
        run_dir = Path(root) / testbed / "runs"
        if not run_dir.exists():
            return []
        return [cls.load(p) for p in sorted(run_dir.glob("*.pkl"))]


def _to_storable(x: Any) -> Any:
    """Recursively convert tensors to numpy / scalars for pickling."""
    if isinstance(x, Tensor):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.numpy()
    if isinstance(x, dict):
        return {k: _to_storable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_storable(v) for v in x)
    return x


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Internal correctness checks. Run with ``python gauge_lib.py``."""
    print("Running gauge_lib self-tests...")
    set_seed(0)

    # ---- 1. Controller forward shapes and gauge-invariant magnitudes ----
    B, n = 4, 6
    z = torch.randn(B, n)
    s = torch.rand(B) + 0.1  # positive

    for mode in ALL_MODES:
        ctrl = Controller(mode=mode, lam=0.1, fixed_magnitude=2.0)
        e, info = ctrl(z, s)
        assert e.shape == (B, n), f"{mode}: e shape {e.shape}"
        # Logged ||e|| should match actual ||e||.
        actual_e_norm = e.norm(dim=-1)
        torch.testing.assert_close(info.e_norm, actual_e_norm, rtol=1e-5, atol=1e-6)
        # m == ||e||.
        torch.testing.assert_close(info.m, actual_e_norm, rtol=1e-5, atol=1e-6)
        if mode == "dir_norm":
            torch.testing.assert_close(actual_e_norm, torch.full_like(actual_e_norm, 2.0), rtol=1e-5, atol=1e-6)
        elif mode == "gauge_fixed":
            torch.testing.assert_close(actual_e_norm, s, rtol=1e-5, atol=1e-6)
        else:
            torch.testing.assert_close(actual_e_norm, s * z.norm(dim=-1), rtol=1e-5, atol=1e-6)
    print("  [ok] Controller shapes and magnitude semantics")

    # ---- 2. Penalty values match the paper formulas ----
    z_norm = z.norm(dim=-1)
    e_amp_norm = s * z_norm
    expected = {
        "none": 0.0,
        "amp_only": 0.1 * (s.pow(2)).mean().item(),
        "agg_only": 0.1 * (z_norm.pow(2)).mean().item(),
        "dir_norm": 0.0,
        "raw_disp": 0.1 * (e_amp_norm.pow(2)).mean().item(),
        "gauge_fixed": 0.1 * (s.pow(2)).mean().item(),
    }
    for mode in ALL_MODES:
        ctrl = Controller(mode=mode, lam=0.1, fixed_magnitude=2.0)
        _, info = ctrl(z, s)
        assert math.isclose(info.penalty.item(), expected[mode], rel_tol=1e-5, abs_tol=1e-7), \
            f"{mode}: penalty {info.penalty.item()} vs expected {expected[mode]}"
    print("  [ok] Penalty formulas")

    # ---- 3. Exact gauge invariance for amplitude variants ----
    # Apply (a, z) -> (c^{-1} a, c z); check e is preserved for amp variants.
    c = 3.7
    z2 = c * z
    s2 = s / c
    for mode in ("none", "amp_only", "agg_only", "raw_disp"):
        ctrl = Controller(mode=mode, lam=0.0)
        e1, _ = ctrl(z, s)
        e2, _ = ctrl(z2, s2)
        torch.testing.assert_close(e1, e2, rtol=1e-5, atol=1e-6)
    # gauge_fixed is invariant under z -> c*z too (since e = m * d, d unchanged).
    ctrl = Controller(mode="gauge_fixed", lam=0.0)
    e1, _ = ctrl(z, s)
    e2, _ = ctrl(z2, s)  # m unchanged, only z scales
    torch.testing.assert_close(e1, e2, rtol=1e-5, atol=1e-6)
    print("  [ok] Exact gauge invariance")

    # ---- 4. delta_gauge ~ 0 for an exact pass ----
    e_before, _ = Controller("amp_only", lam=0.0)(z, s)
    e_after, _ = Controller("amp_only", lam=0.0)(z2, s2)
    dg = delta_gauge(e_before, e_after)
    assert dg.item() < 1e-5, f"delta_gauge too large: {dg.item()}"
    print(f"  [ok] delta_gauge under exact transform: {dg.item():.2e}")

    # ---- 5. Linearized prediction & quadratic prediction agree on a quadratic loss ----
    # L(e) = 0.5 e^T A e + b^T e  has H = A, grad = A e + b. At e=0: g = b.
    torch.manual_seed(1)
    n = 5
    A = torch.randn(n, n)
    A = A @ A.T + torch.eye(n)  # PSD
    b = torch.randn(n)

    def Lfn(e: Tensor) -> Tensor:
        return 0.5 * e @ A @ e + b @ e

    e0 = torch.zeros(n)
    g = task_grad_wrt_e(Lfn, e0)
    torch.testing.assert_close(g, b, rtol=1e-5, atol=1e-6)
    H = task_hessian_explicit(Lfn, e0)
    torch.testing.assert_close(H, 0.5 * (A + A.T), rtol=1e-5, atol=1e-5)

    lam = 0.5
    e_lin = predict_e_lin(g, lam)
    e_quad = predict_e_quad_explicit(H, g, lam)
    # Sanity: stationarity at e_quad means (H + 2 lam I) e_quad + g ~ 0.
    res = (H + 2 * lam * torch.eye(n)) @ e_quad + g
    assert res.norm().item() < 1e-4, f"e_quad stationarity residual {res.norm().item():.2e}"
    # HVP+CG path agrees with explicit solve.
    hvp = task_hvp_fn(Lfn, e0)
    e_quad_cg, info = predict_e_quad_hvp(hvp, g, lam)
    torch.testing.assert_close(e_quad_cg, e_quad, rtol=1e-3, atol=1e-4)
    assert info["converged"], f"CG did not converge: {info}"
    print(f"  [ok] e_lin, e_quad, HVP+CG (CG iters={info['iters']})")

    # ---- 6. Inverse-lambda scaling on the linearized prediction ----
    for lam in [0.1, 1.0, 10.0]:
        e_lin = predict_e_lin(g, lam)
        m = e_lin.norm()
        ratio = (2 * lam * m / g.norm()).item()
        assert math.isclose(ratio, 1.0, rel_tol=1e-5), f"2*lam*m/||g|| = {ratio} for lam={lam}"
    print("  [ok] 2 lambda m / ||g|| = 1 in linearized regime")

    # ---- 7. RunLogger round-trip ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        meta = RunMeta(testbed="synthetic", variant="gauge_fixed", lam=1.0, seed=0)
        log = RunLogger(tmp, meta, config={"hello": "world"})
        log.log_step(step=0, loss=1.5, m_mean=torch.tensor(0.7))
        log.log_step(step=1, loss=1.2)
        log.set_final("g_norm", torch.randn(8))
        log.set_final("nested", {"cos": torch.tensor(0.99), "arr": torch.randn(3)})
        path = log.save()
        loaded = RunLogger.load(path)
        assert loaded["config"] == {"hello": "world"}
        assert len(loaded["training_log"]) == 2
        assert math.isclose(loaded["training_log"][0]["m_mean"], 0.7, rel_tol=1e-5)
        assert math.isclose(loaded["final"]["nested"]["cos"], 0.99, rel_tol=1e-5)
        all_runs = RunLogger.load_testbed(tmp, "synthetic")
        assert len(all_runs) == 1
    print("  [ok] RunLogger round-trip")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
