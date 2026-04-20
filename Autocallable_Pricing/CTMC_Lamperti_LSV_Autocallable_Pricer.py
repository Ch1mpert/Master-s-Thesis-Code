#!/usr/bin/env python3
"""
autocallable_pricer_lamperti.py — CTMC-Lamperti-LSV Autocallable Pricer (ρ≠0)
================================================================================
Prices autocallable structured products using the calibrated CTMC-Lamperti-LSV
model with ρ≠0 coupling.

Key design principles:
  1. Uses calibrated pillar densities as checkpoints (no re-evolution from t=0)
  2. Bucket-aware propagation matching calibration substep grid exactly
  3. Batch propagation: builds generator once per substep, applies to all slices
  4. Barrier application in z-space via state-dependent X→z mapping

State space is (v_ℓ, X_j) with X = g(z) - ρv/ξ (Lamperti transform).
Uses coupled generator (FP + CTMC combined) — no operator splitting.
Converts between X-space (propagation) and z-space (barriers/payoffs)
via z = g⁻¹(X + ρv/ξ).

Usage:
  python3 autocallable_pricer_lamperti_fixed.py \
    --lsv_result coupled_v2/lamperti_lsv_model.npz \
    --forward_curve data/forward_curve_interpolated_daily.csv \
    --discount_curve data/discount_curve_grid.csv
"""
from __future__ import annotations
import argparse, time, csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _GPU = False; cp = None; cp_sparse = None

# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════
@dataclass
class AutocallableSpec:
    notional: float = 1.0; maturity_years: float = 1.5; ac_barrier: float = 1.0
    coupon_barrier: float = 0.70; ki_barrier: float = 0.60; coupon_rate: float = 0.08
    put_strike: float = 1.0; memory: bool = True; obs_freq: str = "quarterly"
    no_call_periods: int = 0; ki_continuous: bool = False; ac_step_down: float = 0.0

@dataclass
class LampertiModel:
    """All data needed to propagate density under CTMC-Lamperti-LSV."""
    z_grid: np.ndarray; X_grid: np.ndarray; dz: float; dX: float
    n_states: int; v_states: np.ndarray; Q: np.ndarray; pi0: np.ndarray
    mart_corr: np.ndarray; S0: float
    v0: float; kappa: float; theta: float; xi: float; rho: float
    pillar_T: np.ndarray; pillar_forwards: np.ndarray; pillar_dfs: np.ndarray
    pillar_labels: np.ndarray; pillar_dt: np.ndarray
    leverage: List[np.ndarray]; sigma_lv: List[np.ndarray]
    g_pillars: List[np.ndarray]; densities_X: List[np.ndarray]
    n_substeps: int; omega: float; lcap: float
    dgdt_clip: float = 160.0
    leverage_time: Optional[List[np.ndarray]] = None

@dataclass
class PricingResult:
    price: float; notional: float; price_pct: float
    autocall_probabilities: np.ndarray; stop_probabilities: np.ndarray
    coupon_contributions: np.ndarray; autocall_contributions: np.ndarray
    terminal_par_contribution: float; terminal_put_contribution: float
    survival_probability: float; ki_probability: float
    observation_dates: np.ndarray; memory_enabled: bool
    fair_coupon: Optional[float] = None; expected_expiry_years: float = 0.0

@dataclass
class TermStructurePoint:
    maturity_years: float; coupon_rate: float; price: float; price_pct: float
    price_diff: float; price_diff_bps: float; survival_probability: float
    terminal_par_contribution: float; terminal_put_contribution: float
    expected_expiry_years: float; obs_freq: str

# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
def load_lamperti_model(npz_path, leverage_time_stride=1, dgdt_clip=160.0):
    d = np.load(npz_path, allow_pickle=True)
    z_grid = np.asarray(d["z_grid"], dtype=np.float64)
    X_grid = np.asarray(d["X_grid"], dtype=np.float64)
    dz = float(d["dz"]); dX = float(d["dX"])
    n_states = int(d["ctmc_n_states"])
    v_states = np.asarray(d["ctmc_states"], dtype=np.float64)
    Q = np.asarray(d["ctmc_generator"], dtype=np.float64)
    pi0 = np.asarray(d["ctmc_pi0"], dtype=np.float64)
    mart_corr = np.asarray(d["mart_corr"], dtype=np.float64)
    S0 = float(d["heston_S0"])
    v0 = float(d["heston_v0"]); kappa = float(d["heston_kappa"])
    theta = float(d["heston_theta"]); xi = float(d["heston_xi"]); rho = float(d["heston_rho"])
    pillar_T = np.asarray(d["pillar_T"], dtype=np.float64)
    pillar_forwards = np.asarray(d["pillar_forward"], dtype=np.float64)
    pillar_dfs = np.asarray(d["pillar_df"], dtype=np.float64)
    pillar_labels = np.asarray(d["pillar_labels"])
    pillar_dt = np.asarray(d["pillar_dt"], dtype=np.float64)
    n_sub = int(d["n_substeps"]); omega = float(d["omega"]); lcap = float(d["lcap"])
    nb = int(d["n_buckets"])
    lev, slv, gpil, den = [], [], [], []
    lt = [] if bool(d.get("has_leverage_time", 0)) else None
    for k in range(nb):
        lev.append(np.asarray(d[f"leverage_{k}"], dtype=np.float64))
        slv.append(np.asarray(d[f"sigma_lv_{k}"], dtype=np.float64))
        gpil.append(np.asarray(d[f"g_{k}"], dtype=np.float64))
        den.append(np.asarray(d[f"density_{k}"], dtype=np.float64))
        if lt is not None:
            key = f"leverage_time_{k}"
            if key in d:
                arr = np.asarray(d[key], dtype=np.float64)
                if leverage_time_stride > 1:
                    idx = np.arange(0, arr.shape[0], leverage_time_stride)
                    if idx[-1] != arr.shape[0] - 1:
                        idx = np.append(idx, arr.shape[0] - 1)
                    arr = arr[idx]
                lt.append(arr)
            else:
                lt.append(None)
    if lt is not None:
        tot = sum(x.shape[0] for x in lt if x is not None)
        print(f"  [load] leverage_time: {tot} slices (stride={leverage_time_stride})")
    return LampertiModel(
        z_grid=z_grid, X_grid=X_grid, dz=dz, dX=dX,
        n_states=n_states, v_states=v_states, Q=Q, pi0=pi0,
        mart_corr=mart_corr, S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        pillar_T=pillar_T, pillar_forwards=pillar_forwards, pillar_dfs=pillar_dfs,
        pillar_labels=pillar_labels, pillar_dt=pillar_dt,
        leverage=lev, sigma_lv=slv, g_pillars=gpil, densities_X=den,
        n_substeps=n_sub, omega=omega, lcap=lcap, dgdt_clip=dgdt_clip,
        leverage_time=lt)

# ══════════════════════════════════════════════════════════════
# CURVE LOADING
# ══════════════════════════════════════════════════════════════
def load_forward_curve(p):
    T, F = [], []
    with open(p) as f:
        for r in csv.DictReader(f): T.append(float(r["T_years"])); F.append(float(r["forward_interp"]))
    return np.array(T), np.array(F)

def load_discount_curve(p):
    T, D = [], []
    with open(p) as f:
        for r in csv.DictReader(f): T.append(float(r["T_years"])); D.append(float(r["discount_factor"]))
    return np.array(T), np.array(D)

def build_interpolators(fT, fF, dT, dD):
    return (interp1d(fT, fF, kind="linear", fill_value="extrapolate"),
            interp1d(dT, dD, kind="linear", fill_value="extrapolate"))

# ══════════════════════════════════════════════════════════════
# OBSERVATION DATES
# ══════════════════════════════════════════════════════════════
def normalize_obs_freq(f):
    f = f.strip().lower()
    if f in ("monthly","month","m","1m"): return "monthly"
    if f in ("quarterly","quarter","q","3m"): return "quarterly"
    if f in ("semi-annual","semiannual","semi","sa","6m","semi_annual"): return "semi-annual"
    if f in ("annual","yearly","a","12m","1y"): return "annual"
    raise ValueError(f)

def obs_freq_to_months(f):
    return {"monthly":1, "quarterly":3, "semi-annual":6, "annual":12}[normalize_obs_freq(f)]

def generate_observation_dates(mat, freq):
    dt = {"monthly":1/12, "quarterly":0.25, "semi-annual":0.5, "annual":1.0}[normalize_obs_freq(freq)]
    r = np.arange(dt, mat - 1e-12, dt)
    return np.array([mat]) if r.size == 0 else np.concatenate([r, [mat]])

def parse_obs_freq_list(t):
    if not t or not t.strip(): return None
    v = [normalize_obs_freq(x.strip()) for x in t.split(",") if x.strip()]
    o, s = [], set()
    for x in v:
        if x not in s: o.append(x); s.add(x)
    return o or None

# ══════════════════════════════════════════════════════════════
# LAMPERTI FUNCTIONS (matching calibration code exactly)
# ══════════════════════════════════════════════════════════════
def compute_g(z_grid, L_z):
    """g(z) = ∫₀ᶻ dz'/L(z'), g(0)=0. Simpson's rule O(dz⁴).
    Matches idk.py compute_g exactly (same L => same g)."""
    inv_L = 1.0 / np.maximum(L_z, 0.01)
    dz_ = z_grid[1] - z_grid[0]
    i0 = int(np.argmin(np.abs(z_grid)))
    N = len(z_grid)
    g = np.zeros_like(z_grid)
    # Forward from i0: Simpson where possible, trapezoid for odd remainder
    for j in range(i0 + 1, N):
        if j >= i0 + 2 and (j - i0) % 2 == 0:
            g[j] = g[j-2] + dz_/3.0 * (inv_L[j-2] + 4*inv_L[j-1] + inv_L[j])
        else:
            g[j] = g[j-1] + 0.5*dz_*(inv_L[j-1] + inv_L[j])
    # Backward from i0
    for j in range(i0 - 1, -1, -1):
        if j <= i0 - 2 and (i0 - j) % 2 == 0:
            g[j] = g[j+2] - dz_/3.0 * (inv_L[j] + 4*inv_L[j+1] + inv_L[j+2])
        else:
            g[j] = g[j+1] - 0.5*dz_*(inv_L[j] + inv_L[j+1])
    return g

def interp_density(X_grid, u_X_row, Xz_query):
    """Linear interp with zero extrapolation; clamps negatives and NaN/Inf.
    Matches idk.py interp_density exactly."""
    y = np.maximum(np.nan_to_num(u_X_row, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return np.maximum(np.interp(Xz_query, X_grid, y, left=0.0, right=0.0), 0.0)

def remap_density_at_boundary(p_vec, g_old, L_old, g_new, L_new,
                               v_shifts, X_grid, z_grid, M, Nx, dX):
    """Remap X-space density from (g_old, L_old) to (g_new, L_new).
    Matches idk.py remap_density_at_boundary exactly:
      - interp u_old at X_old (zero outside X-grid)
      - multiply by Jacobian L_new / L_old, clipped to [0.1, 10.0]
      - per-state mass renormalization"""
    p_new = np.zeros_like(p_vec)
    u_X = p_vec.reshape(M, Nx)
    for ell in range(M):
        shift = v_shifts[ell]
        mass_old = float(np.sum(np.maximum(u_X[ell], 0)) * dX)

        z_at_X = np.interp(X_grid + shift, g_new, z_grid)
        X_old  = np.interp(z_at_X, z_grid, g_old) - shift

        u_interp = np.interp(X_old, X_grid, np.maximum(u_X[ell], 0),
                             left=0.0, right=0.0)
        L_old_z = np.interp(z_at_X, z_grid, L_old)
        L_new_z = np.interp(z_at_X, z_grid, L_new)
        J = L_new_z / np.maximum(L_old_z, 1e-6)
        J = np.clip(J, 0.1, 10.0)

        u_new_ell = np.maximum(u_interp * J, 0.0)
        mass_new = float(np.sum(u_new_ell) * dX)
        if mass_new > 1e-15 and mass_old > 1e-15:
            u_new_ell *= (mass_old / mass_new)

        p_new[ell*Nx:(ell+1)*Nx] = u_new_ell
    return p_new

def build_forward_gen_coo(mu_all, v_states, Q, M, Nx, dX, rp2):
    """Vectorized COO backward generator + transpose → forward generator.
    Identical to test_coupled.py build_generator."""
    from scipy.sparse import coo_matrix
    N = M * Nx; dX2 = dX * dX
    D2_all = 0.5 * rp2 * v_states / dX2
    mu2_all = mu_all / (2 * dX)
    offsets = np.arange(M) * Nx; j_idx = np.arange(Nx)
    _, j_grid = np.meshgrid(np.arange(M), np.arange(Nx), indexing='ij')
    flat_idx = offsets[:, None] + j_idx[None, :]
    d_vals = (-2 * D2_all[:, None] * np.ones((1, Nx)) + np.diag(Q)[:, None]).ravel()
    d_rows = flat_idx.ravel(); d_cols = d_rows.copy()
    m_lo = j_grid > 0
    l_rows = flat_idx[m_lo]; l_cols = (flat_idx - 1)[m_lo]
    l_vals = (D2_all[:, None] - mu2_all)[m_lo]
    m_up = j_grid < Nx - 1
    u_rows = flat_idx[m_up]; u_cols = (flat_idx + 1)[m_up]
    u_vals = (D2_all[:, None] + mu2_all)[m_up]
    cr, cc, cv = [], [], []
    for ell in range(M):
        for m in range(M):
            if m != ell and abs(Q[ell, m]) > 1e-30:
                cr.append(offsets[ell] + j_idx); cc.append(offsets[m] + j_idx)
                cv.append(np.full(Nx, Q[ell, m]))
    if cr: cr = np.concatenate(cr); cc = np.concatenate(cc); cv = np.concatenate(cv)
    else: cr = np.array([], dtype=int); cc = np.array([], dtype=int); cv = np.array([])
    rows = np.concatenate([d_rows, l_rows, u_rows, cr])
    cols = np.concatenate([d_cols, l_cols, u_cols, cc])
    vals = np.concatenate([d_vals, l_vals, u_vals, cv])
    return csr_matrix(coo_matrix((vals, (rows, cols)), shape=(N, N)).T)

# ══════════════════════════════════════════════════════════════
# UNIFORMIZATION (matrix exponential for sparse generators)
# ══════════════════════════════════════════════════════════════
def unif_cpu(A, v, t, tol=1e-13):
    diag = np.array(A.diagonal()); lam = float(np.max(-diag))
    if lam < 1e-30: return v.copy()
    P = A.copy(); P.setdiag(P.diagonal() + lam); P = P * (1.0 / lam)
    tgt = 30.0; ns = max(1, int(np.ceil(lam * t / tgt))); dt_ = t / ns
    ld = lam * dt_; K = int(ld + 6 * np.sqrt(max(ld, 1))) + 5; K = max(K, 10)
    w = v.copy()
    for _ in range(ns):
        r = w * np.exp(-ld); term = w.copy(); c = np.exp(-ld)
        for k in range(1, K + 1):
            term = P.dot(term); c *= ld / k; r += c * term
            if c * np.max(np.abs(term)) < tol * (np.max(np.abs(r)) + 1e-30): break
        w = r
    return w

def unif_gpu(A, v, t, tol=1e-13):
    if not _GPU: return unif_cpu(A, v, t, tol)
    diag = np.array(A.diagonal()); lam = float(np.max(-diag))
    if lam < 1e-30: return v.copy()
    P_cpu = A.copy(); P_cpu.setdiag(P_cpu.diagonal() + lam); P_cpu = P_cpu * (1.0 / lam)
    P_g = cp_sparse.csr_matrix((cp.asarray(P_cpu.data), cp.asarray(P_cpu.indices),
                                 cp.asarray(P_cpu.indptr)), shape=P_cpu.shape)
    tgt = 30.0; ns = max(1, int(np.ceil(lam * t / tgt))); dt_ = t / ns
    ld = lam * dt_; K = int(ld + 6 * np.sqrt(max(ld, 1))) + 5; K = max(K, 10)
    w = cp.asarray(v); enld = np.exp(-ld)
    for _ in range(ns):
        r = w * enld; term = w.copy(); c = enld
        for k in range(1, K + 1):
            term = P_g.dot(term); c *= ld / k; r = r + c * term
            if k > 5 and c < tol: break
        w = r
    return cp.asnumpy(w)

_propagate_vec = unif_gpu if _GPU else unif_cpu

def _propagate_batch(A, p_batch, dt):
    """Apply the same generator A to multiple density vectors.
    p_batch: list of 1D arrays. Returns list of propagated arrays."""
    return [_propagate_vec(A, p, dt) for p in p_batch]

# ══════════════════════════════════════════════════════════════
# BUCKET-AWARE DENSITY PROPAGATOR
# ══════════════════════════════════════════════════════════════
class LampertiPropagator:
    """
    Propagates joint density in (v, X) space using the coupled generator.
    
    Key design: bucket-aware propagation matching calibration structure.
    
    The calibration evolved density bucket-by-bucket with n_sub substeps
    per bucket, storing leverage_time[k][s] at each substep s of bucket k.
    This propagator replicates that structure exactly:
      - Splits propagation at pillar boundaries
      - Uses native n_sub substeps per bucket
      - Accesses leverage_time[k][s] at stored resolution
      - Computes dgdt from consecutive native snapshots
    
    For barrier application:
      - Convert X-space density to z-space via z = g⁻¹(X + ρv/ξ)
      - Apply barriers in z-space (z = log(S/F))
    """
    
    def __init__(self, model: LampertiModel):
        self.model = model
        self.M = model.n_states
        self.Nx = len(model.X_grid)
        self.Nz = len(model.z_grid)
        self.N = self.M * self.Nx
        self.n_sub_per_bucket = model.n_substeps
        self.rp2 = 1 - model.rho**2
        self.v_shifts = model.rho * model.v_states / model.xi
        self.dgdt_clip = float(getattr(model, 'dgdt_clip', 160.0))
        
        # Precompute bucket structure
        self.pillar_T = model.pillar_T
        self.n_buckets = len(self.pillar_T)
        # Bucket k covers [T_start_k, T_end_k] where T_end_k = pillar_T[k]
        self.bucket_T_start = np.zeros(self.n_buckets)
        self.bucket_T_start[1:] = self.pillar_T[:-1]
        self.bucket_T_end = self.pillar_T.copy()
        self.bucket_dt = self.bucket_T_end - self.bucket_T_start
        
        # Leverage time arrays
        self.has_lt = model.leverage_time is not None
        if self.has_lt:
            self.lt_arrays = model.leverage_time
            self.lt_n = [arr.shape[0] if arr is not None else 0 for arr in model.leverage_time]
    
    def _find_bucket(self, t):
        """Find which bucket time t belongs to for leverage lookup.
        Inclusive: t at bucket end is IN that bucket (for leverage access)."""
        for k in range(self.n_buckets):
            if t <= self.bucket_T_end[k] + 1e-12:
                return k
        return self.n_buckets - 1
    
    def _get_leverage_native(self, bucket_k, substep_s):
        """Get leverage L(z) at native substep s of bucket k.
        This is the EXACT leverage the calibration used."""
        if self.has_lt and self.lt_arrays[bucket_k] is not None:
            lt = self.lt_arrays[bucket_k]
            n = lt.shape[0]
            s_clamped = min(max(substep_s, 0), n - 1)
            return lt[s_clamped].copy()
        return self.model.leverage[bucket_k].copy()
    
    def _get_leverage_at_time(self, t):
        """Get L(z) at arbitrary time t by interpolating stored leverage snapshots."""
        T = self.pillar_T
        
        if self.has_lt:
            k = self._find_bucket(t)
            lt = self.lt_arrays[k]
            if lt is None:
                return self.model.leverage[k].copy()
            n = lt.shape[0]
            if n < 2:
                return lt[0].copy()
            Ts = self.bucket_T_start[k]
            dk = self.bucket_dt[k]
            fr = max(0.0, min(1.0, (t - Ts) / dk if dk > 1e-15 else 0.0))
            fi = fr * (n - 1)
            il = min(max(int(fi), 0), n - 2)
            w = fi - il
            return (1 - w) * lt[il] + w * lt[il + 1]
        
        # Fallback: pillar-end leverage interpolation
        if t <= T[0]: return self.model.leverage[0].copy()
        if t >= T[-1]: return self.model.leverage[-1].copy()
        i = max(0, min(np.searchsorted(T, t) - 1, len(T) - 2))
        w = (t - T[i]) / (T[i+1] - T[i]) if T[i+1] > T[i] else 0.0
        return (1 - w) * self.model.leverage[i] + w * self.model.leverage[i+1]
    
    def _build_drift_and_generator(self, L_prev, L_new, g_prev, g_new, dt_sub):
        """Build drift mu(ell,j) and coupled generator at one substep.
        Matches idk.py calibration formula exactly:
            L_for_drift = 0.5 * (L_prev + L_new)          (midpoint L)
            dLdz        = gradient(L_for_drift)            (on midpoint)
            g_for_map   = 0.5 * (g_prev + g_new)           (midpoint g)
            dgdt        = clip((g_new - g_prev) / dt_sub, ± dgdt_clip)
        """
        m = self.model
        M, Nx = self.M, self.Nx
        X_grid, z_grid, dz = m.X_grid, m.z_grid, m.dz
        vs, Q = m.v_states, m.Q
        v_shifts = self.v_shifts
        rho, kappa, theta, xi = m.rho, m.kappa, m.theta, m.xi
        mart_corr = m.mart_corr

        L_for_drift = 0.5 * (L_prev + L_new)
        dLdz = np.gradient(L_for_drift, dz)
        g_for_map = 0.5 * (g_prev + g_new)
        dgdt = np.clip((g_new - g_prev) / dt_sub,
                       -self.dgdt_clip, self.dgdt_clip)

        mu_all = np.zeros((M, Nx))
        for ell in range(M):
            gt = X_grid + v_shifts[ell]
            zX = np.interp(gt, g_for_map, z_grid)
            mu_all[ell] = (-0.5 * (np.interp(zX, z_grid, L_for_drift)
                                   + np.interp(zX, z_grid, dLdz)) * vs[ell]
                           + np.interp(zX, z_grid, dgdt)
                           - rho * kappa * (theta - vs[ell]) / xi
                           + mart_corr[ell])

        A_fwd = build_forward_gen_coo(mu_all, vs, Q, M, Nx, m.dX, self.rp2)
        return A_fwd
    
    def _get_L_prev_for_bucket_start(self, bucket_k):
        """Get the correct L_prev for the start of bucket k.
        
        During calibration, L_prev at the start of each bucket was:
          - Bucket 0: L_init = sigma_LV / sqrt(v0) (clipped, smoothed)
          - Bucket k>0: pillar-end leverage of previous bucket = leverage[k-1]
        
        leverage_time[k][0] stores L_NEW (output of forward induction at step 0),
        NOT L_prev (input to step 0). Using it as L_prev causes dgdt=0 at the
        first substep, missing the sigma_LV jump at bucket boundaries.
        """
        if bucket_k == 0:
            # L_init = sigma_LV_0 / sqrt(v0), matching calibration line 383
            m = self.model
            sigma_LV_0 = m.sigma_lv[0]
            L_init = np.clip(sigma_LV_0 / np.sqrt(max(m.v0, 1e-6)), 1.0/m.lcap, m.lcap)
            return L_init
        else:
            # Previous bucket's pillar-end leverage
            return self.model.leverage[bucket_k - 1].copy()
    
    def _propagate_within_bucket(self, p_list, bucket_k, s_start, s_end,
                                   L_prev_override=None, g_prev_override=None):
        """Propagate density vectors through substeps [s_start, s_end) of bucket k.
        
        Uses NATIVE leverage snapshots at calibration resolution.
        Correctly initializes L_prev to match calibration's state at each substep.

        If L_prev_override / g_prev_override are provided (post-remap values
        at bucket start), they are used instead of the default
        _get_L_prev_for_bucket_start / compute_g recomputation.
        """
        if s_start >= s_end:
            return [p.copy() for p in p_list]
        
        n_sub_total = self.lt_n[bucket_k] if self.has_lt and self.lt_arrays[bucket_k] is not None else self.n_sub_per_bucket
        dt_sub = self.bucket_dt[bucket_k] / n_sub_total
        
        if dt_sub < 1e-15:
            return [p.copy() for p in p_list]

        # Initialize L_prev / g_prev for this substepping call.
        # Priority: explicit override (post-remap) > bucket-start default > previous substep.
        if L_prev_override is not None:
            L_prev = L_prev_override.copy()
            g_prev = (g_prev_override.copy()
                      if g_prev_override is not None
                      else compute_g(self.model.z_grid, L_prev))
        elif s_start == 0:
            L_prev = self._get_L_prev_for_bucket_start(bucket_k)
            g_prev = compute_g(self.model.z_grid, L_prev)
        else:
            L_prev = self._get_leverage_native(bucket_k, s_start - 1)
            g_prev = compute_g(self.model.z_grid, L_prev)
        
        result = [p.copy() for p in p_list]
        
        for s in range(s_start, s_end):
            # leverage_time[k][s] = L_new from calibration step s
            L_new = self._get_leverage_native(bucket_k, s)
            g_new = compute_g(self.model.z_grid, L_new)
            
            # Build generator (same for all slices)
            A_fwd = self._build_drift_and_generator(L_prev, L_new, g_prev, g_new, dt_sub)
            
            # Propagate all slices with same generator
            result = [np.maximum(_propagate_vec(A_fwd, p, dt_sub), 0) for p in result]
            
            g_prev = g_new.copy()
            L_prev = L_new.copy()
        
        return result
    
    def _find_bucket_for_propagation(self, t):
        """Find the bucket to propagate IN at time t.
        
        When t is exactly at a pillar boundary (end of bucket k = start of bucket k+1),
        we want the NEXT bucket (k+1) since we're propagating forward from that point.
        Uses a small tolerance to handle floating point.
        """
        for k in range(self.n_buckets):
            Te = self.bucket_T_end[k]
            # If t is strictly inside this bucket (not at the end boundary), use it
            if t < Te - 1e-10:
                return k
        # t is at or past the last pillar
        return self.n_buckets - 1
    
    def _recompute_L_post_remap(self, p_vec, g_old, L_old, bucket_k):
        """Reproduce the calibrator's post-remap L at the start of bucket k:
            u_z_bnd[ell] = interp(u_X_bnd[ell], g_old - shift_ell) / max(L_old, 0.1)
            L_new_bnd    = sigma_LV_k / sqrt(E[v|z])  (clipped to [1/lcap, lcap])
        Matches idk.py compute_leverage behavior at boundary (omega applied
        against L_old if omega < 1; otherwise pure new L)."""
        m = self.model
        sigma_LV = m.sigma_lv[bucket_k]
        u_X = p_vec.reshape(self.M, self.Nx)
        u_z = np.zeros((self.M, self.Nz))
        for ell in range(self.M):
            u_z[ell] = interp_density(
                m.X_grid, u_X[ell], g_old - self.v_shifts[ell]
            ) / np.maximum(L_old, 0.1)
        p_z = np.sum(u_z, axis=0)
        mom_z = np.sum(m.v_states[:, None] * u_z, axis=0)
        total_mass = float(np.sum(u_z))
        vm = float(np.sum(m.v_states * np.sum(u_z, axis=1)) / max(total_mass, 1e-12))
        Ev = np.full(self.Nz, vm)
        reliable = p_z > 1e-10
        Ev[reliable] = mom_z[reliable] / p_z[reliable]
        Ev = np.maximum(Ev, 1e-6)
        if int(np.sum(reliable)) > 10:
            idx = np.where(reliable)[0]
            Ev[:idx[0]] = Ev[idx[0]]
            Ev[idx[-1]+1:] = Ev[idx[-1]]
        L = sigma_LV / np.sqrt(Ev)
        L = np.clip(L, 1.0 / m.lcap, m.lcap)
        if m.omega < 1.0:
            L = (1.0 - m.omega) * L_old + m.omega * L
        return L

    def _apply_boundary_remap_if_needed(self, p_list, t_start):
        """If t_start coincides with pillar time T_{k-1} (start of bucket k,
        k >= 1), apply the calibrator's inline pillar-boundary sequence:

            L_old, g_old = pillar-end L/g of bucket k-1 (= leverage[k-1], g_pillars[k-1])
            u_z_bnd[ell] = interp(u_X[ell], g_old - shift_ell) / max(L_old, 0.1)
            L_new_bnd    = compute_leverage(u_z_bnd, sigma_LV[k], L_old)
            g_new_bnd    = compute_g(z, L_new_bnd)
            p_new[ell]   = remap_density_at_boundary(p, g_old, L_old, g_new_bnd, L_new_bnd)

        Returns (p_list_new, bucket_entered, L_post, g_post). If no remap fires,
        returns (p_list, None, None, None).
        """
        tol = 1e-8
        m = self.model
        for k in range(1, self.n_buckets):
            # Bucket k starts at pillar_T[k-1]
            if abs(t_start - self.pillar_T[k - 1]) < tol:
                L_old = m.leverage[k - 1]
                g_old = m.g_pillars[k - 1]
                # Recompute L from the first slice (all slices share coords)
                L_new = self._recompute_L_post_remap(p_list[0], g_old, L_old, k)
                g_new = compute_g(m.z_grid, L_new)
                p_list_new = [
                    remap_density_at_boundary(
                        p, g_old, L_old, g_new, L_new,
                        self.v_shifts, m.X_grid, m.z_grid,
                        self.M, self.Nx, m.dX
                    )
                    for p in p_list
                ]
                return p_list_new, k, L_new, g_new
        return p_list, None, None, None

    def propagate_batch(self, p_list, t_start, t_end):
        """Propagate a list of density vectors from t_start to t_end.
        
        Bucket-aware: splits at pillar boundaries and uses native substep
        resolution within each bucket, matching the calibration exactly.
        Applies the calibrator's pillar-entry boundary remap at t_start
        if it coincides with a pillar time (k >= 1).
        
        For times beyond the last pillar, uses the last bucket's end-of-pillar
        leverage with uniform substeps (frozen leverage extrapolation).
        
        p_list: list of 1D arrays, each of length M*Nx.
        Returns: list of propagated 1D arrays.
        """
        if t_end - t_start <= 1e-12 or len(p_list) == 0:
            return [p.copy() for p in p_list]

        # Step 1: apply the pillar-entry boundary remap if t_start is at a pillar
        result, remap_bucket, L_post, g_post = \
            self._apply_boundary_remap_if_needed(p_list, t_start)
        t_cur = t_start
        
        max_iters = self.n_buckets + 5  # safety bound (extra for beyond-last-pillar)
        iters = 0
        
        while t_cur < t_end - 1e-12:
            iters += 1
            if iters > max_iters:
                raise RuntimeError(
                    f"propagate_batch: too many iterations ({iters}). "
                    f"t_cur={t_cur:.10f}, t_end={t_end:.10f}. Likely a bug."
                )
            
            # Beyond the last pillar: use frozen leverage extrapolation
            if t_cur >= self.bucket_T_end[-1] - 1e-10:
                dt_remain = t_end - t_cur
                result = self._propagate_beyond_last_pillar(result, t_cur, t_end)
                t_cur = t_end
                break
            
            k = self._find_bucket_for_propagation(t_cur)
            
            # Bucket boundaries
            Ts = self.bucket_T_start[k]
            Te = self.bucket_T_end[k]
            dk = self.bucket_dt[k]
            n_sub_k = self.lt_n[k] if self.has_lt and self.lt_arrays[k] is not None else self.n_sub_per_bucket
            
            if dk < 1e-15 or n_sub_k == 0:
                # Degenerate bucket, skip to its end
                t_cur = Te
                continue
            
            dt_sub_k = dk / n_sub_k
            
            # Substep index at t_cur within this bucket
            frac_start = max(0.0, min(1.0, (t_cur - Ts) / dk))
            s_start = int(round(frac_start * n_sub_k))
            s_start = min(max(s_start, 0), n_sub_k)
            
            # Where do we stop in this bucket?
            t_bucket_end = min(t_end, Te)
            frac_end = max(0.0, min(1.0, (t_bucket_end - Ts) / dk))
            s_end = int(round(frac_end * n_sub_k))
            s_end = min(max(s_end, 0), n_sub_k)
            
            if s_end > s_start:
                # Pass post-remap override only on the first bucket entry
                # (when k == remap_bucket and we're at s_start == 0).
                if remap_bucket is not None and k == remap_bucket and s_start == 0:
                    result = self._propagate_within_bucket(
                        result, k, s_start, s_end,
                        L_prev_override=L_post, g_prev_override=g_post)
                    # Clear override so subsequent buckets use their own logic
                    remap_bucket = None; L_post = None; g_post = None
                else:
                    result = self._propagate_within_bucket(result, k, s_start, s_end)
            
            # Always advance past this bucket's endpoint to avoid re-entering it
            t_cur = Te if t_bucket_end >= Te - 1e-12 else t_bucket_end
        
        return result
    
    def _propagate_beyond_last_pillar(self, p_list, t_start, t_end):
        """Propagate beyond the last calibration pillar using frozen leverage.
        
        Uses the leverage at the end of the last bucket (pillar end) and
        a uniform substep grid scaled to match the last bucket's resolution.
        """
        dt_total = t_end - t_start
        if dt_total <= 1e-12:
            return [p.copy() for p in p_list]
        
        # Use last bucket's end leverage and g
        last_k = self.n_buckets - 1
        L_frozen = self._get_leverage_native(last_k, self.lt_n[last_k] - 1 if self.has_lt and self.lt_arrays[last_k] is not None else 0)
        g_frozen = compute_g(self.model.z_grid, L_frozen)
        
        # Scale substep count: use same dt_sub as the last bucket
        last_dt_sub = self.bucket_dt[last_k] / max(self.n_sub_per_bucket, 1)
        n_sub = max(1, int(np.ceil(dt_total / last_dt_sub)))
        dt_sub = dt_total / n_sub
        
        # Build ONE generator with frozen leverage (no dgdt since L is constant)
        m = self.model
        M, Nx = self.M, self.Nx
        mu_all = np.zeros((M, Nx))
        dLdz = np.gradient(L_frozen, m.dz)
        for ell in range(M):
            gt = m.X_grid + self.v_shifts[ell]
            zX = np.interp(gt, g_frozen, m.z_grid)
            mu_all[ell] = (-0.5 * (np.interp(zX, m.z_grid, L_frozen)
                                   + np.interp(zX, m.z_grid, dLdz)) * m.v_states[ell]
                           + 0.0  # dgdt = 0 (frozen leverage)
                           - m.rho * m.kappa * (m.theta - m.v_states[ell]) / m.xi
                           + m.mart_corr[ell])
        
        A_fwd = build_forward_gen_coo(mu_all, m.v_states, m.Q, M, Nx, m.dX, self.rp2)
        
        result = [p.copy() for p in p_list]
        for _ in range(n_sub):
            result = [np.maximum(_propagate_vec(A_fwd, p, dt_sub), 0) for p in result]
        
        return result
    
    def propagate(self, p_vec, t_start, t_end):
        """Propagate a single density vector from t_start to t_end."""
        return self.propagate_batch([p_vec], t_start, t_end)[0]
    
    def get_g_at_time(self, t):
        """Get the Lamperti transform g(z) at time t.
        
        At pillar times: use stored g_pillars[k] for exact match.
        Otherwise: compute from leverage at time t.
        """
        tol = 1e-6
        for k in range(len(self.pillar_T)):
            if abs(t - self.pillar_T[k]) < tol:
                return self.model.g_pillars[k].copy()
        
        # Compute from leverage at time t
        L = self._get_leverage_at_time(t)
        return compute_g(self.model.z_grid, L)
    
    def mass(self, p_vec):
        return float(np.sum(p_vec) * self.model.dX)
    
    def z_at_Xv(self, g):
        """Compute z(ell, j) = g⁻¹(X_j + ρv_ℓ/ξ) for all (ell, j).
        Returns (M, Nx) array of z-values."""
        M, Nx = self.M, self.Nx
        z_grid = self.model.z_grid
        X_grid = self.model.X_grid
        z_all = np.zeros((M, Nx))
        for ell in range(M):
            gt = X_grid + self.v_shifts[ell]
            z_all[ell] = np.interp(gt, g, z_grid)
        return z_all

# ══════════════════════════════════════════════════════════════
# CHECKPOINT DENSITY RETRIEVAL
# ══════════════════════════════════════════════════════════════
def get_density_at_time(model, t, propagator):
    """Get the joint density at time t using calibrated pillar densities as checkpoints.
    
    Strategy:
      - If t matches a pillar time (within 1e-6): use stored density directly.
      - Otherwise: find nearest preceding pillar, propagate from there.
      - If t < first pillar: build Dirac delta IC and propagate.
    
    Matches the rho=0 pricer's checkpoint logic. The bucket-aware
    propagation handles small gaps (e.g. 0.25 vs 0.2411) efficiently
    with just a few native-resolution substeps.
    """
    pillar_T = model.pillar_T
    tol = 1e-6
    
    # Exact match: use stored density directly
    for k, T_k in enumerate(pillar_T):
        if abs(t - T_k) < tol:
            den = model.densities_X[k]
            return den.reshape(-1).copy()
    
    # Find nearest preceding pillar
    preceding_idx = -1
    for k, T_k in enumerate(pillar_T):
        if T_k < t - tol:
            preceding_idx = k
    
    if preceding_idx >= 0:
        den = model.densities_X[preceding_idx]
        p_start = den.reshape(-1).copy()
        t_start = pillar_T[preceding_idx]
        return propagator.propagate(p_start, t_start, t)
    
    # t is before first pillar: propagate from Dirac delta
    # With the L_prev off-by-one fix, this should now reproduce the
    # calibration's density correctly.
    M = model.n_states; Nx = len(model.X_grid); dX = model.dX
    p0 = np.zeros(M * Nx)
    for ell in range(M):
        X0 = -model.rho * model.v_states[ell] / model.xi
        fx = (X0 - model.X_grid[0]) / dX; il = int(fx); ir = il + 1
        if 0 <= il < Nx and 0 <= ir < Nx:
            w = fx - il
            p0[ell*Nx + il] = model.pi0[ell] * (1 - w) / dX
            p0[ell*Nx + ir] = model.pi0[ell] * w / dX
        elif 0 <= il < Nx:
            p0[ell*Nx + il] = model.pi0[ell] / dX
    
    if t < 1e-6:
        return p0
    return propagator.propagate(p0, 0.0, t)

# ══════════════════════════════════════════════════════════════
# SLICE MANAGEMENT
# ══════════════════════════════════════════════════════════════
def propagate_slices(slices, t_start, t_end, propagator, mass_threshold=1e-12):
    """Propagate all live slices from t_start to t_end using batch propagation.
    
    Builds the generator once per substep and applies to all slices together.
    """
    if len(slices) == 0 or t_end - t_start <= 1e-12:
        return {k: v.copy() for k, v in slices.items()}
    
    live_keys = []
    live_arrays = []
    for key, u in slices.items():
        if propagator.mass(u) >= mass_threshold:
            live_keys.append(key)
            live_arrays.append(u)
    
    if len(live_keys) == 0:
        return {}
    
    # Batch propagate: same generator applied to all slices
    propagated = propagator.propagate_batch(live_arrays, t_start, t_end)
    
    out = {}
    for j, key in enumerate(live_keys):
        out[key] = propagated[j]
    return out

# ══════════════════════════════════════════════════════════════
# BARRIER APPLICATION
# ══════════════════════════════════════════════════════════════
def apply_z_split(p_vec, z_all, z_threshold, M, Nx, dX, above=True):
    """Split density at z-threshold. Returns (kept, removed, removed_mass).
    
    z_all: precomputed z(ell,j) array of shape (M, Nx).
    above=True: keep mass where z >= threshold, remove where z < threshold.
    above=False: keep mass where z < threshold, remove where z >= threshold.
    """
    u = p_vec.reshape(M, Nx).copy()
    removed = np.zeros_like(u)
    
    if above:
        mask = z_all < z_threshold
    else:
        mask = z_all >= z_threshold
    
    removed[mask] = u[mask]
    u[mask] = 0.0
    
    removed_mass = float(np.sum(removed) * dX)
    return u.ravel(), removed.ravel(), removed_mass

def compute_put_payoff(p_vec, z_all, F, put_strike, S0, M, Nx, dX):
    """Compute E[min(S/(K*S0), 1)] under the density."""
    u = p_vec.reshape(M, Nx)
    S_ratio = F * np.exp(z_all) / (put_strike * S0)
    payoff = np.minimum(S_ratio, 1.0)
    return float(np.sum(payoff * np.maximum(u, 0.0)) * dX)

# ══════════════════════════════════════════════════════════════
# PRICING ENGINE
# ══════════════════════════════════════════════════════════════
def price_autocallable(model, spec, fwd, disc, propagator=None, verbose=True):
    t0_wall = time.time()
    S0 = model.S0; dX = model.dX
    M = model.n_states; Nx = len(model.X_grid); N = M * Nx
    obs = generate_observation_dates(spec.maturity_years, spec.obs_freq)
    K = len(obs)
    
    if propagator is None:
        propagator = LampertiPropagator(model)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"AUTOCALLABLE PRICER (CTMC-Lamperti-LSV, ρ={model.rho:.4f})")
        print(f"{'='*70}")
        print(f"  S0={S0:.2f} Mat={spec.maturity_years:.4f}y Freq={spec.obs_freq} Obs={K}")
        print(f"  Barriers: AC={spec.ac_barrier} Cpn={spec.coupon_barrier} KI={spec.ki_barrier}")
        print(f"  Coupon={spec.coupon_rate*100:.4f}%/period Memory={spec.memory}")
        print(f"  Leverage: {'substep' if propagator.has_lt else 'pillar-interp'}")
        print(f"  dgdt_clip={propagator.dgdt_clip:.0f} (match calibrator --clip)")
        print(f"  n_sub/bucket={propagator.n_sub_per_bucket} Buckets={propagator.n_buckets}")
        print(f"  Pillar T: {list(model.pillar_T)}")
    
    # ── Get density at first observation using checkpoint ──
    p_first = get_density_at_time(model, obs[0], propagator)
    init_mass = propagator.mass(p_first)
    
    if verbose:
        # Check if we used a checkpoint or propagated
        best_d = min(abs(obs[0] - T_k) for T_k in model.pillar_T)
        used_checkpoint = best_d < 1e-6
        if used_checkpoint:
            best_k = int(np.argmin([abs(obs[0] - T_k) for T_k in model.pillar_T]))
            print(f"  First obs T={obs[0]:.4f}: checkpoint density (pillar {model.pillar_labels[best_k]}M) mass={init_mass:.6f}")
        else:
            prec_k = -1
            for kk, T_k in enumerate(model.pillar_T):
                if T_k < obs[0] - 1e-6: prec_k = kk
            if prec_k >= 0:
                print(f"  First obs T={obs[0]:.4f}: propagated from pillar {model.pillar_labels[prec_k]}M "
                      f"(T={model.pillar_T[prec_k]:.4f}, gap={obs[0]-model.pillar_T[prec_k]:.6f}) mass={init_mass:.6f}")
            else:
                print(f"  First obs T={obs[0]:.4f}: propagated from t=0, mass={init_mass:.6f}")
    
    slices: Dict[Tuple[int, int], np.ndarray] = {(0, 0): p_first}
    ap = np.zeros(K); ac = np.zeros(K); cc = np.zeros(K)
    tp = tpu = price = 0.0
    tv = obs[0]
    
    for k in range(K):
        to = float(obs[k]); D = float(disc(to)); F = float(fwd(to))
        ab = spec.ac_barrier - spec.ac_step_down * k
        g_now = propagator.get_g_at_time(to)
        z_all = propagator.z_at_Xv(g_now)  # (M, Nx) precomputed z-mapping
        
        # z thresholds for barriers
        za = np.log(max(ab * S0 / F, 1e-12))
        zc = np.log(max(spec.coupon_barrier * S0 / F, 1e-12))
        zk = np.log(max(spec.ki_barrier * S0 / F, 1e-12))
        
        fin = (k == K - 1)
        can = (k >= spec.no_call_periods) and not fin
        
        # ── Propagate all slices to this observation date ──
        if k > 0:
            slices = propagate_slices(slices, tv, to, propagator)
        
        # ── KI barrier check ──
        us = {}
        for (b, m), u in slices.items():
            if b == 0:
                u_survived, u_ki_flat, ki_mass = apply_z_split(u, z_all, zk, M, Nx, dX, above=True)
                if ki_mass > 1e-12:
                    k2 = (1, m)
                    us[k2] = us.get(k2, np.zeros(N)) + u_ki_flat
                if propagator.mass(u_survived) > 1e-12:
                    k2 = (0, m)
                    us[k2] = us.get(k2, np.zeros(N)) + u_survived
            else:
                k2 = (b, m)
                us[k2] = us.get(k2, np.zeros(N)) + u.copy()
        slices = us
        
        # ── Autocall + coupon processing ──
        post = {}
        for (b, m), u in slices.items():
            if can:
                u_survived, u_ac_flat, ac_mass = apply_z_split(u, z_all, za, M, Nx, dX, above=False)
                if ac_mass > 1e-12:
                    nc = (m + 1) if spec.memory else 1
                    cv = D * spec.notional * (1 + nc * spec.coupon_rate) * ac_mass
                    price += cv; ap[k] += ac_mass; ac[k] += cv
                u = u_survived
            
            if fin:
                u_above_cpn, _, _ = apply_z_split(u, z_all, zc, M, Nx, dX, above=True)
                cpn_mass = propagator.mass(u_above_cpn)
                if cpn_mass > 0:
                    nc = (m + 1) if spec.memory else 1
                    cv = D * spec.notional * nc * spec.coupon_rate * cpn_mass
                    cc[k] += cv; price += cv
                
                tm = propagator.mass(u)
                if b == 0:
                    pv = D * spec.notional * tm
                    tp += pv; price += pv
                else:
                    pv = D * spec.notional * compute_put_payoff(u, z_all, F, spec.put_strike, S0, M, Nx, dX)
                    tpu += pv; price += pv
                continue
            
            u_above_cpn, u_below_cpn, _ = apply_z_split(u, z_all, zc, M, Nx, dX, above=True)
            ma = propagator.mass(u_above_cpn)
            mb = propagator.mass(u_below_cpn)
            
            if ma > 1e-12:
                nc = (m + 1) if spec.memory else 1
                cv = D * spec.notional * nc * spec.coupon_rate * ma
                cc[k] += cv; price += cv
                kr = (b, 0)
                post[kr] = post.get(kr, np.zeros(N)) + u_above_cpn
            if mb > 1e-12:
                ki = (b, m + 1 if spec.memory else 0)
                post[ki] = post.get(ki, np.zeros(N)) + u_below_cpn
        
        if not fin:
            slices = post
        
        slices = {k_: v for k_, v in slices.items() if propagator.mass(v) > 1e-12}
        
        if verbose:
            sm = sum(propagator.mass(v) for v in slices.values())
            print(f"  Obs {k+1}/{K}: T={to:.4f} F={F:.2f} D={D:.6f} surv={sm:.6f} AC={ap[k]:.6f} sl={len(slices)}")
        
        tv = to
    
    # ── Final summary ──
    su = sum(propagator.mass(v) for v in slices.values())
    sp = np.zeros(K); sp[:-1] = ap[:-1]; sp[-1] = su
    ts = float(np.sum(sp))
    ee = float(np.dot(obs, sp) / ts) if ts > 1e-15 else 0.0
    
    if verbose:
        print(f"\n  Price={price:.8f} ({price/spec.notional*100:.4f}%) Surv={su:.6f} "
              f"E[T*]={ee:.4f} {time.time()-t0_wall:.2f}s")
    
    return PricingResult(
        price=price, notional=spec.notional, price_pct=price/spec.notional*100,
        autocall_probabilities=ap, stop_probabilities=sp,
        coupon_contributions=cc, autocall_contributions=ac,
        terminal_par_contribution=tp, terminal_put_contribution=tpu,
        survival_probability=su, ki_probability=0.0,
        observation_dates=obs, memory_enabled=spec.memory,
        expected_expiry_years=ee)

# ══════════════════════════════════════════════════════════════
# FAIR COUPON SOLVER
# ══════════════════════════════════════════════════════════════
def solve_fair_coupon(model, spec, fwd, disc, verbose=True):
    prop = LampertiPropagator(model)
    s0 = AutocallableSpec(**{**spec.__dict__, "coupon_rate": 0.0})
    s1 = AutocallableSpec(**{**spec.__dict__, "coupon_rate": 1.0})
    r0 = price_autocallable(model, s0, fwd, disc, propagator=prop, verbose=verbose)
    r1 = price_autocallable(model, s1, fwd, disc, propagator=prop, verbose=verbose)
    Vu = r1.price - r0.price
    if abs(Vu) < 1e-15: return 0.0, r0
    fc = (spec.notional - r0.price) / Vu
    if verbose:
        npy = len(generate_observation_dates(spec.maturity_years, spec.obs_freq)) / spec.maturity_years
        print(f"  Fair coupon={fc*100:.4f}%/period = {fc*npy*100:.4f}% p.a.")
    sf = AutocallableSpec(**{**spec.__dict__, "coupon_rate": fc})
    rf = price_autocallable(model, sf, fwd, disc, propagator=prop, verbose=verbose)
    rf.fair_coupon = fc
    return fc, rf

# ══════════════════════════════════════════════════════════════
# TERM STRUCTURE
# ══════════════════════════════════════════════════════════════
def price_ts(model, base, mats, cpns, fwd, disc, verbose=True):
    pairs = sorted(zip(mats, cpns)); pts = []
    prop = LampertiPropagator(model)
    for j, (T, c) in enumerate(pairs, 1):
        if verbose: print(f"\n  TERM {j}/{len(pairs)}: T={T:.4f} c={100*c:.4f}%")
        sp = AutocallableSpec(**{**base.__dict__, "maturity_years": T, "coupon_rate": c})
        r = price_autocallable(model, sp, fwd, disc, propagator=prop, verbose=verbose)
        pd = r.price / sp.notional - 1.0
        pts.append(TermStructurePoint(T, c, r.price, r.price_pct, pd, 1e4*pd,
            r.survival_probability, r.terminal_par_contribution,
            r.terminal_put_contribution, r.expected_expiry_years, sp.obs_freq))
    return pts

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════
def parse_float_list(t, n):
    if not t or not t.strip(): return None
    try: return [float(x.strip()) for x in t.split(",") if x.strip()]
    except ValueError as e: raise ValueError(f"Cannot parse {n}") from e

def resolve_cpn(freq, mats, common, mo, qu, sa, an, fb):
    freq = normalize_obs_freq(freq)
    sp = {"monthly":mo, "quarterly":qu, "semi-annual":sa, "annual":an}.get(freq)
    cl = sp if sp is not None else common
    if cl is None: return [fb] * len(mats)
    if len(cl) == 1 and len(mats) > 1: return cl * len(mats)
    if len(cl) != len(mats): raise ValueError(f"Coupon list mismatch for {freq}")
    return cl

def save_csv(pts, path):
    fn = ["obs_freq","maturity_years","coupon_rate","price","price_diff_bps",
          "expected_expiry_years","survival_probability"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
        for p in pts: w.writerow({k: getattr(p, k) for k in fn})

def print_summary(pts, hdr="TERM STRUCTURE"):
    print(f"\n{'='*120}\n{hdr}\n{'='*120}")
    print(f"{'Freq':>12}{'Mat':>8}{'Cpn%':>8}{'Price':>14}{'D(bps)':>10}{'E[T*]':>8}{'Surv':>8}")
    for p in pts:
        print(f"{p.obs_freq:>12}{p.maturity_years:8.4f}{100*p.coupon_rate:7.4f}%"
              f"{p.price:14.8f}{p.price_diff_bps:10.2f}{p.expected_expiry_years:8.4f}"
              f"{p.survival_probability:8.4f}")

def plot_multi(curves, path=None, title="Price difference"):
    fig = plt.figure(figsize=(9, 5))
    for f in sorted(curves, key=obs_freq_to_months):
        pts = curves[f]
        m = np.array([p.maturity_years for p in pts])
        d = np.array([p.price_diff_bps for p in pts])
        o = np.argsort(m); plt.plot(m[o], d[o], "o-", lw=1.2, label=f)
    plt.axhline(0, ls="--", c="gray", alpha=.7)
    plt.xlabel("T (years)"); plt.ylabel("bps"); plt.title(title)
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="CTMC-Lamperti-LSV Autocallable Pricer (ρ≠0)")
    p.add_argument("--lsv_result", required=True, help="Path to lamperti_lsv_model.npz")
    p.add_argument("--forward_curve", required=True)
    p.add_argument("--discount_curve", required=True)
    p.add_argument("--notional", type=float, default=1.0)
    p.add_argument("--maturity_years", type=float, default=1.5)
    p.add_argument("--coupon_rate", type=float, default=0.026744)
    p.add_argument("--maturity_years_list", default="0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    p.add_argument("--coupon_rates_list", default="")
    p.add_argument("--coupon_rates_list_monthly", default="0.00841059,0.010805,0.011079,0.01084965,0.0104426,0.01003036,0.00964965,0.00930935")
    p.add_argument("--coupon_rates_list_quarterly", default="0.01908626,0.02461403,0.02670627,0.02742142,0.02727404,0.0267436,0.02608444,0.02540501")
    p.add_argument("--coupon_rates_list_semi_annual", default="0.01908626,0.04162856,0.04122418,0.04768614,0.04594846,0.04834639,0.04589727,0.04702281")
    p.add_argument("--coupon_rates_list_annual", default="0.01908626,0.04162856,0.06213436,0.08211667,0.07232821,0.07759105,0.08177649,0.08541534")
    p.add_argument("--obs_freq", default="quarterly")
    p.add_argument("--obs_freqs_list", default="monthly,quarterly")
    p.add_argument("--ac_barrier", type=float, default=1.0)
    p.add_argument("--coupon_barrier", type=float, default=0.0)
    p.add_argument("--ki_barrier", type=float, default=0.8)
    p.add_argument("--put_strike", type=float, default=1.0)
    p.add_argument("--no_call_periods", type=int, default=0)
    p.add_argument("--ac_step_down", type=float, default=0.0)
    mg = p.add_mutually_exclusive_group()
    mg.add_argument("--memory", dest="memory", action="store_true", default=True)
    mg.add_argument("--no_memory", dest="memory", action="store_false")
    p.add_argument("--solve_coupon", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--output_prefix", default="lamperti_autocallable")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--leverage_time_stride", type=int, default=1)
    p.add_argument("--dgdt_clip", type=float, default=160.0,
                   help="Clip bound for dg/dt within buckets (match calibrator --clip)")
    return p.parse_args()

def main():
    args = parse_args(); verbose = not args.quiet
    if verbose: print(f"Loading {args.lsv_result}...")
    model = load_lamperti_model(args.lsv_result, args.leverage_time_stride,
                                 dgdt_clip=args.dgdt_clip)
    if verbose:
        print(f"  S0={model.S0} states={model.n_states} Nx={len(model.X_grid)} Nz={len(model.z_grid)}")
        print(f"  ρ={model.rho:.4f} κ={model.kappa:.4f} θ={model.theta:.6f} ξ={model.xi:.4f}")
        print(f"  Pillars: {list(model.pillar_labels)}")
    
    fT, fF = load_forward_curve(args.forward_curve)
    dT, dD = load_discount_curve(args.discount_curve)
    fi, di = build_interpolators(fT, fF, dT, dD)
    
    base = AutocallableSpec(
        notional=args.notional, maturity_years=args.maturity_years,
        ac_barrier=args.ac_barrier, coupon_barrier=args.coupon_barrier,
        ki_barrier=args.ki_barrier, coupon_rate=args.coupon_rate,
        put_strike=args.put_strike, memory=args.memory,
        obs_freq=normalize_obs_freq(args.obs_freq),
        no_call_periods=args.no_call_periods, ac_step_down=args.ac_step_down)
    
    mats = parse_float_list(args.maturity_years_list, "mats")
    common = parse_float_list(args.coupon_rates_list, "cpn")
    mo = parse_float_list(args.coupon_rates_list_monthly, "mo")
    qu = parse_float_list(args.coupon_rates_list_quarterly, "qu")
    sa = parse_float_list(args.coupon_rates_list_semi_annual, "sa")
    an = parse_float_list(args.coupon_rates_list_annual, "an")
    freqs = parse_obs_freq_list(args.obs_freqs_list)
    
    if freqs and mats:
        curves = {}
        for freq in freqs:
            cl = resolve_cpn(freq, mats, common, mo, qu, sa, an, base.coupon_rate)
            sf = AutocallableSpec(**{**base.__dict__, "obs_freq": freq})
            if verbose: print(f"\n{'='*80}\nSWEEP: {freq}\n{'='*80}")
            curves[freq] = price_ts(model, sf, mats, cl, fi, di, verbose)
        all_pts = []
        for f in sorted(curves, key=obs_freq_to_months):
            print_summary(curves[f], f"TERM STRUCTURE — {f}")
            all_pts.extend(curves[f])
        save_csv(all_pts, f"{args.output_prefix}.csv")
        if not args.no_plot:
            plot_multi(curves, f"{args.output_prefix}.png")
            plt.close("all")
        return curves
    
    if mats:
        cl = resolve_cpn(base.obs_freq, mats, common, mo, qu, sa, an, base.coupon_rate)
        pts = price_ts(model, base, mats, cl, fi, di, verbose)
        print_summary(pts)
        return pts
    
    if args.solve_coupon:
        return solve_fair_coupon(model, base, fi, di, verbose)
    
    return price_autocallable(model, base, fi, di, verbose=verbose)

if __name__ == "__main__":
    main()