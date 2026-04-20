#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU accelerations:
  1) Batched tridiagonal PDE solves — all N_states systems solved in parallel on GPU
  2) CTMC mixing via cuBLAS dense matrix multiply (QT @ u_joint)
  3) Fused leverage computation kernel (reduction over states per z)
  4) Fused PDE operator construction + RHS assembly
  5) Minimized host<->device transfers: densities stay on GPU during calibration

Falls back to CPU (NumPy) when CUDA is unavailable.

Requirements:
  pip install cupy-cuda12x   # or cupy-cuda11x for CUDA 11
  pip install scipy numpy
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.linalg import expm

# --------------- GPU backend selection ---------------
try:
    import cupy as cp
    from cupy import RawKernel

    _HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False
    cp = None
    RawKernel = None

if _HAS_CUDA:
    print(f"[CUDA] GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"[CUDA] CuPy {cp.__version__}, CUDA {cp.cuda.runtime.runtimeGetVersion()}")
else:
    print("[CUDA] No GPU detected — falling back to CPU (NumPy)")


# ============================================================
# CUDA Kernels (compiled once, reused)
# ============================================================
if _HAS_CUDA:
    # ---- Kernel 1: Build forward Fokker-Planck operator for ALL states at once ----
    _KERNEL_BUILD_FWD_OP = RawKernel(
        r'''
    extern "C" __global__
    void build_forward_op_batched(
        const double* __restrict__ L2,        // [Nz]
        const double* __restrict__ v_states,  // [N_states]
        double* __restrict__ sub,             // [N_states, Nz]
        double* __restrict__ diag,            // [N_states, Nz]
        double* __restrict__ sup,             // [N_states, Nz]
        int N_states, int Nz, double dz)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int total = N_states * Nz;
        if (tid >= total) return;

        int state = tid / Nz;
        int j = tid % Nz;
        int base = state * Nz;

        double dz2 = dz * dz;
        double v_i = v_states[state];
        double a_j = L2[j] * v_i;

        if (j == 0) {
            double a_next = L2[1] * v_i;
            sub[base + j]  = 0.0;
            diag[base + j] = -a_j / dz2 - a_j / (4.0 * dz);
            sup[base + j]  =  a_next / dz2 + a_next / (4.0 * dz);
        } else if (j == Nz - 1) {
            double a_prev = L2[Nz - 2] * v_i;
            sub[base + j]  =  a_prev / dz2 - a_prev / (4.0 * dz);
            diag[base + j] = -a_j / dz2 + a_j / (4.0 * dz);
            sup[base + j]  = 0.0;
        } else {
            double a_prev = L2[j - 1] * v_i;
            double a_next = L2[j + 1] * v_i;
            sub[base + j]  = 0.5 * a_prev / dz2 - a_prev / (4.0 * dz);
            diag[base + j] = -a_j / dz2;
            sup[base + j]  = 0.5 * a_next / dz2 + a_next / (4.0 * dz);
        }
    }
    ''',
        "build_forward_op_batched",
    )

    # ---- Kernel 2: Build RHS for all states (explicit part of theta-scheme) ----
    _KERNEL_BUILD_RHS = RawKernel(
        r'''
    extern "C" __global__
    void build_rhs_batched(
        const double* __restrict__ phi,   // [N_states, Nz]
        const double* __restrict__ sub,
        const double* __restrict__ diag,
        const double* __restrict__ sup,
        double* __restrict__ rhs,         // [N_states, Nz]
        int N_states, int Nz, double expl_factor)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int total = N_states * Nz;
        if (tid >= total) return;

        int s = tid / Nz;
        int j = tid % Nz;
        int base = s * Nz;

        double p = phi[base + j];
        double val;

        if (j == 0) {
            val = p * (1.0 + expl_factor * diag[base]);
            if (Nz > 1) val += phi[base + 1] * expl_factor * sup[base];
        } else if (j == Nz - 1) {
            val = p * (1.0 + expl_factor * diag[base + j]);
            if (Nz > 1) val += phi[base + j - 1] * expl_factor * sub[base + j];
        } else {
            val = phi[base + j - 1] * expl_factor * sub[base + j]
                + p * (1.0 + expl_factor * diag[base + j])
                + phi[base + j + 1] * expl_factor * sup[base + j];
        }
        rhs[base + j] = val;
    }
    ''',
        "build_rhs_batched",
    )

    # ---- Kernel 3: Batched Thomas with scratch buffer ----
    _KERNEL_THOMAS_V2 = RawKernel(
        r'''
    extern "C" __global__
    void thomas_batched_v2(
        const double* __restrict__ a_in,
        const double* __restrict__ b_in,
        const double* __restrict__ c_in,
        const double* __restrict__ d_in,
        double* __restrict__ x_out,
        double* __restrict__ scratch,
        int N_states, int Nz)
    {
        int s = blockDim.x * blockIdx.x + threadIdx.x;
        if (s >= N_states) return;

        int base = s * Nz;

        double denom = b_in[base];
        if (fabs(denom) < 1e-30) denom = (denom >= 0) ? 1e-30 : -1e-30;
        scratch[base] = c_in[base] / denom;
        x_out[base]   = d_in[base] / denom;

        for (int i = 1; i < Nz; i++) {
            double a_i = a_in[base + i];
            denom = b_in[base + i] - a_i * scratch[base + i - 1];
            if (fabs(denom) < 1e-30) denom = (denom >= 0) ? 1e-30 : -1e-30;
            scratch[base + i] = c_in[base + i] / denom;
            x_out[base + i]  = (d_in[base + i] - a_i * x_out[base + i - 1]) / denom;
        }

        for (int i = Nz - 2; i >= 0; i--) {
            x_out[base + i] -= scratch[base + i] * x_out[base + i + 1];
        }
    }
    ''',
        "thomas_batched_v2",
    )

    # ---- Kernel 4: Compute leverage from joint density (Gyoengy projection) ----
    _KERNEL_LEVERAGE = RawKernel(
        r'''
    extern "C" __global__
    void compute_leverage(
        const double* __restrict__ u_joint,
        const double* __restrict__ v_states,
        const double* __restrict__ sigma_LV,
        double* __restrict__ L_out,
        double* __restrict__ E_v_out,
        double* __restrict__ p_z_out,
        int N_states, int Nz,
        double eps, double v_mean)
    {
        int j = blockDim.x * blockIdx.x + threadIdx.x;
        if (j >= Nz) return;

        double pz = 0.0;
        double mz = 0.0;
        for (int i = 0; i < N_states; i++) {
            double u = u_joint[i * Nz + j];
            pz += u;
            mz += v_states[i] * u;
        }

        double ev = (pz > eps) ? (mz / pz) : v_mean;
        ev = fmax(ev, eps);
        double L = sigma_LV[j] / sqrt(ev);

        p_z_out[j] = pz;
        E_v_out[j] = ev;
        L_out[j] = L;
    }
    ''',
        "compute_leverage",
    )

    # ---- Kernel 5: Clamp to non-negative ----
    _KERNEL_CLAMP_NONNEG = RawKernel(
        r'''
    extern "C" __global__
    void clamp_nonneg(double* __restrict__ arr, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            if (arr[i] < 0.0) arr[i] = 0.0;
        }
    }
    ''',
        "clamp_nonneg",
    )

    # ---- Kernel 6: Single-system Thomas for LV marginal ----
    _KERNEL_THOMAS_SINGLE = RawKernel(
        r'''
    extern "C" __global__
    void thomas_single(
        const double* __restrict__ a_in,
        const double* __restrict__ b_in,
        const double* __restrict__ c_in,
        const double* __restrict__ d_in,
        double* __restrict__ x_out,
        double* __restrict__ scratch,
        int Nz)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) return;

        double denom = b_in[0];
        if (fabs(denom) < 1e-30) denom = 1e-30;
        scratch[0] = c_in[0] / denom;
        x_out[0]   = d_in[0] / denom;

        for (int i = 1; i < Nz; i++) {
            denom = b_in[i] - a_in[i] * scratch[i-1];
            if (fabs(denom) < 1e-30) denom = 1e-30;
            scratch[i] = c_in[i] / denom;
            x_out[i]   = (d_in[i] - a_in[i] * x_out[i-1]) / denom;
        }
        for (int i = Nz - 2; i >= 0; i--) {
            x_out[i] -= scratch[i] * x_out[i + 1];
        }
    }
    ''',
        "thomas_single",
    )


# ============================================================
# Data Structures
# ============================================================
@dataclass
class PillarData:
    tenor_label: str; T: float; dt: float; forward: float; df: float
    z_grid: np.ndarray; sigma_z: np.ndarray; S_grid: np.ndarray; sigma_S: np.ndarray

@dataclass
class HestonParams:
    S0: float; v0: float; kappa: float; theta: float; xi: float; rho: float

@dataclass
class CTMCSpec:
    n_states: int; states: np.ndarray; generator: np.ndarray; pi0: np.ndarray

@dataclass
class CalibConfig:
    Nz: int = 500; z_min: float = -3.0; z_max: float = 3.0
    n_substeps_per_bucket: int = 10
    omega: float = 0.7; smooth_leverage: bool = True; smooth_width: int = 5
    theta_pde: float = 1.0; rannacher_steps: int = 4; eps: float = 1e-12
    backend: str = "cuda"; leverage_cap: float = 10.0; density_threshold_frac: float = 1e-6
    splitting: str = "lie_trotter"; store_leverage_time: bool = True
    expm_substep_threshold: float = 5.0  # sub-step expm when max(lambda)*dt exceeds this

@dataclass
class MarketRND:
    tenor_label: str; T: float; forward: float; S_grid: np.ndarray; q: np.ndarray

@dataclass
class CalibResult:
    z_grid: np.ndarray; pillars: List[PillarData]; leverage: List[np.ndarray]
    densities: List[np.ndarray]; marginals: List[np.ndarray]; E_sig2: List[np.ndarray]
    lv_marginals: List[np.ndarray]; ctmc: CTMCSpec; config: CalibConfig
    leverage_time: List[Optional[np.ndarray]] = field(default_factory=list)
    elapsed_sec: float = 0.0


# ============================================================
# Load inputs
# ============================================================
def load_pillars(npz_files):
    pillars = []
    for fpath in npz_files:
        d = np.load(fpath, allow_pickle=True)
        label = str(d["tenor_months"].item()) + "M"
        pillars.append(PillarData(
            tenor_label=label, T=float(d["T"].item()), dt=float(d["dt"].item()),
            forward=float(d["forward"].item()), df=float(d["df"].item()),
            z_grid=d["z"].astype(np.float64), sigma_z=d["sigma_z"].astype(np.float64),
            S_grid=d["xg"].astype(np.float64), sigma_S=d["sigma_S"].astype(np.float64)))
    pillars.sort(key=lambda p: p.T)
    return pillars

def load_heston(json_file):
    with open(json_file) as f:
        data = json.load(f)
    cp_ = data["calibrated_params"]
    return HestonParams(S0=float(data["S0"]), v0=float(cp_["v0"]),
                        kappa=float(cp_["kappa"]), theta=float(cp_["theta"]),
                        xi=float(cp_["sigma"]), rho=float(cp_["rho"]))

def load_market_rnds(rnd_files):
    rnds = []
    for fpath in rnd_files:
        d = np.load(fpath, allow_pickle=True)
        T = float(d["T"].item())
        basename = os.path.splitext(os.path.basename(fpath))[0]
        rnds.append(MarketRND(tenor_label=basename, T=T,
                              forward=float(d["forward"].item()),
                              S_grid=d["xg"].astype(np.float64),
                              q=d["q"].astype(np.float64)))
    rnds.sort(key=lambda r: r.T)
    return rnds


# ============================================================
# CTMC Construction helpers
# ============================================================
def _choose_t_short(kappa, t_short=None):
    if t_short is not None:
        return float(t_short)
    if kappa > 0.01:
        ts = min(1.0 / (2.0 * kappa), 1.0 / 12.0)
    else:
        ts = 1.0 / 12.0
    return float(max(ts, 7.0 / 365.25))

def _cir_transition_params(kappa, theta, xi, v0, t):
    t = float(t)
    ekt = np.exp(-kappa * t)
    denom = 1.0 - ekt
    if denom < 1e-15:
        denom = kappa * t
    c_t = 4.0 * kappa / (xi * xi * denom)
    d_cir = 4.0 * kappa * theta / (xi * xi)
    nc_cir = c_t * v0 * ekt
    return float(c_t), float(d_cir), float(nc_cir)


def _compute_transition_matrix(generator, dt_sub, expm_threshold=5.0):
    """
    Compute transition matrix Q = expm(generator * dt_sub) with sub-stepping
    when max exit rate * dt_sub is large, to maintain numerical accuracy.
    """
    lam_max = float(np.max(-np.diag(generator)))
    product = lam_max * dt_sub

    if product <= expm_threshold:
        # Direct expm is accurate enough
        n_expm_steps = 1
    else:
        # Sub-step: split into m steps so that lam_max * (dt_sub/m) <= threshold
        n_expm_steps = int(np.ceil(product / expm_threshold))

    if n_expm_steps == 1:
        Q_sub = expm(generator * dt_sub)
    else:
        dt_micro = dt_sub / n_expm_steps
        Q_micro = expm(generator * dt_micro)
        Q_micro = np.maximum(Q_micro, 0.0)
        Q_micro /= Q_micro.sum(axis=1, keepdims=True)
        Q_sub = np.linalg.matrix_power(Q_micro, n_expm_steps)

    Q_sub = np.maximum(Q_sub, 0.0)
    Q_sub /= Q_sub.sum(axis=1, keepdims=True)
    return Q_sub, n_expm_steps


# ============================================================
# CTMC: Uniform v-grid
# ============================================================
def _build_ctmc_uniform_v(kappa, theta, xi, v0, n_states,
                          t_short=None, p_bounds=(0.0001, 0.9999),
                          v_cap_mult_theta=4.0):
    from scipy.stats import gamma as gamma_dist, ncx2

    N = int(n_states)
    if N < 3:
        raise ValueError("n_states must be >= 3")

    alpha_cir = 2.0 * kappa * theta / (xi * xi)
    beta_cir = 2.0 * kappa / (xi * xi)
    rv_stat = gamma_dist(a=alpha_cir, scale=1.0 / beta_cir)

    if alpha_cir < 0.5:
        v_floor = max(0.15 * v0, 0.15 * theta, 1e-6)
    else:
        v_floor = max(float(rv_stat.ppf(0.005)), 0.05 * v0, 1e-6)

    ts = _choose_t_short(kappa, t_short)
    c_t, d_cir, nc_cir = _cir_transition_params(kappa, theta, xi, v0, ts)

    p_lo = min(max(float(p_bounds[0]), 1e-8), 0.49)
    p_hi = max(min(float(p_bounds[1]), 1.0 - 1e-8), 0.51)

    v_lo_stat = float(rv_stat.ppf(p_lo)) if alpha_cir > 0 else v_floor
    v_hi_stat = float(rv_stat.ppf(p_hi)) if alpha_cir > 0 else (10.0 * theta)
    v_lo_tr = float(ncx2.ppf(p_lo, df=d_cir, nc=nc_cir) / c_t)
    v_hi_tr = float(ncx2.ppf(p_hi, df=d_cir, nc=nc_cir) / c_t)

    v_min = max(v_floor, min(v_lo_stat, v_lo_tr))
    v_max_raw = max(v_hi_stat, v_hi_tr, v_min * 1.5)
    v_cap = float(v_cap_mult_theta) * theta
    v_max = min(v_max_raw, v_cap)
    if not (v_max > v_min * 1.0001):
        v_max = v_min * 2.0

    v_states = np.linspace(v_min, v_max, N, dtype=np.float64)
    dv = float(v_states[1] - v_states[0])

    # Build generator with full upwind and 0.5 diffusion factor
    Lambda = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        v_i = float(v_states[i])
        mu_i = kappa * (theta - v_i)
        gam2_i = xi * xi * v_i
        diff = 0.5 * gam2_i / (dv * dv)

        if i == 0:
            up = diff + (mu_i / dv if mu_i >= 0.0 else 0.0)
            up = max(up, 1e-12)
            Lambda[0, 1] = up
            Lambda[0, 0] = -up
        elif i == N - 1:
            dn = diff + ((-mu_i) / dv if mu_i < 0.0 else 0.0)
            dn = max(dn, 1e-12)
            Lambda[N - 1, N - 2] = dn
            Lambda[N - 1, N - 1] = -dn
        else:
            up = diff + (mu_i / dv if mu_i >= 0.0 else 0.0)
            dn = diff + ((-mu_i) / dv if mu_i < 0.0 else 0.0)
            up = max(up, 0.0)
            dn = max(dn, 0.0)
            Lambda[i, i + 1] = up
            Lambda[i, i - 1] = dn
            Lambda[i, i] = -(up + dn)

    # pi0 from CIR transition bins at t_short
    edges = np.empty(N + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:N] = 0.5 * (v_states[:-1] + v_states[1:])
    edges[N] = max(v_states[-1] + 0.5 * dv, 5.0 * v_states[-1], 100.0 * theta)

    cdf_vals = ncx2.cdf(c_t * edges, df=d_cir, nc=nc_cir)
    pi0 = np.maximum(np.diff(cdf_vals), 0.0)
    s = float(pi0.sum())
    if s > 1e-30:
        pi0 /= s
    else:
        dists = np.abs(v_states - v0)
        w = np.exp(-dists / max(0.5 * np.std(v_states), 1e-8))
        pi0 = w / w.sum()

    eff = 1.0 / float(np.sum(pi0 ** 2))
    print(f"  [CTMC uniform_v] N={N}, v=[{v_states[0]:.6g}, {v_states[-1]:.6g}], "
          f"dv={dv:.3g}, ratio={v_states[-1]/v_states[0]:.1f}, eff={eff:.1f}, t_short={ts:.4f}")
    print(f"    alpha={alpha_cir:.3f}, v_floor={v_floor:.6g}, v_cap={v_cap:.6g}, "
          f"max|row_sum|={np.max(np.abs(Lambda.sum(axis=1))):.2e}")

    return CTMCSpec(n_states=N, states=v_states, generator=Lambda, pi0=pi0)


# ============================================================
# CTMC: Hybrid (gamma quantile)
# ============================================================
def _build_ctmc_hybrid(kappa, theta, xi, v0, n_states,
                       t_short=None,
                       frac_center=0.8,
                       low_tail_frac=0.55,
                       p_center_band=(0.03, 0.97),
                       p_tail_lo_band=(1e-6, None),
                       p_tail_hi_band=(None, 0.99995),
                       v_floor_abs=1e-6,
                       v_cap_mult_theta=20.0,
                       v_cap_mult_center=10.0,
                       log_tol=1e-10):
    """
    Improved hybrid CTMC variance grid.

    Construction:
      - low tail   : short-time CIR transition quantiles (geometric in probability)
      - center     : short-time CIR transition quantiles (linear in probability)
      - high tail  : stationary Gamma quantiles (geometric in survival probability)
    """
    from scipy.stats import ncx2, gamma as gamma_dist

    m0 = int(n_states)
    if m0 < 3:
        raise ValueError("n_states must be >= 3")
    if not (0.0 < frac_center < 1.0):
        raise ValueError("frac_center must be in (0,1)")
    if not (0.0 < low_tail_frac < 1.0):
        raise ValueError("low_tail_frac must be in (0,1)")
    if not (kappa > 0.0 and theta > 0.0 and xi > 0.0 and v0 > 0.0):
        raise ValueError("kappa, theta, xi, v0 must all be positive")

    def _interior_linspace(lo, hi, n):
        lo = float(lo); hi = float(hi)
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        if hi <= lo:
            return np.full(n, lo, dtype=np.float64)
        if n == 1:
            return np.array([0.5 * (lo + hi)], dtype=np.float64)
        return np.linspace(lo, hi, n + 2, dtype=np.float64)[1:-1]

    def _interior_geomspace(lo, hi, n):
        lo = float(lo); hi = float(hi)
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        lo = max(lo, 1e-300)
        hi = max(hi, lo * (1.0 + 1e-12))
        if n == 1:
            return np.array([np.sqrt(lo * hi)], dtype=np.float64)
        return np.geomspace(lo, hi, n + 2, dtype=np.float64)[1:-1]

    def _dedup_sorted(x, tol):
        x = np.sort(np.asarray(x, dtype=np.float64))
        if x.size == 0:
            return x
        out = [x[0]]
        for i in range(1, x.size):
            if x[i] - out[-1] > tol:
                out.append(x[i])
        return np.asarray(out, dtype=np.float64)

    def _fill_largest_log_gaps(x, target_n):
        x = list(np.sort(np.asarray(x, dtype=np.float64)))
        if len(x) == 0:
            raise ValueError("Cannot refill an empty state set.")
        while len(x) < target_n:
            if len(x) == 1:
                x.append(x[0] + 1e-8)
                continue
            gaps = np.diff(x)
            i = int(np.argmax(gaps))
            mid = 0.5 * (x[i] + x[i + 1])
            if not (x[i] < mid < x[i + 1]):
                mid = np.nextafter(x[i], x[i + 1])
            x.insert(i + 1, mid)
        return np.asarray(x, dtype=np.float64)

    alpha_cir = 2.0 * kappa * theta / (xi * xi)
    beta_cir = 2.0 * kappa / (xi * xi)

    ts = _choose_t_short(kappa, t_short)
    c_t, d_cir, nc_cir = _cir_transition_params(kappa, theta, xi, v0, ts)
    rv_stat = gamma_dist(a=alpha_cir, scale=1.0 / beta_cir)

    v_floor = max(float(v_floor_abs), 1e-12)

    p_cen_lo, p_cen_hi = float(p_center_band[0]), float(p_center_band[1])
    p_cen_lo = min(max(p_cen_lo, 1e-12), 1.0 - 1e-8)
    p_cen_hi = min(max(p_cen_hi, p_cen_lo + 1e-8), 1.0 - 1e-12)

    p_tail_lo = float(p_tail_lo_band[0]) if p_tail_lo_band[0] is not None else 1e-6
    p_tail_lo = min(max(p_tail_lo, 1e-12), p_cen_lo - 1e-12)

    p_tl_hi = p_tail_lo_band[1]
    p_tl_hi = p_cen_lo if p_tl_hi is None else float(p_tl_hi)
    p_tl_hi = min(max(p_tl_hi, p_tail_lo + 1e-12), p_cen_lo)

    p_th_lo = p_tail_hi_band[0]
    p_th_lo = p_cen_hi if p_th_lo is None else float(p_th_lo)
    p_th_lo = min(max(p_th_lo, p_cen_hi), 1.0 - 1e-10)

    p_tail_hi = p_tail_hi_band[1]
    p_tail_hi = 0.99995 if p_tail_hi is None else float(p_tail_hi)
    p_tail_hi = min(max(p_tail_hi, p_th_lo + 1e-12), 1.0 - 1e-12)

    p_floor_trans = float(ncx2.cdf(c_t * v_floor, df=d_cir, nc=nc_cir))
    p_cen_lo = max(p_cen_lo, min(p_floor_trans + 1e-10, p_cen_hi - 1e-8))
    if p_cen_lo >= p_cen_hi:
        p_cen_lo = max(min(p_cen_hi - 1e-6, 0.10), 1e-10)

    if m0 == 3:
        m_tail_lo, m_center, m_tail_hi = 1, 1, 1
    else:
        m_center = int(round(frac_center * m0))
        m_center = max(3, min(m_center, m0 - 2))
        m_tail = m0 - m_center
        if m_tail == 1:
            m_tail_lo, m_tail_hi = 1, 0
        else:
            m_tail_lo = int(round(low_tail_frac * m_tail))
            m_tail_lo = max(1, min(m_tail_lo, m_tail - 1))
            m_tail_hi = m_tail - m_tail_lo

    p_lo = _interior_geomspace(p_tail_lo, p_tl_hi, m_tail_lo)
    v_lo_cand = np.maximum(ncx2.ppf(p_lo, df=d_cir, nc=nc_cir) / c_t, v_floor)

    p_cen = _interior_linspace(p_cen_lo, p_cen_hi, m_center)
    v_center = np.maximum(ncx2.ppf(p_cen, df=d_cir, nc=nc_cir) / c_t, v_floor)

    surv_lo = max(1.0 - p_th_lo, 1e-12)
    surv_hi = max(1.0 - p_tail_hi, 1e-12)
    surv_hi = min(surv_hi, surv_lo * (1.0 - 1e-12))
    s_hi = _interior_geomspace(surv_lo, surv_hi, m_tail_hi)
    p_hi = 1.0 - s_hi
    v_hi_raw = rv_stat.ppf(p_hi)

    v_cap = max(float(v_cap_mult_theta) * theta,
                float(v_cap_mult_center) * float(np.max(v_center)))
    v_hi_cand = np.minimum(np.maximum(v_hi_raw, v_floor), v_cap)

    all_v = np.concatenate([v_lo_cand, v_center, v_hi_cand]).astype(np.float64)
    all_v = np.maximum(all_v, v_floor)
    all_x = _dedup_sorted(np.log(all_v), log_tol)

    if all_x.size == 0:
        raise ValueError("Failed to build any CTMC variance states.")

    if all_x.size < m0:
        all_x = _fill_largest_log_gaps(all_x, m0)
    elif all_x.size > m0:
        idx = np.unique(np.round(np.linspace(0, all_x.size - 1, m0)).astype(int))
        all_x = all_x[idx]
        if all_x.size < m0:
            all_x = _fill_largest_log_gaps(all_x, m0)

    x_states = np.sort(all_x[:m0])
    for i in range(1, x_states.size):
        if x_states[i] <= x_states[i - 1]:
            x_states[i] = np.nextafter(x_states[i - 1], np.inf)

    v_states = np.exp(x_states)
    N = int(v_states.size)

    # Generator on nonuniform v-grid with full upwind
    Lambda = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        v_i = float(v_states[i])
        mu_i = kappa * (theta - v_i)
        gam2_i = xi * xi * v_i

        if i == 0:
            dv = float(v_states[1] - v_states[0])
            c = 0.5 * gam2_i / (dv * dv)
            if mu_i >= 0.0:
                c += mu_i / dv
            c = max(c, 1e-8)
            Lambda[0, 1] = c
            Lambda[0, 0] = -c

        elif i == N - 1:
            dv = float(v_states[-1] - v_states[-2])
            c = 0.5 * gam2_i / (dv * dv)
            if mu_i < 0.0:
                c += (-mu_i) / dv
            c = max(c, 1e-8)
            Lambda[-1, -2] = c
            Lambda[-1, -1] = -c

        else:
            dv_f = float(v_states[i + 1] - v_states[i])
            dv_b = float(v_states[i] - v_states[i - 1])
            dv_a = 0.5 * (dv_f + dv_b)

            c_up = 0.5 * gam2_i / (dv_f * dv_a)
            c_dn = 0.5 * gam2_i / (dv_b * dv_a)

            if mu_i >= 0.0:
                c_up += mu_i / dv_f
            else:
                c_dn += (-mu_i) / dv_b

            c_up = max(c_up, 0.0)
            c_dn = max(c_dn, 0.0)

            Lambda[i, i + 1] = c_up
            Lambda[i, i - 1] = c_dn
            Lambda[i, i] = -(c_up + c_dn)

    # pi0 from transition bins
    be = np.zeros(N + 1, dtype=np.float64)
    be[0] = 0.0
    be[1:N] = 0.5 * (v_states[:-1] + v_states[1:])
    be[-1] = max(v_cap, 5.0 * v_states[-1], 100.0 * theta)

    pi0 = np.maximum(np.diff(ncx2.cdf(c_t * be, df=d_cir, nc=nc_cir)), 0.0)
    s = float(pi0.sum())
    if s > 1e-30:
        pi0 /= s
    else:
        dists = np.abs(v_states - v0)
        scale = max(0.25 * np.std(v_states), 1e-10)
        w = np.exp(-dists / scale)
        pi0 = w / w.sum()

    eff = 1.0 / float(np.sum(pi0 ** 2))
    first_mass = float(pi0[0])
    low1 = float(pi0[:min(3, N)].sum())

    print(
        f"  [CTMC hybrid] N={N}, "
        f"v=[{v_states[0]:.6g}, {v_states[-1]:.6g}], "
        f"ratio={v_states[-1]/v_states[0]:.1f}, "
        f"eff={eff:.1f}, t_short={ts:.4f}"
    )
    print(
        f"    alpha={alpha_cir:.3f}, floor={v_floor:.3e}, "
        f"first_pi0={first_mass:.3%}, first3_pi0={low1:.3%}, "
        f"max|row_sum|={np.max(np.abs(Lambda.sum(axis=1))):.2e}"
    )

    return CTMCSpec(n_states=N, states=v_states, generator=Lambda, pi0=pi0)


# ============================================================
# CTMC: Generator method  (FIX #4: full upwind at boundaries)
# ============================================================
def _build_ctmc_generator(kappa, theta, xi, v0, n_states):
    from scipy.stats import norm
    v_mean = theta
    v_std = np.sqrt(theta * xi * xi / (2.0 * kappa))

    # Improved grid bounds: use CIR stationary distribution quantiles
    # instead of fragile heuristics
    alpha_cir = 2.0 * kappa * theta / (xi * xi)
    if alpha_cir > 1.0:
        from scipy.stats import gamma as gamma_dist
        beta_cir = 2.0 * kappa / (xi * xi)
        rv_stat = gamma_dist(a=alpha_cir, scale=1.0 / beta_cir)
        v_lo = max(float(rv_stat.ppf(0.001)), 0.05 * v0, 1e-6)
        v_hi = max(float(rv_stat.ppf(0.999)), v0 * 3.0)
    else:
        # For low alpha (near Feller boundary), use conservative bounds
        v_lo = max(min(0.05 * v0, 0.05 * theta), 1e-6)
        v_hi = max(v_mean + 4.0 * v_std, v0 * 3.0, 5.0 * theta)

    ln_lo, ln_hi = np.log(v_lo), np.log(v_hi)
    ln_mid = 0.5 * (ln_lo + ln_hi)
    ln_std = (ln_hi - ln_lo) / 5.0
    qs = np.array([(i + 1) / (n_states + 1) for i in range(n_states)])
    xs = np.clip(ln_mid + ln_std * norm.ppf(qs), ln_lo, ln_hi)
    v_states = np.exp(xs)
    N = n_states

    # FIX #4: Use full upwind at boundaries (matching uniform_v and hybrid)
    Lambda = np.zeros((N, N))
    for i in range(N):
        v_i = v_states[i]
        mu_i = kappa * (theta - v_i)
        gam2_i = xi * xi * v_i

        if i == 0:
            dv = v_states[1] - v_states[0]
            # Full upwind: only add positive drift to upward rate
            c = 0.5 * gam2_i / (dv * dv)
            if mu_i >= 0.0:
                c += mu_i / dv
            c = max(c, 1e-8)
            Lambda[0, 1] = c
            Lambda[0, 0] = -c
        elif i == N - 1:
            dv = v_states[N - 1] - v_states[N - 2]
            # Full upwind: only add negative drift to downward rate
            c = 0.5 * gam2_i / (dv * dv)
            if mu_i < 0.0:
                c += (-mu_i) / dv
            c = max(c, 1e-8)
            Lambda[N - 1, N - 2] = c
            Lambda[N - 1, N - 1] = -c
        else:
            dv_f = v_states[i + 1] - v_states[i]
            dv_b = v_states[i] - v_states[i - 1]
            dv_a = 0.5 * (dv_f + dv_b)
            c_up = 0.5 * gam2_i / (dv_f * dv_a)
            c_dn = 0.5 * gam2_i / (dv_b * dv_a)
            if mu_i >= 0:
                c_up += mu_i / dv_f
            else:
                c_dn += (-mu_i) / dv_b
            c_up = max(c_up, 0.0)
            c_dn = max(c_dn, 0.0)
            Lambda[i, i + 1] = c_up
            Lambda[i, i - 1] = c_dn
            Lambda[i, i] = -(c_up + c_dn)

    dists = np.abs(v_states - v0)
    w = np.exp(-dists / (0.5 * np.std(v_states)))
    pi0 = w / w.sum()

    print(f"  [CTMC generator] N={N}, v=[{v_states[0]:.6g}, {v_states[-1]:.6g}], "
          f"max|row_sum|={np.max(np.abs(Lambda.sum(axis=1))):.2e}")

    return CTMCSpec(n_states=N, states=v_states, generator=Lambda, pi0=pi0)


# ============================================================
# CTMC: Gauss-Hermite (uses transition matrix, not FD generator)
# ============================================================
def _build_ctmc_gauss_hermite(kappa, theta, xi, v0, n_states):
    from numpy.polynomial.hermite_e import hermegauss
    x_nodes, w_nodes = hermegauss(n_states)
    w_nodes /= np.sqrt(2 * np.pi)
    v_inf = xi / np.sqrt(2.0 * kappa)
    sig2_states = theta * np.exp(v_inf * x_nodes - 0.5 * v_inf * v_inf)
    dt_ref = 1.0 / 252.0
    rho_ref = np.exp(-kappa * dt_ref)

    def lb(j, x, nodes):
        val = 1.0
        for k in range(len(nodes)):
            if k != j:
                val *= (x - nodes[k]) / (nodes[j] - nodes[k])
        return val

    n = n_states
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(n):
                y = rho_ref * x_nodes[i] + np.sqrt(1.0 - rho_ref ** 2) * x_nodes[k]
                val += w_nodes[k] * lb(j, y, x_nodes)
            Q[i, j] = val
    Q = np.maximum(Q, 0.0)
    Q /= Q.sum(axis=1, keepdims=True)
    Lambda = (Q - np.eye(n)) / dt_ref
    target_x = np.log(v0 / theta) / max(v_inf, 1e-8) + 0.5 * v_inf
    dists = np.abs(x_nodes - target_x)
    w = np.exp(-0.5 * (dists / 0.5) ** 2)
    pi0 = w / w.sum()
    return CTMCSpec(n_states=n, states=sig2_states, generator=Lambda, pi0=pi0)


# ============================================================
# CTMC: Tavella-Randall sinh grid
# ============================================================
def _build_ctmc_tavella_randall(kappa, theta, xi, v0, n_states,
                                gamma=12.0, Tmax=2.0):
    """
    Tavella-Randall sinh-based variance grid.
    
    Concentrates points around v0 using a sinh transformation,
    with bounds from CIR transition distribution at T=Tmax/2.
    Non-uniform grid generator with full upwind discretization.
    """
    M = int(n_states)
    Tr = Tmax / 2.0
    ekt = np.exp(-kappa * Tr)

    # CIR mean and std at reference time
    mu_v = ekt * v0 + theta * (1.0 - ekt)
    sig_v = np.sqrt(xi**2 / kappa * v0 * (ekt - np.exp(-2*kappa*Tr))
                    + theta * xi**2 / (2*kappa) * (1.0 - ekt)**2)

    # Grid bounds
    v_lo = max(1e-6, mu_v - gamma * sig_v)
    v_hi = mu_v + gamma * sig_v

    # Sinh transformation concentrated around v0
    alpha_b = (v_hi - v_lo) / 5.0
    c1 = np.arcsinh((v_lo - v0) / alpha_b)
    c2 = np.arcsinh((v_hi - v0) / alpha_b)

    u = np.linspace(0, 1, M)
    v_states = v0 + alpha_b * np.sinh(c2 * u + c1 * (1.0 - u))
    v_states = np.maximum(v_states, 1e-6)
    v_states = np.sort(np.unique(v_states))

    # Fill gaps if dedup reduced count
    while len(v_states) < M:
        gaps = np.diff(v_states)
        i = int(np.argmax(gaps))
        v_states = np.sort(np.append(v_states, 0.5*(v_states[i] + v_states[i+1])))
    if len(v_states) > M:
        idx = np.round(np.linspace(0, len(v_states)-1, M)).astype(int)
        v_states = v_states[idx]

    N = len(v_states)

    # Generator on non-uniform grid with full upwind
    Lambda = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        v_i = float(v_states[i])
        mu_i = kappa * (theta - v_i)
        gam2_i = xi * xi * v_i

        if i == 0:
            dv = float(v_states[1] - v_states[0])
            c = 0.5 * gam2_i / (dv*dv)
            if mu_i >= 0.0:
                c += mu_i / dv
            c = max(c, 1e-12)
            Lambda[0, 1] = c
            Lambda[0, 0] = -c
        elif i == N - 1:
            dv = float(v_states[-1] - v_states[-2])
            c = 0.5 * gam2_i / (dv*dv)
            if mu_i < 0.0:
                c += (-mu_i) / dv
            c = max(c, 1e-12)
            Lambda[-1, -2] = c
            Lambda[-1, -1] = -c
        else:
            dv_f = float(v_states[i+1] - v_states[i])
            dv_b = float(v_states[i] - v_states[i-1])
            dv_a = 0.5 * (dv_f + dv_b)
            c_up = 0.5 * gam2_i / (dv_f * dv_a)
            c_dn = 0.5 * gam2_i / (dv_b * dv_a)
            if mu_i >= 0.0:
                c_up += mu_i / dv_f
            else:
                c_dn += (-mu_i) / dv_b
            c_up = max(c_up, 0.0)
            c_dn = max(c_dn, 0.0)
            Lambda[i, i+1] = c_up
            Lambda[i, i-1] = c_dn
            Lambda[i, i] = -(c_up + c_dn)

    # pi0: linear interpolation around v0
    pi0 = np.zeros(N)
    ir = min(max(int(np.searchsorted(v_states, v0)), 1), N-1)
    il = ir - 1
    w = (v_states[ir] - v0) / (v_states[ir] - v_states[il])
    pi0[il] = w
    pi0[ir] = 1.0 - w

    eff = 1.0 / float(np.sum(pi0**2))
    print(f"  [CTMC tavella_randall] N={N}, v=[{v_states[0]:.4e},{v_states[-1]:.4e}], "
          f"E[V0]={np.sum(v_states*pi0):.6f}, eff={eff:.1f}, gamma={gamma:.1f}, "
          f"max|row_sum|={np.max(np.abs(Lambda.sum(axis=1))):.2e}")

    return CTMCSpec(n_states=N, states=v_states, generator=Lambda, pi0=pi0)


def build_ctmc_from_heston(heston, n_states=50, method="uniform_v",
                           frac_center=0.8, t_short=None,
                           gamma=12.0, Tmax=2.0):
    kappa, theta, xi, v0 = heston.kappa, heston.theta, heston.xi, heston.v0
    method = str(method).lower()
    if method == "tavella_randall":
        return _build_ctmc_tavella_randall(kappa, theta, xi, v0, n_states,
                                           gamma=gamma, Tmax=Tmax)
    if method == "uniform_v":
        return _build_ctmc_uniform_v(kappa, theta, xi, v0, n_states, t_short=t_short)
    if method == "gamma_quantile":
        return _build_ctmc_hybrid(kappa, theta, xi, v0, n_states,
                                  t_short=t_short, frac_center=frac_center)
    if method == "generator":
        return _build_ctmc_generator(kappa, theta, xi, v0, n_states)
    if method == "gauss_hermite":
        return _build_ctmc_gauss_hermite(kappa, theta, xi, v0, n_states)
    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Dirac delta
# ============================================================
def _dirac_delta_z(z_grid, dz, z0=0.0):
    idx = int(np.argmin(np.abs(z_grid - z0)))
    u0 = np.zeros_like(z_grid)
    u0[idx] = 1.0 / dz
    return u0


# ============================================================
# CPU fallback PDE helpers
# ============================================================
def _thomas_solve_cpu(a, b, c, d):
    N = len(d)
    cp_ = np.empty(N); dp = np.empty(N)
    cp_[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, N):
        dn = b[i] - a[i] * cp_[i - 1]
        cp_[i] = c[i] / dn
        dp[i] = (d[i] - a[i] * dp[i - 1]) / dn
    x = np.empty(N)
    x[N - 1] = dp[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = dp[i] - cp_[i] * x[i + 1]
    return x


def _build_forward_op_cpu(local_var, dz):
    """
    Forward FP for z = log(e^z):
      dp/dt = (1/2) d^2(a*p)/dz^2 + (1/2) d(a*p)/dz
    with drift-inclusive boundaries.
    """
    Nz = len(local_var)
    dz2 = dz * dz
    a = local_var

    sub = np.zeros(Nz)
    diag = np.zeros(Nz)
    sup = np.zeros(Nz)

    # Interior
    sub[1:-1] = 0.5 * a[:-2] / dz2 - a[:-2] / (4.0 * dz)
    diag[1:-1] = -a[1:-1] / dz2
    sup[1:-1] = 0.5 * a[2:] / dz2 + a[2:] / (4.0 * dz)

    # Boundaries include drift terms for consistency
    # Left boundary
    diag[0] = -a[0] / dz2 - a[0] / (4.0 * dz)
    sup[0] = a[1] / dz2 + a[1] / (4.0 * dz)
    # Right boundary
    sub[-1] = a[-2] / dz2 - a[-2] / (4.0 * dz)
    diag[-1] = -a[-1] / dz2 + a[-1] / (4.0 * dz)

    return sub, diag, sup


def _build_rhs_cpu(phi, sub, diag, sup, dt, theta_pde):
    N = len(phi)
    expl = (1.0 - theta_pde) * dt
    rhs = np.empty(N)
    rhs[0] = phi[0] * (1.0 + expl * diag[0])
    if N > 1:
        rhs[0] += phi[1] * expl * sup[0]
    rhs[1:-1] = (phi[:-2] * expl * sub[1:-1]
                 + phi[1:-1] * (1.0 + expl * diag[1:-1])
                 + phi[2:] * expl * sup[1:-1])
    rhs[-1] = phi[-1] * (1.0 + expl * diag[-1])
    if N > 1:
        rhs[-1] += phi[-2] * expl * sub[-1]
    return rhs


def _advance_1d_pde_cpu(phi, local_var, dz, dt, theta_pde, n_rannacher=0):
    sub, diag, sup = _build_forward_op_cpu(local_var, dz)
    if n_rannacher > 0:
        dt_r = dt / n_rannacher
        for _ in range(n_rannacher):
            rhs = _build_rhs_cpu(phi, sub, diag, sup, dt_r, 1.0)
            phi = _thomas_solve_cpu(-dt_r * sub, 1.0 - dt_r * diag, -dt_r * sup, rhs)
    else:
        rhs = _build_rhs_cpu(phi, sub, diag, sup, dt, theta_pde)
        impl = theta_pde * dt
        phi = _thomas_solve_cpu(-impl * sub, 1.0 - impl * diag, -impl * sup, rhs)
    return phi


def _gaussian_smooth(arr, half_width=5):
    if half_width <= 0:
        return arr
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / (half_width / 2.5)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(arr, half_width, mode="edge"), kernel, mode="valid")


# ============================================================
# GPU PDE solver class
# ============================================================
class GPUSolver:
    def __init__(self, N_states, Nz, config):
        assert _HAS_CUDA, "GPU not available"
        self.N_states = int(N_states)
        self.Nz = int(Nz)
        self.config = config
        self.total = self.N_states * self.Nz

        self.block_total = 256
        self.grid_total = (self.total + self.block_total - 1) // self.block_total
        self.block_nz = 256
        self.grid_nz = (self.Nz + self.block_nz - 1) // self.block_nz
        self.block_states = min(256, self.N_states)
        self.grid_states = (self.N_states + self.block_states - 1) // self.block_states

        self.d_sub = cp.zeros((self.N_states, self.Nz), dtype=cp.float64)
        self.d_diag = cp.zeros((self.N_states, self.Nz), dtype=cp.float64)
        self.d_sup = cp.zeros((self.N_states, self.Nz), dtype=cp.float64)
        self.d_rhs = cp.zeros((self.N_states, self.Nz), dtype=cp.float64)
        self.d_scratch = cp.zeros((self.N_states, self.Nz), dtype=cp.float64)

        self.d_L = cp.zeros(self.Nz, dtype=cp.float64)
        self.d_Ev = cp.zeros(self.Nz, dtype=cp.float64)
        self.d_pz = cp.zeros(self.Nz, dtype=cp.float64)

        self.d_lv_sub = cp.zeros(self.Nz, dtype=cp.float64)
        self.d_lv_diag = cp.zeros(self.Nz, dtype=cp.float64)
        self.d_lv_sup = cp.zeros(self.Nz, dtype=cp.float64)
        self.d_lv_rhs = cp.zeros(self.Nz, dtype=cp.float64)
        self.d_lv_scratch = cp.zeros(self.Nz, dtype=cp.float64)

        if config.smooth_leverage and config.smooth_width > 0:
            hw = int(config.smooth_width)
            x_k = np.arange(-hw, hw + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * (x_k / (hw / 2.5)) ** 2)
            kernel /= kernel.sum()
            self.d_smooth_kernel = cp.asarray(kernel)
            self.smooth_hw = hw
        else:
            self.d_smooth_kernel = None
            self.smooth_hw = 0

    def compute_leverage(self, d_u_joint, d_sigma_LV, d_v_states, L_prev_gpu, v_mean):
        cfg = self.config
        Nz = self.Nz

        _KERNEL_LEVERAGE(
            (self.grid_nz,), (self.block_nz,),
            (d_u_joint, d_v_states, d_sigma_LV,
             self.d_L, self.d_Ev, self.d_pz,
             np.int32(self.N_states), np.int32(Nz),
             np.float64(cfg.eps), np.float64(v_mean)))

        dp = float(self.d_pz.max())
        thr = cfg.density_threshold_frac * dp
        sig_mask = self.d_pz > thr
        sig_count = int(sig_mask.sum())

        if sig_count > 10:
            idx = cp.where(sig_mask)[0]
            il = int(idx[0]); ih = int(idx[-1])
            if il > 0:
                self.d_L[:il] = self.d_L[il]
            if ih < Nz - 1:
                self.d_L[ih + 1:] = self.d_L[ih]

        cp.clip(self.d_L, 1.0 / cfg.leverage_cap, cfg.leverage_cap, out=self.d_L)

        if self.d_smooth_kernel is not None:
            hw = self.smooth_hw
            padded = cp.pad(self.d_L, hw, mode="edge")
            smoothed = cp.convolve(padded, self.d_smooth_kernel, mode="valid")
            cp.copyto(self.d_L, smoothed[:Nz])

        if L_prev_gpu is not None:
            self.d_L[:] = (1.0 - cfg.omega) * L_prev_gpu + cfg.omega * self.d_L

        return self.d_L, self.d_Ev, self.d_pz

    def advance_pde_batched(self, d_u_joint, d_L2, d_v_states, dz, dt,
                            theta_pde, use_rannacher, rannacher_steps):
        Nz = self.Nz; N_states = self.N_states

        _KERNEL_BUILD_FWD_OP(
            (self.grid_total,), (self.block_total,),
            (d_L2, d_v_states, self.d_sub, self.d_diag, self.d_sup,
             np.int32(N_states), np.int32(Nz), np.float64(dz)))

        if use_rannacher and rannacher_steps > 0:
            dt_r = dt / rannacher_steps
            for _ in range(rannacher_steps):
                _KERNEL_BUILD_RHS(
                    (self.grid_total,), (self.block_total,),
                    (d_u_joint, self.d_sub, self.d_diag, self.d_sup, self.d_rhs,
                     np.int32(N_states), np.int32(Nz), np.float64(0.0)))
                d_a = -dt_r * self.d_sub
                d_b = 1.0 - dt_r * self.d_diag
                d_c = -dt_r * self.d_sup
                _KERNEL_THOMAS_V2(
                    (self.grid_states,), (self.block_states,),
                    (d_a, d_b, d_c, self.d_rhs, d_u_joint, self.d_scratch,
                     np.int32(N_states), np.int32(Nz)))
                _KERNEL_CLAMP_NONNEG(
                    (self.grid_total,), (self.block_total,),
                    (d_u_joint, np.int32(self.total)))
        else:
            expl_factor = (1.0 - theta_pde) * dt
            impl = theta_pde * dt
            _KERNEL_BUILD_RHS(
                (self.grid_total,), (self.block_total,),
                (d_u_joint, self.d_sub, self.d_diag, self.d_sup, self.d_rhs,
                 np.int32(N_states), np.int32(Nz), np.float64(expl_factor)))
            d_a = -impl * self.d_sub
            d_b = 1.0 - impl * self.d_diag
            d_c = -impl * self.d_sup
            _KERNEL_THOMAS_V2(
                (self.grid_states,), (self.block_states,),
                (d_a, d_b, d_c, self.d_rhs, d_u_joint, self.d_scratch,
                 np.int32(N_states), np.int32(Nz)))
            _KERNEL_CLAMP_NONNEG(
                (self.grid_total,), (self.block_total,),
                (d_u_joint, np.int32(self.total)))

        return d_u_joint

    def advance_lv_marginal(self, d_lv, d_sigma_LV2, dz, dt, theta_pde,
                            use_rannacher, rannacher_steps):
        """
        FIX #5: Boundary operator now matches CPU _build_forward_op_cpu exactly.
        Previously had spurious 0.5 factors on boundary diffusion and drift terms.
        """
        Nz = self.Nz
        dz2 = dz * dz
        a = d_sigma_LV2

        self.d_lv_sub.fill(0.0); self.d_lv_diag.fill(0.0); self.d_lv_sup.fill(0.0)

        # Interior with drift — matches CPU exactly
        self.d_lv_sub[1:-1] = 0.5 * a[:-2] / dz2 - a[:-2] / (4.0 * dz)
        self.d_lv_diag[1:-1] = -a[1:-1] / dz2
        self.d_lv_sup[1:-1] = 0.5 * a[2:] / dz2 + a[2:] / (4.0 * dz)

        # FIX #5: Boundaries now match CPU _build_forward_op_cpu exactly
        # Left boundary: no spurious 0.5 on diffusion
        self.d_lv_diag[0] = -a[0] / dz2 - a[0] / (4.0 * dz)
        self.d_lv_sup[0] = a[1] / dz2 + a[1] / (4.0 * dz)
        # Right boundary: no spurious 0.5 on diffusion or drift
        self.d_lv_sub[-1] = a[-2] / dz2 - a[-2] / (4.0 * dz)
        self.d_lv_diag[-1] = -a[-1] / dz2 + a[-1] / (4.0 * dz)

        if use_rannacher and rannacher_steps > 0:
            dt_r = dt / rannacher_steps
            for _ in range(rannacher_steps):
                self.d_lv_rhs[:] = d_lv
                a_imp = -dt_r * self.d_lv_sub
                b_imp = 1.0 - dt_r * self.d_lv_diag
                c_imp = -dt_r * self.d_lv_sup
                _KERNEL_THOMAS_SINGLE(
                    (1,), (1,),
                    (a_imp, b_imp, c_imp, self.d_lv_rhs,
                     d_lv, self.d_lv_scratch, np.int32(Nz)))
                cp.maximum(d_lv, 0.0, out=d_lv)
        else:
            expl = (1.0 - theta_pde) * dt
            impl = theta_pde * dt
            self.d_lv_rhs[0] = (d_lv[0] * (1.0 + expl * self.d_lv_diag[0])
                                + d_lv[1] * expl * self.d_lv_sup[0])
            self.d_lv_rhs[1:-1] = (d_lv[:-2] * expl * self.d_lv_sub[1:-1]
                                   + d_lv[1:-1] * (1.0 + expl * self.d_lv_diag[1:-1])
                                   + d_lv[2:] * expl * self.d_lv_sup[1:-1])
            self.d_lv_rhs[-1] = (d_lv[-2] * expl * self.d_lv_sub[-1]
                                 + d_lv[-1] * (1.0 + expl * self.d_lv_diag[-1]))
            a_imp = -impl * self.d_lv_sub
            b_imp = 1.0 - impl * self.d_lv_diag
            c_imp = -impl * self.d_lv_sup
            _KERNEL_THOMAS_SINGLE(
                (1,), (1,),
                (a_imp, b_imp, c_imp, self.d_lv_rhs,
                 d_lv, self.d_lv_scratch, np.int32(Nz)))
            cp.maximum(d_lv, 0.0, out=d_lv)

        return d_lv


# ============================================================
# Calibration entry point
# ============================================================
def calibrate_lsv_ctmc_forward_induction(pillars, ctmc, config):
    use_gpu = _HAS_CUDA and config.backend != "cpu"
    if use_gpu:
        return _calibrate_gpu(pillars, ctmc, config)
    return _calibrate_cpu(pillars, ctmc, config)


def _calibrate_gpu(pillars, ctmc, config):
    t_start = time.time()
    N_states = ctmc.n_states; Nz = config.Nz
    splitting = config.splitting.lower()
    z_grid = np.linspace(config.z_min, config.z_max, Nz)
    dz = float(z_grid[1] - z_grid[0])

    solver = GPUSolver(N_states, Nz, config)
    d_v_states = cp.asarray(ctmc.states, dtype=cp.float64)
    v_mean = float(np.mean(ctmc.states))

    u0_z = _dirac_delta_z(z_grid, dz)
    u_joint_np = np.zeros((N_states, Nz), dtype=np.float64)
    for i in range(N_states):
        u_joint_np[i, :] = ctmc.pi0[i] * u0_z
    d_u_joint = cp.asarray(u_joint_np)
    d_lv_phi = cp.asarray(u0_z.copy())

    lev_all, den_all, marg_all, esig_all, lv_marg_all = [], [], [], [], []
    lev_time_all = []

    print(f"LSV-CTMC Forward-Induction [GPU]")
    print(f"  states={N_states}, Nz={Nz}, z=[{config.z_min:.2f},{config.z_max:.2f}]")
    print(f"  Splitting: {splitting.upper()}, L_cap={config.leverage_cap:.1f}, omega={config.omega:.2f}")
    print(f"  Pillars: {[p.tenor_label for p in pillars]}")
    print("-" * 70)

    for k, pillar in enumerate(pillars):
        t0 = time.time()
        dt_bucket = float(pillar.dt)
        n_sub = int(config.n_substeps_per_bucket)
        dt_sub = dt_bucket / n_sub

        sigma_LV_np = np.maximum(np.interp(z_grid, pillar.z_grid, pillar.sigma_z), 1e-6)
        d_sigma_LV = cp.asarray(sigma_LV_np)
        d_sigma_LV2 = d_sigma_LV * d_sigma_LV

        # FIX #7: Use sub-stepped expm for numerical stability
        Q_sub, n_expm_steps = _compute_transition_matrix(
            ctmc.generator, dt_sub, config.expm_substep_threshold)
        d_QT = cp.asarray(Q_sub.T, dtype=cp.float64)
        if n_expm_steps > 1:
            print(f"    [{pillar.tenor_label}] expm sub-stepped: {n_expm_steps} micro-steps")

        L_time_np = np.zeros((n_sub + 1, Nz)) if config.store_leverage_time else None

        # FIX #6: Reset leverage relaxation at bucket boundaries
        # Each bucket has its own sigma_LV, so stale leverage from previous bucket
        # would slow convergence. Start fresh each bucket.
        d_L_prev_time = None

        for s in range(n_sub):
            d_L_s, d_Ev_s, d_pz_s = solver.compute_leverage(
                d_u_joint, d_sigma_LV, d_v_states, d_L_prev_time, v_mean)
            if L_time_np is not None:
                L_time_np[s, :] = cp.asnumpy(d_L_s)

            d_L2 = d_L_s * d_L_s
            use_rann = (s == 0)

            if splitting == "strang":
                d_u_joint = solver.advance_pde_batched(
                    d_u_joint, d_L2, d_v_states, dz, dt_sub / 2.0,
                    config.theta_pde, use_rann, config.rannacher_steps)
                d_u_joint = d_QT @ d_u_joint
                d_u_joint = solver.advance_pde_batched(
                    d_u_joint, d_L2, d_v_states, dz, dt_sub / 2.0,
                    config.theta_pde, False, 0)
            else:
                d_u_joint = d_QT @ d_u_joint
                d_u_joint = solver.advance_pde_batched(
                    d_u_joint, d_L2, d_v_states, dz, dt_sub,
                    config.theta_pde, use_rann, config.rannacher_steps)

            nr_lv = config.rannacher_steps if (s == 0) else 0
            d_lv_phi = solver.advance_lv_marginal(
                d_lv_phi, d_sigma_LV2, dz, dt_sub,
                config.theta_pde, nr_lv > 0, nr_lv)

            d_L_prev_time = d_L_s.copy()

        d_L_end, d_Ev_end, _ = solver.compute_leverage(
            d_u_joint, d_sigma_LV, d_v_states, d_L_prev_time, v_mean)
        if L_time_np is not None:
            L_time_np[n_sub, :] = cp.asnumpy(d_L_end)

        cp.cuda.Stream.null.synchronize()
        L_end_np = cp.asnumpy(d_L_end)
        u_joint_np = cp.asnumpy(d_u_joint)
        marg_np = u_joint_np.sum(axis=0)
        Ev_np = cp.asnumpy(d_Ev_end)
        lv_np = cp.asnumpy(d_lv_phi)

        mass = float(np.sum(marg_np) * dz)
        lv_mass = float(np.sum(lv_np) * dz)
        print(f"  [{pillar.tenor_label}] T={pillar.T:.4f}, dt={dt_bucket:.4f}, "
              f"mass={mass:.6f}, lv_mass={lv_mass:.6f}, "
              f"L=[{L_end_np.min():.4f},{L_end_np.max():.4f}], {time.time()-t0:.2f}s")

        lev_all.append(L_end_np.copy())
        den_all.append(u_joint_np.copy())
        marg_all.append(marg_np.copy())
        lv_marg_all.append(lv_np.copy())
        esig_all.append(Ev_np.copy())
        lev_time_all.append(L_time_np.copy() if L_time_np is not None else None)

    elapsed = time.time() - t_start
    print("-" * 70)
    print(f"Calibration completed in {elapsed:.2f}s [GPU {splitting.upper()}]")

    return CalibResult(z_grid=z_grid, pillars=pillars, leverage=lev_all,
                       densities=den_all, marginals=marg_all, E_sig2=esig_all,
                       lv_marginals=lv_marg_all, ctmc=ctmc, config=config,
                       leverage_time=lev_time_all, elapsed_sec=elapsed)


def _compute_leverage_from_joint_cpu(u_joint, sigma_LV, v_states, L_prev_time, config):
    eps = config.eps
    p_z = np.sum(u_joint, axis=0)
    m_z = np.sum(v_states[:, None] * u_joint, axis=0)
    E_v = np.where(p_z > eps, m_z / p_z, float(np.mean(v_states)))
    L_raw = sigma_LV / np.sqrt(np.maximum(E_v, eps))

    dp = float(p_z.max())
    sig = p_z > config.density_threshold_frac * dp
    if np.sum(sig) > 10:
        si = np.where(sig)[0]
        il, ih = int(si[0]), int(si[-1])
        Lu = L_raw.copy()
        if il > 0: Lu[:il] = Lu[il]
        if ih < len(Lu) - 1: Lu[ih + 1:] = Lu[ih]
    else:
        Lu = L_raw.copy()

    np.clip(Lu, 1.0 / config.leverage_cap, config.leverage_cap, out=Lu)
    if config.smooth_leverage and config.smooth_width > 0:
        Lu = _gaussian_smooth(Lu, config.smooth_width)
    if L_prev_time is not None:
        return (1.0 - config.omega) * L_prev_time + config.omega * Lu, E_v, p_z
    return Lu, E_v, p_z


def _calibrate_cpu(pillars, ctmc, config):
    t_start = time.time()
    N_states = ctmc.n_states; Nz = config.Nz
    splitting = config.splitting.lower()
    z_grid = np.linspace(config.z_min, config.z_max, Nz)
    dz = float(z_grid[1] - z_grid[0])

    u0_z = _dirac_delta_z(z_grid, dz)
    u_joint = np.zeros((N_states, Nz))
    for i in range(N_states):
        u_joint[i, :] = ctmc.pi0[i] * u0_z
    lv_phi = u0_z.copy()

    lev_all, den_all, marg_all, esig_all, lv_marg_all = [], [], [], [], []
    lev_time_all = []

    print(f"LSV-CTMC Forward-Induction [CPU]")
    print(f"  states={N_states}, Nz={Nz}, Splitting: {splitting.upper()}")
    print(f"  Pillars: {[p.tenor_label for p in pillars]}")
    print("-" * 70)

    for k, pillar in enumerate(pillars):
        dt_bucket = float(pillar.dt)
        n_sub = int(config.n_substeps_per_bucket)
        dt_sub = dt_bucket / n_sub

        sigma_LV = np.maximum(np.interp(z_grid, pillar.z_grid, pillar.sigma_z), 1e-6)
        sigma_LV2 = sigma_LV * sigma_LV

        # FIX #7: Use sub-stepped expm for numerical stability
        Q_sub, n_expm_steps = _compute_transition_matrix(
            ctmc.generator, dt_sub, config.expm_substep_threshold)
        QT = Q_sub.T.copy()
        if n_expm_steps > 1:
            print(f"    [{pillar.tenor_label}] expm sub-stepped: {n_expm_steps} micro-steps")

        L_time = np.zeros((n_sub + 1, Nz)) if config.store_leverage_time else None
        uw = u_joint.copy()
        lv_curr = lv_phi.copy()

        # FIX #6: Reset leverage relaxation at bucket boundaries
        L_prev_time = None

        for s in range(n_sub):
            L_s, E_v_s, p_z_s = _compute_leverage_from_joint_cpu(
                uw, sigma_LV, ctmc.states, L_prev_time, config)
            if L_time is not None:
                L_time[s, :] = L_s

            use_rann = (s == 0)

            if splitting == "strang":
                dt_half = dt_sub / 2.0
                for i in range(N_states):
                    lv_i = (L_s * L_s) * ctmc.states[i]
                    nr = config.rannacher_steps if use_rann else 0
                    uw[i, :] = _advance_1d_pde_cpu(uw[i, :], lv_i, dz, dt_half, config.theta_pde, n_rannacher=nr)
                    np.maximum(uw[i, :], 0.0, out=uw[i, :])
                uw = QT @ uw
                for i in range(N_states):
                    lv_i = (L_s * L_s) * ctmc.states[i]
                    uw[i, :] = _advance_1d_pde_cpu(uw[i, :], lv_i, dz, dt_half, config.theta_pde, n_rannacher=0)
                    np.maximum(uw[i, :], 0.0, out=uw[i, :])
            else:
                uw = QT @ uw
                for i in range(N_states):
                    lv_i = (L_s * L_s) * ctmc.states[i]
                    nr = config.rannacher_steps if use_rann else 0
                    uw[i, :] = _advance_1d_pde_cpu(uw[i, :], lv_i, dz, dt_sub, config.theta_pde, n_rannacher=nr)
                    np.maximum(uw[i, :], 0.0, out=uw[i, :])

            nr_lv = config.rannacher_steps if (s == 0) else 0
            lv_curr = _advance_1d_pde_cpu(lv_curr, sigma_LV2, dz, dt_sub, config.theta_pde, n_rannacher=nr_lv)
            np.maximum(lv_curr, 0.0, out=lv_curr)
            L_prev_time = L_s

        L_end, E_v_end, _ = _compute_leverage_from_joint_cpu(
            uw, sigma_LV, ctmc.states, L_prev_time, config)
        if L_time is not None:
            L_time[n_sub, :] = L_end

        marg = np.sum(uw, axis=0)
        mass = float(np.sum(marg) * dz)
        lv_mass = float(np.sum(lv_curr) * dz)
        print(f"  [{pillar.tenor_label}] T={pillar.T:.4f}, mass={mass:.6f}, "
              f"lv_mass={lv_mass:.6f}, L=[{L_end.min():.4f},{L_end.max():.4f}]")

        lev_all.append(L_end.copy())
        den_all.append(uw.copy())
        marg_all.append(marg.copy())
        lv_marg_all.append(lv_curr.copy())
        esig_all.append(E_v_end.copy())
        lev_time_all.append(L_time.copy() if L_time is not None else None)

        u_joint = uw; lv_phi = lv_curr

    elapsed = time.time() - t_start
    print("-" * 70)
    print(f"CPU calibration completed in {elapsed:.2f}s")

    return CalibResult(z_grid=z_grid, pillars=pillars, leverage=lev_all,
                       densities=den_all, marginals=marg_all, E_sig2=esig_all,
                       lv_marginals=lv_marg_all, ctmc=ctmc, config=config,
                       leverage_time=lev_time_all, elapsed_sec=elapsed)


# ============================================================
# Utility / Save / Plot
# ============================================================
def z_density_to_S_density(z_grid, phi_z, F, S_grid):
    return np.interp(np.log(S_grid / F), z_grid, phi_z, left=0.0, right=0.0) / np.maximum(S_grid, 1e-12)


def save_lsv_ctmc_result(result, heston, filepath):
    K = len(result.pillars); Nz = len(result.z_grid)
    dz = float(result.z_grid[1] - result.z_grid[0])
    u0 = _dirac_delta_z(result.z_grid, dz)

    sd = {
        "z_grid": result.z_grid, "dz": np.float64(dz),
        "n_buckets": np.int32(K),
        "n_substeps": np.int32(result.config.n_substeps_per_bucket),
        "pillar_labels": np.array([p.tenor_label for p in result.pillars]),
        "pillar_T": np.array([p.T for p in result.pillars]),
        "pillar_dt": np.array([p.dt for p in result.pillars]),
        "pillar_forward": np.array([p.forward for p in result.pillars]),
        "pillar_df": np.array([p.df for p in result.pillars]),
        "ctmc_n_states": np.int32(result.ctmc.n_states),
        "ctmc_states": result.ctmc.states,
        "ctmc_generator": result.ctmc.generator,
        "ctmc_pi0": result.ctmc.pi0,
        "elapsed_sec": np.float64(result.elapsed_sec),
        "splitting": np.array([result.config.splitting], dtype="U20"),
        "theta_pde": np.float64(result.config.theta_pde),
        "rannacher_steps": np.int32(result.config.rannacher_steps),
    }
    for attr in ["S0", "v0", "kappa", "theta", "xi", "rho"]:
        sd[f"heston_{attr}"] = np.float64(getattr(heston, attr))

    p0 = np.zeros(result.ctmc.n_states * Nz)
    for i in range(result.ctmc.n_states):
        p0[i * Nz:(i + 1) * Nz] = result.ctmc.pi0[i] * u0
    sd["initial_density_vec"] = p0

    has_lt = False
    for k, p in enumerate(result.pillars):
        sd[f"leverage_{k}"] = result.leverage[k]
        sd[f"sigma_lv_{k}"] = np.interp(result.z_grid, p.z_grid, p.sigma_z)
        sd[f"density_{k}"] = result.densities[k]
        sd[f"marginal_{k}"] = result.marginals[k]
        sd[f"lv_marginal_{k}"] = result.lv_marginals[k]
        sd[f"E_sig2_{k}"] = result.E_sig2[k]
        if result.leverage_time and result.leverage_time[k] is not None:
            sd[f"leverage_time_{k}"] = result.leverage_time[k]
            has_lt = True
    sd["has_leverage_time"] = np.int32(1 if has_lt else 0)

    np.savez_compressed(filepath, **sd)
    fsize = os.path.getsize(filepath) / (1024 * 1024)
    print(f"LSV-CTMC result saved -> {filepath}  ({fsize:.2f} MB)")


def plot_results(result, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = result.z_grid
    X = np.exp(z)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for k, p in enumerate(result.pillars):
        axes[0, 0].plot(X, result.leverage[k], label=p.tenor_label, lw=1.2)
    axes[0, 0].set_xlabel("e^z (moneyness)")
    axes[0, 0].set_ylabel("L(e^z)")
    axes[0, 0].set_title(f"Leverage ({result.config.splitting.upper()} FI)")
    axes[0, 0].legend(); axes[0, 0].set_xlim(0.3, 2.0); axes[0, 0].grid(True, alpha=0.3)

    for k, p in enumerate(result.pillars):
        axes[0, 1].plot(X, result.marginals[k], "-", color=f"C{k}",
                        label=f"{p.tenor_label} model", lw=1.2)
        axes[0, 1].plot(X, result.lv_marginals[k], "--", color=f"C{k}",
                        label=f"{p.tenor_label} LV", lw=0.8, alpha=0.6)
    axes[0, 1].set_xlabel("e^z"); axes[0, 1].set_ylabel("p(z)")
    axes[0, 1].set_title("Model vs LV marginals")
    axes[0, 1].legend(fontsize=6); axes[0, 1].set_xlim(0.3, 2.0)
    axes[0, 1].grid(True, alpha=0.3)

    for k, p in enumerate(result.pillars):
        axes[0, 2].plot(X, np.sqrt(result.E_sig2[k]), label=p.tenor_label, lw=1.2)
    axes[0, 2].set_xlabel("e^z"); axes[0, 2].set_ylabel("sqrt(E[v|z])")
    axes[0, 2].set_title("Conditional Vol")
    axes[0, 2].legend(); axes[0, 2].set_xlim(0.3, 2.0); axes[0, 2].grid(True, alpha=0.3)

    for k, p in enumerate(result.pillars):
        st = np.interp(z, p.z_grid, p.sigma_z)
        sm = result.leverage[k] * np.sqrt(result.E_sig2[k])
        axes[1, 0].plot(X, st, "--", color=f"C{k}", alpha=0.5, lw=0.8)
        axes[1, 0].plot(X, sm, "-", color=f"C{k}", label=p.tenor_label, lw=1.2)
    axes[1, 0].set_xlabel("e^z"); axes[1, 0].set_ylabel("LV")
    axes[1, 0].set_title("Target (--) vs Model (-) LV")
    axes[1, 0].legend(); axes[1, 0].set_xlim(0.3, 2.0); axes[1, 0].set_ylim(0, 5)
    axes[1, 0].grid(True, alpha=0.3)

    kl = len(result.pillars) - 1
    n_states = result.ctmc.n_states
    n_show = min(n_states, 12)
    show_idx = np.round(np.linspace(0, n_states - 1, n_show)).astype(int)
    cmap = plt.cm.plasma
    for ii, si in enumerate(show_idx):
        color = cmap(ii / max(n_show - 1, 1))
        vol_label = f"v={result.ctmc.states[si]:.4f} ({np.sqrt(result.ctmc.states[si])*100:.0f}%)"
        axes[1, 1].plot(X, result.densities[kl][si, :], lw=1.0, color=color, label=vol_label)
    axes[1, 1].set_xlabel("e^z")
    axes[1, 1].set_title(f"Regime densities ({result.pillars[kl].tenor_label})")
    axes[1, 1].set_xlim(0.3, 2.0); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=5, ncol=2)

    for k, p in enumerate(result.pillars):
        diff = result.marginals[k] - result.lv_marginals[k]
        axes[1, 2].plot(X, diff, label=p.tenor_label, lw=1.2)
    axes[1, 2].set_xlabel("e^z"); axes[1, 2].set_ylabel("model - LV")
    axes[1, 2].set_title("Marginal difference")
    axes[1, 2].legend(); axes[1, 2].set_xlim(0.3, 2.0)
    axes[1, 2].axhline(0, color="k", ls=":", alpha=0.3); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lsv_ctmc_fi_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_leverage_evolution(result, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not result.leverage_time:
        return
    z = result.z_grid
    X = np.exp(z)
    os.makedirs(out_dir, exist_ok=True)

    for k, pillar in enumerate(result.pillars):
        lt = result.leverage_time[k]
        if lt is None:
            continue
        n_sub = lt.shape[0] - 1
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        n_show = min(10, n_sub + 1)
        idxs = np.round(np.linspace(0, n_sub, n_show)).astype(int)
        cmap = plt.cm.viridis
        for ii, idx in enumerate(idxs):
            frac = idx / n_sub if n_sub > 0 else 0
            color = cmap(ii / max(len(idxs) - 1, 1))
            ax.plot(X, lt[idx, :], lw=1.0, alpha=0.8, color=color,
                    label=f"s={idx} ({frac:.0%})")
        ax.set_xlabel("e^z (moneyness)")
        ax.set_ylabel("L(e^z)")
        ax.set_title(f"Leverage evolution: {pillar.tenor_label} ({n_sub} substeps)")
        ax.set_xlim(0.3, 2.0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f"lsv_ctmc_fi_leverage_evo_{pillar.tenor_label}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def plot_joint_density(result, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.cm as cm

    z = result.z_grid
    v_states = result.ctmc.states
    vol_states = np.sqrt(v_states) * 100
    N_states = len(v_states)

    os.makedirs(out_dir, exist_ok=True)
    n_pillars = len(result.pillars)

    for k, pillar in enumerate(result.pillars):
        u_joint = result.densities[k]
        marg = result.marginals[k]

        active = marg > marg.max() * 1e-4
        if active.sum() < 10:
            continue
        zlo_idx = max(np.where(active)[0][0] - 20, 0)
        zhi_idx = min(np.where(active)[0][-1] + 20, len(z) - 1)

        z_crop = z[zlo_idx:zhi_idx + 1]
        X_crop = np.exp(z_crop)
        u_crop = u_joint[:, zlo_idx:zhi_idx + 1]

        max_z_pts = 200
        if len(X_crop) > max_z_pts:
            step = max(1, len(X_crop) // max_z_pts)
            X_ds = X_crop[::step]
            u_ds = u_crop[:, ::step]
        else:
            X_ds = X_crop
            u_ds = u_crop

        max_v_pts = 80
        if N_states > max_v_pts:
            v_step = max(1, N_states // max_v_pts)
            v_idx = np.arange(0, N_states, v_step)
            vol_ds = vol_states[v_idx]
            u_ds = u_ds[v_idx, :]
        else:
            vol_ds = vol_states

        XX, VV = np.meshgrid(X_ds, vol_ds)
        ZZ = np.maximum(u_ds, 0.0)

        zz_floor = ZZ[ZZ > 0].min() * 0.1 if (ZZ > 0).any() else 1e-30
        zz_color = np.maximum(ZZ, zz_floor)
        log_zz = np.log10(zz_color)
        log_range = max(log_zz.max() - log_zz.min(), 1e-30)
        log_zz_norm = (log_zz - log_zz.min()) / log_range
        colors = cm.inferno(log_zz_norm)

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(XX, VV, ZZ, facecolors=colors, shade=True,
                        alpha=0.9, rstride=1, cstride=1, linewidth=0,
                        antialiased=True)
        ax.set_xlabel("e^z (moneyness)", fontsize=10, labelpad=10)
        ax.set_ylabel("Vol (%)", fontsize=10, labelpad=10)
        ax.set_zlabel("Density", fontsize=10, labelpad=8)
        ax.set_title(f"Joint density p(e^z, vol) — {pillar.tenor_label}\n"
                     f"({len(vol_ds)} vol states shown, {len(X_ds)} z-points)",
                     fontsize=12)
        ax.view_init(elev=25, azim=-55)

        mappable = cm.ScalarMappable(cmap='inferno',
                                     norm=plt.Normalize(vmin=log_zz.min(), vmax=log_zz.max()))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1)
        cbar.set_label("log10(density)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f"lsv_ctmc_fi_joint_density_{pillar.tenor_label}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        fig2, axes2 = plt.subplots(1, 2, figsize=(24, 7), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.08})

        fig2.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.10)
        axes2[0].remove()
        ax3d = fig2.add_subplot(1, 2, 1, projection='3d')

        ax3d.plot_surface(XX, VV, ZZ, facecolors=colors, shade=True,
                          alpha=0.85, rstride=1, cstride=1, linewidth=0,
                          antialiased=True)
        ax3d.set_xlabel("e^z", fontsize=9, labelpad=8)
        ax3d.set_ylabel("Vol (%)", fontsize=9, labelpad=8)
        ax3d.set_zlabel("Density", fontsize=9, labelpad=6)
        ax3d.set_title(f"3D surface — {pillar.tenor_label}", fontsize=11)
        ax3d.view_init(elev=25, azim=-55)

        ax2 = axes2[1]
        n_slices = min(10, N_states)
        slice_idx = np.round(np.linspace(0, N_states - 1, n_slices)).astype(int)
        cmap_lines = plt.cm.plasma
        for ii, si in enumerate(slice_idx):
            color = cmap_lines(ii / max(n_slices - 1, 1))
            ax2.plot(X_crop, u_joint[si, zlo_idx:zhi_idx + 1],
                     lw=1.3, color=color,
                     label=f"{vol_states[si]:.1f}%")
        ax2.set_xlabel("e^z (moneyness)", fontsize=10)
        ax2.set_ylabel("p(z | v)", fontsize=10)
        ax2.set_title(f"Density slices — {pillar.tenor_label}", fontsize=11)
        ax2.legend(fontsize=6, title="Vol state", title_fontsize=7)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f"lsv_ctmc_fi_joint_3d_slices_{pillar.tenor_label}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    fig, axes_flat = plt.subplots(1, n_pillars, figsize=(5.5 * n_pillars, 5),
                                  subplot_kw={"projection": "3d"}, squeeze=False)
    for k, pillar in enumerate(result.pillars):
        ax = axes_flat[0, k]
        u_joint = result.densities[k]
        marg = result.marginals[k]
        active = marg > marg.max() * 1e-4
        if active.sum() < 10:
            ax.set_title(pillar.tenor_label); continue
        zlo_idx = max(np.where(active)[0][0] - 10, 0)
        zhi_idx = min(np.where(active)[0][-1] + 10, len(z) - 1)
        X_crop = np.exp(z[zlo_idx:zhi_idx + 1])
        u_crop = u_joint[:, zlo_idx:zhi_idx + 1]

        max_pts = 120
        if len(X_crop) > max_pts:
            step = max(1, len(X_crop) // max_pts)
            X_ds = X_crop[::step]; u_s = u_crop[:, ::step]
        else:
            X_ds = X_crop; u_s = u_crop

        max_v = 60
        if N_states > max_v:
            vs = max(1, N_states // max_v)
            vi = np.arange(0, N_states, vs)
            vol_ds = vol_states[vi]; u_s = u_s[vi, :]
        else:
            vol_ds = vol_states

        XX, VV = np.meshgrid(X_ds, vol_ds)
        ZZ = np.maximum(u_s, 0.0)
        zf = ZZ[ZZ > 0].min() * 0.1 if (ZZ > 0).any() else 1e-30
        lc = np.log10(np.maximum(ZZ, zf))
        ln = (lc - lc.min()) / max(lc.max() - lc.min(), 1e-30)

        ax.plot_surface(XX, VV, ZZ, facecolors=cm.inferno(ln), shade=True,
                        alpha=0.85, rstride=1, cstride=1, linewidth=0)
        ax.set_xlabel("e^z", fontsize=7, labelpad=4)
        ax.set_ylabel("Vol%", fontsize=7, labelpad=4)
        ax.set_title(pillar.tenor_label, fontsize=10)
        ax.view_init(elev=25, azim=-55)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lsv_ctmc_fi_joint_density_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_leverage_surface(result, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.cm as cm

    if not result.leverage_time:
        return

    z = result.z_grid
    X = np.exp(z)
    os.makedirs(out_dir, exist_ok=True)

    for k, pillar in enumerate(result.pillars):
        lt = result.leverage_time[k]
        if lt is None:
            continue
        n_sub = lt.shape[0]

        marg = result.marginals[k]
        active = marg > marg.max() * 1e-4
        if active.sum() < 10:
            continue
        zlo = max(np.where(active)[0][0] - 80, 0)
        zhi = min(np.where(active)[0][-1] + 80, len(z) - 1)

        X_crop = X[zlo:zhi + 1]
        lt_crop = lt[:, zlo:zhi + 1]

        max_z = 200
        if len(X_crop) > max_z:
            step = max(1, len(X_crop) // max_z)
            X_ds = X_crop[::step]
            lt_ds = lt_crop[:, ::step]
        else:
            X_ds = X_crop
            lt_ds = lt_crop

        max_t = 60
        if n_sub > max_t:
            t_step = max(1, n_sub // max_t)
            t_idx = np.arange(0, n_sub, t_step)
            lt_ds = lt_ds[t_idx, :]
        else:
            t_idx = np.arange(n_sub)

        t_frac = t_idx / max(n_sub - 1, 1)
        T_start = pillar.T - pillar.dt
        t_abs = T_start + t_frac * pillar.dt

        TT, XX_mesh = np.meshgrid(t_abs, X_ds, indexing='ij')
        LL = lt_ds

        l_norm = (LL - LL.min()) / max(LL.max() - LL.min(), 1e-30)
        colors = cm.coolwarm(l_norm)

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(XX_mesh, TT, LL, facecolors=colors, shade=True,
                        alpha=0.9, rstride=1, cstride=1, linewidth=0,
                        antialiased=True)
        ax.set_xlabel("e^z (moneyness)", fontsize=10, labelpad=10)
        ax.set_ylabel("Time (years)", fontsize=10, labelpad=10)
        ax.set_zlabel("L(t, e^z)", fontsize=10, labelpad=8)
        ax.set_title(f"Leverage surface — {pillar.tenor_label}\n"
                     f"T=[{T_start:.4f}, {pillar.T:.4f}], {len(t_idx)} time steps (downsampled)",
                     fontsize=12)
        ax.view_init(elev=25, azim=-45)

        mappable = cm.ScalarMappable(cmap='coolwarm',
                                     norm=plt.Normalize(vmin=LL.min(), vmax=LL.max()))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1)
        cbar.set_label("Leverage L")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f"lsv_ctmc_fi_leverage_surface_{pillar.tenor_label}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # Combined: all pillars stitched together
    all_t = []
    all_L = []
    z_active_lo = 0
    z_active_hi = len(z) - 1

    for k, pillar in enumerate(result.pillars):
        lt = result.leverage_time[k]
        if lt is None:
            continue
        n_sub = lt.shape[0]
        T_start = pillar.T - pillar.dt

        marg = result.marginals[k]
        active = marg > marg.max() * 1e-4
        if active.sum() >= 10:
            lo = np.where(active)[0][0]
            hi = np.where(active)[0][-1]
            z_active_lo = max(z_active_lo, 0)
            z_active_hi = min(max(z_active_hi, hi + 80), len(z) - 1)

        max_t_per = 30
        if n_sub > max_t_per:
            t_step = max(1, n_sub // max_t_per)
            t_idx = np.arange(0, n_sub, t_step)
        else:
            t_idx = np.arange(n_sub)

        for ti in t_idx:
            t_abs = T_start + (ti / max(n_sub - 1, 1)) * pillar.dt
            all_t.append(t_abs)
            all_L.append(lt[ti, :])

    if len(all_t) < 3:
        return

    X_plot_min = 0.3
    X_plot_max = 3.0

    zlo = max(z_active_lo - 60, 0)
    zhi = min(z_active_hi + 60, len(z) - 1)

    mask_x = (X >= X_plot_min) & (X <= X_plot_max)
    if np.any(mask_x):
        xlo = np.where(mask_x)[0][0]
        xhi = np.where(mask_x)[0][-1]
        zlo = max(zlo, xlo)
        zhi = min(zhi, xhi)

    X_crop = X[zlo:zhi + 1]

    max_z = 200
    if len(X_crop) > max_z:
        step = max(1, len(X_crop) // max_z)
        X_ds = X_crop[::step]
        z_slice = slice(zlo, zhi + 1, step)
    else:
        X_ds = X_crop
        z_slice = slice(zlo, zhi + 1)

    all_t = np.array(all_t)
    L_matrix = np.array([L[z_slice] for L in all_L])

    TT, XX_mesh = np.meshgrid(all_t, X_ds, indexing='ij')
    LL = L_matrix

    l_norm = (LL - LL.min()) / max(LL.max() - LL.min(), 1e-30)
    colors = cm.coolwarm(l_norm)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(XX_mesh, TT, LL, facecolors=colors, shade=True,
                    alpha=0.85, rstride=1, cstride=1, linewidth=0,
                    antialiased=True)

    for pillar in result.pillars:
        ax.plot([X_ds[0], X_ds[-1]], [pillar.T, pillar.T],
                [LL.min(), LL.min()], 'g--', lw=1.0, alpha=0.5)

    ax.set_xlabel("e^z (moneyness)", fontsize=10, labelpad=10)
    ax.set_ylabel("Time (years)", fontsize=10, labelpad=10)
    ax.set_zlabel("L(t, e^z)", fontsize=10, labelpad=8)
    ax.set_title(f"Full leverage surface L(t, e^z)\n"
                 f"{len(all_t)} time points, {len(X_ds)} moneyness points",
                 fontsize=13)
    ax.view_init(elev=20, azim=-50)

    mappable = cm.ScalarMappable(cmap='coolwarm',
                                 norm=plt.Normalize(vmin=LL.min(), vmax=LL.max()))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label("Leverage L")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lsv_ctmc_fi_leverage_surface_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="LSV-CTMC Forward-Induction Calibrator (CUDA)")
    p.add_argument("--data_dir", type=str, default="./data", help="Directory containing pillar .npz files and heston parameters")
    p.add_argument("--splitting", type=str, default="lie_trotter", choices=["lie_trotter", "strang"])
    p.add_argument("--n_states", type=int, default=100)
    p.add_argument("--ctmc_method", type=str, default="tavella_randall",
                   choices=["uniform_v", "gamma_quantile", "tavella_randall"])
    p.add_argument("--frac_center", type=float, default=0.8)
    p.add_argument("--ctmc_gamma", type=float, default=10.0)
    p.add_argument("--Tmax", type=float, default=2.0)
    p.add_argument("--Nz", type=int, default=1201)
    p.add_argument("--z_min", type=float, default=-3.0)
    p.add_argument("--z_max", type=float, default=3.0)
    p.add_argument("--n_substeps", type=int, default=1200)
    p.add_argument("--omega", type=float, default=0.6)
    p.add_argument("--theta_pde", type=float, default=1.0)
    p.add_argument("--smooth_width", type=int, default=0)
    p.add_argument("--leverage_cap", type=float, default=25.0)
    p.add_argument("--density_threshold", type=float, default=1e-30)
    p.add_argument("--no_store_leverage_time", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out_dir", type=str, default="CTMC_FI_Output")
    p.add_argument("--backend", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    npz_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.startswith("localvol_") and f.endswith(".npz")])
    print(f"Found {len(npz_files)} pillar files:")
    for f in npz_files:
        print(f"  {os.path.basename(f)}")

    pillars = load_pillars(npz_files)
    heston = load_heston(os.path.join(data_dir, "heston_rho0_parameters.json"))
    print(f"\nHeston: v0={heston.v0:.6f}, kappa={heston.kappa:.4f}, "
          f"theta={heston.theta:.6f}, xi={heston.xi:.4f}, rho={heston.rho:.4f}")

    ctmc = build_ctmc_from_heston(heston, n_states=args.n_states,
                                  method=args.ctmc_method, frac_center=args.frac_center,
                                  gamma=args.ctmc_gamma, Tmax=args.Tmax)
    print(f"\nCTMC ({args.ctmc_method}): {ctmc.n_states} states")
    print(f"  v range: [{ctmc.states.min():.6g}, {ctmc.states.max():.6g}]")

    config = CalibConfig(
        Nz=args.Nz, z_min=args.z_min, z_max=args.z_max,
        n_substeps_per_bucket=args.n_substeps,
        omega=args.omega, theta_pde=args.theta_pde,
        smooth_leverage=True, smooth_width=args.smooth_width,
        rannacher_steps=4, backend=args.backend,
        leverage_cap=args.leverage_cap,
        density_threshold_frac=args.density_threshold,
        splitting=args.splitting,
        store_leverage_time=not args.no_store_leverage_time)

    Lambda = ctmc.generator
    lam = -np.diag(Lambda)
    dt_sub = pillars[0].dt / config.n_substeps_per_bucket
    print(f"\n=== CTMC JUMP-RATE DIAGNOSTICS ===")
    print(f"Exit rate: min/med/90%/max = {np.quantile(lam, [0, 0.5, 0.9, 1.0])}")
    print(f"dt_sub = {dt_sub:.6g}, lambda*dt_sub max = {np.max(lam * dt_sub):.4f}")
    print(f"==================================\n")

    print("=" * 70)
    result = calibrate_lsv_ctmc_forward_induction(pillars, ctmc, config)
    print("=" * 70)

    z = result.z_grid; dz = float(z[1] - z[0])
    print("\n=== DIAGNOSTICS ===")
    for k, pillar in enumerate(result.pillars):
        L = result.leverage[k]; marg = result.marginals[k]
        lv_marg = result.lv_marginals[k]; Es = result.E_sig2[k]
        st = np.interp(z, pillar.z_grid, pillar.sigma_z)
        sm = L * np.sqrt(Es)
        rel = marg > 1e-6
        if rel.sum() > 0:
            err = np.abs(sm[rel] - st[rel])
            rmse_val = float(np.sqrt(np.mean(err ** 2)))
            mx = float(np.max(err))
        else:
            rmse_val = mx = float("nan")
        marg_l1 = float(np.sum(np.abs(marg - lv_marg)) * dz)
        print(f"  [{pillar.tenor_label}] mass={np.sum(marg)*dz:.6f}, "
              f"LV_RMSE={rmse_val:.6f}, max_err={mx:.6f}, "
              f"L=[{L.min():.4f},{L.max():.4f}], marg_L1_diff={marg_l1:.6f}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "lsv_ctmc_fi_result.npz")
    save_lsv_ctmc_result(result, heston, result_path)

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(result, out_dir)
        plot_leverage_evolution(result, out_dir)
        plot_joint_density(result, out_dir)
        plot_leverage_surface(result, out_dir)

    print(f"\nAll done. Results in {out_dir}/")


if __name__ == "__main__":
    main()