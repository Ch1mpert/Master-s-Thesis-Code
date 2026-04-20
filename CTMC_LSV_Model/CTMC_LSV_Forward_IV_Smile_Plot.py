#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.stats import norm

try:
    from numba import njit
except Exception:
    njit = None


# -----------------------------------------------------------------------------
# Black forward-start IV utilities
# -----------------------------------------------------------------------------
def black_forward_call(discount: float, forward: float, strike: float, tau: float, vol: float) -> float:
    if tau <= 0.0 or vol <= 0.0:
        return discount * max(forward - strike, 0.0)
    srt = vol * math.sqrt(tau)
    d1 = (math.log(max(forward, 1e-300) / max(strike, 1e-300)) + 0.5 * vol * vol * tau) / srt
    d2 = d1 - srt
    return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def black_forward_put(discount: float, forward: float, strike: float, tau: float, vol: float) -> float:
    if tau <= 0.0 or vol <= 0.0:
        return discount * max(strike - forward, 0.0)
    srt = vol * math.sqrt(tau)
    d1 = (math.log(max(forward, 1e-300) / max(strike, 1e-300)) + 0.5 * vol * vol * tau) / srt
    d2 = d1 - srt
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol_forward_option(
    norm_call_price: float,
    discount: float,
    forward: float,
    strike: float,
    tau: float,
) -> float:
    """
    Invert the forward-start Black smile from the normalized call price:
        norm_call_price = C0 / (DF(0,T1) * F(0,T1)).

    For numerical stability, use puts on the left wing through parity.
    """
    if tau <= 0.0:
        return 0.0

    call_intrinsic = discount * max(forward - strike, 0.0)
    call_upper = discount * forward
    c = float(np.clip(norm_call_price, call_intrinsic, call_upper))

    # Use OTM side for inversion.
    if strike < forward:
        p = c - discount * (forward - strike)
        p = max(p, 0.0)
        intrinsic = discount * max(strike - forward, 0.0)
        upper = discount * strike
        price = float(np.clip(p, intrinsic, upper))
        if abs(price - intrinsic) < 1e-14:
            return 0.0
        if price >= upper - 1e-12:
            return np.nan

        def f(sig: float) -> float:
            return black_forward_put(discount, forward, strike, tau, sig) - price
    else:
        intrinsic = call_intrinsic
        upper = call_upper
        price = c
        if abs(price - intrinsic) < 1e-14:
            return 0.0
        if price >= upper - 1e-12:
            return np.nan

        def f(sig: float) -> float:
            return black_forward_call(discount, forward, strike, tau, sig) - price

    for lo, hi in ((1e-8, 5.0), (1e-10, 10.0)):
        try:
            return float(brentq(f, lo, hi, maxiter=300))
        except ValueError:
            pass
    return np.nan


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
@dataclass
class CTMCLSVResult:
    z: np.ndarray
    dz: float
    n_buckets: int
    n_substeps: int
    pillar_labels: List[str]
    pillar_T: np.ndarray
    pillar_dt: np.ndarray
    pillar_forward: np.ndarray
    pillar_df: np.ndarray
    v_states: np.ndarray
    ctmc_Q: np.ndarray
    ctmc_pi0: np.ndarray
    sigma_lv: List[np.ndarray]
    lv_marginals: List[np.ndarray]
    densities: List[np.ndarray]
    leverage_paths: List[np.ndarray]


def load_ctmc_lsv_result(path: str) -> CTMCLSVResult:
    d = np.load(path, allow_pickle=True)
    n_buckets = int(d['n_buckets'])
    required = []
    for k in range(n_buckets):
        required += [f'sigma_lv_{k}', f'leverage_time_{k}', f'density_{k}', f'lv_marginal_{k}']
    missing = [k for k in required if k not in d.files]
    if missing:
        raise ValueError(
            'The CTMC result file does not contain all LV/LSV bucket arrays needed for operator pricing. '
            f'Missing keys: {missing}'
        )
    return CTMCLSVResult(
        z=np.asarray(d['z_grid'], dtype=float),
        dz=float(d['dz']),
        n_buckets=n_buckets,
        n_substeps=int(d['n_substeps']),
        pillar_labels=[str(x) for x in np.asarray(d['pillar_labels'])],
        pillar_T=np.asarray(d['pillar_T'], dtype=float),
        pillar_dt=np.asarray(d['pillar_dt'], dtype=float),
        pillar_forward=np.asarray(d['pillar_forward'], dtype=float),
        pillar_df=np.asarray(d['pillar_df'], dtype=float),
        v_states=np.asarray(d['ctmc_states'], dtype=float),
        ctmc_Q=np.asarray(d['ctmc_generator'], dtype=float),
        ctmc_pi0=np.asarray(d['ctmc_pi0'], dtype=float),
        sigma_lv=[np.asarray(d[f'sigma_lv_{k}'], dtype=float) for k in range(n_buckets)],
        lv_marginals=[np.asarray(d[f'lv_marginal_{k}'], dtype=float) for k in range(n_buckets)],
        densities=[np.asarray(d[f'density_{k}'], dtype=float) for k in range(n_buckets)],
        leverage_paths=[np.asarray(d[f'leverage_time_{k}'], dtype=float) for k in range(n_buckets)],
    )


# -----------------------------------------------------------------------------
# Discrete operators (same forward-grid convention as the saved calibration)
# -----------------------------------------------------------------------------
def build_lv_forward_coefficients(sig: np.ndarray, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = sig * sig
    nz = len(a)
    lower = np.empty(nz - 1, dtype=float)
    diag = np.empty(nz, dtype=float)
    upper = np.empty(nz - 1, dtype=float)

    c_l = 0.5 / dz**2 - 0.25 / dz
    c_u = 0.5 / dz**2 + 0.25 / dz
    c_d0 = -0.5 / dz**2 + 0.25 / dz
    c_d = -1.0 / dz**2
    c_dn = -0.5 / dz**2 - 0.25 / dz

    lower[:-1] = c_l * a[:-2]
    upper[1:] = c_u * a[2:]
    diag[1:-1] = c_d * a[1:-1]
    diag[0] = c_d0 * a[0]
    upper[0] = c_u * a[1]
    lower[-1] = c_l * a[-2]
    diag[-1] = c_dn * a[-1]
    return lower, diag, upper


def build_ctmc_lsv_forward_coefficients(v_states: np.ndarray, L: np.ndarray, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = v_states[:, None] * (L[None, :] ** 2)
    nv, nz = a.shape
    lower = np.empty((nv, nz - 1), dtype=float)
    diag = np.empty((nv, nz), dtype=float)
    upper = np.empty((nv, nz - 1), dtype=float)

    c_l = 0.5 / dz**2 - 0.25 / dz
    c_u = 0.5 / dz**2 + 0.25 / dz
    c_d0 = -0.5 / dz**2 + 0.25 / dz
    c_d = -1.0 / dz**2
    c_dn = -0.5 / dz**2 - 0.25 / dz

    lower[:, :-1] = c_l * a[:, :-2]
    upper[:, 1:] = c_u * a[:, 2:]
    diag[:, 1:-1] = c_d * a[:, 1:-1]
    diag[:, 0] = c_d0 * a[:, 0]
    upper[:, 0] = c_u * a[:, 1]
    lower[:, -1] = c_l * a[:, -2]
    diag[:, -1] = c_dn * a[:, -1]
    return lower, diag, upper


# -----------------------------------------------------------------------------
# Tridiagonal solves for backward propagation (transpose of the saved forward step)
# -----------------------------------------------------------------------------
if njit is not None:
    @njit(cache=True)
    def solve_tridiag_transpose_numba(lower, diag, upper, rhs):
        n = diag.shape[0]
        m = rhs.shape[1]
        lp = upper.copy()
        dp = diag.copy()
        up = lower.copy()
        rr = rhs.copy()

        for j in range(1, n):
            mult = lp[j - 1] / dp[j - 1]
            dp[j] -= mult * up[j - 1]
            for k in range(m):
                rr[j, k] -= mult * rr[j - 1, k]

        x = np.empty_like(rr)
        for k in range(m):
            x[n - 1, k] = rr[n - 1, k] / dp[n - 1]
        for j in range(n - 2, -1, -1):
            for k in range(m):
                x[j, k] = (rr[j, k] - up[j] * x[j + 1, k]) / dp[j]
        return x

    @njit(cache=True)
    def solve_batched_tridiag_transpose_numba(lower, diag, upper, rhs):
        nv, nz = diag.shape
        c = rhs.shape[2]
        lp = upper.copy()
        dp = diag.copy()
        up = lower.copy()
        rr = rhs.copy()

        for i in range(nv):
            for j in range(1, nz):
                mult = lp[i, j - 1] / dp[i, j - 1]
                dp[i, j] -= mult * up[i, j - 1]
                for k in range(c):
                    rr[i, j, k] -= mult * rr[i, j - 1, k]

        x = np.empty_like(rr)
        for i in range(nv):
            for k in range(c):
                x[i, nz - 1, k] = rr[i, nz - 1, k] / dp[i, nz - 1]
            for j in range(nz - 2, -1, -1):
                for k in range(c):
                    x[i, j, k] = (rr[i, j, k] - up[i, j] * x[i, j + 1, k]) / dp[i, j]
        return x
else:
    solve_tridiag_transpose_numba = None
    solve_batched_tridiag_transpose_numba = None


def solve_tridiag_transpose(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if solve_tridiag_transpose_numba is not None:
        return solve_tridiag_transpose_numba(lower, diag, upper, rhs)
    lp = upper.copy()
    dp = diag.copy()
    up = lower.copy()
    rr = rhs.copy()
    n = dp.shape[0]
    for j in range(1, n):
        mult = lp[j - 1] / dp[j - 1]
        dp[j] -= mult * up[j - 1]
        rr[j, :] -= mult * rr[j - 1, :]
    x = np.empty_like(rr)
    x[-1, :] = rr[-1, :] / dp[-1]
    for j in range(n - 2, -1, -1):
        x[j, :] = (rr[j, :] - up[j] * x[j + 1, :]) / dp[j]
    return x


def solve_batched_tridiag_transpose(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if solve_batched_tridiag_transpose_numba is not None:
        return solve_batched_tridiag_transpose_numba(lower, diag, upper, rhs)
    lp = upper.copy()
    dp = diag.copy()
    up = lower.copy()
    rr = rhs.copy()
    nv, nz = dp.shape
    for j in range(1, nz):
        mult = lp[:, j - 1] / dp[:, j - 1]
        dp[:, j] -= mult * up[:, j - 1]
        rr[:, j, :] -= mult[:, None] * rr[:, j - 1, :]
    x = np.empty_like(rr)
    x[:, -1, :] = rr[:, -1, :] / dp[:, -1][:, None]
    for j in range(nz - 2, -1, -1):
        x[:, j, :] = (rr[:, j, :] - up[:, j][:, None] * x[:, j + 1, :]) / dp[:, j][:, None]
    return x


# -----------------------------------------------------------------------------
# Bucket resampling / preprocessing
# -----------------------------------------------------------------------------
def resampled_time_index(n_saved: int, n_substeps: int) -> np.ndarray:
    if n_substeps == n_saved:
        return np.arange(n_saved + 1, dtype=int)
    idx = np.rint(np.linspace(0, n_saved, n_substeps + 1)).astype(int)
    idx[0] = 0
    idx[-1] = n_saved
    idx = np.maximum.accumulate(idx)
    for j in range(1, len(idx)):
        if idx[j] <= idx[j - 1]:
            idx[j] = min(n_saved, idx[j - 1] + 1)
    return idx


def normalize_density_1d(p: np.ndarray, dz: float) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    mass = float(np.sum(p) * dz)
    if mass <= 0.0 or not np.isfinite(mass):
        raise RuntimeError('Non-positive 1D density mass encountered.')
    return p / mass


def normalize_density_2d(p: np.ndarray, dz: float) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    mass = float(np.sum(p) * dz)
    if mass <= 0.0 or not np.isfinite(mass):
        raise RuntimeError('Non-positive 2D density mass encountered.')
    return p / mass


@dataclass
class BucketPrepLV:
    z: np.ndarray
    dz: float
    F1: float
    F2: float
    DF1: float
    DF2: float
    tau: float
    start_density: np.ndarray
    lower_steps: List[np.ndarray]
    diag_steps: List[np.ndarray]
    upper_steps: List[np.ndarray]


@dataclass
class BucketPrepCTMC:
    z: np.ndarray
    dz: float
    F1: float
    F2: float
    DF1: float
    DF2: float
    tau: float
    start_density: np.ndarray
    trans_v_back: np.ndarray
    lower_steps: List[np.ndarray]
    diag_steps: List[np.ndarray]
    upper_steps: List[np.ndarray]


def prepare_bucket_lv(res: CTMCLSVResult, bucket_idx: int, z_stride: int, max_substeps: int) -> BucketPrepLV:
    z = res.z[::z_stride]
    dz = res.dz * z_stride
    n_saved = int(res.n_substeps)
    n_substeps = n_saved if max_substeps <= 0 else min(n_saved, int(max_substeps))
    dt = float(res.pillar_dt[bucket_idx]) / n_substeps

    sigma = np.asarray(res.sigma_lv[bucket_idx], dtype=float)[::z_stride]
    lower, diag, upper = build_lv_forward_coefficients(sigma, dz)
    lower = -dt * lower
    diag = 1.0 - dt * diag
    upper = -dt * upper

    if bucket_idx == 0:
        start_density = np.zeros_like(z)
        start_density[np.argmin(np.abs(z))] = 1.0 / dz
    else:
        start_density = normalize_density_1d(np.asarray(res.lv_marginals[bucket_idx - 1], dtype=float)[::z_stride], dz)

    return BucketPrepLV(
        z=z,
        dz=dz,
        F1=float(res.pillar_forward[bucket_idx - 1]) if bucket_idx > 0 else float(res.pillar_forward[0]) / np.exp(0.0),
        F2=float(res.pillar_forward[bucket_idx]),
        DF1=float(res.pillar_df[bucket_idx - 1]) if bucket_idx > 0 else 1.0,
        DF2=float(res.pillar_df[bucket_idx]),
        tau=float(res.pillar_dt[bucket_idx]),
        start_density=start_density,
        lower_steps=[lower] * n_substeps,
        diag_steps=[diag] * n_substeps,
        upper_steps=[upper] * n_substeps,
    )


def prepare_bucket_ctmc(res: CTMCLSVResult, bucket_idx: int, z_stride: int, max_substeps: int) -> BucketPrepCTMC:
    z = res.z[::z_stride]
    dz = res.dz * z_stride
    n_saved = int(res.n_substeps)
    n_substeps = n_saved if max_substeps <= 0 else min(n_saved, int(max_substeps))
    dt = float(res.pillar_dt[bucket_idx]) / n_substeps
    step_idx = resampled_time_index(n_saved, n_substeps)

    lower_steps: List[np.ndarray] = []
    diag_steps: List[np.ndarray] = []
    upper_steps: List[np.ndarray] = []
    lev_path = np.asarray(res.leverage_paths[bucket_idx], dtype=float)[:, ::z_stride]
    for step in range(n_substeps, 0, -1):
        i0 = step_idx[step - 1]
        i1 = step_idx[step]
        L_step = 0.5 * (lev_path[i0] + lev_path[i1])
        lower, diag, upper = build_ctmc_lsv_forward_coefficients(res.v_states, L_step, dz)
        lower_steps.append(-dt * lower)
        diag_steps.append(1.0 - dt * diag)
        upper_steps.append(-dt * upper)

    if bucket_idx == 0:
        start_density = np.zeros((len(res.v_states), len(z)), dtype=float)
        start_density[:, np.argmin(np.abs(z))] = np.maximum(res.ctmc_pi0, 0.0) / dz
        start_density = normalize_density_2d(start_density, dz)
        F1 = 1.0
        DF1 = 1.0
    else:
        start_density = normalize_density_2d(np.asarray(res.densities[bucket_idx - 1], dtype=float)[:, ::z_stride], dz)
        F1 = float(res.pillar_forward[bucket_idx - 1])
        DF1 = float(res.pillar_df[bucket_idx - 1])

    return BucketPrepCTMC(
        z=z,
        dz=dz,
        F1=F1,
        F2=float(res.pillar_forward[bucket_idx]),
        DF1=DF1,
        DF2=float(res.pillar_df[bucket_idx]),
        tau=float(res.pillar_dt[bucket_idx]),
        start_density=start_density,
        trans_v_back=expm(np.asarray(res.ctmc_Q, dtype=float) * dt),
        lower_steps=lower_steps,
        diag_steps=diag_steps,
        upper_steps=upper_steps,
    )


# -----------------------------------------------------------------------------
# Deterministic forward-start prices (operator based)
# -----------------------------------------------------------------------------
def forward_start_call_prices_lv(prep: BucketPrepLV, kappas: np.ndarray, chunk_size: int) -> np.ndarray:
    z = prep.z
    expz = np.exp(z)
    prices = np.empty(len(kappas), dtype=float)
    for ik, kappa in enumerate(kappas):
        strikes = kappa * prep.F1 * expz
        total = 0.0
        for j0 in range(0, len(z), chunk_size):
            js = np.arange(j0, min(len(z), j0 + chunk_size))
            payoff = np.maximum(prep.F2 * expz[:, None] - strikes[js][None, :], 0.0)
            V = payoff
            for lower, diag, upper in zip(prep.lower_steps, prep.diag_steps, prep.upper_steps):
                V = solve_tridiag_transpose(lower, diag, upper, V)
            vals = V[js, np.arange(len(js))]
            total += np.sum(prep.start_density[js] * vals) * prep.dz
        prices[ik] = prep.DF2 * total
    return prices


def forward_start_call_prices_ctmc(prep: BucketPrepCTMC, kappas: np.ndarray, chunk_size: int) -> np.ndarray:
    z = prep.z
    expz = np.exp(z)
    nv = prep.start_density.shape[0]
    prices = np.empty(len(kappas), dtype=float)
    for ik, kappa in enumerate(kappas):
        strikes = kappa * prep.F1 * expz
        total = 0.0
        for j0 in range(0, len(z), chunk_size):
            js = np.arange(j0, min(len(z), j0 + chunk_size))
            payoff_1v = np.maximum(prep.F2 * expz[None, :, None] - strikes[js][None, None, :], 0.0)
            V = np.broadcast_to(payoff_1v, (nv, len(z), len(js))).copy()
            for lower, diag, upper in zip(prep.lower_steps, prep.diag_steps, prep.upper_steps):
                V = solve_batched_tridiag_transpose(lower, diag, upper, V)
                V2 = V.reshape(nv, -1)
                V = (prep.trans_v_back @ V2).reshape(V.shape)
            vals = V[:, js, np.arange(len(js))]
            total += np.sum(prep.start_density[:, js] * vals) * prep.dz
        prices[ik] = prep.DF2 * total
    return prices


# -----------------------------------------------------------------------------
# Driver / plotting
# -----------------------------------------------------------------------------
def bucket_pair_label(res: CTMCLSVResult, bucket_idx: int) -> str:
    if bucket_idx <= 0:
        return f'0 -> {res.pillar_labels[0]}'
    return f'{res.pillar_labels[bucket_idx-1]} → {res.pillar_labels[bucket_idx]}'


def run_all_pairs(
    ctmc_file: str,
    kappa_min: float,
    kappa_max: float,
    n_kappa: int,
    z_stride: int,
    max_substeps: int,
    chunk_size: int,
    include_first_bucket: bool,
) -> tuple[pd.DataFrame, List[tuple[str, np.ndarray, np.ndarray, np.ndarray]]]:
    res = load_ctmc_lsv_result(ctmc_file)
    kappas = np.linspace(kappa_min, kappa_max, n_kappa)
    bucket_range = range(0 if include_first_bucket else 1, res.n_buckets)

    rows = []
    plot_data: List[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for bucket_idx in bucket_range:
        label = bucket_pair_label(res, bucket_idx)
        print(f'[bucket {bucket_idx}] {label}: preparing operators...')
        prep_lv = prepare_bucket_lv(res, bucket_idx, z_stride=z_stride, max_substeps=max_substeps)
        prep_ct = prepare_bucket_ctmc(res, bucket_idx, z_stride=z_stride, max_substeps=max_substeps)

        print(f'[bucket {bucket_idx}] {label}: pricing LV...')
        prices_lv = forward_start_call_prices_lv(prep_lv, kappas, chunk_size=chunk_size)
        print(f'[bucket {bucket_idx}] {label}: pricing CTMC-LSV...')
        prices_ct = forward_start_call_prices_ctmc(prep_ct, kappas, chunk_size=chunk_size)

        discount = prep_lv.DF2 / prep_lv.DF1
        forward = prep_lv.F2 / prep_lv.F1
        tau = prep_lv.tau
        iv_lv = np.array([implied_vol_forward_option(p / (prep_lv.DF1 * prep_lv.F1), discount, forward, k, tau) for p, k in zip(prices_lv, kappas)])
        iv_ct = np.array([implied_vol_forward_option(p / (prep_ct.DF1 * prep_ct.F1), discount, forward, k, tau) for p, k in zip(prices_ct, kappas)])

        for k, p_lv, p_ct, v_lv, v_ct in zip(kappas, prices_lv, prices_ct, iv_lv, iv_ct):
            rows.append({
                'bucket_index': bucket_idx,
                'pair': label,
                'tau': tau,
                'forward_ratio_F2_over_F1': forward,
                'strike_ratio_K_over_S_T1': k,
                'lv_price_t0': p_lv,
                'ctmc_lsv_price_t0': p_ct,
                'lv_forward_iv': v_lv,
                'ctmc_lsv_forward_iv': v_ct,
            })
        plot_data.append((label, kappas.copy(), iv_lv, iv_ct))
    return pd.DataFrame(rows), plot_data


def make_plot(plot_data: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]], out_png: str) -> None:
    n = len(plot_data)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (label, kappas, iv_lv, iv_ct) in zip(axes_flat, plot_data):
        ax.plot(kappas, iv_lv, linewidth=2.0, label='LV')
        ax.plot(kappas, iv_ct, linewidth=2.0, linestyle='--', label='CTMC-LSV')
        ax.set_title(label)
        ax.set_xlabel(r'$K/S(T_1)$')
        ax.set_ylabel('Forward implied vol')
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[len(plot_data):]:
        ax.axis('off')

    fig.suptitle('Operator-based forward-start IV smiles: LV vs CTMC-LSV', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Plot operator-based forward-start implied-volatility smiles for the LV and CTMC-LSV models '\
            'stored in lsv_ctmc_fi_result.npz. '\
            'The script prices true forward-start options deterministically by backward propagation on the saved operator grid.'
        )
    )
    p.add_argument('--ctmc_file', default='lsv_ctmc_fi_result.npz')
    p.add_argument('--out_png', default='forward_start_smiles_operator_lv_vs_ctmc.png')
    p.add_argument('--out_csv', default='forward_start_smiles_operator_lv_vs_ctmc.csv')
    p.add_argument('--kappa_min', type=float, default=0.70)
    p.add_argument('--kappa_max', type=float, default=1.30)
    p.add_argument('--n_kappa', type=int, default=25)
    p.add_argument('--z_stride', type=int, default=4,
                   help='Subsample the saved z-grid by this factor. Lower is more accurate but much slower.')
    p.add_argument('--max_substeps', type=int, default=24,
                   help='Use at most this many operator substeps per bucket. 0 means all saved substeps.')
    p.add_argument('--chunk_size', type=int, default=24,
                   help='Number of start-z columns solved together in one backward batch.')
    p.add_argument('--include_first_bucket', action='store_true',
                   help='Also include 0 -> first pillar. By default only adjacent pillar pairs are shown.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, plot_data = run_all_pairs(
        ctmc_file=args.ctmc_file,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        n_kappa=args.n_kappa,
        z_stride=args.z_stride,
        max_substeps=args.max_substeps,
        chunk_size=args.chunk_size,
        include_first_bucket=args.include_first_bucket,
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    make_plot(plot_data, args.out_png)
    print(f'Saved CSV: {args.out_csv}')
    print(f'Saved PNG: {args.out_png}')


if __name__ == '__main__':
    main()
