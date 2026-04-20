#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local volatility calibration using Schrödinger bridge / KL divergence.
v2: Added warm start, stronger martingale constraint, two-phase optimizer.

REGHAI'S APPROACH (from "Schrödinger Local Volatility" paper):
  - For each interval [T_i, T_{i+1}], we calibrate σ_i(x) such that
    the density evolves from μ_{T_i}(x) to μ_{T_{i+1}}(x).
  - The INITIAL CONDITION for each interval is the MARKET RND at T_i
    (not a propagated model density, not a Dirac delta).
  - This guarantees that calibration is always feasible since we're
    finding a bridge between two known, well-behaved distributions.

v2 changes:
  1) WARM_START_PREV_INTERVAL = True (init from previous interval's sigma)
  2) MEAN_WEIGHT = 50.0 (enforce E[S/F]=1 more strictly)
  3) Two-phase optimizer: Adam (60%) then L-BFGS (40%)
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
NPZ_BY_TENOR = {
    1:  "./data/1M.npz",
    3:  "./data/3M.npz",
    6:  "./data/6M.npz",
    12: "./data/12M.npz",
    24: "./data/24M.npz",
}
OPTIONS_CSV = "./data/^SPX_options_cleaned.csv"

# Fixed logX grid
N_Z   = 1201
X_MIN = 0.0001
X_MAX = 10.0

# Numerics
EPS = 1e-18
EPS_STATE = 1e-18
EPS_LOG   = 1e-18
N_SUBSTEPS = 1

# Optimization
EPOCHS_PER_INTERVAL = 2500
LR = 5e-3
KL_WEIGHT = 1.0
SMOOTH_WEIGHT = 0.0
MEAN_WEIGHT = 10.0
PRINT_EVERY = 50

# Frechet derivative quadrature
GL_N = 12

# Plotting + saving
SAVE_PLOTS = True
PLOTS_DIR = "plots_reghai"
PLOT_DPI = 1000
PLOT_SIGMA_POINTS = 2000
SHOW_PLOTS = False

WARM_START_PREV_INTERVAL = False  # moment-based warmup is better for Reghai bridge
WARMUP_TO_MOMENT_GUESS = True

SAVE_MODEL_NPZ = True
MODEL_NPZ_DIR = "model_rnds_reghai_npz"

SAVE_LV_NPZ = True
LV_NPZ_DIR = "model_localvol_reghai_npz"

SAVE_TRIDIAG_Q_NPZ = True
TRIDIAG_Q_NPZ_DIR = "model_generator_reghai_npz"

# Log-moneyness IV plot window
PLOT_IV_LOGMNY_ONLY = True
LOGMNY_MIN = -0.7
LOGMNY_MAX =  0.35

# Implied vol
IV_MAX = 8.0
IV_TOL = 1e-8
IV_MAX_ITERS = 350

# Initial Gaussian prior
SIGMA0_Z_INIT = 0.015

# Device + precision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

print(f"[Info] Using device: {DEVICE}, default dtype: {torch.get_default_dtype()}")


# -----------------------------
# Small utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def save_or_show(fig: plt.Figure, path: str | None):
    if SAVE_PLOTS and path is not None:
        ensure_dir(os.path.dirname(path))
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def trapz_integral_np(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


# -----------------------------
# IO utils
# -----------------------------
def load_npz_market(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    for k in ["xg", "q", "w"]:
        if k in out:
            out[k] = np.asarray(out[k], dtype=np.float64)
    for k in ["T", "r", "df", "forward"]:
        if k in out:
            out[k] = float(np.asarray(out[k]))
    if "expiry" in out:
        out["expiry"] = str(out["expiry"])
    return out


# -----------------------------
# Pricing from densities
# -----------------------------
def price_from_rnd_call_put(S_grid, qS, K, df):
    calls = np.zeros_like(K, dtype=np.float64)
    puts  = np.zeros_like(K, dtype=np.float64)
    for i, k in enumerate(K):
        calls[i] = df * trapz_integral_np(np.maximum(S_grid - k, 0.0) * qS, S_grid)
        puts[i]  = df * trapz_integral_np(np.maximum(k - S_grid, 0.0) * qS, S_grid)
    return calls, puts


def price_from_pz_call_put(z_grid, pz, K, df, Fwd, dz):
    with torch.no_grad():
        S = float(Fwd) * torch.exp(z_grid)
        K_t = torch.tensor(K, device=z_grid.device, dtype=z_grid.dtype)
        S_col = S.reshape(-1, 1)
        K_row = K_t.reshape(1, -1)
        call_pay = torch.clamp(S_col - K_row, min=0.0)
        put_pay  = torch.clamp(K_row - S_col, min=0.0)
        w = (pz.reshape(-1, 1) * dz)
        calls = float(df) * torch.sum(call_pay * w, dim=0)
        puts  = float(df) * torch.sum(put_pay  * w, dim=0)
    return calls.cpu().numpy(), puts.cpu().numpy()


# -----------------------------
# Priors
# -----------------------------
def make_gaussian_prior_pz(z, dz, sigma0_z):
    with torch.no_grad():
        pz = torch.exp(-0.5 * (z / float(sigma0_z)) ** 2)
        pz = torch.clamp(pz, min=0.0)
        pz = pz / (torch.sum(pz) * dz + EPS)
    return pz

def make_dirac_prior_pz(z, dz, z0=0.0):
    with torch.no_grad():
        pz = torch.zeros_like(z)
        idx = int(torch.argmin(torch.abs(z - float(z0))).item())
        pz[idx] = 1.0 / float(dz)
    return pz


# -----------------------------
# Saving outputs
# -----------------------------
def save_model_rnd_npz(out_path, *, expiry, T, df, forward, z_grid, pz, S_grid, qS):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dz = float(z_grid[1] - z_grid[0])
    pz = np.maximum(pz, 0.0)
    pz = pz / (np.sum(pz) * dz + 1e-18)
    qS = np.maximum(qS, 0.0)
    qS = qS / (np.trapezoid(qS, S_grid) + 1e-18)
    np.savez(out_path, expiry=np.array(str(expiry)), T=np.array(float(T)),
             df=np.array(float(df)), forward=np.array(float(forward)),
             z=z_grid.astype(np.float64), pz=pz.astype(np.float64),
             xg=S_grid.astype(np.float64), q=qS.astype(np.float64))

def save_localvol_npz(out_path, *, expiry, tenor_months, T, dt, df, forward,
                       z_grid, sigma_z, S_grid=None, sigma_S=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = dict(
        expiry=np.array(str(expiry)), tenor_months=np.array(int(tenor_months)),
        T=np.array(float(T)), dt=np.array(float(dt)), df=np.array(float(df)),
        forward=np.array(float(forward)), z=z_grid.astype(np.float64),
        sigma_z=np.asarray(sigma_z, dtype=np.float64),
    )
    if S_grid is not None and sigma_S is not None:
        payload["xg"] = np.asarray(S_grid, dtype=np.float64)
        payload["sigma_S"] = np.asarray(sigma_S, dtype=np.float64)
    np.savez(out_path, **payload)

def save_tridiag_generator_npz(out_path, *, expiry, tenor_months, T, dt,
                                n_substeps, z_grid, dz, lower, diag, upper):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, expiry=np.array(str(expiry)),
             tenor_months=np.array(int(tenor_months)), T=np.array(float(T)),
             dt=np.array(float(dt)), n_substeps=np.array(int(n_substeps)),
             dz=np.array(float(dz)), z=z_grid.astype(np.float64),
             Q_lower=np.asarray(lower, dtype=np.float64),
             Q_diag=np.asarray(diag, dtype=np.float64),
             Q_upper=np.asarray(upper, dtype=np.float64))


# -----------------------------
# Black76 pricing and IV
# -----------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black76_price(df, Fwd, K, T, vol, is_call):
    if T <= 0 or vol <= 0:
        intrinsic = max(Fwd - K, 0.0) if is_call else max(K - Fwd, 0.0)
        return df * intrinsic
    if Fwd <= 0 or K <= 0:
        return float("nan")
    srt = vol * math.sqrt(T)
    d1 = (math.log(Fwd / K) + 0.5 * vol * vol * T) / srt
    d2 = d1 - srt
    if is_call:
        return df * (Fwd * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        return df * (K * norm_cdf(-d2) - Fwd * norm_cdf(-d1))

def black76_implied_vol(price, df, Fwd, K, T, is_call,
                         vol_lo=1e-8, vol_hi=IV_MAX, tol=IV_TOL, max_iter=IV_MAX_ITERS):
    if not (np.isfinite(price) and np.isfinite(df) and np.isfinite(Fwd) and np.isfinite(K) and np.isfinite(T)):
        return float("nan")
    if df <= 0 or Fwd <= 0 or K <= 0 or T <= 0:
        return float("nan")
    intrinsic = df * (max(Fwd - K, 0.0) if is_call else max(K - Fwd, 0.0))
    upper = df * (Fwd if is_call else K)
    if price < intrinsic - 1e-10 or price > upper + 1e-10:
        return float("nan")
    price = min(max(price, intrinsic), upper)
    if abs(price - intrinsic) <= 1e-12:
        return 0.0
    lo, hi = vol_lo, vol_hi
    plo = black76_price(df, Fwd, K, T, lo, is_call)
    phi = black76_price(df, Fwd, K, T, hi, is_call)
    if not (plo <= price <= phi):
        for _ in range(10):
            hi *= 1.5
            phi = black76_price(df, Fwd, K, T, hi, is_call)
            if plo <= price <= phi:
                break
        else:
            return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pm = black76_price(df, Fwd, K, T, mid, is_call)
        if abs(pm - price) <= tol * max(1.0, price):
            return mid
        if pm < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def compute_iv_vector(prices, df, Fwd, K, T, is_call):
    out = np.full_like(prices, np.nan, dtype=np.float64)
    for i in range(len(prices)):
        out[i] = black76_implied_vol(float(prices[i]), df, Fwd, float(K[i]), T, bool(is_call[i]))
    return out


# -----------------------------
# Grid / interpolation
# -----------------------------
def make_fixed_logX_grid(xmin, xmax, n):
    z_min = math.log(xmin)
    z_max = math.log(xmax)
    z = np.linspace(z_min, z_max, n, dtype=np.float64)
    dz = float(z[1] - z[0])
    return z, dz

def interp_1d_torch_zero(x_src, y_src, x_tgt):
    n = x_src.numel()
    out = torch.zeros_like(x_tgt)
    in_range = (x_tgt >= x_src[0]) & (x_tgt <= x_src[-1])
    if torch.any(in_range):
        xt = x_tgt[in_range]
        idx = torch.searchsorted(x_src, xt).clamp(1, n - 1)
        x0 = x_src[idx - 1]; x1 = x_src[idx]
        y0 = y_src[idx - 1]; y1 = y_src[idx]
        w = (xt - x0) / (x1 - x0 + EPS)
        out[in_range] = y0 * (1.0 - w) + y1 * w
    return out

def normalize_density_z(p_z, dz):
    p = torch.clamp(p_z, min=0.0)
    Z = torch.sum(p) * dz
    return p / (Z + EPS_STATE)

def filter_df_by_logmny(df, Fwd, kmin, kmax):
    d = df.copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["strike"])
    F = float(Fwd)
    if not (np.isfinite(F) and F > 0):
        return d.iloc[0:0].copy()
    d["log_mny"] = np.log(d["strike"].to_numpy(dtype=np.float64) / F)
    d = d[(d["log_mny"] >= float(kmin)) & (d["log_mny"] <= float(kmax))].copy()
    return d


# -----------------------------
# Local vol NN
# -----------------------------
class LocalVolNN(nn.Module):
    def __init__(self, hidden=64, depth=3, sigma_floor=1e-3, sigma_cap=25.0):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.sigma_floor = float(sigma_floor)
        self.sigma_cap = float(sigma_cap)

    def forward(self, logX):
        z = self.net(logX.reshape(-1, 1)).reshape(-1)
        sig = F.softplus(z) + self.sigma_floor
        return torch.clamp(sig, 1e-4, self.sigma_cap)


# -----------------------------
# Gauss-Legendre quadrature
# -----------------------------
def gauss_legendre_01(n):
    xs, ws = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (xs + 1.0)
    w = 0.5 * ws
    return s.astype(np.float64), w.astype(np.float64)

GL_S, GL_W = gauss_legendre_01(GL_N)
GL_S_T = torch.tensor(GL_S, device=DEVICE, dtype=torch.float32)
GL_W_T = torch.tensor(GL_W, device=DEVICE, dtype=torch.float32)


# -----------------------------
# Matrix exponential with Frechet derivative
# -----------------------------
class MatrixExpFrechet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        P = torch.matrix_exp(A)
        ctx.save_for_backward(A)
        return P

    @staticmethod
    def backward(ctx, grad_P):
        (A,) = ctx.saved_tensors
        with torch.no_grad():
            G = grad_P
            accum = torch.zeros_like(A)
            for s, w in zip(GL_S_T, GL_W_T):
                E_left  = torch.matrix_exp((1.0 - s) * A).transpose(0, 1)
                E_right = torch.matrix_exp(s * A).transpose(0, 1)
                accum += w * (E_left @ G @ E_right)
        return (accum,)

def expm_frechet(A):
    return MatrixExpFrechet.apply(A)


# -----------------------------
# Driftless generator in logX
# -----------------------------
def build_backward_generator_logX(z, dz, sigma_X):
    n = z.numel()
    a = sigma_X**2
    b = -0.5 * a
    up   = 0.5 * a / (dz * dz) + b / (2.0 * dz)
    down = 0.5 * a / (dz * dz) - b / (2.0 * dz)
    diag = -(up + down)
    B = torch.zeros((n, n), device=z.device, dtype=z.dtype)
    i = torch.arange(1, n - 1, device=z.device)
    B[i, i - 1] = down[i]
    B[i, i]     = diag[i]
    B[i, i + 1] = up[i]
    B[0, 0] = diag[0] - down[0]
    B[0, 1] = up[0] + down[0]
    B[n - 1, n - 2] = up[n - 1] + down[n - 1]
    B[n - 1, n - 1] = diag[n - 1] - up[n - 1]
    return B


# -----------------------------
# Density transforms
# -----------------------------
def market_pz_from_qS(z_grid, dz, S_mkt, qS_mkt, Fwd_T):
    F_T = float(Fwd_T)
    X_pts = torch.exp(z_grid)
    S_pts = F_T * X_pts
    qS_on = interp_1d_torch_zero(S_mkt, qS_mkt, S_pts)
    pz_raw = F_T * qS_on * X_pts
    mass_before_norm = torch.sum(torch.clamp(pz_raw, min=0.0)) * dz
    pz = torch.clamp(pz_raw, min=0.0)
    pz = normalize_density_z(pz, dz)
    return pz, float(mass_before_norm.detach().cpu().item())

def qS_from_pz_on_mktS(z_grid, pz, S_mkt, Fwd_T):
    F_T = float(Fwd_T)
    z_pts = torch.log(torch.clamp(S_mkt / F_T, min=EPS))
    p_on = interp_1d_torch_zero(z_grid, pz, z_pts)
    qS = p_on / torch.clamp(S_mkt, min=EPS)
    qS = torch.clamp(qS, min=0.0)
    qS = qS / (torch.trapz(qS, S_mkt) + EPS)
    return qS


# -----------------------------
# Reghai Initial Guess (moment-based)
# -----------------------------
def compute_reghai_initial_guess(z_grid, pz_start, pz_end, dz, dt):
    with torch.no_grad():
        mean_start = torch.sum(z_grid * pz_start) * dz
        mean_end   = torch.sum(z_grid * pz_end) * dz
        var_start  = torch.sum((z_grid - mean_start)**2 * pz_start) * dz
        var_end    = torch.sum((z_grid - mean_end)**2 * pz_end) * dz
        delta_var  = var_end - var_start
        if delta_var > 0 and dt > 0:
            sigma_uniform = torch.sqrt(delta_var / dt)
        else:
            sigma_uniform = torch.sqrt(var_end / max(dt, 0.01))
        sigma_uniform = torch.clamp(sigma_uniform, min=0.05, max=2.0)
        sigma_guess = sigma_uniform * torch.ones_like(z_grid)
    return sigma_guess


# =====================================================================
# v2: Loss function (factored out for reuse in Adam and L-BFGS phases)
# =====================================================================
def compute_loss(nn_sigma, z, dz_t, pz_start, pz_target, dt, smooth_weight):
    """Compute KL + mean_pen + smooth loss. Returns loss, kl, mean_pen, EX, smooth."""
    sigma = nn_sigma(z)
    B = build_backward_generator_logX(z, dz_t, sigma_X=sigma)
    Q = B.transpose(0, 1)

    p_tmp = pz_start.clone()
    dt_sub = dt / float(N_SUBSTEPS)
    for _ in range(N_SUBSTEPS):
        A = dt_sub * Q
        P = expm_frechet(A)
        p_tmp = normalize_density_z(P @ p_tmp, dz_t)
        p_tmp = torch.clamp(p_tmp, min=EPS_STATE)
        p_tmp = p_tmp / (torch.sum(p_tmp) * dz_t + EPS_STATE)
    pz_model = p_tmp

    log_pz_model  = torch.log(torch.clamp(pz_model,  min=EPS_LOG))
    log_pz_target = torch.log(torch.clamp(pz_target, min=EPS_LOG))
    kl = torch.sum(pz_target * (log_pz_target - log_pz_model)) * dz_t

    X_grid = torch.exp(z)
    EX = torch.sum(X_grid * pz_model) * dz_t
    mean_pen = (EX - 1.0) ** 2

    d2 = (sigma[:-2] - 2.0 * sigma[1:-1] + sigma[2:]) / (dz_t * dz_t)
    smooth = torch.mean(d2 * d2)

    loss = KL_WEIGHT * kl + smooth_weight * smooth + MEAN_WEIGHT * mean_pen
    return loss, kl, mean_pen, EX, smooth


# -----------------------------
# Plot makers (unchanged from v1)
# -----------------------------
def make_price_compare_plot(df, title, ycol_model, ycol_market="mid", market_label=None):
    if market_label is None:
        market_label = ycol_market
    fig = plt.figure(figsize=(10, 5))
    for opt_type in ["call", "put"]:
        sub = df[df["type"] == opt_type]
        if sub.empty:
            continue
        plt.scatter(sub["strike"], sub[ycol_market], s=16, alpha=0.7, label=f"{market_label} ({opt_type})")
        plt.scatter(sub["strike"], sub[ycol_model],   s=16, alpha=0.7, label=f"{ycol_model} ({opt_type})")
    plt.xlabel("Strike"); plt.ylabel("Option price"); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.25); plt.tight_layout()
    return fig

def make_rnd_compare_plot(S, q_mkt, q_model, title):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(S, q_mkt, linewidth=1.5, label="market RND (NPZ)")
    plt.plot(S, q_model, linewidth=1.5, linestyle="--", label="model RND")
    plt.xlabel("S"); plt.ylabel("Density"); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.25); plt.tight_layout(); plt.xlim(0, 12500)
    return fig

def make_sigma_vs_S_plot(nn_sigma, Fwd, S_min, S_max, title):
    fig = plt.figure(figsize=(10, 5))
    S_plot = np.linspace(max(S_min, 1.0), S_max, PLOT_SIGMA_POINTS)
    S_t = torch.tensor(S_plot, device=DEVICE, dtype=torch.float32)
    z_plot = torch.log(S_t / float(Fwd))
    with torch.no_grad():
        sig_plot = nn_sigma(z_plot).cpu().numpy()
    plt.plot(S_plot, sig_plot, linewidth=1.5)
    plt.xlabel("S"); plt.ylabel("σ(S)"); plt.title(title)
    plt.grid(True, alpha=0.25); plt.tight_layout(); plt.xlim(0, 12500)
    return fig

def make_sigma_vs_z_plot(nn_sigma, z_grid_np, title):
    fig = plt.figure(figsize=(10, 5))
    z_t = torch.tensor(z_grid_np, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        sig = nn_sigma(z_t).cpu().numpy()
    plt.plot(z_grid_np, sig, linewidth=1.5)
    plt.xlabel("z = log(S/F)"); plt.ylabel("σ(z)"); plt.title(title)
    plt.grid(True, alpha=0.25); plt.tight_layout(); plt.xlim(-2, 2)
    return fig

def make_iv_compare_plot_logmny_scatter(df, title, Fwd, opt_type):
    d = df[df["type"] == opt_type].copy()
    if d.empty:
        fig = plt.figure(figsize=(10, 5))
        plt.title(title + f" [{opt_type}] (no data)")
        plt.xlabel("log-moneyness  log(K/F)"); plt.ylabel("Implied vol (Black-76)")
        plt.grid(True, alpha=0.25); plt.tight_layout()
        return fig
    F = float(Fwd)
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["strike"])
    d["log_mny"] = np.log(d["strike"].to_numpy(dtype=np.float64) / F)
    d["iv_mkt_b76"] = pd.to_numeric(d["iv_mkt_b76"], errors="coerce")
    d["iv_model_b76"] = pd.to_numeric(d["iv_model_b76"], errors="coerce")
    d = d[np.isfinite(d["log_mny"]) & np.isfinite(d["iv_mkt_b76"]) & np.isfinite(d["iv_model_b76"])].copy()
    d = d.sort_values("log_mny")
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(d["log_mny"], d["iv_mkt_b76"], s=14, alpha=0.7, label="Market IV")
    plt.scatter(d["log_mny"], d["iv_model_b76"], s=14, alpha=0.7, label="Model IV")
    plt.xlabel("log-moneyness  log(K/F)"); plt.ylabel("Implied vol (Black-76)")
    plt.title(title + f" [{opt_type}]  N={len(d)}")
    plt.legend(); plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.xlim(LOGMNY_MIN, LOGMNY_MAX)
    return fig


# =====================================================================
# Main
# =====================================================================
def main():
    # ---------- Load market NPZs ----------
    tenors_sorted = sorted(NPZ_BY_TENOR.keys())
    market = {}
    for m in tenors_sorted:
        fname = NPZ_BY_TENOR[m]
        if not os.path.isfile(fname):
            print(f"[Warning] NPZ not found: {fname}, skipping {m}M")
            continue
        market[m] = load_npz_market(fname)
        print(f"[Loaded] {m}M: {fname}  T={market[m]['T']:.6f}")

    tenors_sorted = [m for m in tenors_sorted if m in market]
    if not tenors_sorted:
        raise RuntimeError("No NPZ files found.")

    # ---------- Load options CSV ----------
    if os.path.isfile(OPTIONS_CSV):
        opt = pd.read_csv(OPTIONS_CSV)
        opt.columns = opt.columns.str.strip().str.lower()
        if "tenor_months" not in opt.columns:
            opt["tenor_months"] = pd.to_numeric(opt.get("tenor_months", pd.Series([np.nan]*len(opt))), errors="coerce")
    else:
        opt = pd.DataFrame()
        print(f"[Warning] Options CSV not found: {OPTIONS_CSV}")

    # ---------- Fixed logX grid ----------
    z_np, dz = make_fixed_logX_grid(X_MIN, X_MAX, N_Z)
    z = torch.tensor(z_np, device=DEVICE, dtype=torch.float32)
    dz_t = float(dz)

    # ---------- Convert all market RNDs to z-grid ----------
    pz_market_by_tenor = {}
    trunc_mass_by_tenor = {}

    for m in tenors_sorted:
        d = market[m]
        Fwd = float(d.get("forward", np.nan))
        if not (np.isfinite(Fwd) and Fwd > 0):
            raise ValueError(f"Invalid forward for {m}M")
        S_mkt = torch.tensor(d["xg"], device=DEVICE, dtype=torch.float32)
        q_mkt = torch.tensor(d["q"], device=DEVICE, dtype=torch.float32)
        q_mkt = q_mkt / (torch.trapz(q_mkt, S_mkt) + EPS)
        pz_mkt, mass_before_norm = market_pz_from_qS(z, dz_t, S_mkt, q_mkt, Fwd_T=Fwd)
        pz_market_by_tenor[m] = pz_mkt
        trunc_mass_by_tenor[m] = mass_before_norm
        print(f"[Info] Market pz for {m}M: sum={torch.sum(pz_mkt).item()*dz_t:.6f} | "
              f"mass_before_norm(trunc)={mass_before_norm:.6f}")

    T_by_tenor   = {m: float(market[m]["T"]) for m in tenors_sorted}
    df_by_tenor  = {m: float(market[m]["df"]) for m in tenors_sorted}
    fwd_by_tenor = {m: float(market[m].get("forward", np.nan)) for m in tenors_sorted}

    # ======================================================================
    # Reghai calibration (v2: with warm start + two-phase optimizer)
    # ======================================================================
    print("\n" + "="*80)
    print("REGHAI'S SCHRÖDINGER LOCAL VOLATILITY CALIBRATION (v2)")
    print("  Warm start: ON | Mean weight: " + str(MEAN_WEIGHT) + " | Two-phase: Adam+L-BFGS")
    print("="*80)

    interval_models = []
    prev_T = 0.0
    prev_tenor = None

    for idx, m in enumerate(tenors_sorted):
        T = T_by_tenor[m]
        dt = T - prev_T
        if dt <= 0:
            raise ValueError(f"Non-increasing maturities: prev_T={prev_T}, T={T} at {m}M")

        df_T  = df_by_tenor[m]
        Fwd_T = fwd_by_tenor[m]
        d_mkt = market[m]
        expiry = d_mkt.get("expiry")

        if prev_tenor is None:
            pz_start = make_dirac_prior_pz(z, dz_t, z0=0.0)
            print(f"\n  [Interval 0 -> {m}M] Starting from Dirac prior at z=0 (S=F)")
        else:
            pz_start = pz_market_by_tenor[prev_tenor].clone()
            print(f"\n  [Interval {prev_tenor}M -> {m}M] Starting from MARKET RND at {prev_tenor}M")

        pz_target = pz_market_by_tenor[m].clone()
        print(f"    dt={dt:.6f} T={T:.6f} df={df_T:.6f} Fwd={Fwd_T:.3f} expiry={expiry}")

        nn_sigma = LocalVolNN(hidden=64, depth=5, sigma_floor=1e-4, sigma_cap=25.0).to(DEVICE)

        # v2: Warm start from previous interval
        if WARM_START_PREV_INTERVAL and interval_models:
            nn_sigma.load_state_dict(interval_models[-1].state_dict())
            print("    [Init] warm-started from previous interval sigma")

        if WARMUP_TO_MOMENT_GUESS:
            sigma_guess = compute_reghai_initial_guess(z, pz_start, pz_target, dz_t, dt)
            warmup_optim = torch.optim.Adam(nn_sigma.parameters(), lr=1e-2)
            for _ in range(100):
                warmup_optim.zero_grad()
                warmup_loss = torch.mean((nn_sigma(z) - sigma_guess)**2)
                warmup_loss.backward()
                warmup_optim.step()
            print(f"    After warmup: σ mean={nn_sigma(z).mean().item():.4f}")

        smooth_weight = SMOOTH_WEIGHT if dt > 0.2 else SMOOTH_WEIGHT * 0.1
        best_loss = float("inf")
        best_state = None

        # ============================================================
        # Phase 1: Adam
        # ============================================================
        adam_epochs = EPOCHS_PER_INTERVAL - 50  # 2450 Adam steps
        optm = torch.optim.Adam(nn_sigma.parameters(), lr=LR)

        print(f"    Phase 1: Adam for {adam_epochs} epochs...")
        for epoch in range(1, adam_epochs + 1):
            optm.zero_grad(set_to_none=True)
            loss, kl, mean_pen, EX, smooth = compute_loss(
                nn_sigma, z, dz_t, pz_start, pz_target, dt, smooth_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_sigma.parameters(), 10.0)
            optm.step()

            L = float(loss.item())
            if L < best_loss:
                best_loss = L
                best_state = {k: v.detach().clone() for k, v in nn_sigma.state_dict().items()}

            if epoch % PRINT_EVERY == 0 or epoch == 1:
                print(
                    f"    epoch {epoch:4d} | loss={L:.6e} | KL={float(kl.item()):.6e} "
                    f"| mean_pen={float(mean_pen.item()):.3e} EX={float(EX.item()):.6f} "
                    f"| smooth={float(smooth.item()):.6e}"
                )

        # ============================================================
        # Phase 2: L-BFGS refinement (v2)
        # ============================================================
        lbfgs_epochs = EPOCHS_PER_INTERVAL - adam_epochs
        if best_state is not None:
            nn_sigma.load_state_dict(best_state)  # start from best Adam result

        lbfgs_optim = torch.optim.LBFGS(
            nn_sigma.parameters(), lr=1e-2,
            max_iter=20, line_search_fn="strong_wolfe"
        )

        print(f"    Phase 2: L-BFGS for {lbfgs_epochs} steps...")
        for lbfgs_ep in range(1, lbfgs_epochs + 1):
            def closure():
                lbfgs_optim.zero_grad(set_to_none=True)
                loss, _, _, _, _ = compute_loss(
                    nn_sigma, z, dz_t, pz_start, pz_target, dt, smooth_weight)
                loss.backward()
                return loss

            L_val = lbfgs_optim.step(closure)
            L = float(L_val.item()) if L_val is not None else float('inf')

            if L < best_loss:
                best_loss = L
                best_state = {k: v.detach().clone() for k, v in nn_sigma.state_dict().items()}

            if lbfgs_ep % 100 == 0 or lbfgs_ep == 1 or lbfgs_ep == lbfgs_epochs:
                with torch.no_grad():
                    _, kl, mean_pen, EX, smooth = compute_loss(
                        nn_sigma, z, dz_t, pz_start, pz_target, dt, smooth_weight)
                print(
                    f"    L-BFGS {lbfgs_ep:4d} | loss={L:.6e} | KL={float(kl.item()):.6e} "
                    f"| mean_pen={float(mean_pen.item()):.3e} EX={float(EX.item()):.6f}"
                )

        print(f"    Best loss: {best_loss:.6e}")

        if best_state is None:
            raise RuntimeError("Optimization failed: best_state is None.")
        nn_sigma.load_state_dict(best_state)
        nn_sigma.eval()

        # Save generator
        if SAVE_TRIDIAG_Q_NPZ:
            with torch.no_grad():
                sigma = nn_sigma(z)
                B = build_backward_generator_logX(z, dz_t, sigma_X=sigma)
                Q = B.transpose(0, 1)
                q_lower = torch.diagonal(Q, offset=-1).detach().cpu().numpy()
                q_diag  = torch.diagonal(Q, offset=0).detach().cpu().numpy()
                q_upper = torch.diagonal(Q, offset=+1).detach().cpu().numpy()
            out_name = f"Q_tridiag_{m}M_{sanitize(expiry)}.npz"
            out_path = os.path.join(TRIDIAG_Q_NPZ_DIR, out_name)
            save_tridiag_generator_npz(
                out_path, expiry=str(expiry), tenor_months=int(m),
                T=float(T), dt=float(dt), n_substeps=int(N_SUBSTEPS),
                z_grid=z_np, dz=float(dz_t),
                lower=q_lower, diag=q_diag, upper=q_upper)
            print(f"    [Saved] Generator: {out_path}")

        # Save local vol
        if SAVE_LV_NPZ:
            with torch.no_grad():
                sigma_z = nn_sigma(z).detach().cpu().numpy()
                S_mkt_np = d_mkt["xg"].astype(np.float64)
                S_t = torch.tensor(S_mkt_np, device=DEVICE, dtype=torch.float32)
                zS = torch.log(torch.clamp(S_t / float(Fwd_T), min=EPS))
                sigma_S = nn_sigma(zS).detach().cpu().numpy()
            out_name = f"localvol_{m}M_{sanitize(expiry)}.npz"
            out_path = os.path.join(LV_NPZ_DIR, out_name)
            save_localvol_npz(
                out_path, expiry=str(expiry), tenor_months=int(m),
                T=float(T), dt=float(dt), df=float(df_T), forward=float(Fwd_T),
                z_grid=z_np, sigma_z=sigma_z, S_grid=S_mkt_np, sigma_S=sigma_S)
            print(f"    [Saved] Local vol: {out_path}")

        interval_models.append(nn_sigma)
        prev_T = T
        prev_tenor = m

    # ======================================================================
    # Post-calibration chaining
    # ======================================================================
    print("\n[Step] Post-calibration: compute model densities by chaining intervals...")
    pz0 = make_dirac_prior_pz(z, dz_t, z0=0.0)
    pz_by_tenor = {}
    p_curr = pz0.clone()
    prev_T = 0.0

    for k, m in enumerate(tenors_sorted):
        T = T_by_tenor[m]
        dt = T - prev_T
        nn_sigma = interval_models[k].to(DEVICE)
        with torch.no_grad():
            sigma = nn_sigma(z)
            B = build_backward_generator_logX(z, dz_t, sigma_X=sigma)
            Q = B.transpose(0, 1)
            dt_sub = dt / float(N_SUBSTEPS)
            for _ in range(N_SUBSTEPS):
                p_curr = torch.matrix_exp(dt_sub * Q) @ p_curr
                p_curr = normalize_density_z(p_curr, dz_t)
                p_curr = torch.clamp(p_curr, min=EPS)
                p_curr = p_curr / (torch.sum(p_curr) * dz_t + EPS)
            pz_by_tenor[m] = p_curr.detach().clone()

            # v2: Report E[S/F] at each pillar
            EX = torch.sum(torch.exp(z) * p_curr) * dz_t
            print(f"  {m:2d}M: mass={torch.sum(p_curr).item()*dz_t:.6f} E[S/F]={EX.item():.6f}")
        prev_T = T

    # ======================================================================
    # Post-calibration comparisons
    # ======================================================================
    print("\n[Step] Post-calibration: model prices + implied vols vs CSV...")
    print("-" * 132)
    print(f"{'Tenor':>6} {'Expiry':>12} {'RMSE(CSV,Model)':>16} {'RMSE(CSV,MktPZ)':>16} {'RMSE(Model,MktPZ)':>18} {'N':>6}  {'trunc_mass':>10}")
    print("-" * 132)

    for m in tenors_sorted:
        d = market[m]
        T = float(d["T"]); dfT = float(d["df"]); FwdT = float(d.get("forward", np.nan))
        expiry = d.get("expiry")
        trunc_mass = float(trunc_mass_by_tenor.get(m, float("nan")))
        if not (np.isfinite(FwdT) and FwdT > 0):
            continue

        S_mkt_np = d["xg"]
        q_mkt_np = d["q"].astype(np.float64)
        q_mkt_np = q_mkt_np / (np.trapezoid(q_mkt_np, S_mkt_np) + EPS)

        S_mkt_t = torch.tensor(S_mkt_np, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            q_model_t = qS_from_pz_on_mktS(z, pz_by_tenor[m].to(DEVICE), S_mkt_t, Fwd_T=FwdT)
        q_model_np = q_model_t.detach().cpu().numpy()

        if SAVE_MODEL_NPZ:
            out_name = f"model_rnd_{m}M_{sanitize(expiry)}.npz"
            out_path = os.path.join(MODEL_NPZ_DIR, out_name)
            save_model_rnd_npz(out_path, expiry=str(expiry), T=float(T), df=float(dfT),
                forward=float(FwdT), z_grid=z_np, pz=pz_by_tenor[m].detach().cpu().numpy(),
                S_grid=S_mkt_np, qS=q_model_np)

        fig_rnd = make_rnd_compare_plot(S_mkt_np, q_mkt_np, q_model_np,
            title=f"{m}M RND: market vs model | expiry {expiry}")
        save_or_show(fig_rnd, os.path.join(PLOTS_DIR, f"post_rnd_compare_{m}M_{sanitize(expiry)}.png"))

        nn_sigma = interval_models[tenors_sorted.index(m)]
        fig_sig = make_sigma_vs_S_plot(nn_sigma, FwdT, float(np.min(S_mkt_np)),
            float(np.max(S_mkt_np)), title=f"{m}M σ(S) | expiry {expiry}")
        save_or_show(fig_sig, os.path.join(PLOTS_DIR, f"post_sigma_S_{m}M_{sanitize(expiry)}.png"))

        fig_sigz = make_sigma_vs_z_plot(nn_sigma, z_grid_np=z_np,
            title=f"{m}M σ(z) | expiry {expiry}")
        save_or_show(fig_sigz, os.path.join(PLOTS_DIR, f"post_sigma_z_{m}M_{sanitize(expiry)}.png"))

        sub = opt[opt["tenor_months"] == m].copy() if not opt.empty else pd.DataFrame()
        if sub.empty:
            print(f"{m:>6} {str(expiry):>12} {'N/A':>16} {'N/A':>16} {'N/A':>18} {0:>6}  {trunc_mass:>10.6f}")
            continue

        MID_COL = "adjusted_mid" if ("adjusted_mid" in sub.columns) else "mid"
        sub["strike"] = pd.to_numeric(sub["strike"], errors="coerce")
        sub["type"] = sub["type"].astype(str).str.lower()
        sub[MID_COL] = pd.to_numeric(sub[MID_COL], errors="coerce")
        sub = sub.dropna(subset=["strike", MID_COL, "type"])
        sub = sub[sub["type"].isin(["call", "put"])].copy()
        sub = sub[np.isfinite(sub[MID_COL]) & (sub[MID_COL] > 0.0)].copy()
        if sub.empty:
            print(f"{m:>6} {str(expiry):>12} {'N/A':>16} {'N/A':>16} {'N/A':>18} {0:>6}  {trunc_mass:>10.6f}")
            continue

        K = sub["strike"].to_numpy(dtype=np.float64)
        is_call = (sub["type"].values == "call")

        pz_T = pz_by_tenor[m].to(DEVICE)
        call_model, put_model = price_from_pz_call_put(z, pz_T, K, dfT, FwdT, dz_t)
        model_price = np.where(is_call, call_model, put_model)
        sub["model_price"] = model_price

        pz_mkt = pz_market_by_tenor[m].to(DEVICE)
        call_mktpz, put_mktpz = price_from_pz_call_put(z, pz_mkt, K, dfT, FwdT, dz_t)
        mktpz_price = np.where(is_call, call_mktpz, put_mktpz)
        sub["mktpz_price"] = mktpz_price

        mid_pv   = sub[MID_COL].to_numpy(np.float64)
        model_pv = sub["model_price"].to_numpy(np.float64)
        mid_fwd   = mid_pv   / float(dfT)
        model_fwd = model_pv / float(dfT)
        iv_mkt_b76   = compute_iv_vector(mid_fwd,   df=1.0, Fwd=FwdT, K=K, T=T, is_call=is_call)
        iv_model_b76 = compute_iv_vector(model_fwd, df=1.0, Fwd=FwdT, K=K, T=T, is_call=is_call)
        sub["iv_mkt_b76"]   = iv_mkt_b76
        sub["iv_model_b76"] = iv_model_b76

        both_ok = np.isfinite(sub["iv_mkt_b76"].to_numpy()) & np.isfinite(sub["iv_model_b76"].to_numpy())
        sub_ok = sub[both_ok].copy()
        if PLOT_IV_LOGMNY_ONLY:
            sub_ok = filter_df_by_logmny(sub_ok, Fwd=FwdT, kmin=LOGMNY_MIN, kmax=LOGMNY_MAX)

        rmse_csv_model = float(np.sqrt(np.nanmean((sub[MID_COL].to_numpy() - sub["model_price"].to_numpy()) ** 2)))
        rmse_csv_mktpz = float(np.sqrt(np.nanmean((sub[MID_COL].to_numpy() - sub["mktpz_price"].to_numpy()) ** 2)))
        rmse_model_mktpz = float(np.sqrt(np.nanmean((sub["model_price"].to_numpy() - sub["mktpz_price"].to_numpy()) ** 2)))

        print(f"{m:>6} {str(expiry):>12} {rmse_csv_model:>16.6f} {rmse_csv_mktpz:>16.6f} {rmse_model_mktpz:>18.6f} {len(sub):>6}  {trunc_mass:>10.6f}")

        fig_p = make_price_compare_plot(sub,
            title=f"{m}M: market vs MODEL | RMSE={rmse_csv_model:.4f} | {expiry}",
            ycol_model="model_price", ycol_market=MID_COL, market_label=f"CSV {MID_COL}")
        save_or_show(fig_p, os.path.join(PLOTS_DIR, f"post_price_model_vs_csv_{m}M_{sanitize(expiry)}.png"))

        iv_rmse = float(np.sqrt(np.nanmean((sub_ok["iv_mkt_b76"].to_numpy() - sub_ok["iv_model_b76"].to_numpy())**2))) if len(sub_ok) else float("nan")
        title_iv = f"{m}M IV: Market vs Model | IV_RMSE={iv_rmse:.6f} | {expiry}"
        for ot in ["call", "put"]:
            fig_iv = make_iv_compare_plot_logmny_scatter(sub_ok, title_iv, Fwd=FwdT, opt_type=ot)
            save_or_show(fig_iv, os.path.join(PLOTS_DIR, f"post_iv_{ot}_{m}M_{sanitize(expiry)}.png"))

    print("-" * 132)
    if SAVE_PLOTS:
        print(f"\n[Done] All plots saved to: {PLOTS_DIR}")
    else:
        print("\n[Done]")


if __name__ == "__main__":
    main()