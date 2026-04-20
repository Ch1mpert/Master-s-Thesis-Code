#!/usr/bin/env python3
"""
lv_pricer.py
============
Price European options using the LV (Local Volatility) tridiagonal generator
matrices, propagating density via scipy expm_multiply (one big step per
interval), and compare to market prices.

Adds bid/ask diagnostics (same idea as your LSV/MC “inside the spread”):
  - Per option: bid, ask, spread, inside_spread (0/1)
  - Per expiry (calls/puts separately): inside_pct, median_spread, median_half_spread
  - Plot: LV_inside_spread_rate_vs_time.png  (% inside bid-ask vs T)

Uses IDENTICAL conventions to the LSV pricer for fair comparison:
  - Day count: days / 365.0
  - Init density: SIGMA_INIT_Z = 0.01
  - Forward: log-linear interpolation
  - Discount: linear interpolation
  - Strikes: /1000, median agg, +/-50% band
  - Payoff: DF * sum(pT * payoff) * dz
"""

from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Configuration — IDENTICAL to LSV pricer
# ============================================================
DAY_COUNT_DENOM = 365.0
SIGMA_INIT_Z = 0.01
STRIKE_MONEYNESS_BAND = 0.75
MIN_T_YEARS = 0.0
MAX_T_YEARS = 2.0
OUTDIR = './market_vs_model_plot'
DPI = 150

# Propagation mode
PROPAGATION_MODE = "right_anchor"

# Q files
Q_FILES = [
    './data/Q_tridiag_1M_2025-02-03.npz',
    './data/Q_tridiag_3M_2025-03-31.npz',
    './data/Q_tridiag_6M_2025-06-30.npz',
    './data/Q_tridiag_12M_2025-12-31.npz',
    './data/Q_tridiag_24M_2026-12-18.npz',
]


# ============================================================
# Load generators
# ============================================================

def load_q_slices(data_dir, q_files):
    """Load tridiagonal Q generators from NPZ files."""
    slices = []
    for fname in q_files:
        path = os.path.join(data_dir, fname)
        d = np.load(path)
        z = d['z'].astype(float)
        dz = float(d['dz'])
        n = len(z)

        Q = sp.diags(
            [d['Q_lower'].astype(float),
             d['Q_diag'].astype(float),
             d['Q_upper'].astype(float)],
            [-1, 0, 1], shape=(n, n), format='csr')

        tenor = int(d['tenor_months']) if 'tenor_months' in d.files else 0
        T_anchor = float(d['T']) if 'T' in d.files else tenor / 12.0
        expiry = str(d['expiry']) if 'expiry' in d.files else fname

        slices.append({
            'tenor_months': tenor,
            'T_anchor': T_anchor,
            'expiry': expiry,
            'z': z, 'dz': dz, 'n': n,
            'Q': Q,
        })

    slices.sort(key=lambda s: s['T_anchor'])

    # Verify all share the same z-grid
    z0 = slices[0]['z']
    for s in slices[1:]:
        assert np.allclose(s['z'], z0), "All Q slices must share the same z-grid"

    return slices


# ============================================================
# Market data — IDENTICAL to LSV pricer, but keeps bid/ask too
# ============================================================

def act365_yearfrac(d0, d1):
    return max(0.0, (d1 - d0).days / DAY_COUNT_DENOM)


def load_forward_curve(path):
    fwd = pd.read_csv(path)
    T_grid = fwd['T_years'].values.astype(float)
    F_grid = fwd['forward_interp'].values.astype(float)
    logF = np.log(F_grid)

    def forward_at(T):
        if T <= T_grid[0]: return float(F_grid[0])
        if T >= T_grid[-1]: return float(F_grid[-1])
        return float(np.exp(np.interp(T, T_grid, logF)))
    return forward_at


def load_discount_curve(path):
    dc = pd.read_csv(path)
    T_grid = dc['T_years'].values.astype(float)
    DF_grid = dc['discount_factor'].values.astype(float)

    def df_at(T):
        if T <= T_grid[0]: return float(DF_grid[0])
        if T >= T_grid[-1]: return float(DF_grid[-1])
        return float(np.interp(T, T_grid, DF_grid))
    return df_at


def load_options(path, val_date_str='2025-01-02'):
    """
    Loads OptionMetrics-style options.csv, matching your LSV conventions.
    Keeps bid/ask columns (cleaned) so we can compute inside-spread metrics.
    """
    opt = pd.read_csv(path)
    opt['date'] = pd.to_datetime(opt['date'], errors='coerce').dt.date
    opt['exdate'] = pd.to_datetime(opt['exdate'], errors='coerce').dt.date

    val_date = dt.date.fromisoformat(val_date_str)
    opt = opt[opt['date'] == val_date].copy()

    # Strike: /1000 if large (OptionMetrics)
    raw_strikes = opt['strike_price'].astype(float).values
    if np.nanmedian(raw_strikes) > 50000:
        opt['strike'] = opt['strike_price'].astype(float) / 1000.0
    else:
        opt['strike'] = opt['strike_price'].astype(float)

    opt['cp_flag'] = opt['cp_flag'].astype(str).str.upper().str.strip()
    opt = opt[opt['cp_flag'].isin(['C', 'P'])].copy()

    # Clean bid/ask (do NOT fallback; if invalid, set NaN)
    bid = pd.to_numeric(opt['best_bid'], errors='coerce')
    ask = pd.to_numeric(opt['best_offer'], errors='coerce')
    valid_ba = np.isfinite(bid) & np.isfinite(ask) & (ask > 0) & (bid >= 0) & (ask >= bid)
    opt['bid'] = bid.where(valid_ba, np.nan)
    opt['ask'] = ask.where(valid_ba, np.nan)

    # --- Mid price used for calibration/RMSE ---
    # Fallback mid from bid/ask (kept for robustness and for when adjusted_mid is missing)
    mid_ba = (bid + ask) / 2.0
    mid_ba = mid_ba.where(np.isfinite(mid_ba) & (mid_ba > 0), np.nan)
    mid_ba = mid_ba.fillna(ask.where(np.isfinite(ask) & (ask > 0), np.nan))
    mid_ba = mid_ba.fillna(bid.where(np.isfinite(bid) & (bid > 0), np.nan))

    if 'adjusted_mid' in opt.columns:
        adj = pd.to_numeric(opt['adjusted_mid'], errors='coerce')
        adj = adj.where(np.isfinite(adj) & (adj > 0), np.nan)

        # Use adjusted_mid where available; fallback to bid/ask mid otherwise
        opt['mid'] = adj.fillna(mid_ba)

        # (optional) keep both for debugging
        opt['mid_raw_ba'] = mid_ba
        opt['mid_adjusted'] = adj
    else:
        opt['mid'] = mid_ba

    # T using act/365
    opt['T_years'] = opt['exdate'].apply(
        lambda d: act365_yearfrac(val_date, d) if isinstance(d, dt.date) else np.nan)

    # T range filter + valid mid
    opt = opt[(opt['T_years'] > MIN_T_YEARS) & (opt['T_years'] <= MAX_T_YEARS)].copy()
    opt = opt[np.isfinite(opt['mid']) & (opt['mid'] > 0)].copy()

    return opt, val_date


def get_chain(opt, T_target, fwd_at):
    """
    Extract call/put chains for a specific T, matching your conventions.
    Aggregates duplicate strikes by median. Keeps bid/ask medians too.
    """
    sub = opt[np.abs(opt['T_years'] - T_target) < 1e-8].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Aggregate duplicates by median (mid), and also median bid/ask for spread checks
    sub = sub.groupby(['cp_flag', 'strike'], as_index=False).agg(
        mid=('mid', 'median'),
        bid=('bid', 'median'),
        ask=('ask', 'median'),
    )

    # Moneyness band filter (same as before)
    F = fwd_at(T_target)
    K_lo = F * (1.0 - STRIKE_MONEYNESS_BAND)
    K_hi = F * (1.0 + STRIKE_MONEYNESS_BAND)
    sub = sub[(sub['strike'] >= K_lo) & (sub['strike'] <= K_hi)].copy()

    sub['spread'] = sub['ask'] - sub['bid']
    sub['half_spread'] = 0.5 * sub['spread']

    calls = sub[sub['cp_flag'] == 'C'].sort_values('strike').copy()
    puts = sub[sub['cp_flag'] == 'P'].sort_values('strike').copy()
    return calls, puts


# ============================================================
# Pricing — IDENTICAL formula to LSV pricer
# ============================================================

def price_options(z, dz, pT, F, DF, strikes, is_call):
    ST = F * np.exp(z)
    if is_call:
        payoff = np.maximum(ST[:, None] - strikes[None, :], 0.0)
    else:
        payoff = np.maximum(strikes[None, :] - ST[:, None], 0.0)
    prices = DF * (pT[:, None] * payoff).sum(axis=0) * dz
    return prices


def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.sum() else np.nan


def mae_fn(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.sum() else np.nan


def bias_fn(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[mask] - b[mask])) if mask.sum() else np.nan


def inside_spread_pct(model, bid, ask):
    """Fraction of strikes where bid <= model <= ask (ignores missing bid/ask)."""
    model = np.asarray(model, float)
    bid = np.asarray(bid, float)
    ask = np.asarray(ask, float)
    valid = np.isfinite(model) & np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)
    if valid.sum() == 0:
        return np.nan
    inside = (model[valid] >= bid[valid]) & (model[valid] <= ask[valid])
    return float(np.mean(inside))


def median_spread(bid, ask):
    bid = np.asarray(bid, float)
    ask = np.asarray(ask, float)
    valid = np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)
    if valid.sum() == 0:
        return np.nan, np.nan
    spr = ask[valid] - bid[valid]
    return float(np.nanmedian(spr)), float(np.nanmedian(0.5 * spr))


# ============================================================
# Propagation
# ============================================================

def active_slice_index(t, anchors, mode):
    """
    Return index of active generator at time t.
    left_anchor: Q_k active for (T_k, T_{k+1}]
    right_anchor: Q_k active for (T_{k-1}, T_k]
    """
    if mode == "left_anchor":
        idx = int(np.searchsorted(anchors, t, side='right') - 1)
        return min(max(idx, 0), len(anchors) - 1)
    else:  # right_anchor
        idx = int(np.searchsorted(anchors, t, side='left'))
        return min(max(idx, 0), len(anchors) - 1)


def propagate_to(p, t_cur, t_target, q_slices, anchors, mode):
    """
    Propagate density from t_cur to t_target using the appropriate Q.
    Uses one big expm_multiply call per generator interval.
    """
    while t_cur < t_target - 1e-15:
        idx = active_slice_index(t_cur + 1e-14, anchors, mode)
        Q = q_slices[idx]['Q']

        # Determine how far this Q is valid
        if mode == "left_anchor":
            if idx + 1 < len(anchors):
                t_boundary = anchors[idx + 1]
            else:
                t_boundary = np.inf
        else:
            t_boundary = anchors[idx]
            if t_boundary <= t_cur + 1e-15:
                t_boundary = anchors[min(idx + 1, len(anchors) - 1)]

        t_step_end = min(t_target, t_boundary)
        dt_step = t_step_end - t_cur

        if dt_step > 1e-15:
            p = expm_multiply(Q * dt_step, p)

        t_cur = t_step_end

    return p, t_cur


# ============================================================
# Main
# ============================================================

def main():
    t0_wall = time.time()

    data_dir = ''
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 70)
    print("LV PRICER (tridiagonal Q, expm_multiply)")
    print(f"  SIGMA_INIT_Z = {SIGMA_INIT_Z}")
    print(f"  DAY_COUNT    = days / {DAY_COUNT_DENOM}")
    print(f"  FWD INTERP   = log-linear")
    print(f"  MODE         = {PROPAGATION_MODE}")
    print("=" * 70)

    # Load Q slices
    q_slices = load_q_slices(data_dir, Q_FILES)
    z = q_slices[0]['z']
    dz = q_slices[0]['dz']
    n = len(z)
    anchors = np.array([s['T_anchor'] for s in q_slices])

    print(f"  z-grid: n={n}, range=[{z[0]:.4f}, {z[-1]:.4f}], dz={dz:.8f}")
    print(f"  Q anchors: {anchors}")
    for s in q_slices:
        print(f"    {s['tenor_months']:2d}M  T={s['T_anchor']:.6f}  "
              f"expiry={s['expiry']}  Q: {s['Q'].shape}")

    # Load market data
    fwd_at = load_forward_curve(os.path.join(data_dir, 'forward_curve_interpolated_daily.csv'))
    df_at  = load_discount_curve(os.path.join(data_dir, 'discount_curve_grid.csv'))
    opts, val_date = load_options(os.path.join(data_dir, 'options.csv'))

    unique_T = np.sort(opts['T_years'].unique())
    print(f"\n  Val date:  {val_date}")
    print(f"  Expiries:  {len(unique_T)}")
    print(f"  T range:   [{unique_T.min():.6f}, {unique_T.max():.6f}]")

    i0 = np.argmin(np.abs(z))    # grid point closest to z=0
    p0 = np.zeros(n)
    p0[i0] = 1.0 / dz            # delta approximation: integral = 1

    # ================================================================
    # Sequential propagation and pricing
    # ================================================================
    print("\n" + "-" * 70)
    print("PROPAGATING & PRICING (calls + puts)")
    print("-" * 70)

    option_rows = []
    summary_rows = []
    p_cur = p0.copy()
    t_cur = 0.0
    n_expiries = len(unique_T)

    for idx_T, T_target in enumerate(unique_T):
        # Propagate to this expiry
        t0_prop = time.time()
        p_cur, t_cur = propagate_to(p_cur, t_cur, T_target, q_slices, anchors, PROPAGATION_MODE)
        dt_prop = time.time() - t0_prop

        marginal = p_cur
        mass = marginal.sum() * dz
        F = fwd_at(T_target)
        DF = df_at(T_target)

        calls, puts = get_chain(opts, T_target, fwd_at)
        if calls.empty and puts.empty:
            continue

        call_rmse = call_mae = call_bias = np.nan
        put_rmse  = put_mae  = put_bias  = np.nan

        call_inside = put_inside = np.nan
        call_med_sp = call_med_hs = np.nan
        put_med_sp  = put_med_hs  = np.nan

        if not calls.empty:
            Kc = calls['strike'].values
            mkt_c = calls['mid'].values
            bid_c = calls['bid'].values
            ask_c = calls['ask'].values

            mdl_c = price_options(z, dz, marginal, F, DF, Kc, is_call=True)
            err_c = mdl_c - mkt_c

            call_rmse = rmse(mdl_c, mkt_c)
            call_mae  = mae_fn(mdl_c, mkt_c)
            call_bias = bias_fn(mdl_c, mkt_c)

            call_inside = inside_spread_pct(mdl_c, bid_c, ask_c)
            call_med_sp, call_med_hs = median_spread(bid_c, ask_c)

            valid_ba = np.isfinite(bid_c) & np.isfinite(ask_c) & (ask_c >= bid_c)
            inside_flags = np.zeros_like(mdl_c, dtype=int)
            inside_flags[valid_ba] = ((mdl_c[valid_ba] >= bid_c[valid_ba]) & (mdl_c[valid_ba] <= ask_c[valid_ba])).astype(int)

            for kk, m, bd, ak, md, e, ins in zip(Kc, mkt_c, bid_c, ask_c, mdl_c, err_c, inside_flags):
                spr = (ak - bd) if (np.isfinite(ak) and np.isfinite(bd)) else np.nan
                option_rows.append({
                    'T': T_target, 'F': F, 'DF': DF,
                    'cp_flag': 'C', 'strike': kk,
                    'bid': bd, 'ask': ak, 'spread': spr, 'half_spread': 0.5 * spr if np.isfinite(spr) else np.nan,
                    'mkt_mid': m, 'model': md, 'err': e,
                    'inside_spread': int(ins),
                    'mass': mass
                })

        if not puts.empty:
            Kp = puts['strike'].values
            mkt_p = puts['mid'].values
            bid_p = puts['bid'].values
            ask_p = puts['ask'].values

            mdl_p = price_options(z, dz, marginal, F, DF, Kp, is_call=False)
            err_p = mdl_p - mkt_p

            put_rmse = rmse(mdl_p, mkt_p)
            put_mae  = mae_fn(mdl_p, mkt_p)
            put_bias = bias_fn(mdl_p, mkt_p)

            put_inside = inside_spread_pct(mdl_p, bid_p, ask_p)
            put_med_sp, put_med_hs = median_spread(bid_p, ask_p)

            valid_ba = np.isfinite(bid_p) & np.isfinite(ask_p) & (ask_p >= bid_p)
            inside_flags = np.zeros_like(mdl_p, dtype=int)
            inside_flags[valid_ba] = ((mdl_p[valid_ba] >= bid_p[valid_ba]) & (mdl_p[valid_ba] <= ask_p[valid_ba])).astype(int)

            for kk, m, bd, ak, md, e, ins in zip(Kp, mkt_p, bid_p, ask_p, mdl_p, err_p, inside_flags):
                spr = (ak - bd) if (np.isfinite(ak) and np.isfinite(bd)) else np.nan
                option_rows.append({
                    'T': T_target, 'F': F, 'DF': DF,
                    'cp_flag': 'P', 'strike': kk,
                    'bid': bd, 'ask': ak, 'spread': spr, 'half_spread': 0.5 * spr if np.isfinite(spr) else np.nan,
                    'mkt_mid': m, 'model': md, 'err': e,
                    'inside_spread': int(ins),
                    'mass': mass
                })

        summary_rows.append({
            'T': T_target, 'F': F, 'DF': DF,
            'n_calls': len(calls), 'n_puts': len(puts),
            'call_rmse': call_rmse, 'call_mae': call_mae, 'call_bias': call_bias,
            'put_rmse': put_rmse, 'put_mae': put_mae, 'put_bias': put_bias,
            'call_inside_pct': call_inside,
            'put_inside_pct': put_inside,
            'call_median_spread': call_med_sp,
            'put_median_spread': put_med_sp,
            'call_median_half_spread': call_med_hs,
            'put_median_half_spread': put_med_hs,
            'mass': mass
        })

        if idx_T % 8 == 0 or idx_T < 2 or idx_T == n_expiries - 1:
            ci = (100.0 * call_inside) if np.isfinite(call_inside) else np.nan
            pi = (100.0 * put_inside) if np.isfinite(put_inside) else np.nan
            print(f"  [{idx_T+1:3d}/{n_expiries}] T={T_target:.6f}  "
                  f"F={F:.2f}  DF={DF:.6f}  mass={mass:.6f}  "
                  f"C({len(calls)})={call_rmse:.4f}  P({len(puts)})={put_rmse:.4f}  "
                  f"inside% C={ci:.1f} P={pi:.1f}  prop={dt_prop:.2f}s")

    # ================================================================
    # Save
    # ================================================================
    df_err = pd.DataFrame(option_rows)
    df_sum = pd.DataFrame(summary_rows).sort_values('T').reset_index(drop=True)

    csv_err = os.path.join(OUTDIR, 'LV_all_maturities_option_errors.csv')
    csv_sum = os.path.join(OUTDIR, 'LV_error_by_expiry.csv')
    df_err.to_csv(csv_err, index=False)
    df_sum.to_csv(csv_sum, index=False)
    print(f"\nSaved: {csv_err}")
    print(f"Saved: {csv_sum}")

    # ================================================================
    # Print summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY BY EXPIRY")
    print("=" * 70)
    print(f"{'T':>9} {'#C':>4} {'#P':>4} {'C_RMSE':>9} {'C_MAE':>9} "
          f"{'P_RMSE':>9} {'P_MAE':>9} {'C_in%':>7} {'P_in%':>7} {'mass':>8}")
    print("-" * 70)
    for _, row in df_sum.iterrows():
        ci = 100.0 * row['call_inside_pct'] if np.isfinite(row['call_inside_pct']) else np.nan
        pi = 100.0 * row['put_inside_pct']  if np.isfinite(row['put_inside_pct'])  else np.nan
        print(f"{row['T']:9.6f} {int(row['n_calls']):4d} {int(row['n_puts']):4d} "
              f"{row['call_rmse']:9.4f} {row['call_mae']:9.4f} "
              f"{row['put_rmse']:9.4f} {row['put_mae']:9.4f} "
              f"{ci:7.1f} {pi:7.1f} "
              f"{row['mass']:8.6f}")

    # ================================================================
    # Plots: LV standalone
    # ================================================================
    print("\nGenerating LV plots...")
    _plot_lv(df_sum, df_err, anchors, OUTDIR)
    _plot_inside_spread(df_sum, anchors, OUTDIR)

    # NEW: OTM-only inside-spread plot (computed from df_err; keeps everything else unchanged)
    _plot_inside_spread_otm(df_err, anchors, OUTDIR)

    # ================================================================
    # Combined LV vs LSV plot (if LSV results exist)
    # ================================================================
    lsv_sum_path = os.path.join(OUTDIR, 'LSV_error_by_expiry.csv')
    lsv_err_path = os.path.join(OUTDIR, 'LSV_all_maturities_option_errors.csv')
    if os.path.exists(lsv_sum_path):
        print("\nGenerating combined LV vs LSV comparison...")
        lsv_sum = pd.read_csv(lsv_sum_path)
        lsv_err = pd.read_csv(lsv_err_path) if os.path.exists(lsv_err_path) else pd.DataFrame()
        _plot_combined(df_sum, df_err, lsv_sum, lsv_err, anchors, OUTDIR)
    else:
        print("\n  (LSV results not found — run lsv_ctmc_pricer.py first for combined plot)")

    print(f"\nTotal elapsed: {time.time() - t0_wall:.1f}s")


# ============================================================
# Plotting
# ============================================================

def _plot_lv(df_sum, df_err, anchors, outdir):
    """LV-only T vs error plots."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    ax = axes[0]
    ax.plot(df_sum['T'], df_sum['call_rmse'], 'o-', ms=4, lw=1.5,
            color='steelblue', label='Call RMSE')
    ax.plot(df_sum['T'], df_sum['put_rmse'], 's-', ms=4, lw=1.5,
            color='darkred', label='Put RMSE')
    ax.plot(df_sum['T'], df_sum['call_mae'], 'o--', ms=3, lw=1, alpha=0.7,
            color='steelblue', label='Call MAE')
    ax.plot(df_sum['T'], df_sum['put_mae'], 's--', ms=3, lw=1, alpha=0.7,
            color='darkred', label='Put MAE')
    for a in anchors:
        ax.axvline(a, color='green', alpha=0.35, lw=1, ls=':')
    ax.set_xlabel('T (years)'); ax.set_ylabel('Pricing Error')
    ax.set_title('LV: T vs Pricing Error (Calls + Puts)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if not df_err.empty:
        c = df_err[df_err['cp_flag'] == 'C']
        p = df_err[df_err['cp_flag'] == 'P']
        ax.scatter(c['T'], c['err'], s=6, alpha=0.15, c='steelblue', label='Calls')
        ax.scatter(p['T'], p['err'], s=6, alpha=0.15, c='darkred', label='Puts')
    ax.axhline(0, color='k', lw=0.5)
    for a in anchors:
        ax.axvline(a, color='green', alpha=0.35, lw=1, ls=':')
    ax.set_xlabel('T (years)'); ax.set_ylabel('Model - Market')
    ax.set_title('LV: Individual Option Errors vs T')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if len(df_err) > 0:
        lo, hi = np.percentile(df_err['err'].dropna(), [1, 99])
        ax.set_ylim(lo * 1.3, hi * 1.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'LV_T_vs_pricing_error.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def _plot_inside_spread(df_sum, anchors, outdir):
    """% inside bid-ask vs T (calls and puts)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    if 'call_inside_pct' in df_sum.columns:
        ax.plot(df_sum['T'], 100.0 * df_sum['call_inside_pct'],
                'o-', ms=4, lw=1.8, color='steelblue', label='Calls inside bid-ask (%)')
    if 'put_inside_pct' in df_sum.columns:
        ax.plot(df_sum['T'], 100.0 * df_sum['put_inside_pct'],
                's-', ms=4, lw=1.8, color='darkred', label='Puts inside bid-ask (%)')

    for a in anchors:
        ax.axvline(a, color='green', alpha=0.30, lw=1, ls=':')

    ax.set_xlabel('T (years)')
    ax.set_ylabel('Inside bid-ask (%)')
    ax.set_title('LV: Inside-the-Spread Rate vs T')
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, 'LV_inside_spread_rate_vs_time.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def _plot_inside_spread_otm(df_err, anchors, outdir):
    """
    NEW: OTM-only % inside bid-ask vs T (calls and puts), computed from df_err
    so that the core pricing loop + summary CSVs remain unchanged.

      - Call is OTM if strike > F
      - Put  is OTM if strike < F
      - Requires finite bid/ask with ask>=bid and finite model.
    """
    if df_err is None or df_err.empty:
        print("  (Skipping OTM inside-spread plot: df_err empty)")
        return

    d = df_err.copy()

    # Valid bid/ask and model
    valid = (
        np.isfinite(d['bid'].values) &
        np.isfinite(d['ask'].values) &
        np.isfinite(d['model'].values) &
        (d['ask'].values >= d['bid'].values)
    )
    d = d.loc[valid].copy()
    if d.empty:
        print("  (Skipping OTM inside-spread plot: no valid bid/ask rows)")
        return

    # OTM mask
    is_call = (d['cp_flag'].values.astype(str) == 'C')
    K = d['strike'].values.astype(float)
    F = d['F'].values.astype(float)
    otm = np.where(is_call, K > F, K < F)

    d = d.loc[otm].copy()
    if d.empty:
        print("  (Skipping OTM inside-spread plot: no OTM rows with valid bid/ask)")
        return

    inside = (d['model'].values >= d['bid'].values) & (d['model'].values <= d['ask'].values)
    d['inside_otm'] = inside.astype(float)

    g = d.groupby(['T', 'cp_flag'], as_index=False)['inside_otm'].mean()

    # Prepare series
    calls = g[g['cp_flag'] == 'C'].sort_values('T')
    puts  = g[g['cp_flag'] == 'P'].sort_values('T')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    if not calls.empty:
        ax.plot(calls['T'], 100.0 * calls['inside_otm'],
                'o-', ms=4, lw=1.8, color='steelblue', label='OTM Calls inside bid-ask (%)')
    if not puts.empty:
        ax.plot(puts['T'], 100.0 * puts['inside_otm'],
                's-', ms=4, lw=1.8, color='darkred', label='OTM Puts inside bid-ask (%)')

    for a in anchors:
        ax.axvline(a, color='green', alpha=0.30, lw=1, ls=':')

    ax.set_xlabel('T (years)')
    ax.set_ylabel('Inside bid-ask (%)')
    ax.set_title('LV: OTM Inside-the-Spread Rate vs T')
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, 'LV_inside_spread_rate_vs_time_OTM.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def _plot_combined(lv_sum, lv_err, lsv_sum, lsv_err, anchors, outdir):
    """Combined LV vs LSV comparison on the same axes (existing plot)."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # --- Call RMSE/MAE ---
    ax = axes[0, 0]
    ax.plot(lv_sum['T'], lv_sum['call_rmse'], 'o-', ms=4, lw=1.5,
            color='steelblue', label='LV Call RMSE')
    ax.plot(lsv_sum['T'], lsv_sum['call_rmse'], 's-', ms=4, lw=1.5,
            color='red', label='LSV Call RMSE')
    ax.plot(lv_sum['T'], lv_sum['call_mae'], 'o--', ms=3, lw=1, alpha=0.6,
            color='steelblue', label='LV Call MAE')
    ax.plot(lsv_sum['T'], lsv_sum['call_mae'], 's--', ms=3, lw=1, alpha=0.6,
            color='red', label='LSV Call MAE')
    for a in anchors:
        ax.axvline(a, color='green', alpha=0.3, lw=1, ls=':')
    ax.set_xlabel('T (years)'); ax.set_ylabel('Pricing Error')
    ax.set_title('Calls: LV vs LSV-CTMC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Put RMSE/MAE ---
    ax = axes[0, 1]
    ax.plot(lv_sum['T'], lv_sum['put_rmse'], 'o-', ms=4, lw=1.5,
            color='steelblue', label='LV Put RMSE')
    ax.plot(lsv_sum['T'], lsv_sum['put_rmse'], 's-', ms=4, lw=1.5,
            color='red', label='LSV Put RMSE')
    ax.plot(lv_sum['T'], lv_sum['put_mae'], 'o--', ms=3, lw=1, alpha=0.6,
            color='steelblue', label='LV Put MAE')
    ax.plot(lsv_sum['T'], lsv_sum['put_mae'], 's--', ms=3, lw=1, alpha=0.6,
            color='red', label='LSV Put MAE')
    for a in anchors:
        ax.axvline(a, color='green', alpha=0.3, lw=1, ls=':')
    ax.set_xlabel('T (years)'); ax.set_ylabel('Pricing Error')
    ax.set_title('Puts: LV vs LSV-CTMC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Call error scatter ---
    ax = axes[1, 0]
    if not lv_err.empty:
        c = lv_err[lv_err['cp_flag'] == 'C']
        ax.scatter(c['T'], c['err'], s=4, alpha=0.12, c='steelblue', label='LV')
    if not lsv_err.empty:
        c = lsv_err[lsv_err['cp_flag'] == 'C']
        ax.scatter(c['T'], c['err'], s=4, alpha=0.12, c='red', label='LSV')
    ax.axhline(0, color='k', lw=0.5)
    for a in anchors:
        ax.axvline(a, color='green', alpha=0.3, lw=1, ls=':')
    ax.set_xlabel('T (years)'); ax.set_ylabel('Model - Market')
    ax.set_title('Call Errors: LV vs LSV-CTMC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    all_errs = []
    if not lv_err.empty: all_errs.extend(lv_err['err'].dropna().values)
    if not lsv_err.empty: all_errs.extend(lsv_err['err'].dropna().values)
    if all_errs:
        lo, hi = np.percentile(all_errs, [1, 99])
        ax.set_ylim(lo * 1.3, hi * 1.3)

    # --- Put error scatter ---
    ax = axes[1, 1]
    if not lv_err.empty:
        p = lv_err[lv_err['cp_flag'] == 'P']
        ax.scatter(p['T'], p['err'], s=4, alpha=0.12, c='steelblue', label='LV')
    if not lsv_err.empty:
        p = lsv_err[lsv_err['cp_flag'] == 'P']
        ax.scatter(p['T'], p['err'], s=4, alpha=0.12, c='red', label='LSV')
    ax.axhline(0, color='k', lw=0.5)
    for a in anchors:
        ax.axvline(a, color='green', alpha=0.3, lw=1, ls=':')
    ax.set_xlabel('T (years)'); ax.set_ylabel('Model - Market')
    ax.set_title('Put Errors: LV vs LSV-CTMC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    if all_errs:
        ax.set_ylim(lo * 1.3, hi * 1.3)

    fig.suptitle('LV vs LSV-CTMC: Pricing Error Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, 'LV_vs_LSV_pricing_error_comparison.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()