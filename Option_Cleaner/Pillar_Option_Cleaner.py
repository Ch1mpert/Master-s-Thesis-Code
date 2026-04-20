#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bid-Ask Aware Options Data Cleaning
====================================

Enforces no-arbitrage constraints (monotonicity + convexity) while respecting
the bid-ask spread. Instead of filtering on mid-price violations, this script
recognises that any violation "healable" within the bid-ask interval is just
quote noise, not a real arbitrage.

Pipeline per (tenor, side):
  1) Basic validity filters (non-null strike/price/tenor, strike >= MIN_STRIKE)
  2) LP feasibility: find prices p_i ∈ [bid_i, ask_i] satisfying:
       - Monotonicity: non-increasing (calls) / non-decreasing (puts) in strike
       - Convexity: non-negative butterfly spreads (non-decreasing slopes)
  3) If feasible with ALL points → keep all, output adjusted arbitrage-free prices
  4) If infeasible → iteratively remove the point contributing most to
     infeasibility, re-solve, until feasible. Then output adjusted prices.
  5) Optionally applies a relative spread filter.

Outputs:
  - All original columns preserved
  - New column `adjusted_mid`: arbitrage-free price closest (L1) to original mid
    that lies within [bid, ask] and satisfies monotonicity + convexity globally.

The adjusted_mid prices are suitable for direct use in vol surface fitting,
risk-neutral density extraction, and other downstream analytics.
"""

from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ------------------ Config --------------------------------------------------
INPUT_PATH  = "./data/options_formatted.csv"
OUTPUT_PATH = "^SPX_options_cleaned.csv"
MIN_STRIKE  = 0

# Numerical tolerance for constraint satisfaction checks
FEAS_TOL = 1e-10

# Optional spread filter (only applied if bid/ask columns are detected).
APPLY_SPREAD_FILTER = False
MAX_REL_SPREAD = 0.50  # e.g. 0.50 = 50 % relative spread cutoff

# Fallback convexity tolerance when bid/ask are unavailable.
# Used only if no bid/ask columns exist — reverts to mid-price filtering.
FALLBACK_CONVEX_TOL = 0.05


# ------------------ Column detection (unchanged) ----------------------------
def _fuzzy_find(patterns, cols_lower):
    for pat in patterns:
        cre = re.compile(pat, re.IGNORECASE)
        for c in cols_lower:
            if cre.search(c):
                return c
    return None


def autodetect_columns(df: pd.DataFrame):
    """Return (strike_col, type_col, price_selector, bid_col, ask_col)."""
    cols_lower = [c.strip().lower() for c in df.columns]

    def orig(col_lower):
        for oc in df.columns:
            if oc.strip().lower() == col_lower:
                return oc
        return None

    strike_l = _fuzzy_find([r"^strike$", r"strike_?price", r"^k$"], cols_lower)
    type_l   = _fuzzy_find([r"option_?type", r"^type$", r"right", r"cp[_-]?flag"], cols_lower)
    price_l  = _fuzzy_find([
        r"call_?price", r"c_?price", r"call_?last", r"call_?mid", r"callmark",
        r"^mid$", r"^mark$", r"^price$", r"last_?price", r"^last$", r"^close$",
        r"settle", r"theo"
    ], cols_lower)
    bid_l = _fuzzy_find([r"^bid$", r"best_?bid", r"c?_?bid"], cols_lower)
    ask_l = _fuzzy_find([r"^ask$", r"best_?ask", r"c?_?ask"], cols_lower)

    strike_col = orig(strike_l) if strike_l else None
    type_col   = orig(type_l)   if type_l   else None
    price_col  = orig(price_l)  if price_l  else None
    bid_col    = orig(bid_l)    if bid_l    else None
    ask_col    = orig(ask_l)    if ask_l    else None

    if strike_col is None:
        raise ValueError("Could not identify Strike column.")

    def select_price(_df: pd.DataFrame) -> pd.Series:
        if price_col is not None:
            return pd.to_numeric(_df[price_col], errors="coerce")
        if bid_col is not None and ask_col is not None:
            return (pd.to_numeric(_df[bid_col], errors="coerce") +
                    pd.to_numeric(_df[ask_col], errors="coerce")) / 2.0
        raise ValueError("No price column (or bid/ask pair) found.")

    return strike_col, type_col, select_price, bid_col, ask_col


def compute_tenor_months_series(df: pd.DataFrame) -> pd.Series:
    """Series[Int64] of tenor (months), aligned to df.index."""
    cols = [c.lower() for c in df.columns]

    def col(name):
        for oc in df.columns:
            if oc.strip().lower() == name:
                return oc
        return None

    for name in ["tenor_months", "tenor_m", "tenor"]:
        if name in cols:
            return pd.to_numeric(df[col(name)], errors="coerce").round().astype("Int64")

    for name in ["dte", "days_to_expiry", "days_to_expiration",
                  "days_to_maturity", "days", "ttm_days"]:
        if name in cols:
            dte = pd.to_numeric(df[col(name)], errors="coerce")
            return (dte / 30.0).round().astype("Int64")

    def _ff(pats):
        return _fuzzy_find(pats, cols)

    trade_l  = _ff([r"^date$", r"trade.*date", r"quote.*date", r"asof", r"timestamp"])
    expiry_l = _ff([r"exp.*date", r"maturity", r"expiry"])

    if trade_l and expiry_l:
        trade_c, expiry_c = col(trade_l), col(expiry_l)
        td = pd.to_datetime(df[trade_c], errors="coerce", utc=True).dt.tz_localize(None)
        ed = pd.to_datetime(df[expiry_c], errors="coerce", utc=True).dt.tz_localize(None)
        dte = (ed - td).dt.days
        return (dte / 30.0).round().astype("Int64")

    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


# ------------------ LP solver -----------------------------------------------
def _build_lp(strikes, bids, asks, mids, side):
    """
    Build the LP for finding adjusted prices closest to mid (L1 norm)
    within [bid, ask] satisfying monotonicity and convexity.

    Variables: [p_0, ..., p_{n-1}, d_0, ..., d_{n-1}]
    Minimize:  Σ d_i
    Subject to:
        p_i - mid_i ≤  d_i   (upper deviation)
        mid_i - p_i ≤  d_i   (lower deviation)
        bid_i ≤ p_i ≤ ask_i  (spread bounds)
        d_i ≥ 0
        monotonicity constraints
        convexity constraints (non-negative butterflies)
    """
    n = len(strikes)
    k = np.asarray(strikes, dtype=float)
    m = np.asarray(mids, dtype=float)
    b = np.asarray(bids, dtype=float)
    a = np.asarray(asks, dtype=float)

    # Objective: minimize sum(d_i)
    c = np.zeros(2 * n)
    c[n:] = 1.0

    rows, rhs = [], []

    # |p_i - mid_i| ≤ d_i  (linearised)
    for i in range(n):
        # p_i - d_i ≤ mid_i
        r = np.zeros(2 * n); r[i] = 1.0; r[n + i] = -1.0
        rows.append(r); rhs.append(m[i])
        # -p_i - d_i ≤ -mid_i
        r = np.zeros(2 * n); r[i] = -1.0; r[n + i] = -1.0
        rows.append(r); rhs.append(-m[i])

    # Monotonicity
    for i in range(n - 1):
        r = np.zeros(2 * n)
        if side == "call":
            r[i + 1] = 1.0; r[i] = -1.0       # p_{i+1} ≤ p_i
        else:
            r[i] = 1.0; r[i + 1] = -1.0        # p_i ≤ p_{i+1}
        rows.append(r); rhs.append(0.0)

    # Convexity: for consecutive triple (i-1, i, i+1),
    # w1·p_{i-1} − (w1+w2)·p_i + w2·p_{i+1} ≥ 0
    # ⇔ −w1·p_{i-1} + (w1+w2)·p_i − w2·p_{i+1} ≤ 0
    for i in range(1, n - 1):
        dk1 = k[i] - k[i - 1]
        dk2 = k[i + 1] - k[i]
        if dk1 > 0 and dk2 > 0:
            w1, w2 = 1.0 / dk1, 1.0 / dk2
            r = np.zeros(2 * n)
            r[i - 1] = -w1; r[i] = w1 + w2; r[i + 1] = -w2
            rows.append(r); rhs.append(0.0)

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    bounds = [(bi, ai) for bi, ai in zip(b, a)] + [(0.0, None)] * n

    return c, A_ub, b_ub, bounds


def solve_adjusted_prices(strikes, bids, asks, mids, side):
    """
    Find arbitrage-free prices closest (L1) to mid within [bid, ask].

    Returns:
        adjusted_prices : ndarray or None if infeasible
        kept_mask       : boolean ndarray (True for kept points)
    """
    n = len(strikes)
    if n <= 2:
        # Trivially feasible; just clamp mid to [bid, ask]
        adj = np.clip(mids, bids, asks)
        return adj, np.ones(n, dtype=bool)

    # --- Attempt 1: keep all points ---
    c, A_ub, b_ub, bounds = _build_lp(strikes, bids, asks, mids, side)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        return res.x[:n], np.ones(n, dtype=bool)

    # --- Attempt 2: iterative removal of infeasible points ---
    # Use dual information / heuristic: remove points near the worst violation.
    mask = np.ones(n, dtype=bool)

    for _attempt in range(n - 2):  # at most remove n-2 points
        idx = np.where(mask)[0]
        if len(idx) <= 2:
            break

        # Find worst butterfly violation using mid prices as proxy
        k_sub = strikes[idx]
        m_sub = mids[idx]

        worst_val = 0.0
        worst_inner = -1  # index into idx array (the middle point of the triplet)

        for j in range(1, len(idx) - 1):
            dk1 = k_sub[j] - k_sub[j - 1]
            dk2 = k_sub[j + 1] - k_sub[j]
            if dk1 > 0 and dk2 > 0:
                w1, w2 = 1.0 / dk1, 1.0 / dk2
                # Butterfly value (negative = violation)
                B = w1 * m_sub[j - 1] - (w1 + w2) * m_sub[j] + w2 * m_sub[j + 1]
                if B < -worst_val:
                    worst_val = -B
                    worst_inner = j

        if worst_inner == -1:
            break  # no mid-price violations remain

        # Remove the middle point of the worst-violating triplet
        mask[idx[worst_inner]] = False

        # Re-solve LP on remaining points
        idx2 = np.where(mask)[0]
        c2, A2, b2, bnd2 = _build_lp(
            strikes[idx2], bids[idx2], asks[idx2], mids[idx2], side
        )
        res2 = linprog(c2, A_ub=A2, b_ub=b2, bounds=bnd2, method="highs")
        if res2.success:
            adj = np.full(n, np.nan)
            adj[idx2] = res2.x[:len(idx2)]
            return adj, mask

    # Last resort: return only the 2 extreme points
    mask[:] = False
    mask[0] = True; mask[-1] = True
    adj = np.full(n, np.nan)
    adj[0] = np.clip(mids[0], bids[0], asks[0])
    adj[-1] = np.clip(mids[-1], bids[-1], asks[-1])
    return adj, mask


# ---------- Fallback: original mid-price LCS (no bid/ask) ------------------
import bisect

def longest_nonincreasing_subsequence(values):
    a = np.asarray(values, dtype=float)
    b = -a
    tails, tails_idx, prev = [], [], np.full(len(a), -1, dtype=int)
    for i, x in enumerate(b):
        j = bisect.bisect_left(tails, x)
        if j == len(tails):
            tails.append(x); tails_idx.append(i)
        else:
            tails[j] = x; tails_idx[j] = i
        if j > 0:
            prev[i] = tails_idx[j - 1]
    seq, k = [], (tails_idx[-1] if tails_idx else -1)
    while k != -1:
        seq.append(k); k = prev[k]
    seq.reverse()
    return np.array(seq, dtype=int)


def longest_nondecreasing_subsequence(values):
    a = np.asarray(values, dtype=float)
    tails, tails_idx, prev = [], [], np.full(len(a), -1, dtype=int)
    for i, x in enumerate(a):
        j = bisect.bisect_right(tails, x)
        if j == len(tails):
            tails.append(x); tails_idx.append(i)
        else:
            tails[j] = x; tails_idx[j] = i
        if j > 0:
            prev[i] = tails_idx[j - 1]
    seq, k = [], (tails_idx[-1] if tails_idx else -1)
    while k != -1:
        seq.append(k); k = prev[k]
    seq.reverse()
    return np.array(seq, dtype=int)


def _is_convex_step(k0, p0, k1, p1, k2, p2, tol):
    if k1 == k0 or k2 == k1:
        return False
    s01 = (p1 - p0) / (k1 - k0)
    s12 = (p2 - p1) / (k2 - k1)
    return (s12 + tol) >= s01


def longest_convex_subsequence(strikes, prices, tol):
    k = np.asarray(strikes, dtype=float)
    p = np.asarray(prices, dtype=float)
    n = len(k)
    if n <= 2:
        return np.arange(n, dtype=int)

    dp = np.ones((n, n), dtype=int) * 2
    prev = np.full((n, n), -1, dtype=int)
    best_len, best_state = 2, (0, 1)

    for j in range(1, n):
        for i in range(j):
            dp[i, j] = 2; prev[i, j] = -1
            for h in range(i):
                if _is_convex_step(k[h], p[h], k[i], p[i], k[j], p[j], tol=tol):
                    cand = dp[h, i] + 1
                    if cand > dp[i, j]:
                        dp[i, j] = cand; prev[i, j] = h
            if dp[i, j] > best_len:
                best_len = dp[i, j]; best_state = (i, j)

    i, j = best_state
    seq = [j, i]
    h = prev[i, j]
    while h != -1:
        seq.append(h); j, i = i, h; h = prev[i, j]
    seq.reverse()
    return np.array(seq, dtype=int)


def fallback_midprice_clean(strikes, prices, side, tol):
    """Original monotone + convex LCS filtering on mid prices."""
    n = len(strikes)
    if n <= 2:
        return np.ones(n, dtype=bool)

    if side == "call":
        keep_mono = longest_nonincreasing_subsequence(prices)
    else:
        keep_mono = longest_nondecreasing_subsequence(prices)

    k_m = strikes[keep_mono]
    p_m = prices[keep_mono]
    if len(k_m) <= 2:
        mask = np.zeros(n, dtype=bool); mask[keep_mono] = True
        return mask

    keep_conv = longest_convex_subsequence(k_m, p_m, tol=tol)
    final_idx = keep_mono[keep_conv]
    mask = np.zeros(n, dtype=bool); mask[final_idx] = True
    return mask


# ------------------ Main ----------------------------------------------------
def main():
    df = pd.read_csv(INPUT_PATH)
    orig_columns = df.columns.tolist()

    strike_col, type_col, select_price, bid_col, ask_col = autodetect_columns(df)
    strike = pd.to_numeric(df[strike_col], errors="coerce")
    price  = select_price(df)
    tenor  = compute_tenor_months_series(df)

    has_bidask = (bid_col is not None) and (ask_col is not None)

    if has_bidask:
        bid_ser = pd.to_numeric(df[bid_col], errors="coerce")
        ask_ser = pd.to_numeric(df[ask_col], errors="coerce")
    else:
        bid_ser = ask_ser = None

    # Optional spread filter
    spread_ok = pd.Series(True, index=df.index)
    if APPLY_SPREAD_FILTER and has_bidask:
        mid = (bid_ser + ask_ser) / 2.0
        rel = (ask_ser - bid_ser) / mid.replace(0.0, np.nan)
        spread_ok = rel.isna() | (rel <= MAX_REL_SPREAD)

    if type_col is None:
        base_mask = (
            strike.notna() & price.notna() & tenor.notna()
            & (strike >= MIN_STRIKE) & (price >= 0) & spread_ok
        )
        df_out = df.loc[base_mask, orig_columns].copy()
        df_out["adjusted_mid"] = price[base_mask]
        df_out.to_csv(OUTPUT_PATH, index=False)
        print("Saved (no type column; only base filters):", OUTPUT_PATH)
        return

    type_ser = df[type_col].astype(str).str.upper()

    base_mask = (
        strike.notna() & price.notna() & tenor.notna()
        & (strike >= MIN_STRIKE) & (price >= 0) & spread_ok
    )
    if has_bidask:
        base_mask = base_mask & bid_ser.notna() & ask_ser.notna() & (bid_ser >= 0)

    keep_mask   = pd.Series(False, index=df.index)
    adjusted_mid = pd.Series(np.nan, index=df.index)

    stats = {"tenors_processed": 0, "lp_used": 0, "fallback_used": 0}

    for t_val, grp_idx in tenor[base_mask].groupby(tenor[base_mask]).groups.items():
        grp_idx = np.array(list(grp_idx))

        for side_label, side_key in [("call", "C"), ("put", "P")]:
            side_mask_vals = type_ser.loc[grp_idx].str.startswith(side_key).values
            idxs = grp_idx[side_mask_vals]
            if len(idxs) == 0:
                continue

            # Sort by strike
            order = np.argsort(strike.loc[idxs].values, kind="mergesort")
            idxs_sorted = idxs[order]

            k_arr = strike.loc[idxs_sorted].values.astype(float)
            p_arr = price.loc[idxs_sorted].values.astype(float)

            if has_bidask:
                b_arr = bid_ser.loc[idxs_sorted].values.astype(float)
                a_arr = ask_ser.loc[idxs_sorted].values.astype(float)

                # --- LP-based cleaning ---
                adj, kept = solve_adjusted_prices(k_arr, b_arr, a_arr, p_arr, side_label)

                if adj is not None:
                    kept_idx = idxs_sorted[kept]
                    keep_mask.loc[kept_idx] = True
                    adjusted_mid.loc[kept_idx] = adj[kept]
                    stats["lp_used"] += 1
                else:
                    # Should not happen, but fallback
                    fmask = fallback_midprice_clean(k_arr, p_arr, side_label, FALLBACK_CONVEX_TOL)
                    keep_mask.loc[idxs_sorted[fmask]] = True
                    adjusted_mid.loc[idxs_sorted[fmask]] = p_arr[fmask]
                    stats["fallback_used"] += 1
            else:
                # --- No bid/ask: use original mid-price LCS with fallback tolerance ---
                fmask = fallback_midprice_clean(k_arr, p_arr, side_label, FALLBACK_CONVEX_TOL)
                keep_mask.loc[idxs_sorted[fmask]] = True
                adjusted_mid.loc[idxs_sorted[fmask]] = p_arr[fmask]
                stats["fallback_used"] += 1

            stats["tenors_processed"] += 1

        # Unknown types: keep with original mid
        others = ~(
            type_ser.loc[grp_idx].str.startswith("C")
            | type_ser.loc[grp_idx].str.startswith("P")
        )
        if others.any():
            other_idx = grp_idx[others.values]
            keep_mask.loc[other_idx] = True
            adjusted_mid.loc[other_idx] = price.loc[other_idx]

    # Build output
    df_out = df.loc[keep_mask, orig_columns].copy()
    df_out["adjusted_mid"] = adjusted_mid[keep_mask].values

    df_out.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print({
        "rows_in":  len(df),
        "rows_kept": len(df_out),
        "rows_dropped": len(df) - len(df_out),
        "columns": df_out.columns.tolist(),
        "strike_col": strike_col,
        "type_col": type_col,
        "bid_ask_mode": has_bidask,
        "spread_filter": bool(APPLY_SPREAD_FILTER and has_bidask),
        **stats,
    })


if __name__ == "__main__":
    main()