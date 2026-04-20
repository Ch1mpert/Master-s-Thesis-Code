#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autocallable_pricer_lv.py — LV Autocallable Pricer (z-marginal only)

Uses tridiagonal infinitesimal generators for density propagation.
No variance dimension - serves as LV baseline for comparison with CTMC-LSV.

FEATURES
--------
- Computes expected stopping / expiry time E[T_*]
- Supports single-frequency maturity sweeps
- Supports multi-frequency overlays (monthly / quarterly / semi-annual / annual)
- Supports separate coupon-rate lists for each observation frequency
- Supports fair-coupon solving in:
    * single-maturity mode
    * single-frequency maturity-list mode
    * multi-frequency maturity-list mode
- Saves CSV summaries
- Saves expected-expiry plots
- Retains original single-maturity pricing and fair-coupon sweep behavior
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.interpolate import interp1d


# ============================================================
# Data structures
# ============================================================
@dataclass
class AutocallableSpec:
    notional: float = 1.0
    maturity_years: float = 1.0
    ac_barrier: float = 1.0
    coupon_barrier: float = 0.70
    ki_barrier: float = 0.60
    coupon_rate: float = 0.02
    put_strike: float = 1.0
    memory: bool = True
    obs_freq: str = "quarterly"
    no_call_periods: int = 0
    ac_step_down: float = 0.0


@dataclass
class LVPillar:
    tenor_months: int
    T: float
    dt: float
    z: np.ndarray
    dz: float
    Q_lower: np.ndarray
    Q_diag: np.ndarray
    Q_upper: np.ndarray


@dataclass
class LVModel:
    pillars: List[LVPillar]
    z_grid: np.ndarray
    dz: float
    S0: float


@dataclass
class PricingResult:
    price: float
    notional: float
    price_pct: float
    autocall_probabilities: np.ndarray
    stop_probabilities: np.ndarray
    coupon_contributions: np.ndarray
    autocall_contributions: np.ndarray
    terminal_par_contribution: float
    terminal_put_contribution: float
    survival_probability: float
    ki_probability: float
    observation_dates: np.ndarray
    memory_enabled: bool
    fair_coupon: Optional[float] = None
    expected_expiry_years: float = 0.0


@dataclass
class TermStructurePoint:
    maturity_years: float
    coupon_rate: float
    price: float
    price_pct: float
    price_diff: float
    price_diff_bps: float
    survival_probability: float
    terminal_par_contribution: float
    terminal_put_contribution: float
    expected_expiry_years: float
    obs_freq: str


# ============================================================
# Loading model / market data
# ============================================================
def load_lv_generators(files: List[str], S0: float) -> LVModel:
    pillars = []
    for f in sorted(files):
        d = np.load(f, allow_pickle=True)
        pillars.append(
            LVPillar(
                tenor_months=int(d["tenor_months"]),
                T=float(d["T"]),
                dt=float(d["dt"]),
                z=d["z"].astype(np.float64),
                dz=float(d["dz"]),
                Q_lower=d["Q_lower"].astype(np.float64),
                Q_diag=d["Q_diag"].astype(np.float64),
                Q_upper=d["Q_upper"].astype(np.float64),
            )
        )
    pillars.sort(key=lambda p: p.T)
    return LVModel(
        pillars=pillars,
        z_grid=pillars[0].z,
        dz=pillars[0].dz,
        S0=S0,
    )


def load_forward_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import csv

    T, F = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            T.append(float(r["T_years"]))
            F.append(float(r["forward_interp"]))
    return np.array(T), np.array(F)


def load_discount_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import csv

    T, D = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            T.append(float(r["T_years"]))
            D.append(float(r["discount_factor"]))
    return np.array(T), np.array(D)


def build_interpolators(fT, fF, dT, dD):
    return (
        interp1d(fT, fF, kind="linear", fill_value="extrapolate"),
        interp1d(dT, dD, kind="linear", fill_value="extrapolate"),
    )


# ============================================================
# Frequency helpers
# ============================================================
def normalize_obs_freq(freq: str) -> str:
    f = freq.strip().lower()
    if f in ("monthly", "month", "m", "1m"):
        return "monthly"
    if f in ("quarterly", "quarter", "q", "3m"):
        return "quarterly"
    if f in ("semi-annual", "semiannual", "semi", "sa", "6m", "semi_annual"):
        return "semi-annual"
    if f in ("annual", "yearly", "a", "12m", "1y"):
        return "annual"
    raise ValueError(f"Unknown observation frequency: {freq}")


def obs_freq_to_months(freq: str) -> int:
    f = normalize_obs_freq(freq)
    if f == "monthly":
        return 1
    if f == "quarterly":
        return 3
    if f == "semi-annual":
        return 6
    if f == "annual":
        return 12
    raise ValueError(f"Unknown observation frequency: {freq}")


def obs_freq_legend_label(freq: str) -> str:
    return rf"$T^O = {obs_freq_to_months(freq)}$"


def parse_obs_freq_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    text = text.strip()
    if text == "":
        return None

    vals = [normalize_obs_freq(tok.strip()) for tok in text.split(",") if tok.strip()]
    if len(vals) == 0:
        return None

    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def generate_observation_dates(maturity: float, freq: str) -> np.ndarray:
    freq = normalize_obs_freq(freq)

    if freq == "monthly":
        dt_obs = 1.0 / 12.0
    elif freq == "quarterly":
        dt_obs = 0.25
    elif freq == "semi-annual":
        dt_obs = 0.5
    elif freq == "annual":
        dt_obs = 1.0
    else:
        raise ValueError(f"Unknown freq: {freq}")

    tol = 1e-12
    regular_dates = np.arange(dt_obs, maturity - tol, dt_obs, dtype=float)

    if regular_dates.size == 0:
        return np.array([maturity], dtype=float)

    return np.concatenate([regular_dates, np.array([maturity], dtype=float)])


# ============================================================
# LV propagation
# ============================================================
def _get_pillar_at_time(model: LVModel, t: float) -> LVPillar:
    for p in model.pillars:
        if t <= p.T + 1e-8:
            return p
    return model.pillars[-1]


def _build_Q_sparse(pillar: LVPillar):
    Nz = len(pillar.z)
    return diags(
        [pillar.Q_lower, pillar.Q_diag, pillar.Q_upper],
        [-1, 0, 1],
        shape=(Nz, Nz),
        format="csc",
    )


def propagate_lv(
    phi: np.ndarray,
    model: LVModel,
    t_start: float,
    t_end: float,
    n_substeps: int = 500,
) -> np.ndarray:
    if t_end - t_start <= 1e-12:
        return phi.copy()

    result = phi.copy()
    ptimes = sorted(set([0.0] + [p.T for p in model.pillars]))
    bounds = [t for t in ptimes if t_start < t < t_end]
    bounds = [t_start] + bounds + [t_end]
    total_dt = t_end - t_start

    for i in range(len(bounds) - 1):
        seg_dt = bounds[i + 1] - bounds[i]
        if seg_dt <= 1e-12:
            continue

        pillar = _get_pillar_at_time(model, 0.5 * (bounds[i] + bounds[i + 1]))
        Q = _build_Q_sparse(pillar)

        nsub = max(1, int(round(n_substeps * seg_dt / total_dt)))
        dt_sub = seg_dt / nsub

        for _ in range(nsub):
            result = expm_multiply(Q * dt_sub, result)
            np.maximum(result, 0.0, out=result)

    return result


def get_lv_density(model: LVModel, t: float, n_substeps: int = 500) -> np.ndarray:
    Nz = len(model.z_grid)
    phi = np.zeros(Nz)
    phi[np.argmin(np.abs(model.z_grid))] = 1.0 / model.dz
    if t < 1e-12:
        return phi
    return propagate_lv(phi, model, 0.0, t, n_substeps)


# ============================================================
# Core pricer
# ============================================================
def price_autocallable(
    model: LVModel,
    spec: AutocallableSpec,
    fwd_interp,
    disc_interp,
    n_substeps: int = 500,
    verbose: bool = True,
    mass_thr: float = 1e-12,
) -> PricingResult:
    t0 = time.time()
    S0, z, dz, Nz = model.S0, model.z_grid, model.dz, len(model.z_grid)
    obs = generate_observation_dates(spec.maturity_years, spec.obs_freq)
    K = len(obs)

    if verbose:
        print(f"\n{'='*70}")
        print("AUTOCALLABLE PRICER — LOCAL VOLATILITY")
        print(f"{'='*70}")
        print(f"  S0={S0:.2f}, Mat={spec.maturity_years:.4f}y, Freq={spec.obs_freq}")
        print(f"  Obs ({K}): {[f'{t:.4f}' for t in obs]}")
        print(
            f"  AC={spec.ac_barrier*100:.1f}%, "
            f"Cpn={spec.coupon_barrier*100:.1f}%, "
            f"KI={spec.ki_barrier*100:.1f}%, "
            f"c={spec.coupon_rate*100:.6f}%/period"
        )
        print(
            f"  Memory={spec.memory}, No-call={spec.no_call_periods}, "
            f"Step-down={spec.ac_step_down*100:.1f}%"
        )
        print(f"  Substeps={n_substeps}, z-grid={Nz}pts")
        print(f"{'='*70}")

    phi_init = get_lv_density(model, obs[0], n_substeps)

    if verbose:
        m = np.sum(phi_init) * dz
        pk = z[np.argmax(phi_init)]
        print(f"  LV density at T={obs[0]:.4f}: mass={m:.8f}, peak z={pk:.4f}")

    slices: Dict[Tuple[int, int], np.ndarray] = {(0, 0): phi_init}

    ac_probs = np.zeros(K)
    ac_cont = np.zeros(K)
    cpn_cont = np.zeros(K)
    term_par = 0.0
    term_put = 0.0
    price = 0.0
    t_prev = obs[0]

    for k in range(K):
        t_obs = obs[k]
        D = float(disc_interp(t_obs))
        F = float(fwd_interp(t_obs))

        ac_b = spec.ac_barrier - spec.ac_step_down * k
        z_ac = np.log(max(ac_b * S0 / F, 1e-12))
        z_cpn = np.log(max(spec.coupon_barrier * S0 / F, 1e-12))
        z_ki = np.log(max(spec.ki_barrier * S0 / F, 1e-12))

        a_ac = z >= z_ac
        a_cpn = z >= z_cpn
        b_ki = z < z_ki

        is_final = (k == K - 1)
        can_ac = (k >= spec.no_call_periods) and (not is_final)

        if verbose:
            tm = sum(float(np.sum(p)) * dz for p in slices.values())
            print(f"\n  Obs {k+1}/{K}: T={t_obs:.4f}, F={F:.6f}, D={D:.6f}, mass={tm:.8f}")

        if k > 0:
            ns = {}
            for key, phi in slices.items():
                if float(np.sum(phi)) * dz < mass_thr:
                    continue
                ns[key] = propagate_lv(phi, model, t_prev, t_obs, n_substeps)
            slices = ns

        # KI check
        us = {}
        for (b, m), phi in slices.items():
            if b == 0:
                ps = phi.copy()
                pk = np.zeros_like(phi)
                pk[b_ki] = phi[b_ki]
                ps[b_ki] = 0.0

                if float(np.sum(pk)) * dz > mass_thr:
                    us[(1, m)] = us.get((1, m), np.zeros_like(phi)) + pk
                if float(np.sum(ps)) * dz > mass_thr:
                    us[(0, m)] = us.get((0, m), np.zeros_like(phi)) + ps
            else:
                us[(b, m)] = us.get((b, m), np.zeros_like(phi)) + phi.copy()
        slices = us

        # AC + coupon
        post = {}
        for (b, m), phi in slices.items():
            if can_ac:
                psurv = phi.copy()
                psurv[a_ac] = 0.0
                am = float(np.sum(phi[a_ac])) * dz

                if am > mass_thr:
                    nc = (m + 1) if spec.memory else 1
                    cv = D * spec.notional * (1.0 + nc * spec.coupon_rate) * am
                    price += cv
                    ac_probs[k] += am
                    ac_cont[k] += cv

                phi = psurv

            if verbose:
                sm = float(np.sum(phi)) * dz
                mac = float(np.sum(phi[a_cpn])) * dz
                mbc = float(np.sum(phi[~a_cpn])) * dz
                print(f"    (b={b},m={m}): mass={sm:.6f}, above_cpn={mac:.6f}, below_cpn={mbc:.6f}")

            if is_final:
                pac = phi.copy()
                pac[~a_cpn] = 0.0
                ma = float(np.sum(pac)) * dz

                if ma > 0:
                    nc = (m + 1) if spec.memory else 1
                    cv = D * spec.notional * nc * spec.coupon_rate * ma
                    cpn_cont[k] += cv
                    price += cv

                ttm = float(np.sum(phi)) * dz
                if b == 0:
                    pv = D * spec.notional * ttm
                    term_par += pv
                    price += pv
                else:
                    sr = np.exp(z) * F / (spec.put_strike * S0)
                    ppz = np.minimum(sr, 1.0)
                    pv = D * spec.notional * float(np.sum(ppz * phi) * dz)
                    term_put += pv
                    price += pv
                continue

            pac = phi.copy()
            pac[~a_cpn] = 0.0
            pbc = phi.copy()
            pbc[a_cpn] = 0.0

            ma = float(np.sum(pac)) * dz
            mb = float(np.sum(pbc)) * dz

            if ma > mass_thr:
                nc = (m + 1) if spec.memory else 1
                cv = D * spec.notional * nc * spec.coupon_rate * ma
                cpn_cont[k] += cv
                price += cv
                kr = (b, 0)
                post[kr] = post.get(kr, np.zeros_like(phi)) + pac

            if mb > mass_thr:
                ki = (b, m + 1) if spec.memory else (b, 0)
                post[ki] = post.get(ki, np.zeros_like(phi)) + pbc

        if not is_final:
            slices = post

        slices = {k_: v for k_, v in slices.items() if float(np.sum(v)) * dz > mass_thr}

        if verbose:
            sm = sum(float(np.sum(p)) * dz for p in slices.values())
            mki = {0: 0.0, 1: 0.0}
            mm = 0
            for (bk, mk), ps in slices.items():
                mki[bk] += float(np.sum(ps)) * dz
                mm = max(mm, mk)
            print(f"    => AC={ac_probs[k]:.6f}, Cpn={cpn_cont[k]:.6f}, AC_val={ac_cont[k]:.6f}")
            print(f"    => Surv={sm:.8f}, slices={len(slices)}, max_m={mm}")
            print(f"    => KI: no={mki[0]:.6f}, yes={mki[1]:.6f}")

        t_prev = t_obs

    surv = sum(float(np.sum(p)) * dz for p in slices.values())

    stop_probs = np.zeros(K, dtype=float)
    if K > 1:
        stop_probs[:-1] = ac_probs[:-1]
    stop_probs[-1] = surv

    total_stop_mass = float(np.sum(stop_probs))
    if total_stop_mass > 1e-15:
        expected_expiry_years = float(np.dot(obs, stop_probs) / total_stop_mass)
    else:
        expected_expiry_years = 0.0

    el = time.time() - t0
    pct = price / spec.notional * 100.0

    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS (LV)")
        print(f"{'='*70}")
        print(f"  Price            = {price:.8f} ({pct:.4f}%)")
        print(f"  Survival         = {surv:.6f}")
        print(f"  Expected expiry  = {expected_expiry_years:.6f} years")
        print(f"  Stop mass total  = {total_stop_mass:.8f}")
        print(f"  Term par         = {term_par:.8f}, Term put = {term_put:.8f}")
        for kk in range(K):
            print(
                f"    t={obs[kk]:.4f}: "
                f"stop={stop_probs[kk]:.6f}, "
                f"AC={ac_probs[kk]:.6f}, "
                f"AC_v={ac_cont[kk]:.6f}, "
                f"Cpn={cpn_cont[kk]:.6f}"
            )
        print(f"  Time             = {el:.2f}s")
        print(f"{'='*70}")

    return PricingResult(
        price=price,
        notional=spec.notional,
        price_pct=pct,
        autocall_probabilities=ac_probs,
        stop_probabilities=stop_probs,
        coupon_contributions=cpn_cont,
        autocall_contributions=ac_cont,
        terminal_par_contribution=term_par,
        terminal_put_contribution=term_put,
        survival_probability=surv,
        ki_probability=0.0,
        observation_dates=obs,
        memory_enabled=spec.memory,
        fair_coupon=None,
        expected_expiry_years=expected_expiry_years,
    )


# ============================================================
# Fair coupon
# ============================================================
def solve_fair_coupon(
    model,
    spec,
    fwd_interp,
    disc_interp,
    n_substeps: int = 500,
    verbose: bool = True,
):
    if verbose:
        print(f"\n{'='*70}")
        print("SOLVING FAIR COUPON (LV)")
        print(f"{'='*70}")

    s0 = AutocallableSpec(**{**spec.__dict__, "coupon_rate": 0.0})
    s1 = AutocallableSpec(**{**spec.__dict__, "coupon_rate": 1.0})

    if verbose:
        print("\n--- c=0 ---")
    r0 = price_autocallable(model, s0, fwd_interp, disc_interp, n_substeps, verbose)

    if verbose:
        print("\n--- c=1 ---")
    r1 = price_autocallable(model, s1, fwd_interp, disc_interp, n_substeps, verbose)

    Vu = r1.price - r0.price
    if abs(Vu) < 1e-15:
        return 0.0, r0

    fc = (spec.notional - r0.price) / Vu

    if verbose:
        npy = len(generate_observation_dates(spec.maturity_years, spec.obs_freq)) / spec.maturity_years
        print(f"\n{'='*70}")
        print("FAIR COUPON (LV)")
        print(f"{'='*70}")
        print(f"  V(0)={r0.price:.8f}, V(1)={r1.price:.8f}, Vu={Vu:.8f}")
        print(f"  Fair coupon = {fc*100:.6f}%/period = {fc*npy*100:.6f}% p.a.")
        print(f"{'='*70}")

    sf = AutocallableSpec(**{**spec.__dict__, "coupon_rate": fc})
    if verbose:
        print(f"\n--- Verify c={fc:.6f} ---")
    rf = price_autocallable(model, sf, fwd_interp, disc_interp, n_substeps, verbose)
    rf.fair_coupon = fc
    return fc, rf


# ============================================================
# Batch utilities
# ============================================================
def parse_float_list(text: Optional[str], arg_name: str) -> Optional[List[float]]:
    if text is None:
        return None
    text = text.strip()
    if text == "":
        return None
    try:
        vals = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse {arg_name}: {text}") from exc
    if len(vals) == 0:
        raise ValueError(f"{arg_name} is empty.")
    return vals


def parse_float_grid(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("coupon maturity grid is empty")
    return np.array(vals, dtype=float)


def resolve_coupon_list_for_freq(
    freq: str,
    maturities: List[float],
    common_coupon_list: Optional[List[float]],
    monthly_coupon_list: Optional[List[float]],
    quarterly_coupon_list: Optional[List[float]],
    semi_annual_coupon_list: Optional[List[float]],
    annual_coupon_list: Optional[List[float]],
    fallback_coupon_rate: float,
) -> List[float]:
    freq = normalize_obs_freq(freq)

    specific = None
    if freq == "monthly":
        specific = monthly_coupon_list
    elif freq == "quarterly":
        specific = quarterly_coupon_list
    elif freq == "semi-annual":
        specific = semi_annual_coupon_list
    elif freq == "annual":
        specific = annual_coupon_list

    coupon_list = specific if specific is not None else common_coupon_list

    if coupon_list is None:
        return [fallback_coupon_rate] * len(maturities)

    if len(coupon_list) == 1 and len(maturities) > 1:
        return coupon_list * len(maturities)

    if len(coupon_list) != len(maturities):
        raise ValueError(
            f"Coupon list length mismatch for freq={freq}: "
            f"need {len(maturities)} entries, got {len(coupon_list)}."
        )

    return coupon_list


def save_term_structure_csv(points: List[TermStructurePoint], csv_path: str):
    import csv

    fieldnames = [
        "obs_freq",
        "obs_tenor_months",
        "maturity_years",
        "n_observations",
        "coupon_rate",
        "coupon_rate_pct",
        "coupon_rate_pa_pct",
        "price",
        "price_pct",
        "price_diff",
        "price_diff_bps",
        "survival_probability",
        "terminal_par_contribution",
        "terminal_put_contribution",
        "expected_expiry_years",
        "expected_expiry_months",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            n_obs = len(generate_observation_dates(p.maturity_years, p.obs_freq))
            coupon_rate_pa_pct = (
                100.0 * p.coupon_rate * n_obs / p.maturity_years
                if p.maturity_years > 0.0 else np.nan
            )

            writer.writerow({
                "obs_freq": p.obs_freq,
                "obs_tenor_months": obs_freq_to_months(p.obs_freq),
                "maturity_years": p.maturity_years,
                "n_observations": n_obs,
                "coupon_rate": p.coupon_rate,
                "coupon_rate_pct": 100.0 * p.coupon_rate,
                "coupon_rate_pa_pct": coupon_rate_pa_pct,
                "price": p.price,
                "price_pct": p.price_pct,
                "price_diff": p.price_diff,
                "price_diff_bps": p.price_diff_bps,
                "survival_probability": p.survival_probability,
                "terminal_par_contribution": p.terminal_par_contribution,
                "terminal_put_contribution": p.terminal_put_contribution,
                "expected_expiry_years": p.expected_expiry_years,
                "expected_expiry_months": 12.0 * p.expected_expiry_years,
            })


def print_term_structure_summary(
    points: List[TermStructurePoint],
    header: str = "AUTOCALLABLE TERM STRUCTURE SUMMARY (LV)",
):
    print("\n" + "=" * 140)
    print(header)
    print("=" * 140)
    print(
        f"{'ObsFreq':>12} {'Maturity':>12} {'Coupon':>12} "
        f"{'Price':>14} {'Diff':>14} {'Diff (bps)':>14} "
        f"{'E[T*]':>12} {'Survival':>12}"
    )
    print("-" * 140)
    for p in points:
        print(
            f"{p.obs_freq:>12} "
            f"{p.maturity_years:12.4f} "
            f"{100.0 * p.coupon_rate:11.4f}% "
            f"{p.price:14.8f} "
            f"{p.price_diff:14.8f} "
            f"{p.price_diff_bps:14.4f} "
            f"{p.expected_expiry_years:12.6f} "
            f"{p.survival_probability:12.6f}"
        )
    print("=" * 140)


def print_fair_coupon_term_structure_summary(
    points: List[TermStructurePoint],
    header: str = "FAIR COUPON TERM STRUCTURE SUMMARY (LV)",
):
    print("\n" + "=" * 160)
    print(header)
    print("=" * 160)
    print(
        f"{'ObsFreq':>12} {'Maturity':>12} "
        f"{'Fair cpn %/period':>20} {'Fair cpn % p.a.':>18} "
        f"{'Verify Price':>14} {'Verify Diff (bps)':>18} "
        f"{'E[T*]':>12} {'Survival':>12}"
    )
    print("-" * 160)

    for p in points:
        n_obs = len(generate_observation_dates(p.maturity_years, p.obs_freq))
        coupon_pa_pct = (
            100.0 * p.coupon_rate * n_obs / p.maturity_years
            if p.maturity_years > 0.0 else np.nan
        )

        print(
            f"{p.obs_freq:>12} "
            f"{p.maturity_years:12.4f} "
            f"{100.0 * p.coupon_rate:19.6f}% "
            f"{coupon_pa_pct:17.6f}% "
            f"{p.price:14.8f} "
            f"{p.price_diff_bps:18.4f} "
            f"{p.expected_expiry_years:12.6f} "
            f"{p.survival_probability:12.6f}"
        )

    print("=" * 160)


def plot_term_structure(
    points: List[TermStructurePoint],
    png_path: Optional[str] = None,
    title: str = r"Autocallable price difference (LV)",
):
    maturities = np.array([p.maturity_years for p in points], dtype=float)
    diff_bps = np.array([p.price_diff_bps for p in points], dtype=float)

    order = np.argsort(maturities)
    maturities = maturities[order]
    diff_bps = diff_bps[order]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(maturities, diff_bps, linewidth=1.2)
    plt.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.7)
    plt.xlabel(r"$T^E$ (years)")
    plt.ylabel(r"$10^4\left(\widetilde{V}^{LV}(0)-1\right)$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return fig


def plot_multi_obs_term_structure(
    curves: Dict[str, List[TermStructurePoint]],
    png_path: Optional[str] = None,
    title: str = r"Autocallable price difference (LV)",
):
    fig = plt.figure(figsize=(9, 5))

    ordered_freqs = sorted(curves.keys(), key=obs_freq_to_months)

    for freq in ordered_freqs:
        points = curves[freq]
        maturities = np.array([p.maturity_years for p in points], dtype=float)
        diff_bps = np.array([p.price_diff_bps for p in points], dtype=float)

        order = np.argsort(maturities)
        maturities = maturities[order]
        diff_bps = diff_bps[order]

        plt.plot(
            maturities,
            diff_bps,
            linewidth=1.2,
            label=obs_freq_legend_label(freq),
        )

    plt.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.7)
    plt.xlabel(r"$T^E$ (years)")
    plt.ylabel(r"$10^4\left(\widetilde{V}^{LV}(0)-1\right)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return fig


def plot_expected_expiry(
    points: List[TermStructurePoint],
    png_path: Optional[str] = None,
    title: str = r"Expected expiry time (LV)",
):
    maturities = np.array([p.maturity_years for p in points], dtype=float)
    exp_stop = np.array([p.expected_expiry_years for p in points], dtype=float)

    order = np.argsort(maturities)
    maturities = maturities[order]
    exp_stop = exp_stop[order]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(maturities, exp_stop, linewidth=1.2)
    plt.xlabel(r"$T^E$ (years)")
    plt.ylabel(r"$\mathbb{E}[T_*]$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return fig


def plot_multi_obs_expected_expiry(
    curves: Dict[str, List[TermStructurePoint]],
    png_path: Optional[str] = None,
    title: str = r"Expected expiry time (LV)",
):
    fig = plt.figure(figsize=(9, 5))

    ordered_freqs = sorted(curves.keys(), key=obs_freq_to_months)

    for freq in ordered_freqs:
        points = curves[freq]
        maturities = np.array([p.maturity_years for p in points], dtype=float)
        exp_stop = np.array([p.expected_expiry_years for p in points], dtype=float)

        order = np.argsort(maturities)
        maturities = maturities[order]
        exp_stop = exp_stop[order]

        plt.plot(
            maturities,
            exp_stop,
            linewidth=1.2,
            label=obs_freq_legend_label(freq),
        )

    plt.xlabel(r"$T^E$ (years)")
    plt.ylabel(r"$\mathbb{E}[T_*]$")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return fig


def plot_fair_coupon_term_structure(
    points: List[TermStructurePoint],
    png_path: Optional[str] = None,
    title: str = r"Fair coupon term structure (LV)",
):
    maturities = np.array([p.maturity_years for p in points], dtype=float)

    fair_coupon_pa_pct = []
    for p in points:
        n_obs = len(generate_observation_dates(p.maturity_years, p.obs_freq))
        fair_coupon_pa_pct.append(
            100.0 * p.coupon_rate * n_obs / p.maturity_years
            if p.maturity_years > 0.0 else np.nan
        )
    fair_coupon_pa_pct = np.array(fair_coupon_pa_pct, dtype=float)

    order = np.argsort(maturities)
    maturities = maturities[order]
    fair_coupon_pa_pct = fair_coupon_pa_pct[order]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(maturities, fair_coupon_pa_pct, linewidth=1.2)
    plt.xlabel(r"$T^E$ (years)")
    plt.ylabel(r"Fair coupon (% p.a.)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return fig


def plot_multi_obs_fair_coupon_term_structure(
    curves: Dict[str, List[TermStructurePoint]],
    png_path: Optional[str] = None,
    title: str = r"Fair coupon term structure (LV)",
):
    fig = plt.figure(figsize=(9, 5))

    ordered_freqs = sorted(curves.keys(), key=obs_freq_to_months)

    for freq in ordered_freqs:
        points = curves[freq]
        maturities = np.array([p.maturity_years for p in points], dtype=float)

        fair_coupon_pa_pct = []
        for p in points:
            n_obs = len(generate_observation_dates(p.maturity_years, p.obs_freq))
            fair_coupon_pa_pct.append(
                100.0 * p.coupon_rate * n_obs / p.maturity_years
                if p.maturity_years > 0.0 else np.nan
            )
        fair_coupon_pa_pct = np.array(fair_coupon_pa_pct, dtype=float)

        order = np.argsort(maturities)
        maturities = maturities[order]
        fair_coupon_pa_pct = fair_coupon_pa_pct[order]

        plt.plot(
            maturities,
            fair_coupon_pa_pct,
            linewidth=1.2,
            label=obs_freq_legend_label(freq),
        )

    plt.xlabel(r"$T^E$ (years)")
    plt.ylabel(r"Fair coupon (% p.a.)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return fig


def price_autocallable_term_structure(
    model: LVModel,
    base_spec: AutocallableSpec,
    maturities: List[float],
    coupon_rates: List[float],
    fwd_interp,
    disc_interp,
    n_substeps: int = 500,
    verbose: bool = True,
) -> List[TermStructurePoint]:
    if len(maturities) != len(coupon_rates):
        raise ValueError("maturities and coupon_rates must have the same length.")

    pairs = sorted(zip(maturities, coupon_rates), key=lambda x: x[0])
    points: List[TermStructurePoint] = []

    for j, (T, c) in enumerate(pairs, start=1):
        if verbose:
            print("\n" + "#" * 88)
            print(
                f"TERM POINT {j}/{len(pairs)}: "
                f"obs_freq={base_spec.obs_freq}, maturity={T:.4f}y, coupon={100*c:.4f}%"
            )
            print("#" * 88)

        spec_j = AutocallableSpec(
            notional=base_spec.notional,
            maturity_years=T,
            ac_barrier=base_spec.ac_barrier,
            coupon_barrier=base_spec.coupon_barrier,
            ki_barrier=base_spec.ki_barrier,
            coupon_rate=c,
            put_strike=base_spec.put_strike,
            memory=base_spec.memory,
            obs_freq=base_spec.obs_freq,
            no_call_periods=base_spec.no_call_periods,
            ac_step_down=base_spec.ac_step_down,
        )

        res = price_autocallable(
            model=model,
            spec=spec_j,
            fwd_interp=fwd_interp,
            disc_interp=disc_interp,
            n_substeps=n_substeps,
            verbose=verbose,
        )

        price_diff = res.price / spec_j.notional - 1.0
        price_diff_bps = 1.0e4 * price_diff

        points.append(
            TermStructurePoint(
                maturity_years=T,
                coupon_rate=c,
                price=res.price,
                price_pct=res.price_pct,
                price_diff=price_diff,
                price_diff_bps=price_diff_bps,
                survival_probability=res.survival_probability,
                terminal_par_contribution=res.terminal_par_contribution,
                terminal_put_contribution=res.terminal_put_contribution,
                expected_expiry_years=res.expected_expiry_years,
                obs_freq=spec_j.obs_freq,
            )
        )

    return points


def price_multi_obs_term_structure(
    model: LVModel,
    base_spec: AutocallableSpec,
    obs_freqs: List[str],
    maturities: List[float],
    common_coupon_list: Optional[List[float]],
    monthly_coupon_list: Optional[List[float]],
    quarterly_coupon_list: Optional[List[float]],
    semi_annual_coupon_list: Optional[List[float]],
    annual_coupon_list: Optional[List[float]],
    fwd_interp,
    disc_interp,
    n_substeps: int = 500,
    verbose: bool = True,
) -> Dict[str, List[TermStructurePoint]]:
    curves: Dict[str, List[TermStructurePoint]] = {}

    for freq in obs_freqs:
        coupon_list = resolve_coupon_list_for_freq(
            freq=freq,
            maturities=maturities,
            common_coupon_list=common_coupon_list,
            monthly_coupon_list=monthly_coupon_list,
            quarterly_coupon_list=quarterly_coupon_list,
            semi_annual_coupon_list=semi_annual_coupon_list,
            annual_coupon_list=annual_coupon_list,
            fallback_coupon_rate=base_spec.coupon_rate,
        )

        spec_freq = AutocallableSpec(
            notional=base_spec.notional,
            maturity_years=base_spec.maturity_years,
            ac_barrier=base_spec.ac_barrier,
            coupon_barrier=base_spec.coupon_barrier,
            ki_barrier=base_spec.ki_barrier,
            coupon_rate=base_spec.coupon_rate,
            put_strike=base_spec.put_strike,
            memory=base_spec.memory,
            obs_freq=freq,
            no_call_periods=base_spec.no_call_periods,
            ac_step_down=base_spec.ac_step_down,
        )

        if verbose:
            print("\n" + "=" * 120)
            print(f"MULTI-FREQUENCY SWEEP (LV): obs_freq={freq}  (T^O={obs_freq_to_months(freq)} months)")
            print("=" * 120)

        curves[freq] = price_autocallable_term_structure(
            model=model,
            base_spec=spec_freq,
            maturities=maturities,
            coupon_rates=coupon_list,
            fwd_interp=fwd_interp,
            disc_interp=disc_interp,
            n_substeps=n_substeps,
            verbose=verbose,
        )

    return curves


def solve_fair_coupon_term_structure(
    model: LVModel,
    base_spec: AutocallableSpec,
    maturities: List[float],
    fwd_interp,
    disc_interp,
    n_substeps: int = 500,
    verbose: bool = True,
) -> List[TermStructurePoint]:
    mats = sorted(float(T) for T in maturities)
    points: List[TermStructurePoint] = []

    if verbose:
        print("\n" + "=" * 120)
        print(f"FAIR-COUPON TERM STRUCTURE SWEEP (LV): obs_freq={base_spec.obs_freq}  "
              f"(T^O={obs_freq_to_months(base_spec.obs_freq)} months)")
        print("=" * 120)

    for j, T in enumerate(mats, start=1):
        spec_j = AutocallableSpec(
            notional=base_spec.notional,
            maturity_years=T,
            ac_barrier=base_spec.ac_barrier,
            coupon_barrier=base_spec.coupon_barrier,
            ki_barrier=base_spec.ki_barrier,
            coupon_rate=base_spec.coupon_rate,
            put_strike=base_spec.put_strike,
            memory=base_spec.memory,
            obs_freq=base_spec.obs_freq,
            no_call_periods=base_spec.no_call_periods,
            ac_step_down=base_spec.ac_step_down,
        )

        fair_coupon, res = solve_fair_coupon(
            model=model,
            spec=spec_j,
            fwd_interp=fwd_interp,
            disc_interp=disc_interp,
            n_substeps=n_substeps,
            verbose=False,
        )

        if res is None:
            price = np.nan
            price_pct = np.nan
            price_diff = np.nan
            price_diff_bps = np.nan
            surv = np.nan
            term_par = np.nan
            term_put = np.nan
            exp_stop = np.nan
        else:
            price = res.price
            price_pct = res.price_pct
            price_diff = price / spec_j.notional - 1.0
            price_diff_bps = 1.0e4 * price_diff
            surv = res.survival_probability
            term_par = res.terminal_par_contribution
            term_put = res.terminal_put_contribution
            exp_stop = res.expected_expiry_years

        points.append(
            TermStructurePoint(
                maturity_years=T,
                coupon_rate=fair_coupon,
                price=price,
                price_pct=price_pct,
                price_diff=price_diff,
                price_diff_bps=price_diff_bps,
                survival_probability=surv,
                terminal_par_contribution=term_par,
                terminal_put_contribution=term_put,
                expected_expiry_years=exp_stop,
                obs_freq=spec_j.obs_freq,
            )
        )

        if verbose:
            n_obs = len(generate_observation_dates(T, spec_j.obs_freq))
            fair_coupon_pa = 100.0 * fair_coupon * n_obs / T if T > 0.0 else np.nan
            print(
                f"  [{j:>2d}/{len(mats):>2d}] "
                f"T={T:>5.2f}y | "
                f"fair cpn={100.0*fair_coupon:>10.6f}%/period | "
                f"{fair_coupon_pa:>10.6f}% p.a. | "
                f"verify V(0)={price:.8f} | "
                f"E[T*]={exp_stop:.6f}"
            )

    return points


def solve_multi_obs_fair_coupon_term_structure(
    model: LVModel,
    base_spec: AutocallableSpec,
    obs_freqs: List[str],
    maturities: List[float],
    fwd_interp,
    disc_interp,
    n_substeps: int = 500,
    verbose: bool = True,
) -> Dict[str, List[TermStructurePoint]]:
    curves: Dict[str, List[TermStructurePoint]] = {}

    for freq in obs_freqs:
        spec_freq = AutocallableSpec(
            notional=base_spec.notional,
            maturity_years=base_spec.maturity_years,
            ac_barrier=base_spec.ac_barrier,
            coupon_barrier=base_spec.coupon_barrier,
            ki_barrier=base_spec.ki_barrier,
            coupon_rate=base_spec.coupon_rate,
            put_strike=base_spec.put_strike,
            memory=base_spec.memory,
            obs_freq=freq,
            no_call_periods=base_spec.no_call_periods,
            ac_step_down=base_spec.ac_step_down,
        )

        if verbose:
            print("\n" + "=" * 120)
            print(f"MULTI-FREQUENCY FAIR-COUPON SWEEP (LV): obs_freq={freq}  "
                  f"(T^O={obs_freq_to_months(freq)} months)")
            print("=" * 120)

        curves[freq] = solve_fair_coupon_term_structure(
            model=model,
            base_spec=spec_freq,
            maturities=maturities,
            fwd_interp=fwd_interp,
            disc_interp=disc_interp,
            n_substeps=n_substeps,
            verbose=verbose,
        )

    return curves


def sweep_fair_coupon_term_structure(
    model,
    base_spec,
    fwd_interp,
    disc_interp,
    maturity_grid,
    n_substeps: int = 500,
    verbose: bool = True,
):
    rows = []

    if verbose:
        print(f"\n{'='*80}")
        print("FAIR COUPON TERM STRUCTURE SWEEP (LV)")
        print(f"{'='*80}")
        print(
            f"Observation convention kept fixed: obs_freq={base_spec.obs_freq}, "
            f"no_call_periods={base_spec.no_call_periods}, memory={base_spec.memory}"
        )
        print(f"Maturity grid: {[float(x) for x in maturity_grid]}")
        print(f"{'-'*80}")

    for mat in maturity_grid:
        spec_i = AutocallableSpec(**{**base_spec.__dict__, "maturity_years": float(mat)})
        fair_coupon, result = solve_fair_coupon(
            model,
            spec_i,
            fwd_interp,
            disc_interp,
            n_substeps=n_substeps,
            verbose=False,
        )

        obs_i = generate_observation_dates(spec_i.maturity_years, spec_i.obs_freq)
        n_obs = len(obs_i)
        coupon_pa = fair_coupon * n_obs / spec_i.maturity_years if spec_i.maturity_years > 0 else np.nan
        check_price = result.price if result is not None else np.nan
        exp_stop = result.expected_expiry_years if result is not None else np.nan

        rows.append({
            "maturity_years": float(mat),
            "n_obs": int(n_obs),
            "fair_coupon_per_period": float(fair_coupon),
            "fair_coupon_pct_per_period": float(100.0 * fair_coupon),
            "fair_coupon_pct_pa": float(100.0 * coupon_pa),
            "verification_price": float(check_price),
            "expected_expiry_years": float(exp_stop),
        })

        if verbose:
            print(
                f"Mat={mat:>4.2f}y | n_obs={n_obs:>2d} | "
                f"fair cpn={100.0*fair_coupon:>10.6f}%/period | "
                f"{100.0*coupon_pa:>10.6f}% p.a. | "
                f"check V(0)={check_price:.8f} | "
                f"E[T*]={exp_stop:.6f}"
            )

    if verbose:
        print(f"{'-'*80}")
        print("Summary table:")
        print(f"{'Mat':>6s} {'Obs':>5s} {'cpn %/period':>16s} {'cpn % p.a.':>14s} {'V(0)':>14s} {'E[T*]':>12s}")
        for r in rows:
            print(
                f"{r['maturity_years']:6.2f} "
                f"{r['n_obs']:5d} "
                f"{r['fair_coupon_pct_per_period']:16.6f} "
                f"{r['fair_coupon_pct_pa']:14.6f} "
                f"{r['verification_price']:14.8f} "
                f"{r['expected_expiry_years']:12.6f}"
            )
        print(f"{'='*80}\n")

    return rows


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--generator_dir", required=True)
    p.add_argument("--forward_curve", required=True)
    p.add_argument("--discount_curve", required=True)

    p.add_argument("--S0", type=float, default=5868.55)
    p.add_argument("--notional", type=float, default=1.0)
    p.add_argument("--maturity_years", type=float, default=1.0)

    p.add_argument("--ac_barrier", type=float, default=1.0)
    p.add_argument("--coupon_barrier", type=float, default=0.0)
    p.add_argument("--ki_barrier", type=float, default=0.8)
    p.add_argument("--coupon_rate", type=float, default=0.02742142)
    p.add_argument("--put_strike", type=float, default=1.0)

    mem_group = p.add_mutually_exclusive_group()
    mem_group.add_argument("--memory", dest="memory", action="store_true", default=True)
    mem_group.add_argument("--no_memory", dest="memory", action="store_false")
    p.set_defaults(memory=True)

    p.add_argument("--obs_freq", default="annual")
    p.add_argument(
        "--obs_freqs_list",
        type=str,
        default="monthly,quarterly,semi-annual",
        help='Comma-separated observation frequencies for multi-frequency batch mode, e.g. "monthly,quarterly,semi-annual". Leave empty for single-frequency mode.',
    )
    p.add_argument("--no_call_periods", type=int, default=0)
    p.add_argument("--ac_step_down", type=float, default=0.0)

    p.add_argument("--n_substeps", type=int, default=1)
    p.add_argument("--solve_coupon", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # Fair-coupon sweep
    p.add_argument(
        "--coupon_maturity_grid",
        default="0.25,0.5,0.75,1,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0",
        help="Comma-separated maturity grid for fair-coupon sweep.",
    )
    p.add_argument(
        "--skip_coupon_sweep",
        action="store_true",
        help="Skip the fair-coupon maturity sweep and run only the pricing task.",
    )

    # Batch pricing / solving
    p.add_argument(
        "--maturity_years_list",
        type=str,
        default="0.25,0.5,0.75,1,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0",
        help='Comma-separated maturities for batch mode, e.g. "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0". Leave empty for single-maturity mode.',
    )
    p.add_argument(
        "--coupon_rates_list",
        type=str,
        default="",
        help="Shared coupon-rate list for all observation frequencies unless frequency-specific lists are supplied. Ignored when --solve_coupon is set.",
    )
    p.add_argument("--coupon_rates_list_monthly", type=str,
                   default="0.00841059,0.010805,0.011079,0.01084965,0.0104426,0.01003036,0.00964965,0.00930935,0.00900903,0.00874538,0.00851183,0.00830338")
    p.add_argument("--coupon_rates_list_quarterly", type=str,
                   default="0.01908626,0.02461403,0.02670627,0.02742142,0.02727404,0.0267436,0.02608444,0.02540501,0.02475552,0.02415619,0.02360912,0.02311159")
    p.add_argument("--coupon_rates_list_semi_annual", type=str,
                   default="0.01908626,0.04162856,0.04122418,0.04768614,0.04594846,0.04834639,0.04589727,0.04702281,0.04470034,0.04534369,0.0433364,0.04376315")
    p.add_argument("--coupon_rates_list_annual", type=str,
                   default="0.01908626,0.04162856,0.06213436,0.08211667,0.07232821,0.07759105,0.08177649,0.08541534,0.07744464,0.07915938,0.08067379,0.08210938")

    p.add_argument("--output_prefix", type=str, default="autocallable_lv_term_structure")
    p.add_argument("--no_plot", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    gf = sorted(glob.glob(os.path.join(args.generator_dir, "Q_tridiag_*.npz")))
    if not gf:
        raise FileNotFoundError(f"No generators found in {args.generator_dir}")

    if verbose:
        print("Loading LV generators...")
        for f in gf:
            print(f"  {os.path.basename(f)}")

    model = load_lv_generators(gf, args.S0)

    if verbose:
        print(
            f"  S0={model.S0}, z={len(model.z_grid)}pts, "
            f"pillars={[f'{p.tenor_months}M' for p in model.pillars]}"
        )

    fT, fF = load_forward_curve(args.forward_curve)
    dT, dD = load_discount_curve(args.discount_curve)
    fi, di = build_interpolators(fT, fF, dT, dD)

    base_spec = AutocallableSpec(
        notional=args.notional,
        maturity_years=args.maturity_years,
        ac_barrier=args.ac_barrier,
        coupon_barrier=args.coupon_barrier,
        ki_barrier=args.ki_barrier,
        coupon_rate=args.coupon_rate,
        put_strike=args.put_strike,
        memory=args.memory,
        obs_freq=normalize_obs_freq(args.obs_freq),
        no_call_periods=args.no_call_periods,
        ac_step_down=args.ac_step_down,
    )

    maturity_list = parse_float_list(args.maturity_years_list, "--maturity_years_list")
    common_coupon_list = parse_float_list(args.coupon_rates_list, "--coupon_rates_list")
    monthly_coupon_list = parse_float_list(args.coupon_rates_list_monthly, "--coupon_rates_list_monthly")
    quarterly_coupon_list = parse_float_list(args.coupon_rates_list_quarterly, "--coupon_rates_list_quarterly")
    semi_annual_coupon_list = parse_float_list(args.coupon_rates_list_semi_annual, "--coupon_rates_list_semi_annual")
    annual_coupon_list = parse_float_list(args.coupon_rates_list_annual, "--coupon_rates_list_annual")
    obs_freqs = parse_obs_freq_list(args.obs_freqs_list)

    if obs_freqs is not None and maturity_list is None:
        raise ValueError("--obs_freqs_list requires --maturity_years_list.")

    # ------------------------------------------------------------
    # Multi-frequency batch mode
    # ------------------------------------------------------------
    if obs_freqs is not None:
        if args.solve_coupon:
            if verbose:
                print("\nBatch fair-coupon mode: supplied coupon-rate lists are ignored because --solve_coupon is set.")

            curves = solve_multi_obs_fair_coupon_term_structure(
                model=model,
                base_spec=base_spec,
                obs_freqs=obs_freqs,
                maturities=maturity_list,
                fwd_interp=fi,
                disc_interp=di,
                n_substeps=args.n_substeps,
                verbose=verbose,
            )

            all_points: List[TermStructurePoint] = []
            for freq in sorted(curves.keys(), key=obs_freq_to_months):
                print_fair_coupon_term_structure_summary(
                    curves[freq],
                    header=f"FAIR COUPON TERM STRUCTURE SUMMARY (LV) — {freq} "
                           f"(T^O={obs_freq_to_months(freq)} months)",
                )
                all_points.extend(curves[freq])

            out_csv = f"{args.output_prefix}_fair_coupon.csv"
            save_term_structure_csv(all_points, out_csv)
            print(f"\nSaved CSV summary to: {out_csv}")

            if not args.no_plot:
                out_png = f"{args.output_prefix}_fair_coupon.png"
                plot_multi_obs_fair_coupon_term_structure(
                    curves,
                    png_path=out_png,
                    title=r"Fair coupon term structure (LV)",
                )
                print(f"Saved fair-coupon plot to: {out_png}")

                out_png_exp = f"{args.output_prefix}_fair_coupon_expected_expiry.png"
                plot_multi_obs_expected_expiry(
                    curves,
                    png_path=out_png_exp,
                    title=r"Expected expiry time at fair coupon (LV)",
                )
                print(f"Saved expected-expiry plot to: {out_png_exp}")

                plt.close("all")

            return curves

        curves = price_multi_obs_term_structure(
            model=model,
            base_spec=base_spec,
            obs_freqs=obs_freqs,
            maturities=maturity_list,
            common_coupon_list=common_coupon_list,
            monthly_coupon_list=monthly_coupon_list,
            quarterly_coupon_list=quarterly_coupon_list,
            semi_annual_coupon_list=semi_annual_coupon_list,
            annual_coupon_list=annual_coupon_list,
            fwd_interp=fi,
            disc_interp=di,
            n_substeps=args.n_substeps,
            verbose=verbose,
        )

        all_points: List[TermStructurePoint] = []
        for freq in sorted(curves.keys(), key=obs_freq_to_months):
            print_term_structure_summary(
                curves[freq],
                header=f"AUTOCALLABLE TERM STRUCTURE SUMMARY (LV) — {freq} "
                       f"(T^O={obs_freq_to_months(freq)} months)",
            )
            all_points.extend(curves[freq])

        out_csv = f"{args.output_prefix}.csv"
        save_term_structure_csv(all_points, out_csv)
        print(f"\nSaved CSV summary to: {out_csv}")

        if not args.no_plot:
            out_png = f"{args.output_prefix}.png"
            plot_multi_obs_term_structure(
                curves,
                png_path=out_png,
                title=r"Autocallable price difference (LV)",
            )
            print(f"Saved plot to: {out_png}")

            out_png_exp = f"{args.output_prefix}_expected_expiry.png"
            plot_multi_obs_expected_expiry(
                curves,
                png_path=out_png_exp,
                title=r"Expected expiry time (LV)",
            )
            print(f"Saved expected-expiry plot to: {out_png_exp}")

            plt.close("all")

        return curves

    # ------------------------------------------------------------
    # Single-frequency batch mode
    # ------------------------------------------------------------
    if maturity_list is not None:
        if args.solve_coupon:
            points = solve_fair_coupon_term_structure(
                model=model,
                base_spec=base_spec,
                maturities=maturity_list,
                fwd_interp=fi,
                disc_interp=di,
                n_substeps=args.n_substeps,
                verbose=verbose,
            )

            print_fair_coupon_term_structure_summary(points)

            out_csv = f"{args.output_prefix}_fair_coupon.csv"
            save_term_structure_csv(points, out_csv)
            print(f"\nSaved CSV summary to: {out_csv}")

            if not args.no_plot:
                out_png = f"{args.output_prefix}_fair_coupon.png"
                plot_fair_coupon_term_structure(
                    points,
                    png_path=out_png,
                    title=r"Fair coupon term structure (LV)",
                )
                print(f"Saved fair-coupon plot to: {out_png}")

                out_png_exp = f"{args.output_prefix}_fair_coupon_expected_expiry.png"
                plot_expected_expiry(
                    points,
                    png_path=out_png_exp,
                    title=r"Expected expiry time at fair coupon (LV)",
                )
                print(f"Saved expected-expiry plot to: {out_png_exp}")

                plt.close("all")

            return points

        coupon_list = resolve_coupon_list_for_freq(
            freq=base_spec.obs_freq,
            maturities=maturity_list,
            common_coupon_list=common_coupon_list,
            monthly_coupon_list=monthly_coupon_list,
            quarterly_coupon_list=quarterly_coupon_list,
            semi_annual_coupon_list=semi_annual_coupon_list,
            annual_coupon_list=annual_coupon_list,
            fallback_coupon_rate=base_spec.coupon_rate,
        )

        points = price_autocallable_term_structure(
            model=model,
            base_spec=base_spec,
            maturities=maturity_list,
            coupon_rates=coupon_list,
            fwd_interp=fi,
            disc_interp=di,
            n_substeps=args.n_substeps,
            verbose=verbose,
        )

        print_term_structure_summary(points)

        out_csv = f"{args.output_prefix}.csv"
        save_term_structure_csv(points, out_csv)
        print(f"\nSaved CSV summary to: {out_csv}")

        if not args.no_plot:
            out_png = f"{args.output_prefix}.png"
            plot_term_structure(
                points,
                png_path=out_png,
                title=r"Autocallable price difference (LV)",
            )
            print(f"Saved plot to: {out_png}")

            out_png_exp = f"{args.output_prefix}_expected_expiry.png"
            plot_expected_expiry(
                points,
                png_path=out_png_exp,
                title=r"Expected expiry time (LV)",
            )
            print(f"Saved expected-expiry plot to: {out_png_exp}")

            plt.close("all")

        return points

    # ------------------------------------------------------------
    # Original fair-coupon sweep
    # ------------------------------------------------------------
    if not args.skip_coupon_sweep:
        maturity_grid = parse_float_grid(args.coupon_maturity_grid)
        sweep_fair_coupon_term_structure(
            model=model,
            base_spec=base_spec,
            fwd_interp=fi,
            disc_interp=di,
            maturity_grid=maturity_grid,
            n_substeps=args.n_substeps,
            verbose=verbose,
        )

    # ------------------------------------------------------------
    # Original single-maturity behaviour
    # ------------------------------------------------------------
    if args.solve_coupon:
        return solve_fair_coupon(model, base_spec, fi, di, args.n_substeps, verbose)

    return price_autocallable(model, base_spec, fi, di, args.n_substeps, verbose)


if __name__ == "__main__":
    main()