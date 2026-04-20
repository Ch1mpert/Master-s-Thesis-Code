#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a DAILY interpolated forward curve from NPZ pillar files and save to CSV,
including a synthetic T=0 pillar where forward(0)=spot.

Inputs (pillars):
  1M.npz
  3M.npz
  6M.npz
  12M.npz
  24M.npz

Each pillar NPZ is expected to contain (at minimum):
  - T (year fraction) OR T_years OR tenor_months (months)
  - forward (forward price) OR one of: F, F_cal, F_parity, forward_price, Fwd

Outputs:
  - forward_curve_pillars_extracted.csv
  - forward_curve_interpolated_daily.csv

Interpolation:
  - Interpolates log-forward linearly vs T_years
  - Produces a DAILY grid using ACT/365: day = 0 .. floor(maxT*365)

Notes:
  - You MUST set SPOT_S0 to the spot price at valuation date.
  - A synthetic pillar row (tenor_months=0, T_years=0.0, forward=SPOT_S0) is
    prepended before interpolation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------
PILLAR_FILES = [
    "./data/1M.npz",
    "./data/3M.npz",
    "./data/6M.npz",
    "./data/12M.npz",
    "./data/24M.npz",
]

OUT_PILLARS_CSV = "forward_curve_pillars_extracted.csv"
OUT_CURVE_DAILY_CSV = "forward_curve_interpolated_daily.csv"

DAYS_IN_YEAR = 365.0  # ACT/365

# >>> Set this to the spot price on your valuation date <<<
SPOT_S0 = 5868.55

# ----------------------------
# Helpers
# ----------------------------
def _as_scalar(x) -> float:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.ravel()[0])
    raise ValueError(f"Expected scalar or length-1 array, got shape={arr.shape}")


def _parse_months_from_filename(fp: str) -> int:
    m = re.search(r"(\d+)M", Path(fp).name)
    if not m:
        raise ValueError(f"Cannot parse tenor months from filename: {fp}")
    return int(m.group(1))


def extract_T_years(npz, fp: str) -> Tuple[float, str]:
    keys = set(npz.files)
    for k in ["T", "T_years", "tau", "ttm"]:
        if k in keys:
            return _as_scalar(npz[k]), k
    for k in ["tenor_months", "months"]:
        if k in keys:
            return _as_scalar(npz[k]) / 12.0, k
    months = _parse_months_from_filename(fp)
    return months / 12.0, "filename_months"


def extract_forward(npz, fp: str) -> Tuple[float, str]:
    keys = set(npz.files)
    if "forward" in keys:
        return _as_scalar(npz["forward"]), "forward"
    for k in ["F", "F_cal", "F_parity", "forward_price", "Fwd", "F0"]:
        if k in keys:
            return _as_scalar(npz[k]), k
    raise KeyError(f"Could not find forward in {fp}. Keys={list(npz.files)}")


def extract_tenor_months(npz, fp: str, T_years: float) -> Tuple[int, str]:
    keys = set(npz.files)
    if "tenor_months" in keys:
        return int(round(_as_scalar(npz["tenor_months"]))), "tenor_months"
    try:
        return _parse_months_from_filename(fp), "filename_months"
    except Exception:
        return int(round(T_years * 12)), "T_years*12"


def load_pillars(pillar_files: List[str]) -> pd.DataFrame:
    rows = []
    for fp in pillar_files:
        npz = np.load(fp, allow_pickle=True)

        T_years, T_key = extract_T_years(npz, fp)
        forward, F_key = extract_forward(npz, fp)
        tenor_months, tenor_key = extract_tenor_months(npz, fp, T_years)
        expiry = str(npz["expiry"]) if "expiry" in npz.files else None

        rows.append({
            "pillar_file": Path(fp).name,
            "tenor_months": int(tenor_months),
            "T_years": float(T_years),
            "forward": float(forward),
            "expiry": expiry,
            "T_key": T_key,
            "forward_key": F_key,
            "tenor_key": tenor_key,
        })

    df = pd.DataFrame(rows).sort_values("T_years").reset_index(drop=True)

    # Drop duplicates by tenor (keep last)
    df = df.drop_duplicates(subset=["tenor_months"], keep="last").sort_values("T_years").reset_index(drop=True)

    if (df["T_years"] <= 0).any():
        raise ValueError("Found non-positive T_years in pillars.")
    if (df["forward"] <= 0).any():
        raise ValueError("Found non-positive forward in pillars.")

    return df


def build_daily_forward_curve(pillars: pd.DataFrame, spot_s0: float) -> pd.DataFrame:
    """
    Daily grid (ACT/365) from day 0 to max pillar maturity.
    Interpolate log-forward linearly vs T_years.
    Includes a synthetic pillar at T=0 with forward=spot_s0.
    """
    if not (np.isfinite(spot_s0) and spot_s0 > 0):
        raise ValueError("SPOT_S0 must be set to a positive spot price to build a 0->1M forward segment.")

    # Synthetic T=0 pillar
    pillars0 = pd.DataFrame([{
        "pillar_file": "SPOT_S0",
        "tenor_months": 0,
        "T_years": 0.0,
        "forward": float(spot_s0),
        "expiry": None,
        "T_key": "synthetic",
        "forward_key": "synthetic",
        "tenor_key": "synthetic",
    }])

    pillars_ext = pd.concat([pillars0, pillars], ignore_index=True)
    pillars_ext = pillars_ext.sort_values("T_years").reset_index(drop=True)

    T_p = pillars_ext["T_years"].to_numpy(dtype=float)
    F_p = pillars_ext["forward"].to_numpy(dtype=float)

    if (F_p <= 0).any():
        raise ValueError("Found non-positive forward in extended pillars (including synthetic spot).")

    logF_p = np.log(F_p)

    max_days = int(np.floor(T_p.max() * DAYS_IN_YEAR))
    day_grid = np.arange(0, max_days + 1, dtype=int)
    T_grid = day_grid / DAYS_IN_YEAR

    # log-linear interpolation of forward
    logF_grid = np.interp(T_grid, T_p, logF_p)
    F_grid = np.exp(logF_grid)

    curve = pd.DataFrame({
        "day": day_grid,
        "T_years": T_grid,
        "forward_interp": F_grid,
    })

    # Mark pillar days (nearest integer day)
    pillar_days = np.rint(pillars_ext["T_years"].to_numpy() * DAYS_IN_YEAR).astype(int)
    pillar_map = dict(zip(pillar_days, pillars_ext["forward"].to_numpy()))
    curve["is_pillar_day"] = curve["day"].isin(pillar_days).astype(int)
    curve["forward_pillar"] = curve["day"].map(pillar_map)

    return curve


def main() -> None:
    pillars = load_pillars(PILLAR_FILES)
    curve_daily = build_daily_forward_curve(pillars, spot_s0=SPOT_S0)

    pillars.to_csv(OUT_PILLARS_CSV, index=False)
    curve_daily.to_csv(OUT_CURVE_DAILY_CSV, index=False)

    print("Saved pillars to:", OUT_PILLARS_CSV)
    print("Saved daily curve to:", OUT_CURVE_DAILY_CSV)

    print("\nExtracted pillars:")
    print(pillars.to_string(index=False))

    print("\nDaily curve head (first 60 rows):")
    print(curve_daily.head(60).to_string(index=False))

    print("\nDaily curve tail (last 10 rows):")
    print(curve_daily.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()