#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_expected_expiry_lv_vs_lsv.py
====================================

Compare expected expiry times E[T_*] between:
    - LV model  : solid lines
    - CTMC-LSV  : dashed lines

The CSV files are expected to contain at least:
    obs_freq
    maturity_years
    expected_expiry_years

This script:
    1) loads the two CSV files,
    2) aligns them on (obs_freq, maturity_years),
    3) plots expected expiry vs maturity for each observation frequency,
    4) saves a merged comparison CSV with differences.

Example
-------
python compare_expected_expiry_lv_vs_lsv.py \
    --lv_csv autocallable_lv_term_structure.csv \
    --lsv_csv autocallable_term_structure.csv \
    --output_png expected_expiry_lv_vs_lsv.png \
    --output_csv expected_expiry_lv_vs_lsv.csv \
    --freqs "monthly,quarterly,semi-annual"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Helpers
# ============================================================
REQUIRED_COLUMNS = {
    "obs_freq",
    "maturity_years",
    "expected_expiry_years",
}


def normalize_obs_freq(freq: str) -> str:
    f = str(freq).strip().lower()
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


def obs_freq_label(freq: str) -> str:
    return rf"$T^O = {obs_freq_to_months(freq)}$"


def parse_freqs(text: str) -> List[str]:
    vals = [normalize_obs_freq(tok.strip()) for tok in text.split(",") if tok.strip()]
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def load_expected_expiry_csv(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{model_name} CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["obs_freq"] = df["obs_freq"].map(normalize_obs_freq)
    df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="raise")
    df["expected_expiry_years"] = pd.to_numeric(df["expected_expiry_years"], errors="raise")

    # Keep only the columns we actually need for comparison, but preserve extras if useful later
    return df


def build_comparison_dataframe(
    lv_df: pd.DataFrame,
    lsv_df: pd.DataFrame,
    freqs: List[str],
) -> pd.DataFrame:
    lv_sub = lv_df[lv_df["obs_freq"].isin(freqs)].copy()
    lsv_sub = lsv_df[lsv_df["obs_freq"].isin(freqs)].copy()

    lv_sub = lv_sub.rename(
        columns={
            "expected_expiry_years": "expected_expiry_years_lv",
        }
    )
    lsv_sub = lsv_sub.rename(
        columns={
            "expected_expiry_years": "expected_expiry_years_lsv",
        }
    )

    keep_lv = ["obs_freq", "maturity_years", "expected_expiry_years_lv"]
    keep_lsv = ["obs_freq", "maturity_years", "expected_expiry_years_lsv"]

    merged = pd.merge(
        lv_sub[keep_lv],
        lsv_sub[keep_lsv],
        on=["obs_freq", "maturity_years"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping (obs_freq, maturity_years) points were found between the two CSV files."
        )

    merged["obs_tenor_months"] = merged["obs_freq"].map(obs_freq_to_months)
    merged["expected_expiry_diff_years"] = (
        merged["expected_expiry_years_lsv"] - merged["expected_expiry_years_lv"]
    )
    merged["expected_expiry_diff_months"] = 12.0 * merged["expected_expiry_diff_years"]

    merged = merged.sort_values(
        by=["obs_tenor_months", "maturity_years"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return merged


def print_summary(comp: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print("EXPECTED EXPIRY COMPARISON SUMMARY")
    print("=" * 110)
    print(
        f"{'ObsFreq':>12} {'Mat':>8} {'LV E[T*]':>14} {'LSV E[T*]':>14} "
        f"{'LSV-LV':>14} {'(months)':>14}"
    )
    print("-" * 110)

    for _, row in comp.iterrows():
        print(
            f"{row['obs_freq']:>12} "
            f"{row['maturity_years']:8.4f} "
            f"{row['expected_expiry_years_lv']:14.6f} "
            f"{row['expected_expiry_years_lsv']:14.6f} "
            f"{row['expected_expiry_diff_years']:14.6f} "
            f"{row['expected_expiry_diff_months']:14.6f}"
        )

    print("=" * 110)

    stats = (
        comp.groupby("obs_freq")["expected_expiry_diff_years"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    print("\nDifference stats by frequency (LSV - LV):")
    for _, row in stats.iterrows():
        print(
            f"  {row['obs_freq']:>12}: "
            f"mean={row['mean']:.6f}, min={row['min']:.6f}, max={row['max']:.6f}"
        )
    print()


def plot_expected_expiry_comparison(
    comp: pd.DataFrame,
    output_png: str,
    title: str,
    x_in_months: bool = False,
) -> None:
    color_map: Dict[str, str] = {
        "monthly": "#1f77b4",
        "quarterly": "#e6550d",
        "semi-annual": "#d99a00",
        "annual": "#7b1fa2",
    }

    ordered_freqs = sorted(comp["obs_freq"].unique(), key=obs_freq_to_months)

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)

    for freq in ordered_freqs:
        df_f = comp[comp["obs_freq"] == freq].sort_values("maturity_years")
        x = 12.0 * df_f["maturity_years"].to_numpy() if x_in_months else df_f["maturity_years"].to_numpy()
        y_lv = df_f["expected_expiry_years_lv"].to_numpy()
        y_lsv = df_f["expected_expiry_years_lsv"].to_numpy()

        color = color_map.get(freq, None)

        ax.plot(
            x,
            y_lv,
            linestyle="-",
            linewidth=1.4,
            color=color,
        )
        ax.plot(
            x,
            y_lsv,
            linestyle="--",
            dashes=(10, 6),
            linewidth=1.4,
            color=color,
        )

    if x_in_months:
        ax.set_xlabel(r"$T^E$")
    else:
        ax.set_xlabel(r"$T^E$ (years)")

    ax.set_ylabel(r"$\mathbb{E}[T_\tau]$")
    ax.set_title(title)
    ax.grid(True, alpha=0.45, linewidth=0.7)

    freq_handles = [
        Line2D([0], [0], color=color_map[f], lw=1.4, linestyle="-", label=obs_freq_label(f))
        for f in ordered_freqs
    ]
    legend1 = ax.legend(
        handles=freq_handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.3",
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=2.0,
        handletextpad=0.35,
    )
    ax.add_artist(legend1)

    style_handles = [
        Line2D([0], [0], color="black", lw=1.4, linestyle="-", label="LV"),
        Line2D([0], [0], color="black", lw=1.4, linestyle="--", dashes=(10, 6), label="LSV"),
    ]
    ax.legend(
        handles=style_handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.3",
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=2.0,
        handletextpad=0.45,
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Compare expected expiry time between LV and CTMC-LSV models."
    )

    p.add_argument(
        "--lv_csv",
        type=str,
        default="autocallable_lv_term_structure.csv",
        help="CSV from the LV autocallable pricer.",
    )
    p.add_argument(
        "--lsv_csv",
        type=str,
        default="autocallable_term_structure.csv",
        help="CSV from the CTMC-LSV autocallable pricer.",
    )
    p.add_argument(
        "--output_png",
        type=str,
        default="expected_expiry_lv_vs_lsv.png",
        help="Output figure path.",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default="expected_expiry_lv_vs_lsv.csv",
        help="Merged comparison CSV path.",
    )
    p.add_argument(
        "--freqs",
        type=str,
        default="monthly,quarterly,semi-annual",
        help='Comma-separated frequencies to include, e.g. "monthly,quarterly,semi-annual".',
    )
    p.add_argument(
        "--title",
        type=str,
        default="Expected expiry time comparison",
        help="Plot title.",
    )
    p.add_argument(
        "--x_in_months",
        action="store_true",
        help="Plot maturity on the x-axis in months instead of years.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    lv_path = Path(args.lv_csv)
    lsv_path = Path(args.lsv_csv)

    if not lv_path.exists():
        raise FileNotFoundError(f"LV CSV not found: {lv_path}")
    if not lsv_path.exists():
        raise FileNotFoundError(f"LSV CSV not found: {lsv_path}")

    freqs = parse_freqs(args.freqs)

    lv_df = load_expected_expiry_csv(str(lv_path), "LV")
    lsv_df = load_expected_expiry_csv(str(lsv_path), "LSV")

    comp = build_comparison_dataframe(lv_df, lsv_df, freqs=freqs)

    comp.to_csv(args.output_csv, index=False)
    print(f"Saved merged comparison CSV to: {args.output_csv}")

    print_summary(comp)

    plot_expected_expiry_comparison(
        comp=comp,
        output_png=args.output_png,
        title=args.title,
        x_in_months=args.x_in_months,
    )
    print(f"Saved comparison plot to: {args.output_png}")


if __name__ == "__main__":
    main()