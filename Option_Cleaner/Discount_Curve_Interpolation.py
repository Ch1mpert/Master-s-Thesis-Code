import pandas as pd
import numpy as np

# -----------------------
# CONFIG
# -----------------------
INPUT_CSV = "discount_rates.csv"

OUT_TENOR_CSV = "discount_shortpoints.csv"
OUT_GRID_CSV  = "discount_curve_grid.csv"

# Set True only if your CSV has rates like 3.72 meaning 3.72% (not 0.0372)
RATE_IN_PERCENT = True

# Convert days -> years
DAY_COUNT = 365.0

# Tenors for file 1
TENORS = [
    ("1M",  1/12),
    ("3M",  3/12),
    ("6M",  6/12),
    ("12M", 1.0),
    ("24M", 2.0),
]

# Grid for file 2 (0.00, 0.01, 0.02, ...)
GRID_STEP = 0.01
GRID_MAX_YEARS = None  # None => use max maturity available per date

# -----------------------
# HELPERS
# -----------------------
def interp_linear(x, xp, fp):
    """Linear interpolation with flat extrapolation at ends."""
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

def build_curve_for_date(df_day_rate):
    """
    Input rows for a single date with columns: days, rate
    rate is CONTINUOUSLY-COMPOUNDED zero rate.
    Returns sorted arrays (t_years, zero_rate_cont).
    """
    t = df_day_rate["days"].to_numpy(dtype=float) / DAY_COUNT
    zc = df_day_rate["rate"].to_numpy(dtype=float)

    if RATE_IN_PERCENT:
        zc = zc / 100.0

    idx = np.argsort(t)
    t, zc = t[idx], zc[idx]

    # Add a T=0 anchor (DF=1). Use same rate as shortest maturity for interpolation stability.
    if t[0] > 0.0:
        t = np.insert(t, 0, 0.0)
        zc = np.insert(zc, 0, zc[0])

    return t, zc

def discount_factor_from_zero_cont(t, zc):
    """DF(T) = exp(-zc(T) * T)"""
    return np.exp(-zc * t)

# -----------------------
# MAIN
# -----------------------
df = pd.read_csv(INPUT_CSV)

required = {"date", "days", "rate"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in input CSV: {missing}")

df["date"] = pd.to_datetime(df["date"])
df["days"] = df["days"].astype(float)
df["rate"] = df["rate"].astype(float)

tenor_rows = []
grid_rows = []

for d, g in df.groupby("date"):
    t_curve, z_curve = build_curve_for_date(g)

    # ---- File 1: tenor table
    for tenor, T in TENORS:
        zT = float(interp_linear(T, t_curve, z_curve))                # continuous zero rate at T
        dfT = float(np.exp(-zT * T))                                  # DF(T) = exp(-z(T)*T)
        tenor_rows.append({
            "date": d.date().isoformat(),
            "tenor": tenor,
            "T_years": float(T),
            "discount_factor": dfT,
            "zero_rate_cont": zT,
        })

    # ---- File 2: dense grid
    tmax = float(t_curve[-1]) if GRID_MAX_YEARS is None else float(GRID_MAX_YEARS)
    grid = np.arange(0.0, tmax + 1e-12, GRID_STEP)

    zgrid = interp_linear(grid, t_curve, z_curve)
    dfgrid = discount_factor_from_zero_cont(grid, zgrid)

    for T, DFv, Zv in zip(grid, dfgrid, zgrid):
        grid_rows.append({
            "date": d.date().isoformat(),
            "T_years": float(T),
            "discount_factor": float(DFv),
            "zero_rate_cont": float(Zv),
        })

out1 = pd.DataFrame(tenor_rows)
out2 = pd.DataFrame(grid_rows)

# If you want outputs exactly like your examples (no date column), uncomment:
# out1 = out1.drop(columns=["date"])
# out2 = out2.drop(columns=["date"])

out1.to_csv(OUT_TENOR_CSV, index=False)
out2.to_csv(OUT_GRID_CSV, index=False)

print("Wrote:", OUT_TENOR_CSV, "and", OUT_GRID_CSV)
