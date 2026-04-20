import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from datetime import timezone

# ============================================================
# USER SETTINGS
# ============================================================
INPUT_CSV  = "./data/options.csv"
OUTPUT_CSV = "options_formatted.csv"

SPOT_PRICE = 5868.55	
TENORS_MONTHS = [1, 3, 6, 12, 24]
TOLERANCE_DAYS = 21

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_CSV, low_memory=False)

# ============================================================
# PARSE DATES
# ============================================================
for c in ["date", "exdate", "last_date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

df = df.dropna(subset=["date", "exdate"]).copy()

# quote date inferred from file (single day assumed)
QUOTE_DATE = df["date"].iloc[0]

# ============================================================
# NUMERIC FIELDS
# ============================================================
num_cols = [
    "strike_price", "best_bid", "best_offer",
    "volume", "open_interest", "impl_volatility", "forward_price"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# normalize strike
df["strike"] = df["strike_price"] / 1000.0

# ============================================================
# CONSTANT / DERIVED FIELDS
# ============================================================
df["ticker"] = df.get("ticker", "^SPX")
df["underlying_last"] = float(SPOT_PRICE)
df["fetched_at_utc"] = pd.Timestamp.utcnow().replace(tzinfo=timezone.utc)

df["type"] = df["cp_flag"].map({"C": "call", "P": "put"}).astype("string")

# ============================================================
# TENOR SELECTION
# ============================================================
expiries = pd.DatetimeIndex(df["exdate"].unique()).dropna().sort_values()
expiry_to_tenor = {}

for m in TENORS_MONTHS:
    target = QUOTE_DATE + DateOffset(months=m)
    diffs = np.abs((expiries - target).days.astype(int))
    nearest = expiries[int(diffs.argmin())]
    if abs((nearest - target).days) <= TOLERANCE_DAYS:
        expiry_to_tenor[nearest] = m

df = df[df["exdate"].isin(expiry_to_tenor.keys())].copy()
df["tenor_months"] = df["exdate"].map(expiry_to_tenor)
df["target_date"] = df["tenor_months"].apply(
    lambda m: (QUOTE_DATE + DateOffset(months=int(m))).normalize()
)

# ============================================================
# BID / ASK / MID / SPREADS
# ============================================================
df["bid"] = df["best_bid"]
df["ask"] = df["best_offer"]

df["lastPrice"] = np.nan  # not available in your header

df["mid"] = np.where(
    df["bid"].notna() & df["ask"].notna(),
    (df["bid"] + df["ask"]) / 2.0,
    np.nan
)

df["spread_abs"] = np.where(
    df["bid"].notna() & df["ask"].notna(),
    df["ask"] - df["bid"],
    np.nan
)

df["spread_pct"] = np.where(
    df["mid"].notna() & (df["mid"] != 0),
    df["spread_abs"] / df["mid"],
    np.nan
)

# ============================================================
# IN-THE-MONEY FLAG
# ============================================================
df["inTheMoney"] = np.where(
    ((df["type"] == "call") & (df["strike"] < df["underlying_last"])) |
    ((df["type"] == "put")  & (df["strike"] > df["underlying_last"])),
    True,
    False
)

# ============================================================
# FINAL OUTPUT SCHEMA
# ============================================================
out = pd.DataFrame({
    "ticker": df["ticker"],
    "tenor_months": df["tenor_months"],
    "target_date": df["target_date"].dt.date,
    "expiration": df["exdate"].dt.date,
    "fetched_at_utc": df["fetched_at_utc"],

    "underlying_last": df["underlying_last"],
    "contractSymbol": df["optionid"],
    "lastTradeDate": df["last_date"],

    "strike": df["strike"],
    "lastPrice": df["lastPrice"],
    "bid": df["bid"],
    "ask": df["ask"],
    "change": np.nan,
    "percentChange": np.nan,
    "volume": df["volume"],
    "openInterest": df["open_interest"],
    "impliedVolatility": df["impl_volatility"],
    "inTheMoney": df["inTheMoney"],
    "contractSize": "REGULAR",
    "currency": "USD",
    "type": df["type"],

    "mid": df["mid"],
    "spread_abs": df["spread_abs"],
    "spread_pct": df["spread_pct"],
})

# ============================================================
# SAVE
# ============================================================
out.to_csv(OUTPUT_CSV, index=False)

print("===================================")
print(f"Quote date inferred: {QUOTE_DATE.date()}")
print(f"Spot price used:     {SPOT_PRICE}")
print(f"Rows written:        {len(out):,}")
print(f"Tenors present:      {sorted(out['tenor_months'].unique())}")
print(f"Saved to:            {OUTPUT_CSV}")
print("===================================")
