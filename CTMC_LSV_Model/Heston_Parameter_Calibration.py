"""
Calibrate Heston parameters to *option mid prices* (not IVs), using:
- Market option chain: ^SPX_options_cleaned.csv  (must contain: expiration, strike, type, mid)
- Per-maturity forwards/discounts: {1M,3M,6M,12M,24M}.npz (must contain: expiry, T, df, forward)

Method:
- Build risk-free discount curve from df(T)
- Build dividend discount curve from (forward * df / S0)
- Price each option under Heston (QuantLib AnalyticHestonEngine)
- Calibrate params by least-squares on price errors (SciPy least_squares)

Outputs:
- Calibrated (v0, kappa, theta, sigma, rho)
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# ---------- user inputs ----------
CSV_PATH = "./data/^SPX_options_cleaned.csv"
NPZ_PATHS = [
    "./data/1M.npz",
    "./data/3M.npz",
    "./data/6M.npz",
    "./data/12M.npz",
    "./data/24M.npz",
]

# option filters (tune as needed)
MAX_SPREAD_PCT = 0.60          # drop very wide markets
MIN_MID = 0.05                 # drop near-zero prices
MAX_REL_PRICE = 0.90           # drop absurd quotes: mid > 90% of spot (usually bad data)
MONEYNESS_BAND = (0.5, 1.5)    # keep strikes in [0.5F, 1.5F] per maturity

# initial guess and bounds: (v0, kappa, theta, sigma, rho)
X0 = np.array([0.04, 2.0, 0.04, 0.6, -0.5], dtype=float)
LB = np.array([1e-6, 1e-6, 1e-6, 1e-6, -0.999], dtype=float)
UB = np.array([2.00, 50.0, 2.00, 10.0,  0.999], dtype=float)

# ---------- imports that require installed packages ----------
import QuantLib as ql
from scipy.optimize import least_squares

def load_maturity_table(npz_paths):
    """
    Returns a dict keyed by expiry string 'YYYY-MM-DD':
      { expiry: {"T": float, "df": float, "forward": float} }
    """
    table = {}
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        expiry = str(np.atleast_1d(z["expiry"])[0])  # stored as 0-d array sometimes
        T = float(np.atleast_1d(z["T"])[0])
        df = float(np.atleast_1d(z["df"])[0])
        fwd = float(np.atleast_1d(z["forward"])[0])
        table[expiry] = {"T": T, "df": df, "forward": fwd}
    return table

def ql_date_from_iso(s: str) -> ql.Date:
    y, m, d = map(int, s.split("-"))
    return ql.Date(d, m, y)

def build_curves_match_forward(eval_date: ql.Date, S0: float, maturity_table: dict):
    """
    Builds:
      - risk-free discount curve from df(T)
      - dividend discount curve from DF_q(T) = F(T)*df(T)/S0
    so that the model implied forward matches the NPZ forward at the input maturities.

    maturity_table: {expiry: {"T": float, "df": float, "forward": float}}
    """
    day_counter = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    # sort by T
    items = sorted(
        [(v["T"], float(v["df"]), float(v["forward"])) for v in maturity_table.values()],
        key=lambda x: x[0],
    )

    dates = [eval_date]
    rf_discounts = [1.0]
    div_discounts = [1.0]

    for T, df, fwd in items:
        if T <= 0:
            continue
        d = eval_date + int(round(T * 365.0))
        dates.append(d)

        # risk-free discount
        rf_discounts.append(df)

        # implied dividend discount from forward
        # DF_q(T) = F(T) * DF_r(T) / S0
        div_df = (fwd * df) / S0
        # basic sanity
        if not (0.0 < div_df <= 1.5):  # allow mild >1 if noisy inputs, but warn later
            # don't hard fail; keep to let you diagnose
            pass
        div_discounts.append(div_df)

    rf_curve = ql.DiscountCurve(dates, rf_discounts, day_counter, calendar)
    div_curve = ql.DiscountCurve(dates, div_discounts, day_counter, calendar)

    rf_ts = ql.YieldTermStructureHandle(rf_curve)
    div_ts = ql.YieldTermStructureHandle(div_curve)
    return rf_ts, div_ts

def check_forward_match(S0: float, rf_ts, div_ts, maturity_table: dict):
    """
    Prints NPZ forward vs model-implied forward:
      F_model(T) = S0 * DF_q(T) / DF_r(T)
    """
    print("\nForward match check (NPZ vs model-implied):")
    rows = []
    for exp, info in sorted(maturity_table.items(), key=lambda kv: kv[1]["T"]):
        T = float(info["T"])
        df_r = rf_ts.discount(T)
        df_q = div_ts.discount(T)
        f_model = S0 * (df_q / df_r)
        f_mkt = float(info["forward"])
        rel = (f_model - f_mkt) / f_mkt
        rows.append((exp, T, f_mkt, f_model, rel))
        print(f"{exp}  T={T:.6f}  F_mkt={f_mkt:.6f}  F_model={f_model:.6f}  rel_diff={rel:+.4%}")
    return rows

def prepare_market_quotes(csv_path: str, maturity_table: dict):
    df = pd.read_csv(csv_path)

    # basic cleaning
    df = df.copy()
    df["expiration"] = df["expiration"].astype(str)
    df["type"] = df["type"].astype(str).str.lower()
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["spread_pct"] = pd.to_numeric(df.get("spread_pct", np.nan), errors="coerce")

    # take spot from file (robust)
    S0 = float(pd.to_numeric(df["underlying_last"], errors="coerce").median())

    # keep only expirations we have term data for
    df = df[df["expiration"].isin(maturity_table.keys())]

    # keep sensible quotes
    df = df[df["mid"].notna() & df["strike"].notna()]
    df = df[df["mid"] > MIN_MID]
    df = df[df["mid"] < MAX_REL_PRICE * S0]

    if "spread_pct" in df.columns:
        df = df[(df["spread_pct"].isna()) | (df["spread_pct"] <= MAX_SPREAD_PCT)]

    # moneyness filter per maturity using forward
    rows = []
    for exp, g in df.groupby("expiration"):
        F = maturity_table[exp]["forward"]
        kmin = MONEYNESS_BAND[0] * F
        kmax = MONEYNESS_BAND[1] * F
        gg = g[(g["strike"] >= kmin) & (g["strike"] <= kmax)]
        rows.append(gg)

    df = pd.concat(rows, axis=0) if rows else df.iloc[0:0]

    # final minimal columns
    df = df[["expiration", "strike", "type", "mid"]].reset_index(drop=True)
    return S0, df

def build_option_instruments(eval_date: ql.Date, market_df: pd.DataFrame, maturity_table: dict):
    """
    For each row, create (option, T, market_mid).
    We map each expiry to an exercise date = eval_date + round(T*365) days.
    """
    # map expiry -> exercise date + time (using T from npz)
    expiry_to_exdate = {}
    expiry_to_T = {}
    for expiry, info in maturity_table.items():
        T = float(info["T"])
        expiry_to_T[expiry] = T
        expiry_to_exdate[expiry] = eval_date + int(round(T * 365.0))

    instruments = []
    for _, row in market_df.iterrows():
        expiry = row["expiration"]
        K = float(row["strike"])
        opt_type = row["type"]
        mid = float(row["mid"])

        ex_date = expiry_to_exdate[expiry]
        T = expiry_to_T[expiry]

        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if opt_type == "call" else ql.Option.Put,
            K,
        )
        exercise = ql.EuropeanExercise(ex_date)
        option = ql.VanillaOption(payoff, exercise)
        instruments.append((option, T, mid, opt_type, K, expiry))

    return instruments

def make_engine_and_process(S0: float, rf_ts, div_ts, params):
    v0, kappa, theta, sigma, rho = params

    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    process = ql.HestonProcess(rf_ts, div_ts, spot, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    return process, model, engine

def residuals(params, instruments, S0, rf_ts, div_ts, weights=None):
    # penalize invalid regions quickly
    v0, kappa, theta, sigma, rho = params
    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
        return 1e6 * np.ones(len(instruments), dtype=float)

    # (optional) enforce Feller condition 2*kappa*theta > sigma^2 softly
    feller = 2.0 * kappa * theta - sigma * sigma
    feller_penalty = 0.0
    if feller <= 0:
        feller_penalty = 10.0 * (-feller)

    _, _, engine = make_engine_and_process(S0, rf_ts, div_ts, params)

    res = np.empty(len(instruments), dtype=float)
    for i, (opt, _T, mid, _typ, _K, _exp) in enumerate(instruments):
        opt.setPricingEngine(engine)
        model_price = opt.NPV()

        err = (model_price - mid)
        if weights is not None:
            err *= weights[i]

        # add soft penalty
        res[i] = err + (feller_penalty if i == 0 else 0.0)

    return res

def plot_market_vs_model_save(
    rep,
    out_dir="heston_fit_plots",
    expirations=None,
    types=("call", "put"),
    model_label="Model (Heston)",
    fname_prefix="heston_price_fit_",
):
    """
    Saves strike vs option price plots:
      - Market (scatter) vs Model (line)
      - One figure per expiration
      - One subplot per option type (call / put)

    rep: DataFrame with columns: expiration, type, strike, mid, model
    """
    os.makedirs(out_dir, exist_ok=True)

    rep = rep.copy()
    rep["type"] = rep["type"].str.lower()

    if expirations is None:
        expirations = sorted(rep["expiration"].unique())

    for exp in expirations:
        sub = rep[rep["expiration"] == exp]
        if sub.empty:
            continue

        types_present = [t for t in types if (sub["type"] == t).any()]
        if not types_present:
            continue

        fig, axes = plt.subplots(
            1,
            len(types_present),
            figsize=(7 * len(types_present), 5),
            squeeze=False,
        )

        for i, t in enumerate(types_present):
            s = sub[sub["type"] == t].sort_values("strike")
            ax = axes[0, i]

            ax.scatter(
                s["strike"].values,
                s["mid"].values,
                label="Market (mid)",
                marker="o",
            )
            ax.plot(
                s["strike"].values,
                s["model"].values,
                label=model_label,
                linewidth=2,
            )

            ax.set_title(f"{exp} — {t.upper()}")
            ax.set_xlabel("Strike")
            ax.set_ylabel("Option price")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        fname = f"{fname_prefix}{exp}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {path}")

def _cont_zero_rate_from_df(df: float, T: float) -> float:
    """Continuous-compounded zero rate r such that df = exp(-r*T)."""
    if T <= 0:
        return 0.0
    df = float(df)
    return -np.log(df) / T

# --- add near your other helpers ---

def make_engine_and_process_rho0(S0: float, rf_ts, div_ts, params4):
    """params4 = (v0, kappa, theta, sigma), rho fixed to 0."""
    v0, kappa, theta, sigma = params4
    rho = 0.0
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    process = ql.HestonProcess(rf_ts, div_ts, spot, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    return process, model, engine

def residuals_rho0(params4, instruments, S0, rf_ts, div_ts, weights=None):
    v0, kappa, theta, sigma = params4
    rho = 0.0

    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e6 * np.ones(len(instruments), dtype=float)

    # soft Feller penalty (same idea as before)
    feller = 2.0 * kappa * theta - sigma * sigma
    feller_penalty = 0.0
    if feller <= 0:
        feller_penalty = 10.0 * (-feller)

    _, _, engine = make_engine_and_process_rho0(S0, rf_ts, div_ts, params4)

    res = np.empty(len(instruments), dtype=float)
    for i, (opt, _T, mid, _typ, _K, _exp) in enumerate(instruments):
        opt.setPricingEngine(engine)
        model_price = opt.NPV()
        err = (model_price - mid)
        if weights is not None:
            err *= weights[i]
        res[i] = err + (feller_penalty if i == 0 else 0.0)
    return res

def save_rho0_params_json(
    out_path: str,
    S0: float,
    calibrated_params4: np.ndarray,
    fit_rmse: float,
    optimizer_success: bool,
    optimizer_message: str,
):
    v0, kappa, theta, sigma = map(float, calibrated_params4)
    rho = 0.0
    obj = {
        "S0": float(S0),
        "calibrated_params": {
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "rho": rho,
            "feller_2kappa_theta_minus_sigma2": float(2.0 * kappa * theta - sigma * sigma),
        },
        "fit": {
            "rmse_price": float(fit_rmse),
            "success": bool(optimizer_success),
            "message": str(optimizer_message),
        },
        "notes": {
            "rho_fixed": "rho was fixed to 0.0 during calibration; only (v0,kappa,theta,sigma) were optimized."
        },
    }
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved rho=0 parameters to: {out_path}")

def save_curves_and_params(
    out_dir: str,
    maturity_table: dict,
    S0: float,
    rf_ts: ql.YieldTermStructureHandle,
    div_ts: ql.YieldTermStructureHandle,
    calibrated_params: np.ndarray,
    fit_rmse: float,
    optimizer_success: bool,
    optimizer_message: str,
):
    """
    Saves:
      - curves.csv: per-expiry DF_r, r(T), DF_q, q(T), forwards (NPZ and implied)
      - heston_params.json: calibrated Heston params + fit stats
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # ----- curves table at your NPZ maturities -----
    rows = []
    for exp, info in sorted(maturity_table.items(), key=lambda kv: float(kv[1]["T"])):
        T = float(info["T"])
        df_r_mkt = float(info["df"])
        f_mkt = float(info["forward"])

        # Curve-implied dfs from QuantLib (at time T)
        df_r = float(rf_ts.discount(T))
        df_q = float(div_ts.discount(T))

        # Model-implied forward from curves
        f_model = float(S0 * (df_q / df_r)) if df_r > 0 else np.nan

        # Continuous zero rates from discount factors
        r_mkt = _cont_zero_rate_from_df(df_r_mkt, T)
        r_curve = _cont_zero_rate_from_df(df_r, T)
        q_curve = _cont_zero_rate_from_df(df_q, T)

        # Also the implied dividend DF directly from NPZ (should match df_q at pillars)
        df_q_from_npz = (f_mkt * df_r_mkt) / S0
        q_from_npz = _cont_zero_rate_from_df(df_q_from_npz, T)

        rows.append(
            {
                "expiration": exp,
                "T": T,
                "S0": S0,
                "F_mkt": f_mkt,
                "DF_r_mkt": df_r_mkt,
                "r_mkt_cont": r_mkt,
                "DF_r_curve": df_r,
                "r_curve_cont": r_curve,
                "DF_q_curve": df_q,
                "q_curve_cont": q_curve,
                "DF_q_from_npz": df_q_from_npz,
                "q_from_npz_cont": q_from_npz,
                "F_model_from_curves": f_model,
                "forward_rel_diff": (f_model - f_mkt) / f_mkt if f_mkt != 0 else np.nan,
            }
        )

    curves_df = pd.DataFrame(rows)
    curves_path = os.path.join(out_dir, "curves.csv")
    curves_df.to_csv(curves_path, index=False)

    # ----- params JSON -----
    v0, kappa, theta, sigma, rho = map(float, calibrated_params)

    params_obj = {
        "S0": float(S0),
        "calibrated_params": {
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "rho": rho,
            "feller_2kappa_theta_minus_sigma2": float(2.0 * kappa * theta - sigma * sigma),
        },
        "fit": {
            "rmse_price": float(fit_rmse),
            "success": bool(optimizer_success),
            "message": str(optimizer_message),
        },
        "notes": {
            "rates": "r(T) are continuous-compounded zero rates inferred from discount factors via r=-ln(DF)/T",
            "dividends": "q(T) inferred from dividend discount factors DF_q; DF_q itself implied from NPZ forwards via DF_q=F*DF_r/S0",
        },
    }

    params_path = os.path.join(out_dir, "heston_params.json")
    with open(params_path, "w") as f:
        json.dump(params_obj, f, indent=2)

    print(f"\nSaved curves to: {curves_path}")
    print(f"Saved parameters to: {params_path}")

def main():
    eval_date = ql.Date(30, 12, 2024)
    ql.Settings.instance().evaluationDate = eval_date

    maturity_table = load_maturity_table(NPZ_PATHS)

    S0, market_df = prepare_market_quotes(CSV_PATH, maturity_table)
    if market_df.empty:
        raise RuntimeError("No usable options after filtering (check expirations / filters).")

    rf_ts, div_ts = build_curves_match_forward(eval_date, S0, maturity_table)
    check_forward_match(S0, rf_ts, div_ts, maturity_table)

    instruments = build_option_instruments(eval_date, market_df, maturity_table)

    print(f"Spot S0 = {S0:.4f}")
    print(f"Number of options used = {len(instruments)}")

    # ============================================================
    # 1) Full calibration (v0, kappa, theta, sigma, rho)
    # ============================================================
    print("\nStarting least-squares calibration (FULL: v0,kappa,theta,sigma,rho)...")
    res_full = least_squares(
        fun=lambda x: residuals(x, instruments, S0, rf_ts, div_ts, weights=None),
        x0=X0,
        bounds=(LB, UB),
        method="trf",
        ftol=1e-20,
        xtol=1e-20,
        gtol=1e-10,
        max_nfev=200,
        verbose=2,
    )

    rmse_full = float(np.sqrt(np.mean(res_full.fun**2)))
    print("\n==== Calibrated Heston parameters (FULL) ====")
    v0, kappa, theta, sigma, rho = res_full.x
    print(f"v0    = {v0:.8f}")
    print(f"kappa = {kappa:.8f}")
    print(f"theta = {theta:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"rho   = {rho:.8f}")
    print(f"Feller 2*kappa*theta - sigma^2 = {2*kappa*theta - sigma*sigma:.8f}")
    print(f"RMSE (price) = {rmse_full:.6f}")
    print(f"Success: {res_full.success} | Status: {res_full.status} | Msg: {res_full.message}")

    # Report + plots + save (your existing logic)
    _, _, engine = make_engine_and_process(S0, rf_ts, div_ts, res_full.x)
    rows = []
    for opt, _T, mid, typ, K, exp in instruments:
        opt.setPricingEngine(engine)
        mp = opt.NPV()
        rows.append((exp, typ, K, mid, mp, mp - mid))
    rep = pd.DataFrame(rows, columns=["expiration", "type", "strike", "mid", "model", "err"])
    by_exp = rep.groupby("expiration")["err"].apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
    print("\nRMSE by expiration (FULL):")
    print(by_exp.sort_index().to_string())

    plot_market_vs_model_save(rep)

    out_dir = "calibration_outputs"
    save_curves_and_params(
        out_dir=out_dir,
        maturity_table=maturity_table,
        S0=S0,
        rf_ts=rf_ts,
        div_ts=div_ts,
        calibrated_params=res_full.x,
        fit_rmse=rmse_full,
        optimizer_success=res_full.success,
        optimizer_message=res_full.message,
    )

    # ============================================================
    # 2) Rho-fixed-to-zero calibration (v0, kappa, theta, sigma)
    # ============================================================
    print("\nStarting least-squares calibration (RHO=0: v0,kappa,theta,sigma; rho fixed)...")

    X0_4 = np.array([X0[0], X0[1], X0[2], X0[3]], dtype=float)
    LB_4 = np.array([LB[0], LB[1], LB[2], LB[3]], dtype=float)
    UB_4 = np.array([UB[0], UB[1], UB[2], UB[3]], dtype=float)

    res_rho0 = least_squares(
        fun=lambda x: residuals_rho0(x, instruments, S0, rf_ts, div_ts, weights=None),
        x0=X0_4,
        bounds=(LB_4, UB_4),
        method="trf",
        ftol=1e-20,
        xtol=1e-20,
        gtol=1e-10,
        max_nfev=200,
        verbose=2,
    )

    rmse_rho0 = float(np.sqrt(np.mean(res_rho0.fun**2)))
    v0, kappa, theta, sigma = res_rho0.x
    print("\n==== Calibrated Heston parameters (RHO=0) ====")
    print(f"v0    = {v0:.8f}")
    print(f"kappa = {kappa:.8f}")
    print(f"theta = {theta:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"rho   = 0.00000000 (fixed)")
    print(f"Feller 2*kappa*theta - sigma^2 = {2*kappa*theta - sigma*sigma:.8f}")
    print(f"RMSE (price) = {rmse_rho0:.6f}")
    print(f"Success: {res_rho0.success} | Status: {res_rho0.status} | Msg: {res_rho0.message}")

    # optional per-expiry RMSE report for rho=0
    _, _, engine0 = make_engine_and_process_rho0(S0, rf_ts, div_ts, res_rho0.x)
    rows0 = []
    for opt, _T, mid, typ, K, exp in instruments:
        opt.setPricingEngine(engine0)
        mp = opt.NPV()
        rows0.append((exp, typ, K, mid, mp, mp - mid))
    rep0 = pd.DataFrame(rows0, columns=["expiration", "type", "strike", "mid", "model", "err"])
    plot_market_vs_model_save(
    rep0,
    out_dir=os.path.join(out_dir, "heston_fit_plots_rho0"),
    model_label="Model (Heston, rho=0)",
    fname_prefix="heston_rho0_price_fit",
)
    by_exp0 = rep0.groupby("expiration")["err"].apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
    print("\nRMSE by expiration (RHO=0):")
    print(by_exp0.sort_index().to_string())

    # save rho=0 parameters JSON
    save_rho0_params_json(
        out_path=os.path.join(out_dir, "heston_rho0_parameters.json"),
        S0=S0,
        calibrated_params4=res_rho0.x,
        fit_rmse=rmse_rho0,
        optimizer_success=res_rho0.success,
        optimizer_message=res_rho0.message,
    )

if __name__ == "__main__":
    main()