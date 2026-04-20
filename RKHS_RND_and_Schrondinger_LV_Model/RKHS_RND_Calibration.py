"""
RKHS RND calibration 
"""

import os
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

USE_TORCH = True
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

TORCH_DTYPE = torch.float64 if TORCH_AVAILABLE else None

def get_device():
    if USE_TORCH and TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

CSV_PATH = "./data/SPX_options_cleaned.csv"
RATES_CSV_PATH = "./data/discount_shortpoints.csv"
PLOTS_DIR = "./plots"
OUT_DIR = "./calibrated_rnd"
SAVE_PLOTS = True
SHOW_PLOTS = False
DIVIDEND_YIELD = 0.0

# ============================================================
# Day count convention — must match downstream pricers
# ============================================================
DAY_COUNT_DENOM = 365.0
VAL_DATE = "2025-01-02"   # valuation date (settlement date)

OPT_EXPIRY_COL = "expiration"
OPT_STRIKE_COL = "strike"
OPT_CP_COL = "type"
OPT_MID_COL = "adjusted_mid"
OPT_TENOR_MONTHS_COL = "tenor_months"
OPT_SPOT_COL = "underlying_last"
RATES_TENOR_COL = "tenor"
RATES_ZERO_RATE_CONT_COL = "zero_rate_cont"
MATCH_METHOD = "linear"

@dataclass
class CalibConfig:
    n_grid: int = 5001
    lengthscale_short_base: float = 300
    lengthscale_long: float = 900
    lam: float = 1e-3
    eta_forward: float = 50
    xpad_low: float = 0.001
    xpad_high: float = 5.0
    max_iter: int = 1500
    lr: float = 1e-3
    optimizer: str = "two_phase"
    adam_warmup_iters: int = 500
    adam_lr: float = 2e-3
    smooth_pen_weight: float = 0

CFG = CalibConfig()

# =========================
# Helpers
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize_filename(s: str) -> str:
    keep = "-_.() "
    return "".join(c for c in s if c.isalnum() or c in keep).strip().replace(" ", "_")

def parse_tenor_months(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().upper().replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([MY])", s)
    if m:
        v = float(m.group(1))
        u = m.group(2)
        return v if u == "M" else 12.0 * v
    try:
        return float(s)
    except Exception:
        return None

def act365_yearfrac(d0: date, d1: date) -> float:
    """ACT/365 year fraction — matches downstream pricers exactly."""
    return max(0.0, (d1 - d0).days / DAY_COUNT_DENOM)

def parse_date(s) -> Optional[date]:
    """Parse a date string or datetime to a date object."""
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, str):
        try:
            return datetime.strptime(s.strip()[:10], "%Y-%m-%d").date()
        except Exception:
            return None
    return None

def build_rates_curve_months(rts: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    tenor_m = rts[RATES_TENOR_COL].apply(parse_tenor_months)
    zr = pd.to_numeric(rts[RATES_ZERO_RATE_CONT_COL], errors="coerce")
    tmp = pd.DataFrame({"tenor_months": tenor_m, "zr": zr}).dropna()
    if tmp.empty:
        raise ValueError("Rates curve empty.")
    tmp = tmp.sort_values("tenor_months")
    return tmp["tenor_months"].to_numpy(float), tmp["zr"].to_numpy(float)

def build_discount_rate_dict(opt: pd.DataFrame, rts: pd.DataFrame) -> Dict[str, float]:
    tenors, zrs = build_rates_curve_months(rts)
    opt2 = opt.copy()
    opt2["exp_dt"] = pd.to_datetime(opt2[OPT_EXPIRY_COL], errors="coerce")
    opt2 = opt2.dropna(subset=["exp_dt"])
    exp_key = opt2["exp_dt"].dt.strftime("%Y-%m-%d")
    exp_m = opt2.groupby(exp_key)[OPT_TENOR_MONTHS_COL].median().sort_index()
    if exp_m.empty:
        raise ValueError("Could not compute median tenor_months per expiry.")
    if MATCH_METHOD.lower() == "linear":
        f = lambda m: float(np.interp(m, tenors, zrs))
    elif MATCH_METHOD.lower() == "nearest":
        f = lambda m: float(zrs[int(np.argmin(np.abs(tenors - m)))])
    else:
        raise ValueError("MATCH_METHOD must be 'linear' or 'nearest'.")
    return {e: f(float(m)) for e, m in exp_m.items()}

def compute_forward_from_parity_atm_weighted(
    grp: pd.DataFrame, df: float, atm_sigma: float = 0.10,
) -> float:
    K = pd.to_numeric(grp[OPT_STRIKE_COL], errors="coerce")
    mid = pd.to_numeric(grp[OPT_MID_COL], errors="coerce")
    typ = grp[OPT_CP_COL].astype(str).str.lower()
    tmp = pd.DataFrame({"K": K, "mid": mid, "type": typ}).dropna()
    tmp = tmp[tmp["type"].isin(["call", "put"])]
    calls = tmp[tmp["type"] == "call"].groupby("K")["mid"].median()
    puts  = tmp[tmp["type"] == "put"].groupby("K")["mid"].median()
    common = calls.index.intersection(puts.index)
    if len(common) < 3:
        raise ValueError("Not enough matched call/put strikes for forward (need >=3).")
    Kc = common.to_numpy(float)
    C  = calls.loc[common].to_numpy(float)
    P  = puts.loc[common].to_numpy(float)
    Fk = Kc + (C - P) / df
    F0 = float(np.median(Kc))
    m = np.log(np.clip(Kc / F0, 1e-12, None))
    w = np.exp(-0.5 * (m / atm_sigma) ** 2)
    ok = np.isfinite(Fk) & np.isfinite(w)
    Fk, w = Fk[ok], w[ok]
    if len(Fk) == 0:
        raise ValueError("Forward estimation failed.")
    return float(np.sum(w * Fk) / np.sum(w))

def compute_liquidity_weights(
    df_quotes: pd.DataFrame,
    spread_abs_col: str = "spread_abs", bid_col: str = "bid", ask_col: str = "ask",
    vol_col: str = "volume", oi_col: str = "openInterest",
    beta_oi: float = 0.25, eps: float = 1e-4, clip_q: float = 0.95,
) -> np.ndarray:
    if spread_abs_col in df_quotes.columns:
        spr = pd.to_numeric(df_quotes[spread_abs_col], errors="coerce").to_numpy(float)
    else:
        bid = pd.to_numeric(df_quotes.get(bid_col), errors="coerce").to_numpy(float)
        ask = pd.to_numeric(df_quotes.get(ask_col), errors="coerce").to_numpy(float)
        spr = ask - bid
    spr = np.where(np.isfinite(spr), spr, np.nan)
    if np.any(np.isfinite(spr)):
        spr = np.nan_to_num(spr, nan=float(np.nanmedian(spr)))
    else:
        spr = np.full(len(df_quotes), 0.01, dtype=float)
    spr = np.maximum(spr, eps)
    vol = pd.to_numeric(df_quotes.get(vol_col), errors="coerce").to_numpy(float) if vol_col in df_quotes.columns else np.zeros(len(df_quotes))
    oi  = pd.to_numeric(df_quotes.get(oi_col),  errors="coerce").to_numpy(float) if oi_col in df_quotes.columns else np.zeros(len(df_quotes))
    vol = np.nan_to_num(vol, nan=0.0)
    oi  = np.nan_to_num(oi,  nan=0.0)
    liq = np.log1p(vol) + beta_oi * np.log1p(oi) + 1.0
    w = liq / (spr ** 2)
    w = np.maximum(w, 1e-12)
    cap = float(np.quantile(w, clip_q)) if len(w) > 5 else float(np.max(w))
    w = np.clip(w, 0.0, cap)
    w = w / float(np.mean(w))
    return w


def rbf2_kernel_torch(x, y, ell1, ell2, w1, w2):
    d = x[:, None] - y[None, :]
    K1 = torch.exp(-0.5 * (d / ell1) ** 2)
    K2 = torch.exp(-0.5 * (d / ell2) ** 2)
    return w1 * K1 + w2 * K2


@dataclass
class CalibResult:
    expiry: str
    tenor_months: int
    T: float
    r: float
    df: float
    forward: float
    xg: np.ndarray
    q: np.ndarray
    w: np.ndarray
    info: Dict


def calibrate_expiry_rnd(
    grp: pd.DataFrame, expiry: str, r: float, val_date: date, cfg: CalibConfig
) -> CalibResult:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required.")

    # ================================================================
    # T from ACT/365 (matching downstream pricers)
    # ================================================================
    expiry_date = parse_date(expiry)
    if expiry_date is None:
        raise ValueError(f"Cannot parse expiry date: {expiry}")

    T = act365_yearfrac(val_date, expiry_date)
    if T <= 0:
        raise ValueError(f"Non-positive T={T} for expiry={expiry}, val_date={val_date}")

    # Keep tenor_months for display/naming only
    tenor_m_col = pd.to_numeric(grp[OPT_TENOR_MONTHS_COL], errors="coerce").to_numpy(dtype=float)
    tenor_months = int(round(float(np.nanmedian(tenor_m_col))))

    T_old = tenor_months / 12.0
    print(f"    T fix: tenor_months/12={T_old:.6f} → act/365={T:.6f} "
          f"(Δ={abs(T - T_old)*365:.1f} days, {abs(T - T_old)/T*1e4:.0f} bps)")

    df_val = float(np.exp(-r * T))
    F = compute_forward_from_parity_atm_weighted(grp, df=df_val, atm_sigma=0.10)

    # Tenor-adaptive short lengthscale
    ell_short = max(cfg.lengthscale_short_base,
                    50.0 * np.sqrt(T) * F / 6000.0)
    ell_long = cfg.lengthscale_long

    quotes = grp.copy()
    quotes["K"] = pd.to_numeric(quotes[OPT_STRIKE_COL], errors="coerce")
    quotes["mid"] = pd.to_numeric(quotes[OPT_MID_COL], errors="coerce")
    quotes["type"] = quotes[OPT_CP_COL].astype(str).str.lower()
    quotes = quotes.dropna(subset=["K", "mid", "type"])
    quotes = quotes[(quotes["mid"] >= 0) & (quotes["type"].isin(["call", "put"]))]
    quotes["is_call"] = (quotes["type"] == "call")

    keep = (quotes["is_call"] & (quotes["K"] >= F)) | ((~quotes["is_call"]) & (quotes["K"] <= F))
    quotes = quotes.loc[keep].copy()

    if len(quotes) < 10:
        raise ValueError(f"Not enough OTM quotes ({len(quotes)}).")

    K = quotes["K"].to_numpy(float)
    mid = quotes["mid"].to_numpy(float)
    is_call = quotes["is_call"].to_numpy(bool)
    w_opt = compute_liquidity_weights(quotes)

    Kmin, Kmax = float(np.min(K)), float(np.max(K))
    x_low = max(1e-8, cfg.xpad_low * Kmin)
    x_high = cfg.xpad_high * Kmax
    xg = np.linspace(x_low, x_high, cfg.n_grid)
    dx = float(xg[1] - xg[0])

    dev = get_device()
    xg_t = torch.tensor(xg, dtype=TORCH_DTYPE, device=dev)
    K_t = torch.tensor(K, dtype=TORCH_DTYPE, device=dev)
    mid_t = torch.tensor(mid, dtype=TORCH_DTYPE, device=dev)
    is_call_t = torch.tensor(is_call.astype(np.int64), device=dev)
    w_opt_t = torch.tensor(w_opt, dtype=TORCH_DTYPE, device=dev)

    call_mask = is_call_t.bool()
    put_mask = ~call_mask

    x_mat = xg_t[None, :].repeat(K_t.shape[0], 1)
    K_mat = K_t[:, None].repeat(1, xg_t.shape[0])
    call_pay = torch.relu(x_mat - K_mat)
    put_pay  = torch.relu(K_mat - x_mat)
    payoff = torch.where(is_call_t[:, None].bool(), call_pay, put_pay)

    alpha = torch.zeros(cfg.n_grid, dtype=TORCH_DTYPE, device=dev, requires_grad=True)
    raw_w = torch.zeros(2, dtype=TORCH_DTYPE, device=dev, requires_grad=True)

    df_t = torch.tensor(df_val, dtype=TORCH_DTYPE, device=dev)
    F_t  = torch.tensor(F, dtype=TORCH_DTYPE, device=dev)

    def get_w12():
        ww = torch.nn.functional.softplus(raw_w)
        ww = ww / torch.sum(ww)
        return ww[0], ww[1]

    def loss_fn():
        w1, w2 = get_w12()
        Kg = rbf2_kernel_torch(xg_t, xg_t, ell_short, ell_long, w1, w2)
        f = Kg @ alpha
        w_grid = torch.softmax(f, dim=0)

        model = df_t * (payoff @ w_grid)
        err = model - mid_t
        weighted_err2 = w_opt_t * err * err
        data = torch.sum(weighted_err2)

        ridge = cfg.lam * (alpha @ (Kg @ alpha))

        mean_ST = torch.sum(xg_t * w_grid)
        fwd_pen = cfg.eta_forward * (mean_ST - F_t) ** 2

        d2 = f[2:] - 2*f[1:-1] + f[:-2]
        smooth_pen = cfg.smooth_pen_weight * torch.sum(d2 * d2)

        return data + ridge + fwd_pen + smooth_pen

    params = [alpha, raw_w]

    if cfg.optimizer.lower() == "two_phase":
        opt_adam = torch.optim.Adam(params, lr=cfg.adam_lr)
        for _ in range(cfg.adam_warmup_iters):
            opt_adam.zero_grad(set_to_none=True)
            L = loss_fn()
            L.backward()
            opt_adam.step()

        lbfgs_iters = max(cfg.max_iter - cfg.adam_warmup_iters, 200)
        opt_lbfgs = torch.optim.LBFGS(params, lr=cfg.lr,
                                       max_iter=lbfgs_iters,
                                       line_search_fn="strong_wolfe")
        def closure():
            opt_lbfgs.zero_grad(set_to_none=True)
            L = loss_fn()
            L.backward()
            return L
        opt_lbfgs.step(closure)
    elif cfg.optimizer.lower() == "adam":
        opt = torch.optim.Adam(params, lr=cfg.lr)
        for _ in range(cfg.max_iter):
            opt.zero_grad(set_to_none=True)
            L = loss_fn()
            L.backward()
            opt.step()
    else:
        opt = torch.optim.LBFGS(params, lr=cfg.lr, max_iter=cfg.max_iter, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(set_to_none=True)
            L = loss_fn()
            L.backward()
            return L
        opt.step(closure)

    with torch.no_grad():
        w1, w2 = get_w12()
        Kg = rbf2_kernel_torch(xg_t, xg_t, ell_short, ell_long, w1, w2)
        f = Kg @ alpha
        w_grid = torch.softmax(f, dim=0)

        q = (w_grid / dx).detach().cpu().numpy()
        w_np = w_grid.detach().cpu().numpy()

        final_loss = float(loss_fn().detach().cpu().item())
        mean_ST = float(torch.sum(xg_t * w_grid).detach().cpu().item())

        model_prices = (df_t * (payoff @ w_grid)).detach().cpu().numpy()
        dollar_rmse = float(np.sqrt(np.mean((model_prices - mid) ** 2)))
        call_rmse = float(np.sqrt(np.mean((model_prices[is_call] - mid[is_call]) ** 2))) if np.any(is_call) else 0.0
        put_rmse  = float(np.sqrt(np.mean((model_prices[~is_call] - mid[~is_call]) ** 2))) if np.any(~is_call) else 0.0

    info = {
        "n_otm": int(len(K)),
        "n_calls": int(np.sum(is_call)),
        "n_puts": int(np.sum(~is_call)),
        "loss": final_loss,
        "E_ST": mean_ST,
        "F": float(F),
        "fwd_err_bps": abs(mean_ST - F) / F * 1e4,
        "dollar_rmse": dollar_rmse,
        "call_rmse": call_rmse,
        "put_rmse": put_rmse,
        "device": str(dev),
        "kernel_w1": float(w1.detach().cpu().item()),
        "kernel_w2": float(w2.detach().cpu().item()),
        "ell1": float(ell_short),
        "ell2": float(ell_long),
        "T_tenor_months_12": float(tenor_months / 12.0),  # old T for reference
    }

    return CalibResult(
        expiry=expiry, tenor_months=tenor_months, T=T, r=float(r), df=df_val,
        forward=float(F), xg=xg, q=q, w=w_np, info=info,
    )


# =========================
# Plotting
# =========================
def plot_density(res: CalibResult):
    plt.figure(figsize=(9, 5))
    plt.plot(res.xg, res.q)
    plt.title(
        f"{res.expiry} | RND q(S_T) | T={res.T:.4f}y (act/365) | F={res.forward:.2f} | "
        f"w=[{res.info['kernel_w1']:.2f},{res.info['kernel_w2']:.2f}] | "
        f"ell=[{res.info['ell1']:.0f},{res.info['ell2']:.0f}]"
    )
    plt.xlabel("S_T")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.xlim(0,12500)
    if SAVE_PLOTS:
        ensure_dir(PLOTS_DIR)
        fn = os.path.join(PLOTS_DIR, f"density_{sanitize_filename(res.expiry)}.png")
        plt.savefig(fn, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_strike_slice_otm_fit_curve(grp: pd.DataFrame, res: CalibResult, nK_curve: int = 200):
    K = pd.to_numeric(grp[OPT_STRIKE_COL], errors="coerce").to_numpy(float)
    mid = pd.to_numeric(grp[OPT_MID_COL], errors="coerce").to_numpy(float)
    typ = grp[OPT_CP_COL].astype(str).str.lower().to_numpy()

    ok = np.isfinite(K) & np.isfinite(mid) & (mid >= 0) & np.isin(typ, ["call", "put"])
    K, mid, typ = K[ok], mid[ok], typ[ok]
    is_call = (typ == "call")

    F = res.forward
    keep = (is_call & (K >= F)) | ((~is_call) & (K <= F))
    K, mid, is_call = K[keep], mid[keep], is_call[keep]

    Kc_mkt, Pc_mkt = K[is_call], mid[is_call]
    Kp_mkt, Pp_mkt = K[~is_call], mid[~is_call]
    sc, sp = np.argsort(Kc_mkt), np.argsort(Kp_mkt)
    Kc_mkt, Pc_mkt = Kc_mkt[sc], Pc_mkt[sc]
    Kp_mkt, Pp_mkt = Kp_mkt[sp], Pp_mkt[sp]

    Kc_min = float(np.min(Kc_mkt)) if len(Kc_mkt) else F
    Kc_max = float(np.max(Kc_mkt)) if len(Kc_mkt) else F
    Kp_min = float(np.min(Kp_mkt)) if len(Kp_mkt) else F
    Kp_max = float(np.max(Kp_mkt)) if len(Kp_mkt) else F

    K_call_grid = np.linspace(max(F, Kc_min), Kc_max, nK_curve) if Kc_max > max(F, Kc_min) else np.array([max(F, Kc_min)])
    K_put_grid  = np.linspace(Kp_min, min(F, Kp_max), nK_curve) if min(F, Kp_max) > Kp_min else np.array([min(F, Kp_max)])

    xg, w, df_v = res.xg, res.w, res.df

    def model_call_price(K0):
        return df_v * np.sum(np.maximum(xg - K0, 0.0) * w)
    def model_put_price(K0):
        return df_v * np.sum(np.maximum(K0 - xg, 0.0) * w)

    call_curve = np.array([model_call_price(k) for k in K_call_grid])
    put_curve  = np.array([model_put_price(k)  for k in K_put_grid])

    plt.figure(figsize=(10, 6))
    if len(Kc_mkt):
        plt.scatter(Kc_mkt, Pc_mkt, s=45, color="tab:blue", alpha=0.7, marker="o", label="Market Call (OTM)")
    if len(Kp_mkt):
        plt.scatter(Kp_mkt, Pp_mkt, s=45, color="tab:red", alpha=0.7, marker="^", label="Market Put (OTM)")
    if len(K_call_grid):
        plt.plot(K_call_grid, call_curve, color="navy", linewidth=2.2, label="Model Call")
    if len(K_put_grid):
        plt.plot(K_put_grid, put_curve, color="darkred", linewidth=2.2, label="Model Put")
    plt.axvline(F, linestyle="--", linewidth=1, color="black", alpha=0.6, label="Forward")

    model_on_quotes = np.array([
        model_call_price(Ki) if is_call[i] else model_put_price(Ki)
        for i, Ki in enumerate(K)
    ])
    rmse_val = float(np.sqrt(np.mean((model_on_quotes - mid) ** 2)))
    call_rmse = float(np.sqrt(np.mean((model_on_quotes[is_call] - mid[is_call]) ** 2))) if np.any(is_call) else 0
    put_rmse  = float(np.sqrt(np.mean((model_on_quotes[~is_call] - mid[~is_call]) ** 2))) if np.any(~is_call) else 0

    plt.title(
        f"{res.expiry} | OTM fit | RMSE=${rmse_val:.4f} "
        f"(C=${call_rmse:.4f}, P=${put_rmse:.4f}) | "
        f"E[S_T]\u2212F = {res.info['E_ST'] - res.info['F']:+.2f} "
        f"({res.info['fwd_err_bps']:.1f} bps)"
    )
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.grid(True, alpha=0.25)
    plt.legend()

    if SAVE_PLOTS:
        ensure_dir(PLOTS_DIR)
        fn = os.path.join(PLOTS_DIR, f"strike_slice_otm_fit_curve_{sanitize_filename(res.expiry)}.png")
        plt.savefig(fn, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_dir(PLOTS_DIR)
    ensure_dir(OUT_DIR)

    val_date = parse_date(VAL_DATE)
    if val_date is None:
        raise ValueError(f"Cannot parse VAL_DATE: {VAL_DATE}")

    opt = pd.read_csv(CSV_PATH)
    rts = pd.read_csv(RATES_CSV_PATH)

    opt["exp_dt"] = pd.to_datetime(opt[OPT_EXPIRY_COL], errors="coerce")
    opt = opt.dropna(subset=["exp_dt"])
    opt["exp"] = opt["exp_dt"].dt.strftime("%Y-%m-%d")

    discount_rates = build_discount_rate_dict(opt, rts)

    print(f"\nValuation date: {val_date}")
    print(f"Day count: ACT/{int(DAY_COUNT_DENOM)}")
    print("\nDiscount rates by maturity:")
    print("-" * 82)
    print(f"{'Expiry':<12} {'T_act365':>10} {'T_m/12':>10} {'Δ(days)':>8} {'r_cont':>12} {'DF':>12}")
    print("-" * 82)
    for exp in sorted(discount_rates.keys()):
        r = float(discount_rates[exp])
        grp = opt[opt["exp"] == exp]
        tenor_m = pd.to_numeric(grp[OPT_TENOR_MONTHS_COL], errors="coerce").to_numpy(dtype=float)
        tenor = float(np.nanmedian(tenor_m))

        exp_date = parse_date(exp)
        T_act = act365_yearfrac(val_date, exp_date) if exp_date else float("nan")
        T_old = tenor / 12.0
        df_v = float(np.exp(-r * T_act)) if np.isfinite(T_act) and T_act > 0 else float("nan")
        delta_days = abs(T_act - T_old) * 365 if np.isfinite(T_act) else float("nan")

        print(f"{exp:<12} {T_act:10.6f} {T_old:10.6f} {delta_days:8.1f} {r:12.6%} {df_v:12.6f}")
    print("-" * 82)

    dev = get_device()
    print(f"\nDevice: {dev} | torch={TORCH_AVAILABLE}")
    print(f"Kernel: w1*RBF(ell_short_adaptive) + w2*RBF({CFG.lengthscale_long})")
    print(f"Optimizer: {CFG.optimizer} | eta_forward={CFG.eta_forward} | smooth_pen={CFG.smooth_pen_weight}")
    print()

    results: Dict[str, CalibResult] = {}

    for exp, grp in opt.groupby("exp", sort=True):
        r = discount_rates.get(exp, None)
        if r is None or (not np.isfinite(r)):
            print(f"[skip] {exp}: missing rate")
            continue

        try:
            res = calibrate_expiry_rnd(grp, expiry=exp, r=float(r), val_date=val_date, cfg=CFG)
            results[exp] = res

            plot_density(res)
            plot_strike_slice_otm_fit_curve(grp, res, nK_curve=250)

            # Save NPZ — format unchanged, T now contains act/365 value
            npz_path = os.path.join(OUT_DIR, f"rnd_{sanitize_filename(exp)}.npz")
            np.savez(
                npz_path,
                expiry=res.expiry,
                tenor_months=res.tenor_months,
                T=res.T,                          # act/365 (FIXED)
                r=res.r,
                df=res.df,
                forward=res.forward,
                xg=res.xg,
                q=res.q,
                w=res.w,
                kernel_w1=res.info["kernel_w1"],
                kernel_w2=res.info["kernel_w2"],
                ell1=res.info["ell1"],
                ell2=res.info["ell2"],
            )

            # Also save with the legacy filename pattern for compatibility
            tenor_m = res.tenor_months
            legacy_path = os.path.join(OUT_DIR, f"{tenor_m}M.npz")
            np.savez(
                legacy_path,
                expiry=res.expiry,
                tenor_months=res.tenor_months,
                T=res.T,
                r=res.r,
                df=res.df,
                forward=res.forward,
                xg=res.xg,
                q=res.q,
                w=res.w,
                kernel_w1=res.info["kernel_w1"],
                kernel_w2=res.info["kernel_w2"],
                ell1=res.info["ell1"],
                ell2=res.info["ell2"],
            )

            print(
                f"[ok] {exp}: n_otm={res.info['n_otm']}"
                f"  T={res.T:.6f}(act/365)"
                f"  $RMSE={res.info['dollar_rmse']:.4f}"
                f"  (C=${res.info['call_rmse']:.4f}, P=${res.info['put_rmse']:.4f})"
                f"  E[S_T]={res.info['E_ST']:.2f}"
                f"  F={res.info['F']:.2f}"
                f"  fwd_err={res.info['fwd_err_bps']:.1f}bp"
                f"  w=[{res.info['kernel_w1']:.2f},{res.info['kernel_w2']:.2f}]"
                f"  ell=[{res.info['ell1']:.0f},{res.info['ell2']:.0f}]"
                f"  dev={res.info['device']}"
            )
        except Exception as e:
            print(f"[fail] {exp}: {e}")

    print(f"\nDone. Calibrated {len(results)} expiries.")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Densities saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()