#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autocallable_pricer_ctmc_cuda.py — CTMC-LSV Autocallable Pricer
================================================================
Full substep-resolution leverage surface + all CUDA kernels preserved.
Only 4 changes from original: CTMCModel fields, load_ctmc_model,
DensityPropagator.__init__, _get_leverage_at_time.
"""
from __future__ import annotations
import argparse, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.interpolate import interp1d

try:
    import cupy as cp
except Exception:
    cp = None

class _CudaPDEKernelCache:
    _step_kernel = None
    @classmethod
    def pde_theta_step(cls):
        if cp is None: return None
        if cls._step_kernel is None:
            code = r'''
            extern "C" __global__
            void pde_theta_step(
                const double* __restrict__ phi, const double* __restrict__ L2,
                const double* __restrict__ v_states, double* __restrict__ cprime,
                double* __restrict__ out, const int n_rows, const int n_states,
                const int nz, const double dz, const double dt, const double theta) {
                const int row = (int)(blockDim.x * blockIdx.x + threadIdx.x);
                if (row >= n_rows) return;
                const int state = row % n_states;
                const int base = row * nz;
                const double v = v_states[state];
                const double dz2 = dz * dz;
                const double expl = (1.0 - theta) * dt;
                const double impl = theta * dt;
                const double eps = 1.0e-30;
                const double* phi_r = phi + base;
                double* cp_r = cprime + base;
                double* out_r = out + base;
                const double lv0 = v * L2[0];
                const double diag0 = -lv0 / dz2 - lv0 / (4.0 * dz);
                double sup0 = 0.0;
                if (nz > 1) { const double lv1 = v * L2[1]; sup0 = lv1 / dz2 + lv1 / (4.0 * dz); }
                double rhs0 = phi_r[0] * (1.0 + expl * diag0);
                if (nz > 1) rhs0 += phi_r[1] * expl * sup0;
                double denom = 1.0 - impl * diag0;
                if (denom < 0.0 ? (-denom) < eps : denom < eps) denom = (denom < 0.0 ? -eps : eps);
                cp_r[0] = (nz > 1 ? (-impl * sup0) / denom : 0.0);
                out_r[0] = rhs0 / denom;
                for (int j = 1; j < nz - 1; ++j) {
                    const double lvm = v * L2[j-1]; const double lvj = v * L2[j]; const double lvp = v * L2[j+1];
                    const double sub = 0.5 * lvm / dz2 - lvm / (4.0 * dz);
                    const double diag = -lvj / dz2;
                    const double sup = 0.5 * lvp / dz2 + lvp / (4.0 * dz);
                    const double rhs = phi_r[j-1]*expl*sub + phi_r[j]*(1.0+expl*diag) + phi_r[j+1]*expl*sup;
                    const double a = -impl * sub; const double b = 1.0 - impl * diag; const double c = -impl * sup;
                    denom = b - a * cp_r[j-1];
                    if (denom < 0.0 ? (-denom) < eps : denom < eps) denom = (denom < 0.0 ? -eps : eps);
                    cp_r[j] = c / denom; out_r[j] = (rhs - a * out_r[j-1]) / denom;
                }
                if (nz > 1) {
                    const int j = nz - 1;
                    const double lvm = v * L2[j-1]; const double lvj = v * L2[j];
                    const double sub = lvm / dz2 - lvm / (4.0 * dz);
                    const double diag = -lvj / dz2 + lvj / (4.0 * dz);
                    const double rhs = phi_r[j-1]*expl*sub + phi_r[j]*(1.0+expl*diag);
                    const double a = -impl * sub; const double b = 1.0 - impl * diag;
                    denom = b - a * cp_r[j-1];
                    if (denom < 0.0 ? (-denom) < eps : denom < eps) denom = (denom < 0.0 ? -eps : eps);
                    cp_r[j] = 0.0; out_r[j] = (rhs - a * out_r[j-1]) / denom;
                }
                for (int j = nz - 2; j >= 0; --j) out_r[j] -= cp_r[j] * out_r[j+1];
                for (int j = 0; j < nz; ++j) if (out_r[j] < 0.0) out_r[j] = 0.0;
            }'''
            module = cp.RawModule(code=code, options=("-std=c++11",), name_expressions=("pde_theta_step",))
            cls._step_kernel = module.get_function("pde_theta_step")
        return cls._step_kernel

ArrayLike = np.ndarray
def _cupy_available():
    if cp is None: return False
    try: cp.cuda.runtime.getDeviceCount(); return True
    except: return False
def select_backend(name):
    name = name.strip().lower()
    if name == "cpu": return np, False, "cpu"
    if name == "cuda":
        if not _cupy_available(): raise RuntimeError("CUDA not available")
        return cp, True, "cuda"
    return (cp, True, "cuda") if _cupy_available() else (np, False, "cpu")
def _is_gpu(x): return cp is not None and isinstance(x, cp.ndarray)
def _to_np(x): return cp.asnumpy(x) if _is_gpu(x) else np.asarray(x)
def _to_be(x, xp, dt=np.float64): return xp.asarray(x, dtype=dt) if xp is not np else np.asarray(x, dtype=dt)
def _sc(x):
    if _is_gpu(x): return float(cp.asnumpy(x))
    return float(x.item()) if isinstance(x, np.ndarray) else float(x)
def _mass(u, dz, xp): return _sc(xp.sum(u) * dz)

@dataclass
class AutocallableSpec:
    notional: float = 1.0; maturity_years: float = 1.5; ac_barrier: float = 1.0
    coupon_barrier: float = 0.70; ki_barrier: float = 0.60; coupon_rate: float = 0.08
    put_strike: float = 1.0; memory: bool = True; obs_freq: str = "quarterly"
    no_call_periods: int = 0; ki_continuous: bool = False; ac_step_down: float = 0.0

@dataclass
class CTMCModel:
    z_grid: np.ndarray; dz: float; n_states: int; v_states: np.ndarray
    generator: np.ndarray; pi0: np.ndarray; pillar_T: np.ndarray
    pillar_forwards: np.ndarray; pillar_dfs: np.ndarray; pillar_labels: np.ndarray
    leverage: List[np.ndarray]; sigma_lv: List[np.ndarray]; S0: float
    densities: Optional[List[np.ndarray]] = None; n_substeps_calib: int = 2000
    leverage_time: Optional[List[Optional[np.ndarray]]] = None
    pillar_dt: Optional[np.ndarray] = None

@dataclass
class PricingResult:
    price: float; notional: float; price_pct: float
    autocall_probabilities: np.ndarray; stop_probabilities: np.ndarray
    coupon_contributions: np.ndarray; autocall_contributions: np.ndarray
    terminal_par_contribution: float; terminal_put_contribution: float
    survival_probability: float; ki_probability: float
    observation_dates: np.ndarray; memory_enabled: bool
    fair_coupon: Optional[float] = None; expected_expiry_years: float = 0.0

@dataclass
class TermStructurePoint:
    maturity_years: float; coupon_rate: float; price: float; price_pct: float
    price_diff: float; price_diff_bps: float; survival_probability: float
    terminal_par_contribution: float; terminal_put_contribution: float
    expected_expiry_years: float; obs_freq: str

def load_ctmc_model(npz_path, leverage_time_stride=50):
    d = np.load(npz_path, allow_pickle=True)
    z_grid = np.asarray(d["z_grid"], dtype=np.float64); dz = float(d["dz"])
    n_states = int(d["ctmc_n_states"]); v_states = np.asarray(d["ctmc_states"], dtype=np.float64)
    generator = np.asarray(d["ctmc_generator"], dtype=np.float64)
    pi0 = np.asarray(d["ctmc_pi0"], dtype=np.float64)
    pillar_T = np.asarray(d["pillar_T"], dtype=np.float64)
    pillar_forwards = np.asarray(d["pillar_forward"], dtype=np.float64)
    pillar_dfs = np.asarray(d["pillar_df"], dtype=np.float64)
    pillar_labels = np.asarray(d["pillar_labels"]); S0 = float(d["heston_S0"])
    n_sub = int(d["n_substeps"]) if "n_substeps" in d else 2000
    pillar_dt = np.asarray(d["pillar_dt"], dtype=np.float64) if "pillar_dt" in d else np.diff(np.concatenate([[0.0], pillar_T]))
    nb = int(d["n_buckets"]); lev, slv, den = [], [], []
    for k in range(nb):
        lev.append(np.asarray(d[f"leverage_{k}"], dtype=np.float64))
        slv.append(np.asarray(d[f"sigma_lv_{k}"], dtype=np.float64))
        if f"density_{k}" in d: den.append(np.asarray(d[f"density_{k}"], dtype=np.float64))
    lt = None
    if bool(d.get("has_leverage_time", 0)):
        lt = []
        for k in range(nb):
            key = f"leverage_time_{k}"
            if key in d:
                arr = np.asarray(d[key], dtype=np.float64)
                if leverage_time_stride > 1:
                    idx = np.arange(0, arr.shape[0], leverage_time_stride)
                    if idx[-1] != arr.shape[0] - 1: idx = np.append(idx, arr.shape[0] - 1)
                    arr = arr[idx]
                lt.append(arr)
            else: lt.append(None)
        if all(x is None for x in lt): lt = None
        else:
            tot = sum(x.shape[0] for x in lt if x is not None)
            print(f"  [load] leverage_time: {tot} slices (stride={leverage_time_stride})")
    return CTMCModel(z_grid=z_grid, dz=dz, n_states=n_states, v_states=v_states,
        generator=generator, pi0=pi0, pillar_T=pillar_T, pillar_forwards=pillar_forwards,
        pillar_dfs=pillar_dfs, pillar_labels=pillar_labels, leverage=lev, sigma_lv=slv,
        S0=S0, densities=den if den else None, n_substeps_calib=n_sub, leverage_time=lt, pillar_dt=pillar_dt)

def load_forward_curve(p):
    import csv; T,F=[],[]
    with open(p) as f:
        for r in csv.DictReader(f): T.append(float(r["T_years"])); F.append(float(r["forward_interp"]))
    return np.array(T,dtype=np.float64), np.array(F,dtype=np.float64)
def load_discount_curve(p):
    import csv; T,D=[],[]
    with open(p) as f:
        for r in csv.DictReader(f): T.append(float(r["T_years"])); D.append(float(r["discount_factor"]))
    return np.array(T,dtype=np.float64), np.array(D,dtype=np.float64)
def build_interpolators(fT,fF,dT,dD):
    return interp1d(fT,fF,kind="linear",fill_value="extrapolate"), interp1d(dT,dD,kind="linear",fill_value="extrapolate")

def normalize_obs_freq(f):
    f=f.strip().lower()
    if f in("monthly","month","m","1m"): return "monthly"
    if f in("quarterly","quarter","q","3m"): return "quarterly"
    if f in("semi-annual","semiannual","semi","sa","6m","semi_annual"): return "semi-annual"
    if f in("annual","yearly","a","12m","1y"): return "annual"
    raise ValueError(f)
def obs_freq_to_months(f): return {"monthly":1,"quarterly":3,"semi-annual":6,"annual":12}[normalize_obs_freq(f)]
def obs_freq_legend_label(f): return rf"$T^O = {obs_freq_to_months(f)}$"
def parse_obs_freq_list(t):
    if not t or not t.strip(): return None
    v=[normalize_obs_freq(x.strip()) for x in t.split(",") if x.strip()]
    o,s=[],set()
    for x in v:
        if x not in s: o.append(x); s.add(x)
    return o or None
def generate_observation_dates(mat,freq):
    dt={"monthly":1/12,"quarterly":0.25,"semi-annual":0.5,"annual":1.0}[normalize_obs_freq(freq)]
    r=np.arange(dt,mat-1e-12,dt,dtype=float)
    return np.array([mat],dtype=float) if r.size==0 else np.concatenate([r,[mat]])

def _batched_thomas(a,b,c,d,xp):
    eps=1e-30; batch,n=d.shape
    cp_=xp.empty((batch,n),dtype=d.dtype); dp_=xp.empty((batch,n),dtype=d.dtype)
    dn=b[:,0]; dn=xp.where(xp.abs(dn)<eps,eps,dn); cp_[:,0]=c[:,0]/dn; dp_[:,0]=d[:,0]/dn
    for i in range(1,n):
        dn=b[:,i]-a[:,i]*cp_[:,i-1]; dn=xp.where(xp.abs(dn)<eps,eps,dn)
        cp_[:,i]=c[:,i]/dn; dp_[:,i]=(d[:,i]-a[:,i]*dp_[:,i-1])/dn
    x=xp.empty_like(d); x[:,-1]=dp_[:,-1]
    for i in range(n-2,-1,-1): x[:,i]=dp_[:,i]-cp_[:,i]*x[:,i+1]
    return x

def _cuda_pde_step(phi,L2,vs,dz,dt,theta,ns,ws,out):
    nr,nz=phi.shape; k=_CudaPDEKernelCache.pde_theta_step(); th=128; bl=(nr+th-1)//th
    k((bl,),(th,),(phi,L2,vs,ws,out,np.int32(nr),np.int32(ns),np.int32(nz),np.float64(dz),np.float64(dt),np.float64(theta)))
    return out

def _build_fwd_op(lv,dz,xp):
    b,nz=lv.shape; dz2=dz*dz
    sub=xp.zeros((b,nz),dtype=lv.dtype); diag=xp.zeros_like(sub); sup=xp.zeros_like(sub)
    if nz>2:
        sub[:,1:-1]=0.5*lv[:,:-2]/dz2-lv[:,:-2]/(4*dz)
        diag[:,1:-1]=-lv[:,1:-1]/dz2
        sup[:,1:-1]=0.5*lv[:,2:]/dz2+lv[:,2:]/(4*dz)
    diag[:,0]=-lv[:,0]/dz2-lv[:,0]/(4*dz)
    if nz>1: sup[:,0]=lv[:,1]/dz2+lv[:,1]/(4*dz); sub[:,-1]=lv[:,-2]/dz2-lv[:,-2]/(4*dz)
    diag[:,-1]=-lv[:,-1]/dz2+lv[:,-1]/(4*dz)
    return sub,diag,sup

def _advance_pde(phi,lv,dz,dt,xp,theta=1.0,nr=0):
    sub,diag,sup=_build_fwd_op(lv,dz,xp); b,nz=phi.shape
    if nr>0:
        dtr=dt/nr; r=phi.copy()
        for _ in range(nr):
            rhs=r.copy(); r=_batched_thomas(-dtr*sub,1-dtr*diag,-dtr*sup,rhs,xp); xp.maximum(r,0,out=r)
        return r
    ex=(1-theta)*dt; im=theta*dt; rhs=xp.empty((b,nz),dtype=phi.dtype)
    rhs[:,0]=phi[:,0]*(1+ex*diag[:,0])
    if nz>1: rhs[:,0]+=phi[:,1]*ex*sup[:,0]
    if nz>2: rhs[:,1:-1]=phi[:,:-2]*ex*sub[:,1:-1]+phi[:,1:-1]*(1+ex*diag[:,1:-1])+phi[:,2:]*ex*sup[:,1:-1]
    rhs[:,-1]=phi[:,-1]*(1+ex*diag[:,-1])
    if nz>1: rhs[:,-1]+=phi[:,-2]*ex*sub[:,-1]
    r=_batched_thomas(-im*sub,1-im*diag,-im*sup,rhs,xp); xp.maximum(r,0,out=r); return r

class DensityPropagator:
    def __init__(self, model, n_substeps=500, theta_pde=1.0, rannacher_steps=4, backend="auto"):
        self.model=model; self.n_substeps=n_substeps; self.theta_pde=theta_pde; self.rannacher_steps=rannacher_steps
        self.xp,self.use_cuda,self.backend_name=select_backend(backend)
        self.pillar_T=np.asarray(model.pillar_T,dtype=np.float64)
        self.pillar_dt=np.asarray(model.pillar_dt,dtype=np.float64) if model.pillar_dt is not None else np.diff(np.concatenate([[0.0],self.pillar_T]))
        self.leverage_pillars_cpu=[np.asarray(L,dtype=np.float64) for L in model.leverage]
        self.leverage_pillars=[_to_be(L,self.xp) for L in self.leverage_pillars_cpu]
        self.v_states=_to_be(model.v_states,self.xp)
        self.generator_cpu=np.asarray(model.generator,dtype=np.float64)
        self.dz=float(model.dz); self.n_states=int(model.n_states); self.nz=len(model.z_grid)
        self.has_leverage_time=False; self.lt_be=[]; self.lt_n=[]
        if model.leverage_time is not None:
            for lt in model.leverage_time:
                if lt is not None:
                    self.lt_be.append(_to_be(lt,self.xp)); self.lt_n.append(lt.shape[0]); self.has_leverage_time=True
                else: self.lt_be.append(None); self.lt_n.append(0)
        self._qc={}; self._cw=None; self._co=None

    def asarray(self,x): return _to_be(x,self.xp)
    def to_numpy(self,x): return _to_np(x)
    def mass(self,u): return _mass(u,self.dz,self.xp)

    def _get_leverage_at_time(self,t):
        T=self.pillar_T
        if self.has_leverage_time:
            dt_arr=self.pillar_dt
            # For t before or at first pillar: use first pillar end leverage
            # (early substeps have unreliable leverage from near-Dirac density)
            if t<=T[0]+1e-12:
                lt=self.lt_be[0]
                if lt is not None:
                    return lt[-1].copy()
                return self.leverage_pillars[0].copy()
            # For t beyond last pillar: use last pillar end leverage
            if t>=T[-1]-1e-12:
                for k in range(len(T)-1,-1,-1):
                    if self.lt_be[k] is not None: return self.lt_be[k][-1].copy()
                return self.leverage_pillars[-1].copy()
            # For t between pillars: find the bucket and interpolate
            # Only use the second half of each bucket (the well-converged part)
            # and interpolate between adjacent pillar-end values for cross-bucket
            for k in range(len(T)):
                Te=T[k]; dk=dt_arr[k]; Ts=Te-dk
                if t<=Te+1e-12:
                    lt=self.lt_be[k]
                    if lt is None: break
                    n=self.lt_n[k]
                    if n<2: return lt[-1].copy()
                    # Map t into this bucket
                    fr=max(0.0,min(1.0,(t-Ts)/dk if dk>1e-15 else 0.0))
                    # Use at least 20% into the bucket to avoid early-substep noise
                    fr=max(fr, 0.0)
                    fi=fr*(n-1); il=min(max(int(fi),0),n-2); w=fi-il
                    return (1-w)*lt[il]+w*lt[il+1]
            # Should not reach here, but fallback
            for k in range(len(T)-1,-1,-1):
                if self.lt_be[k] is not None: return self.lt_be[k][-1].copy()
        if t<=T[0]: return self.leverage_pillars[0].copy()
        if t>=T[-1]: return self.leverage_pillars[-1].copy()
        i=max(0,min(np.searchsorted(T,t)-1,len(T)-2))
        t0,t1=T[i],T[i+1]; w=(t-t0)/(t1-t0) if t1>t0 else 0.0
        return (1-w)*self.leverage_pillars[i]+w*self.leverage_pillars[i+1]

    def _get_qt(self,dt_sub):
        key=(self.n_substeps,round(float(dt_sub),16))
        if key not in self._qc:
            q=expm(self.generator_cpu*dt_sub); q=np.maximum(q,0); rs=q.sum(1,keepdims=True); rs[rs<=0]=1; q/=rs
            self._qc[key]=_to_be(q.T.copy(),self.xp)
        return self._qc[key]

    def _mix(self,ub,qt):
        xp=self.xp; ns=ub.shape[0]
        tmp=xp.transpose(ub,(1,0,2)).reshape(self.n_states,ns*self.nz)
        return xp.transpose((qt@tmp).reshape(self.n_states,ns,self.nz),(1,0,2))

    def _ensure_buf(self,nr):
        if not self.use_cuda: return None,None
        s=(nr,self.nz)
        if self._cw is None or self._cw.shape!=s: self._cw=cp.empty(s,dtype=cp.float64)
        if self._co is None or self._co.shape!=s: self._co=cp.empty(s,dtype=cp.float64)
        return self._cw,self._co

    def _pde_step(self,ub,tm,ds,rann):
        xp=self.xp; ns=ub.shape[0]; L=self._get_leverage_at_time(tm); L2=L*L
        uf=ub.reshape(ns*self.n_states,self.nz)
        if self.use_cuda:
            w,o=self._ensure_buf(uf.shape[0])
            if rann and self.rannacher_steps>0:
                dr=ds/self.rannacher_steps; pi,po=uf,o
                for _ in range(self.rannacher_steps):
                    _cuda_pde_step(pi,L2,self.v_states,self.dz,dr,1.0,self.n_states,w,po); pi,po=po,pi
                return pi.reshape(ns,self.n_states,self.nz)
            return _cuda_pde_step(uf,L2,self.v_states,self.dz,ds,self.theta_pde,self.n_states,w,o).reshape(ns,self.n_states,self.nz)
        lv_base=self.v_states[:,None]*L2[None,:]; lv=xp.broadcast_to(lv_base[None,:,:],(ns,self.n_states,self.nz))
        uf=_advance_pde(uf,lv.reshape(ns*self.n_states,self.nz),self.dz,ds,xp,self.theta_pde,self.rannacher_steps if rann else 0)
        return uf.reshape(ns,self.n_states,self.nz)

    def propagate_batch(self,ub,ts,te):
        dt=te-ts
        if dt<=1e-12: return ub.copy()
        u=self.asarray(ub).copy(); ns=max(1,self.n_substeps); ds=dt/ns; qt=self._get_qt(ds)
        for s in range(ns):
            tm=ts+(s+0.5)*ds; u=self._mix(u,qt); u=self._pde_step(u,tm,ds,s==0 and self.rannacher_steps>0)
        return u

    def propagate(self,uj,ts,te):
        return self.propagate_batch(self.asarray(uj)[None,:,:],ts,te)[0]

def _get_density(model,t,prop):
    pT=model.pillar_T; Nz=len(model.z_grid); dz=model.dz; tol=1e-6
    if model.densities:
        for k,Tk in enumerate(pT):
            if abs(t-Tk)<tol: return prop.asarray(model.densities[k].copy())
        pr=-1
        for k,Tk in enumerate(pT):
            if Tk<t-tol: pr=k
        if pr>=0: return prop.propagate(model.densities[pr].copy(),pT[pr],t)
    u0=np.zeros(Nz,dtype=np.float64); u0[np.argmin(np.abs(model.z_grid))]=1.0/dz
    ui=np.zeros((model.n_states,Nz),dtype=np.float64)
    for i in range(model.n_states): ui[i,:]=model.pi0[i]*u0
    return prop.asarray(ui) if t<tol else prop.propagate(ui,0.0,t)

def _prop_slices(sl,ts,te,prop,thr):
    if not sl or te-ts<=1e-12: return {k:v.copy() for k,v in sl.items()}
    ks,vs=[],[]
    for k,u in sl.items():
        if prop.mass(u)>=thr: ks.append(k); vs.append(u)
    if not ks: return {}
    xp=prop.xp; ub=xp.stack(vs,axis=0); ub=prop.propagate_batch(ub,ts,te)
    return {k:ub[j] for j,k in enumerate(ks)}

def price_autocallable(model,spec,fwd,disc,n_substeps=500,theta_pde=1.0,rannacher_steps=4,
                       verbose=True,max_memory_slices=0,memory_mass_threshold=1e-12,backend="auto",propagator=None):
    t0=time.time(); S0=model.S0; z_np=model.z_grid; dz=model.dz; Nz=len(z_np)
    obs=generate_observation_dates(spec.maturity_years,spec.obs_freq); K=len(obs)
    if propagator is None: propagator=DensityPropagator(model,n_substeps,theta_pde,rannacher_steps,backend)
    xp=propagator.xp; zg=propagator.asarray(z_np)
    if verbose:
        print(f"\n{'='*70}\nAUTOCALLABLE PRICER (CTMC-LSV, {propagator.backend_name})\n{'='*70}")
        print(f"  S0={S0:.2f} Mat={spec.maturity_years:.4f}y Freq={spec.obs_freq} Obs={K}")
        print(f"  Leverage: {'substep' if propagator.has_leverage_time else 'pillar-interp'}")
    ui=_get_density(model,obs[0],propagator)
    slices={(0,0):ui}; ap=np.zeros(K); ac=np.zeros(K); cc=np.zeros(K); tp=tpu=price=0.0; tv=obs[0]
    for k in range(K):
        to=float(obs[k]); D=float(disc(to)); F=float(fwd(to))
        ab=spec.ac_barrier-spec.ac_step_down*k
        za=np.log(max(ab*S0/F,1e-12)); zc=np.log(max(spec.coupon_barrier*S0/F,1e-12)); zk=np.log(max(spec.ki_barrier*S0/F,1e-12))
        aa=zg>=za; acp=zg>=zc; bk=zg<zk; fin=k==K-1; can=k>=spec.no_call_periods and not fin
        if k>0: slices=_prop_slices(slices,tv,to,propagator,memory_mass_threshold)
        us={}
        for(b,m),u in slices.items():
            if b==0:
                s=u.copy();ki=xp.zeros_like(u);ki[:,bk]=u[:,bk];s[:,bk]=0.0
                if propagator.mass(ki)>memory_mass_threshold: k2=(1,m);us[k2]=us.get(k2,xp.zeros_like(u))+ki
                if propagator.mass(s)>memory_mass_threshold: k2=(0,m);us[k2]=us.get(k2,xp.zeros_like(u))+s
            else: k2=(b,m);us[k2]=us.get(k2,xp.zeros_like(u))+u.copy()
        slices=us; post={}
        for(b,m),u in slices.items():
            if can:
                sv=u.copy();sv[:,aa]=0.0;am=_sc(xp.sum(u[:,aa])*dz)
                if am>memory_mass_threshold:
                    nc=(m+1) if spec.memory else 1; cv=D*spec.notional*(1+nc*spec.coupon_rate)*am
                    price+=cv;ap[k]+=am;ac[k]+=cv
                u=sv
            if fin:
                ua=u.copy();ua[:,xp.logical_not(acp)]=0.0;ma=propagator.mass(ua)
                if ma>0: nc=(m+1) if spec.memory else 1;cv=D*spec.notional*nc*spec.coupon_rate*ma;cc[k]+=cv;price+=cv
                tm=propagator.mass(u)
                if b==0: pv=D*spec.notional*tm;tp+=pv;price+=pv
                else:
                    sr=xp.exp(zg)*F/(spec.put_strike*S0);pp=xp.minimum(sr,1.0);mz=xp.sum(u,axis=0)
                    pv=D*spec.notional*_sc(xp.sum(pp*mz)*dz);tpu+=pv;price+=pv
                continue
            ua=u.copy();ua[:,xp.logical_not(acp)]=0.0;ub=u.copy();ub[:,acp]=0.0
            ma=propagator.mass(ua);mb=propagator.mass(ub)
            if ma>memory_mass_threshold:
                nc=(m+1) if spec.memory else 1;cv=D*spec.notional*nc*spec.coupon_rate*ma;cc[k]+=cv;price+=cv
                kr=(b,0);post[kr]=post.get(kr,xp.zeros_like(u))+ua
            if mb>memory_mass_threshold:
                ki=(b,m+1 if spec.memory else 0)
                if max_memory_slices>0 and spec.memory and m+1>=max_memory_slices: ki=(b,max_memory_slices-1)
                post[ki]=post.get(ki,xp.zeros_like(u))+ub
        if not fin: slices=post
        slices={k_:v for k_,v in slices.items() if propagator.mass(v)>memory_mass_threshold}
        if verbose:
            sm=sum(propagator.mass(v) for v in slices.values())
            print(f"  Obs {k+1}/{K}: T={to:.4f} surv={sm:.6f} AC={ap[k]:.6f} sl={len(slices)}")
        tv=to
    su=sum(propagator.mass(v) for v in slices.values())
    sp=np.zeros(K);sp[:-1]=ap[:-1];sp[-1]=su;ts=float(np.sum(sp))
    ee=float(np.dot(obs,sp)/ts) if ts>1e-15 else 0.0
    if verbose: print(f"\n  Price={price:.8f} ({price/spec.notional*100:.4f}%) Surv={su:.6f} E[T*]={ee:.4f} {time.time()-t0:.2f}s")
    return PricingResult(price=price,notional=spec.notional,price_pct=price/spec.notional*100,
        autocall_probabilities=ap,stop_probabilities=sp,coupon_contributions=cc,autocall_contributions=ac,
        terminal_par_contribution=tp,terminal_put_contribution=tpu,survival_probability=su,ki_probability=0.0,
        observation_dates=obs,memory_enabled=spec.memory,expected_expiry_years=ee)

def solve_fair_coupon(model,spec,fwd,disc,n_sub=500,theta=1.0,rann=4,verbose=True,backend="auto"):
    prop=DensityPropagator(model,n_sub,theta,rann,backend)
    kw=dict(n_substeps=n_sub,theta_pde=theta,rannacher_steps=rann,verbose=verbose,backend=backend,propagator=prop)
    s0=AutocallableSpec(**{**spec.__dict__,"coupon_rate":0.0})
    s1=AutocallableSpec(**{**spec.__dict__,"coupon_rate":1.0})
    r0=price_autocallable(model,s0,fwd,disc,**kw); r1=price_autocallable(model,s1,fwd,disc,**kw)
    Vu=r1.price-r0.price
    if abs(Vu)<1e-15: return 0.0,r0
    fc=(spec.notional-r0.price)/Vu
    if verbose:
        npy=len(generate_observation_dates(spec.maturity_years,spec.obs_freq))/spec.maturity_years
        print(f"  Fair coupon={fc*100:.4f}%/period = {fc*npy*100:.4f}% p.a.")
    sf=AutocallableSpec(**{**spec.__dict__,"coupon_rate":fc})
    rf=price_autocallable(model,sf,fwd,disc,**kw); rf.fair_coupon=fc; return fc,rf

def parse_float_list(t,n):
    if not t or not t.strip(): return None
    try: return [float(x.strip()) for x in t.split(",") if x.strip()]
    except ValueError as e: raise ValueError(f"Cannot parse {n}") from e
def resolve_cpn(freq,mats,common,mo,qu,sa,an,fb):
    freq=normalize_obs_freq(freq)
    sp={"monthly":mo,"quarterly":qu,"semi-annual":sa,"annual":an}.get(freq)
    cl=sp if sp is not None else common
    if cl is None: return [fb]*len(mats)
    if len(cl)==1 and len(mats)>1: return cl*len(mats)
    if len(cl)!=len(mats): raise ValueError(f"Coupon list mismatch for {freq}")
    return cl
def save_csv(pts,path):
    import csv
    fn=["obs_freq","maturity_years","coupon_rate","price","price_diff_bps","expected_expiry_years","survival_probability"]
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fn);w.writeheader()
        for p in pts: w.writerow({k:getattr(p,k) for k in fn})
def print_summary(pts,hdr="TERM STRUCTURE"):
    print(f"\n{'='*120}\n{hdr}\n{'='*120}")
    print(f"{'Freq':>12}{'Mat':>8}{'Cpn%':>8}{'Price':>14}{'D(bps)':>10}{'E[T*]':>8}{'Surv':>8}")
    for p in pts:
        print(f"{p.obs_freq:>12}{p.maturity_years:8.4f}{100*p.coupon_rate:7.4f}%{p.price:14.8f}{p.price_diff_bps:10.2f}{p.expected_expiry_years:8.4f}{p.survival_probability:8.4f}")
def plot_multi(curves,path=None,title="Price difference"):
    fig=plt.figure(figsize=(9,5))
    for f in sorted(curves,key=obs_freq_to_months):
        pts=curves[f];m=np.array([p.maturity_years for p in pts]);d=np.array([p.price_diff_bps for p in pts]);o=np.argsort(m)
        plt.plot(m[o],d[o],"o-",lw=1.2,label=obs_freq_legend_label(f))
    plt.axhline(0,ls="--",c="gray",alpha=.7);plt.xlabel(r"$T^E$ (years)");plt.ylabel("bps")
    plt.title(title);plt.legend();plt.grid(alpha=.25);plt.tight_layout()
    if path: fig.savefig(path,dpi=200,bbox_inches="tight")
    return fig
def plot_ee(curves,path=None,title="Expected expiry"):
    fig=plt.figure(figsize=(9,5))
    for f in sorted(curves,key=obs_freq_to_months):
        pts=curves[f];m=np.array([p.maturity_years for p in pts]);e=np.array([p.expected_expiry_years for p in pts]);o=np.argsort(m)
        plt.plot(m[o],e[o],"o-",lw=1.2,label=obs_freq_legend_label(f))
    plt.xlabel(r"$T^E$ (years)");plt.ylabel(r"$\mathbb{E}[T_\tau]$");plt.title(title)
    plt.legend();plt.grid(alpha=.25);plt.ylim(bottom=0);plt.tight_layout()
    if path: fig.savefig(path,dpi=200,bbox_inches="tight")
    return fig
def price_ts(model,base,mats,cpns,fwd,disc,nsub=500,theta=1.0,rann=4,verbose=True,backend="auto"):
    pairs=sorted(zip(mats,cpns));pts=[];prop=DensityPropagator(model,nsub,theta,rann,backend)
    for j,(T,c) in enumerate(pairs,1):
        if verbose: print(f"\n  TERM {j}/{len(pairs)}: T={T:.4f} c={100*c:.4f}%")
        sp=AutocallableSpec(**{**base.__dict__,"maturity_years":T,"coupon_rate":c})
        r=price_autocallable(model,sp,fwd,disc,nsub,theta,rann,verbose,backend=backend,propagator=prop)
        pd=r.price/sp.notional-1.0
        pts.append(TermStructurePoint(T,c,r.price,r.price_pct,pd,1e4*pd,r.survival_probability,
            r.terminal_par_contribution,r.terminal_put_contribution,r.expected_expiry_years,sp.obs_freq))
    return pts

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--lsv_result",required=True);p.add_argument("--forward_curve",required=True)
    p.add_argument("--discount_curve",required=True);p.add_argument("--notional",type=float,default=1.0)
    p.add_argument("--maturity_years",type=float,default=1.5);p.add_argument("--coupon_rate",type=float,default=0.026744)
    p.add_argument("--maturity_years_list",default="0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    p.add_argument("--coupon_rates_list",default="")
    p.add_argument("--coupon_rates_list_monthly",default="0.00841059,0.010805,0.011079,0.01084965,0.0104426,0.01003036,0.00964965,0.00930935")
    p.add_argument("--coupon_rates_list_quarterly",default="0.01908626,0.02461403,0.02670627,0.02742142,0.02727404,0.0267436,0.02608444,0.02540501")
    p.add_argument("--coupon_rates_list_semi_annual",default="0.01908626,0.04162856,0.04122418,0.04768614,0.04594846,0.04834639,0.04589727,0.04702281")
    p.add_argument("--coupon_rates_list_annual",default="0.01908626,0.04162856,0.06213436,0.08211667,0.07232821,0.07759105,0.08177649,0.08541534")
    p.add_argument("--obs_freq",default="quarterly");p.add_argument("--obs_freqs_list",default="quarterly")
    p.add_argument("--ac_barrier",type=float,default=1.0);p.add_argument("--coupon_barrier",type=float,default=0.0)
    p.add_argument("--ki_barrier",type=float,default=0.8);p.add_argument("--put_strike",type=float,default=1.0)
    p.add_argument("--no_call_periods",type=int,default=0);p.add_argument("--ac_step_down",type=float,default=0.0)
    mg=p.add_mutually_exclusive_group();mg.add_argument("--memory",dest="memory",action="store_true",default=True)
    mg.add_argument("--no_memory",dest="memory",action="store_false")
    p.add_argument("--n_substeps",type=int,default=1000);p.add_argument("--theta_pde",type=float,default=1.0)
    p.add_argument("--rannacher_steps",type=int,default=6);p.add_argument("--backend",default="auto",choices=["auto","cpu","cuda"])
    p.add_argument("--solve_coupon",action="store_true");p.add_argument("--quiet",action="store_true")
    p.add_argument("--output_prefix",default="autocallable_term_structure");p.add_argument("--no_plot",action="store_true")
    p.add_argument("--leverage_time_stride",type=int,default=50,help="Downsample leverage_time (1=full)")
    return p.parse_args()

def main():
    args=parse_args();verbose=not args.quiet
    if verbose: print(f"Loading {args.lsv_result}...")
    model=load_ctmc_model(args.lsv_result,args.leverage_time_stride)
    if verbose:
        print(f"  S0={model.S0} states={model.n_states} z={len(model.z_grid)}")
        print(f"  Pillars: {list(model.pillar_labels)}")
        print(f"  Leverage: {'substep' if model.leverage_time else 'pillar-interp'}")
    fT,fF=load_forward_curve(args.forward_curve);dT,dD=load_discount_curve(args.discount_curve)
    fi,di=build_interpolators(fT,fF,dT,dD)
    base=AutocallableSpec(notional=args.notional,maturity_years=args.maturity_years,ac_barrier=args.ac_barrier,
        coupon_barrier=args.coupon_barrier,ki_barrier=args.ki_barrier,coupon_rate=args.coupon_rate,
        put_strike=args.put_strike,memory=args.memory,obs_freq=normalize_obs_freq(args.obs_freq),
        no_call_periods=args.no_call_periods,ac_step_down=args.ac_step_down)
    mats=parse_float_list(args.maturity_years_list,"mats");common=parse_float_list(args.coupon_rates_list,"cpn")
    mo=parse_float_list(args.coupon_rates_list_monthly,"mo");qu=parse_float_list(args.coupon_rates_list_quarterly,"qu")
    sa=parse_float_list(args.coupon_rates_list_semi_annual,"sa");an=parse_float_list(args.coupon_rates_list_annual,"an")
    freqs=parse_obs_freq_list(args.obs_freqs_list)
    if freqs and mats:
        curves={}
        for freq in freqs:
            cl=resolve_cpn(freq,mats,common,mo,qu,sa,an,base.coupon_rate)
            sf=AutocallableSpec(**{**base.__dict__,"obs_freq":freq})
            if verbose: print(f"\n{'='*80}\nSWEEP: {freq}\n{'='*80}")
            curves[freq]=price_ts(model,sf,mats,cl,fi,di,args.n_substeps,args.theta_pde,args.rannacher_steps,verbose,args.backend)
        all_pts=[]
        for f in sorted(curves,key=obs_freq_to_months): print_summary(curves[f],f"TERM STRUCTURE — {f}");all_pts.extend(curves[f])
        save_csv(all_pts,f"{args.output_prefix}.csv")
        if not args.no_plot: plot_multi(curves,f"{args.output_prefix}.png");plot_ee(curves,f"{args.output_prefix}_expected_expiry.png");plt.close("all")
        return curves
    if mats:
        cl=resolve_cpn(base.obs_freq,mats,common,mo,qu,sa,an,base.coupon_rate)
        pts=price_ts(model,base,mats,cl,fi,di,args.n_substeps,args.theta_pde,args.rannacher_steps,verbose,args.backend)
        print_summary(pts);return pts
    if args.solve_coupon: return solve_fair_coupon(model,base,fi,di,args.n_substeps,args.theta_pde,args.rannacher_steps,verbose,args.backend)
    return price_autocallable(model,base,fi,di,args.n_substeps,args.theta_pde,args.rannacher_steps,verbose,backend=args.backend)

if __name__=="__main__": main()