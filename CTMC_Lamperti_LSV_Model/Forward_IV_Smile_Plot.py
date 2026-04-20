#!/usr/bin/env python3
"""
forward_start_smiles_lamperti.py — Forward-Start IV: LV vs CTMC-Lamperti-LSV (ρ≠0)
=====================================================================================
Adapted for the remap-era idk.py output. Key changes vs. the dg/dt-clipping version:

  1. Boundary remap applied to p_start when bi > 0. The saved density at pillar
     bi-1 lives under (g_{bi-1}, L_{bi-1}). Bucket bi propagates under its own
     leverage path starting at L_{bucket_bi}[0]. Without a remap, the first
     substep's dgdt jumps discontinuously and gets clipped, corrupting the
     Lamperti PDE. This version mirrors the remap in idk.py so the forward-
     start propagation is a faithful re-run of the calibration dynamics.

  2. First-substep L is derived from leverage_time_{bi}[0], not from the
     coarse leverage_{bi} (which was end-of-bucket L). This keeps the
     propagation time-consistent with calibration.

CORRECT METHOD: backward propagation with per-z₁ payoff vectors.

For each κ and each z₁ bin:
  1. Strike K = κ · F₁ · exp(z₁)
  2. Terminal payoff: payoff(ℓ,j) = max(F₂·exp(z_end(ℓ,j)) - K, 0)
     (payoff defined over terminal state space, NOT start space)
  3. Backward propagation: V = expm(G_back·dt)·...·payoff
  4. Price contribution = Σ_{(ℓ,j) in z₁-bin} p_start(ℓ,j) · V(ℓ,j) · dX
  5. Sum over z₁ bins

n_z1 × n_kappa backward propagations. With n_z1~50, n_kappa=25: 1250 propagations.
At M=80, Nx=401: ~15-30s per bucket on GPU.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import norm

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _GPU = False; cp = None; cp_sparse = None

# ── Black IV ──
def black_call(D,F,K,T,s):
    if T<=0 or s<=0: return D*max(F-K,0.)
    srt=s*math.sqrt(T); d1=(math.log(max(F,1e-300)/max(K,1e-300))+.5*s*s*T)/srt
    return D*(F*norm.cdf(d1)-K*norm.cdf(d1-srt))
def black_put(D,F,K,T,s):
    if T<=0 or s<=0: return D*max(K-F,0.)
    srt=s*math.sqrt(T); d1=(math.log(max(F,1e-300)/max(K,1e-300))+.5*s*s*T)/srt
    return D*(K*norm.cdf(-(d1-srt))-F*norm.cdf(-d1))
def impl_vol(np_,D,F,K,T):
    if T<=0: return 0.
    ci=D*max(F-K,0.); c=float(np.clip(np_,ci,D*F))
    if K<F:
        p=max(c-D*(F-K),0.)
        if p<1e-14: return 0.
        if p>=D*K-1e-12: return np.nan
        def f(s): return black_put(D,F,K,T,s)-p
    else:
        if abs(c-ci)<1e-14: return 0.
        if c>=D*F-1e-12: return np.nan
        def f(s): return black_call(D,F,K,T,s)-c
    for lo,hi in ((1e-8,5.),(1e-10,10.)):
        try: return float(brentq(f,lo,hi,maxiter=300))
        except ValueError: pass
    return np.nan

# ── Load ──
def load_result(path):
    d=np.load(path,allow_pickle=True); nb=int(d["n_buckets"])
    class R: pass
    r=R()
    r.z_grid=np.asarray(d["z_grid"],float); r.X_grid=np.asarray(d["X_grid"],float)
    r.dz=float(d["dz"]); r.dX=float(d["dX"])
    r.n_buckets=nb; r.n_substeps=int(d["n_substeps"])
    r.pillar_labels=[str(x) for x in d["pillar_labels"]]
    r.pillar_T=np.asarray(d["pillar_T"],float); r.pillar_dt=np.asarray(d["pillar_dt"],float)
    r.pillar_forward=np.asarray(d["pillar_forward"],float); r.pillar_df=np.asarray(d["pillar_df"],float)
    r.v_states=np.asarray(d["ctmc_states"],float); r.Q=np.asarray(d["ctmc_generator"],float)
    r.pi0=np.asarray(d["ctmc_pi0"],float); r.mart_corr=np.asarray(d["mart_corr"],float)
    r.rho=float(d["heston_rho"]); r.kappa=float(d["heston_kappa"])
    r.theta=float(d["heston_theta"]); r.xi=float(d["heston_xi"]); r.S0=float(d["heston_S0"])
    r.sigma_lv=[np.asarray(d[f"sigma_lv_{k}"],float) for k in range(nb)]
    r.lv_marginals=[np.asarray(d[f"lv_marginal_{k}"],float) for k in range(nb)]
    r.densities_X=[np.asarray(d[f"density_{k}"],float) for k in range(nb)]
    r.leverage_paths=[np.asarray(d[f"leverage_time_{k}"],float) for k in range(nb)]
    r.g_pillars=[np.asarray(d[f"g_{k}"],float) for k in range(nb)]
    # End-of-bucket L surface (for boundary remap)
    r.L_pillars=[np.asarray(d[f"leverage_{k}"],float) for k in range(nb)]
    return r

# ── Lamperti ──
def compute_g(z,L):
    inv_L=1./np.maximum(L,.01); dz_=z[1]-z[0]; i0=np.argmin(np.abs(z)); g=np.zeros_like(z)
    g[i0+1:]=np.cumsum(.5*(inv_L[i0:-1]+inv_L[i0+1:])*dz_)
    g[:i0]=-np.flip(np.cumsum(np.flip(.5*(inv_L[1:i0+1]+inv_L[:i0])*dz_)))
    return g

def compute_mart_corr(vs,Q,dX,rho,kappa,theta,xi):
    rp2=1-rho**2; M=len(vs); h=dX; dm=np.zeros(M)
    for ell in range(M):
        v=vs[ell]; D=rp2*v; D2=.5*D/(h*h)
        mu=-0.5*v-rho*kappa*(theta-v)/xi
        fp=D2*2*(np.cosh(h)-1)+mu*np.sinh(h)/h
        ct=sum(Q[ell,m]*np.exp(rho*(vs[m]-vs[ell])/xi) for m in range(M))
        dm[ell]=-(fp+ct)*h/np.sinh(h)
    if M>4: dm[0]=dm[1]; dm[-1]=dm[-2]
    return dm

def build_ctmc_Q(vs,kappa,theta,xi):
    M=len(vs); Q=np.zeros((M,M))
    for i in range(M):
        vi=vs[i]; mu=kappa*(theta-vi); s2=xi**2*vi
        if i==0: h=vs[1]-vs[0]; r_=max(.5*s2/h**2+max(mu,0)/h,1e-12); Q[0,1]=r_; Q[0,0]=-r_
        elif i==M-1: h=vs[-1]-vs[-2]; r_=max(.5*s2/h**2+max(-mu,0)/h,1e-12); Q[-1,-2]=r_; Q[-1,-1]=-r_
        else:
            hf=vs[i+1]-vs[i]; hb=vs[i]-vs[i-1]; hm=.5*(hf+hb)
            cu=max(.5*s2/(hf*hm)+max(mu,0)/hf,0); cd=max(.5*s2/(hb*hm)+max(-mu,0)/hb,0)
            Q[i,i+1]=cu; Q[i,i-1]=cd; Q[i,i]=-(cu+cd)
    return Q

def coarsen_v(vs_f,pi0_f,M_t):
    M_f=len(vs_f)
    if M_t>=M_f: return vs_f.copy(),np.arange(M_f),pi0_f.copy()
    idx=np.round(np.linspace(0,M_f-1,M_t)).astype(int)
    vs=vs_f[idx]; pi0=pi0_f[idx]; pi0=np.maximum(pi0,0.); pi0/=max(pi0.sum(),1e-30)
    return vs,idx,pi0

def build_backward_gen(mu,vs,Q,M,Nx,dX,rp2):
    N=M*Nx; D2=.5*rp2*vs/(dX*dX); mu2=mu/(2*dX)
    off=np.arange(M)*Nx; ji=np.arange(Nx)
    _,jg=np.meshgrid(np.arange(M),np.arange(Nx),indexing='ij')
    fi=off[:,None]+ji[None,:]
    dv=(-2*D2[:,None]*np.ones((1,Nx))+np.diag(Q)[:,None]).ravel()
    dr=fi.ravel(); dc=dr.copy()
    ml=jg>0; lr=fi[ml]; lc=(fi-1)[ml]; lv=(D2[:,None]-mu2)[ml]
    mu_=jg<Nx-1; ur=fi[mu_]; uc=(fi+1)[mu_]; uv=(D2[:,None]+mu2)[mu_]
    cr,cc,cv=[],[],[]
    for e in range(M):
        for m in range(M):
            if m!=e and abs(Q[e,m])>1e-30:
                cr.append(off[e]+ji); cc.append(off[m]+ji); cv.append(np.full(Nx,Q[e,m]))
    if cr: cr=np.concatenate(cr); cc=np.concatenate(cc); cv=np.concatenate(cv)
    else: cr=np.array([],int); cc=np.array([],int); cv=np.array([])
    r=np.concatenate([dr,lr,ur,cr]); c=np.concatenate([dc,lc,uc,cc]); v=np.concatenate([dv,lv,uv,cv])
    return csr_matrix(coo_matrix((v,(r,c)),shape=(N,N)))

class UnifOp:
    def __init__(self,A,dt,tol=1e-13):
        diag=np.array(A.diagonal()); self.lam=float(np.max(-diag)); self.tol=tol; self.trivial=self.lam<1e-30
        if self.trivial: return
        P=A.copy(); P.setdiag(P.diagonal()+self.lam); P*=(1./self.lam)
        self.ns=max(1,int(np.ceil(self.lam*dt/30.))); dt_=dt/self.ns
        self.ld=self.lam*dt_; self.K=int(self.ld+6*np.sqrt(max(self.ld,1)))+5; self.K=max(self.K,10)
        self.enld=np.exp(-self.ld)
        if _GPU:
            self.Pg=cp_sparse.csr_matrix((cp.asarray(P.data),cp.asarray(P.indices),cp.asarray(P.indptr)),shape=P.shape)
            self.gpu=True
        else: self.Pc=P; self.gpu=False
    def apply(self,v):
        if self.trivial: return v.copy()
        if self.gpu:
            w=cp.asarray(v)
            for _ in range(self.ns):
                r=w*self.enld; t=w.copy(); c=self.enld
                for k in range(1,self.K+1):
                    t=self.Pg.dot(t); c*=self.ld/k; r=r+c*t
                    if k>5 and c<self.tol: break
                w=r
            return cp.asnumpy(w)
        w=v.copy()
        for _ in range(self.ns):
            r=w*self.enld; t=w.copy(); c=self.enld
            for k in range(1,self.K+1):
                t=self.Pc.dot(t); c*=self.ld/k; r+=c*t
                if c*np.max(np.abs(t))<self.tol*(np.max(np.abs(r))+1e-30): break
            w=r
        return w

def resampled_idx(n_saved,n_target):
    if n_target>=n_saved: return np.arange(n_saved+1,dtype=int)
    idx=np.rint(np.linspace(0,n_saved,n_target+1)).astype(int)
    idx[0]=0; idx[-1]=n_saved; idx=np.maximum.accumulate(idx)
    for j in range(1,len(idx)):
        if idx[j]<=idx[j-1]: idx[j]=min(n_saved,idx[j-1]+1)
    return idx

# ══════════════════════════════════════════════════════════════
# BOUNDARY REMAP — mirrors idk.py's remap_density_at_boundary
# ══════════════════════════════════════════════════════════════
def remap_density_at_boundary(p_X, g_old, L_old, g_new, L_new,
                               v_shifts, X_grid, z_grid, M, Nx, dX):
    """Transfer density from (g_old, L_old) coords to (g_new, L_new) coords.
    Matches the remap in idk.py so forward-start dynamics stay consistent
    with calibration dynamics.

    p_X has shape (M, Nx); returns (M, Nx).
    """
    p_new = np.zeros_like(p_X)
    for ell in range(M):
        shift = v_shifts[ell]
        mass_old = float(np.sum(np.maximum(p_X[ell], 0)) * dX)
        z_at_X = np.interp(X_grid + shift, g_new, z_grid)
        X_old = np.interp(z_at_X, z_grid, g_old) - shift
        u_interp = np.interp(X_old, X_grid,
                             np.maximum(p_X[ell], 0), left=0, right=0)
        L_o_z = np.interp(z_at_X, z_grid, L_old)
        L_n_z = np.interp(z_at_X, z_grid, L_new)
        J = np.clip(L_n_z / np.maximum(L_o_z, 1e-6), 0.1, 10.0)
        u_new_ell = np.maximum(u_interp * J, 0)
        mass_new = float(np.sum(u_new_ell) * dX)
        if mass_new > 1e-15 and mass_old > 1e-15:
            u_new_ell *= mass_old / mass_new
        p_new[ell] = u_new_ell
    return p_new

# ══════════════════════════════════════════════════════════════
# LV FORWARD-START
# ══════════════════════════════════════════════════════════════
def forward_start_lv(res,bi,kappas,z_stride=8,max_substeps=0):
    z=res.z_grid[::z_stride]; dz=res.dz*z_stride; Nz=len(z)
    sig=res.sigma_lv[bi][::z_stride]; a=sig*sig
    n_sub=min(res.n_substeps,max_substeps) if max_substeps>0 else res.n_substeps
    dt=float(res.pillar_dt[bi])/n_sub
    c_l=.5/dz**2-.25/dz; c_u=.5/dz**2+.25/dz; c_d=-1./dz**2
    sub=np.zeros(Nz); dia=np.ones(Nz); sup=np.zeros(Nz)
    sub[1:-1]=-dt*c_l*a[:-2]; sup[1:-1]=-dt*c_u*a[2:]; dia[1:-1]=1.-dt*c_d*a[1:-1]
    dia[0]=1.-dt*(-.5/dz**2+.25/dz)*a[0]; sup[0]=-dt*c_u*a[1]
    sub[-1]=-dt*c_l*a[-2]; dia[-1]=1.-dt*(-.5/dz**2-.25/dz)*a[-1]
    cp_=np.empty(Nz); d_=dia.copy()
    cp_[0]=sup[0]/d_[0]
    for i in range(1,Nz): d_[i]=dia[i]-sub[i]*cp_[i-1]; cp_[i]=sup[i]/d_[i] if i<Nz-1 else 0
    def thomas(rhs):
        n,nc=rhs.shape; dp=np.empty_like(rhs)
        dp[0]=rhs[0]/dia[0]
        for i in range(1,n): dp[i]=(rhs[i]-sub[i]*dp[i-1])/d_[i]
        x=np.empty_like(rhs); x[-1]=dp[-1]
        for i in range(n-2,-1,-1): x[i]=dp[i]-cp_[i]*x[i+1]
        return x
    if bi==0:
        start=np.zeros(Nz); start[np.argmin(np.abs(z))]=1./dz; F1=1.; DF1=1.
    else:
        start=np.maximum(res.lv_marginals[bi-1][::z_stride],0.); m=np.sum(start)*dz
        if m>0: start/=m
        F1=float(res.pillar_forward[bi-1]); DF1=float(res.pillar_df[bi-1])
    F2=float(res.pillar_forward[bi]); DF2=float(res.pillar_df[bi])
    sig_mask=start>start.max()*1e-5; z1i=np.where(sig_mask)[0]; nz1=len(z1i)
    print(f"    LV: {nz1} deltas × {n_sub} steps (Nz={Nz})...",end="",flush=True); t0=time.time()
    P=np.zeros((Nz,nz1))
    for k,j1 in enumerate(z1i): P[j1,k]=1./dz
    for _ in range(n_sub): P=np.maximum(thomas(P),0.)
    print(f" {time.time()-t0:.1f}s")
    expz=np.exp(z); prices=np.empty(len(kappas))
    for ik,kap in enumerate(kappas):
        tot=0.
        for k,j1 in enumerate(z1i):
            tot+=start[j1]*np.sum(np.maximum(F2*expz-kap*F1*expz[j1],0.)*P[:,k])*dz*dz
        prices[ik]=DF2*tot
    return prices,F1,F2,DF1,DF2,float(res.pillar_dt[bi])

# ══════════════════════════════════════════════════════════════
# LAMPERTI FORWARD-START — BACKWARD per (κ, z₁)
# ══════════════════════════════════════════════════════════════
def forward_start_lamperti(res,bi,kappas,max_substeps=0,M_price=80,Nx_price=401):
    rp2=1-res.rho**2; z_grid=res.z_grid; Nz=len(z_grid); dz=res.dz
    X_half=(res.X_grid[-1]-res.X_grid[0])/2
    X_grid=np.linspace(-X_half,X_half,Nx_price); dX=X_grid[1]-X_grid[0]; Nx=Nx_price

    vs_full=res.v_states; vs,v_idx,pi0=coarsen_v(vs_full,res.pi0,M_price)
    Q=build_ctmc_Q(vs,res.kappa,res.theta,res.xi)
    mc=compute_mart_corr(vs,Q,dX,res.rho,res.kappa,res.theta,res.xi)
    M=len(vs); N=M*Nx; v_shifts=res.rho*vs/res.xi

    n_sub=min(res.n_substeps,max_substeps) if max_substeps>0 else res.n_substeps
    dt_bucket=float(res.pillar_dt[bi]); dt_sub=dt_bucket/n_sub
    si=resampled_idx(res.n_substeps,n_sub); lp=res.leverage_paths[bi]
    print(f"    Lamperti: M={M} Nx={Nx} N={N} n_sub={n_sub}")

    # ── Start density and coordinate system for bucket bi ──
    # Bucket bi's propagation uses leverage_time_{bi}, whose first entry
    # defines the L at the start of the bucket. The coord system at the
    # start is g_bucket_start = compute_g(z_grid, lp[0]).
    L_bucket_start = lp[0] if lp.shape[0] > 0 else np.ones(Nz)
    g_bucket_start = compute_g(z_grid, L_bucket_start)

    if bi == 0:
        # Dirac at z=0 expressed in Lamperti coords
        p_start = np.zeros((M, Nx))
        for ie in range(M):
            X0 = -res.rho * vs[ie] / res.xi
            fx = (X0 - X_grid[0]) / dX; il = int(fx); ir = il + 1
            if 0 <= il < Nx and 0 <= ir < Nx:
                w = fx - il
                p_start[ie, il] = pi0[ie] * (1 - w) / dX
                p_start[ie, ir] = pi0[ie] * w / dX
            elif 0 <= il < Nx:
                p_start[ie, il] = pi0[ie] / dX
        g_start = g_bucket_start
        F1 = 1.; DF1 = 1.
    else:
        # Saved density at pillar bi-1 is under (g_pillars[bi-1], L_pillars[bi-1]).
        # Need to remap to bucket bi's starting coords (g_bucket_start, L_bucket_start).
        dens_full = res.densities_X[bi - 1]  # shape (M_full, Nx_full) under OLD coords
        X_full = res.X_grid
        g_old = res.g_pillars[bi - 1]
        L_old = res.L_pillars[bi - 1]

        # Step 1: project density from full (M_full, Nx_full) grid to reduced (M, Nx)
        # NB: the v-grid coarsening changes v_shifts, so this interpolation is
        # approximate but matches the original (non-remap) code's approach.
        p_on_old = np.zeros((M, Nx))
        for ie, io in enumerate(v_idx):
            p_on_old[ie] = np.maximum(
                np.interp(X_grid, X_full, np.maximum(dens_full[io], 0.)),
                0.)

        # Step 2: remap from OLD coords (g_old, L_old) to bucket-start coords
        #         (g_bucket_start, L_bucket_start). This mirrors what idk.py
        #         does at the bucket boundary during calibration.
        p_start = remap_density_at_boundary(
            p_on_old, g_old, L_old, g_bucket_start, L_bucket_start,
            v_shifts, X_grid, z_grid, M, Nx, dX)

        # Renormalize (the partial v-grid projection may lose some mass)
        m = np.sum(p_start) * dX
        if m > 0: p_start /= m

        g_start = g_bucket_start
        F1 = float(res.pillar_forward[bi - 1]); DF1 = float(res.pillar_df[bi - 1])

    F2 = float(res.pillar_forward[bi]); DF2 = float(res.pillar_df[bi])
    g_end = res.g_pillars[bi]

    # z at start (for grouping by z₁ = strike determination)
    z_start = np.zeros((M, Nx))
    for e in range(M):
        z_start[e] = np.interp(X_grid + v_shifts[e], g_start, z_grid)
    # z at end (for terminal payoff)
    z_end = np.zeros((M, Nx))
    for e in range(M):
        z_end[e] = np.interp(X_grid + v_shifts[e], g_end, z_grid)
    S_end = F2 * np.exp(z_end)  # (M,Nx)

    # Marginal in z at T₁ for selecting z₁ bins
    z_edges = np.concatenate([[z_grid[0] - dz/2],
                               .5*(z_grid[:-1] + z_grid[1:]),
                               [z_grid[-1] + dz/2]])
    marg_z = np.zeros(Nz)
    for e in range(M):
        iz = np.searchsorted(z_edges, z_start[e]) - 1
        valid = (iz >= 0) & (iz < Nz)
        np.add.at(marg_z, iz[valid], p_start[e, valid] * dX)
    sig_mask = marg_z > marg_z.max() * 1e-4
    z1i = np.where(sig_mask)[0]; nz1 = len(z1i)

    # For each z₁ bin, record which (ℓ,j) states belong to it
    bin_members = [[] for _ in range(nz1)]
    for k, iz1 in enumerate(z1i):
        zlo = z_edges[iz1]; zhi = z_edges[iz1 + 1]
        for e in range(M):
            mask = (z_start[e] >= zlo) & (z_start[e] < zhi)
            js = np.where(mask)[0]
            for j in js:
                bin_members[k].append((e, j, p_start[e, j]))

    # Pre-build backward generators. g_p starts at bucket-start coords.
    print(f"    Building {n_sub} backward generators...", end="", flush=True)
    t0 = time.time()
    ops = []
    g_p = g_bucket_start.copy()
    for s in range(n_sub):
        i1 = si[s + 1]
        L_n = lp[min(i1, lp.shape[0] - 1)]
        g_n = compute_g(z_grid, L_n)
        dgdt = np.clip((g_n - g_p) / dt_sub, -125, 125)
        dLdz = np.gradient(L_n, dz)
        gm = .5 * (g_p + g_n)
        mu = np.zeros((M, Nx))
        for e in range(M):
            gt = X_grid + v_shifts[e]
            zX = np.interp(gt, gm, z_grid)
            mu[e] = (-0.5 * (np.interp(zX, z_grid, L_n)
                              + np.interp(zX, z_grid, dLdz)) * vs[e]
                     + np.interp(zX, z_grid, dgdt)
                     - res.rho * res.kappa * (res.theta - vs[e]) / res.xi
                     + mc[e])
        ops.append(UnifOp(build_backward_gen(mu, vs, Q, M, Nx, dX, rp2), dt_sub))
        g_p = g_n.copy()
    ops_rev = list(reversed(ops))
    print(f" {time.time()-t0:.1f}s")

    # For each (κ, z₁): build payoff, propagate backward, extract values at start states
    print(f"    Pricing {len(kappas)} × {nz1} z-bins...", end="", flush=True)
    t0 = time.time()
    prices = np.empty(len(kappas))
    for ik, kap in enumerate(kappas):
        if ik % max(1, len(kappas) // 5) == 0:
            print(f" κ={kap:.2f}", end="", flush=True)
        total = 0.
        for k, iz1 in enumerate(z1i):
            if not bin_members[k]: continue
            z1_val = z_grid[iz1]
            K_strike = kap * F1 * np.exp(z1_val)
            payoff = np.maximum(S_end - K_strike, 0.).ravel()
            V = payoff.copy()
            for op in ops_rev: V = op.apply(V)
            for e, j, w in bin_members[k]:
                total += w * V[e * Nx + j] * dX
        prices[ik] = DF2 * total
    print(f" done ({time.time()-t0:.1f}s)")
    return prices, F1, F2, DF1, DF2, dt_bucket

# ══════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════
def run(ctmc_file,kmin,kmax,nk,zs,msub,inc_first,Mp,Nxp):
    res=load_result(ctmc_file); kappas=np.linspace(kmin,kmax,nk)
    brange=range(0 if inc_first else 1, res.n_buckets)
    rows=[]; pd_=[]
    for bi in brange:
        lb=f"{res.pillar_labels[bi-1] if bi>0 else '0'} → {res.pillar_labels[bi]}"
        print(f"\n[bucket {bi}] {lb}")
        t0=time.time()
        plv,F1,F2,DF1,DF2,tau=forward_start_lv(res,bi,kappas,z_stride=zs,max_substeps=msub)
        print(f"  LV: {time.time()-t0:.1f}s")
        t0=time.time()
        pct,_,_,_,_,_=forward_start_lamperti(res,bi,kappas,max_substeps=msub,M_price=Mp,Nx_price=Nxp)
        print(f"  Lamperti: {time.time()-t0:.1f}s")
        D=DF2/DF1; F=F2/F1
        ivl=np.array([impl_vol(p/(DF1*F1),D,F,k,tau) for p,k in zip(plv,kappas)])
        ivc=np.array([impl_vol(p/(DF1*F1),D,F,k,tau) for p,k in zip(pct,kappas)])
        for k,pl,pc,vl,vc in zip(kappas,plv,pct,ivl,ivc):
            rows.append(dict(bucket=bi,pair=lb,tau=tau,fwd=F,strike=k,
                             lv_price=pl,lam_price=pc,lv_iv=vl,lam_iv=vc))
        pd_.append((lb,kappas.copy(),ivl,ivc))
    return rows,pd_

def plot_(pd_,out):
    n=len(pd_); nc=2 if n>1 else 1; nr=math.ceil(n/nc)
    fig,axes=plt.subplots(nr,nc,figsize=(7*nc,4.6*nr),squeeze=False)
    for ax,(lb,kp,il,ic) in zip(axes.ravel(),pd_):
        vl=~np.isnan(il); vc=~np.isnan(ic)
        if vl.sum()>0: ax.plot(kp[vl],il[vl],lw=2,label='LV')
        if vc.sum()>0: ax.plot(kp[vc],ic[vc],lw=2,ls='--',label='Lamperti-LSV')
        ax.set_title(lb); ax.set_xlabel(r'$K/S(T_1)$'); ax.set_ylabel('Forward IV')
        ax.grid(True,alpha=.3); ax.legend()
    for ax in axes.ravel()[len(pd_):]: ax.axis('off')
    fig.suptitle('Forward-Start IV: LV vs CTMC-Lamperti-LSV (ρ≠0)',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(out,dpi=180,bbox_inches='tight'); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ctmc_file",default="coupled_v2/lamperti_lsv_model.npz")
    p.add_argument("--out_png",default="forward_start_smiles_lamperti.png")
    p.add_argument("--out_csv",default="forward_start_smiles_lamperti.csv")
    p.add_argument("--kappa_min",type=float,default=.70)
    p.add_argument("--kappa_max",type=float,default=1.30)
    p.add_argument("--n_kappa",type=int,default=25)
    p.add_argument("--z_stride",type=int,default=8)
    p.add_argument("--max_substeps",type=int,default=24)
    p.add_argument("--M_price",type=int,default=80)
    p.add_argument("--Nx_price",type=int,default=401)
    p.add_argument("--include_first_bucket",action="store_true")
    a=p.parse_args()
    rows,pd_=run(a.ctmc_file,a.kappa_min,a.kappa_max,a.n_kappa,a.z_stride,a.max_substeps,
                 a.include_first_bucket,a.M_price,a.Nx_price)
    import pandas; df=pandas.DataFrame(rows)
    Path(a.out_csv).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.out_csv,index=False); plot_(pd_,a.out_png)
    print(f"\nSaved: {a.out_csv}, {a.out_png}")

if __name__=="__main__": main()
