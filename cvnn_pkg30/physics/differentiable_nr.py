"""differentiable_nr.py — sparse vectorised NR + implicit diff + trust region"""
import torch, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from utils.sparse_complex import sparse_complex_mm
from data.ybus_builder import SPARSE_LIMIT, get_GB
from physics.trust_region_nr import clip_nr_step_cluster, clip_step
disablePrint=True
def _csr_to_scipy(t):
    if t.layout==torch.sparse_csr:
        return sp.csr_matrix((t.values().numpy().astype(np.float64),
                               t.col_indices().numpy().astype(np.int32),
                               t.crow_indices().numpy().astype(np.int32)),shape=tuple(t.shape))
    t=t.coalesce(); idx=t.indices().numpy()
    return sp.csr_matrix((t.values().numpy().astype(np.float64),(idx[0],idx[1])),shape=tuple(t.shape))

def _build_sparse_jacobian(Vm,Va,S_calc,Yr,Yi,aa,av,pv,pq,n_pv,n_pq,dP,dQ,reg):
    n_a=len(aa); n_v=len(av)
    Yr_csr=Yr.tocsr() if hasattr(Yr,'tocsr') else Yr
    Yi_csr=Yi.tocsr() if hasattr(Yi,'tocsr') else Yi
    Ycoo=(Yr_csr+1j*Yi_csr).tocoo()
    ib=Ycoo.row; jb=Ycoo.col; yij=Ycoo.data
    Gij=yij.real; Bij=yij.imag
    th=Va[ib]-Va[jb]; Vij=Vm[ib]*Vm[jb]; diag=(ib==jb)
    Yd=np.array(Yr_csr.diagonal()); Bd=np.array(Yi_csr.diagonal())
    mb=Yr_csr.shape[0]
    am=np.full(mb,-1,np.int32); vm=np.full(mb,-1,np.int32)
    am[aa]=np.arange(n_a,dtype=np.int32); vm[av]=np.arange(n_v,dtype=np.int32)
    ki=am[ib]; kja=am[jb]; kiv=vm[ib]; kjv=vm[jb]
    def blk(rok,cok,off,dv):
        msk=rok&cok; val=off.copy(); val[diag&msk]=dv[diag&msk]; return val,msk
    H_off=Vij*(Gij*np.sin(th)-Bij*np.cos(th)); H_d=-S_calc[ib].imag-Bd[ib]*Vm[ib]**2
    hv,hm=blk(ki>=0,kja>=0,H_off,H_d)
    H=sp.csr_matrix((hv[hm],(ki[hm],kja[hm])),shape=(n_a,n_a))
    N_off=Vij*(Gij*np.cos(th)+Bij*np.sin(th)); N_d=S_calc[ib].real+Yd[ib]*Vm[ib]**2
    nv2,nm=blk(ki>=0,kjv>=0,N_off,N_d)
    N=sp.csr_matrix((nv2[nm],(ki[nm],kjv[nm])),shape=(n_a,n_v))
    M_off=-Vij*(Gij*np.cos(th)+Bij*np.sin(th)); M_d=S_calc[ib].real-Yd[ib]*Vm[ib]**2
    mv2,mm=blk(kiv>=0,kja>=0,M_off,M_d)
    M=sp.csr_matrix((mv2[mm],(kiv[mm],kja[mm])),shape=(n_v,n_a))
    L_off=Vij*(Gij*np.sin(th)-Bij*np.cos(th)); L_d=S_calc[ib].imag-Bd[ib]*Vm[ib]**2
    lv2,lm=blk(kiv>=0,kjv>=0,L_off,L_d)
    L=sp.csr_matrix((lv2[lm],(kiv[lm],kjv[lm])),shape=(n_v,n_v))
    J=sp.bmat([[H,N],[M,L]],format='csr')+reg*sp.eye(n_a+n_v,format='csr')
    return J, np.concatenate([dP[aa],dQ[av]])

def _sparse_solve(J,rhs):
    try:
        ilu=spla.spilu(J.tocsc(),fill_factor=10)
        Mpc=spla.LinearOperator(J.shape,ilu.solve)
        dx,info=spla.bicgstab(J,rhs,M=Mpc,atol=1e-8,maxiter=300)
        if not disablePrint:
            print(f"  sparse solve BiCGStab info={info}  ||J*dx - rhs||={np.linalg.norm(J.dot(dx)-rhs):.3e}  ||dx||={np.linalg.norm(dx):.3e}")
        if info==0 and not np.any(np.isnan(dx)): return dx
    except: pass
    return spla.spsolve(J.tocsc(),rhs)

class ImplicitNRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,V_in,Pg,Qg,gd_ref,nr_steps,damping,reg):
        V=V_in.detach().clone().numpy().astype(np.complex128)
        n_bus=V_in.shape[0]
        Yr_np=_csr_to_scipy(gd_ref["Y_real_sp"]); Yi_np=_csr_to_scipy(gd_ref["Y_imag_sp"])
        Yc=Yr_np+1j*Yi_np
        bt=gd_ref["bus_type"].numpy(); gb=gd_ref["gen_bus"].numpy()
        sd=gd_ref["s_demand"].numpy(); Pg_np=Pg.detach().numpy(); Qg_np=Qg.detach().numpy()
        pq=np.where(bt==1)[0]; pv=np.where(bt==2)[0]
        aa=np.concatenate([pv,pq]); av=pq
        n_a=len(aa); n_v=len(av); n_pv=len(pv); n_pq=len(pq)
        J_last=None; _tr=gd_ref.get('_tr_radius',1.0)
        for _ in range(nr_steps):
            # Project stiff internal pairs onto equality manifold
            for gi, gj in gd_ref.get("stiff_pairs", []):
                li = gd_ref["global_to_local_bus"].get(int(gi), -1)
                lj = gd_ref["global_to_local_bus"].get(int(gj), -1)

                if li >= 0 and lj >= 0:
                    V_avg = 0.5 * (V[li] + V[lj])
                    V[li] = V_avg
                    V[lj] = V_avg
            # print(f"  Projecting stiff pair {gi}-{gj} onto equality manifold: V[{li}]={V[li]:.6e}, V[{lj}]={V[lj]:.6e}")
            Sg=np.zeros(n_bus,dtype=np.complex128); np.add.at(Sg,gb,Pg_np+1j*Qg_np)
            Ss=Sg-sd.astype(np.complex128); Sc=V*np.conj(Yc.dot(V))
            dP=Ss.real-Sc.real; dQ=Ss.imag-Sc.imag
            Vm=np.abs(V).clip(1e-4); Va=np.angle(V)
            J,mis=_build_sparse_jacobian(Vm,Va,Sc,Yr_np,Yi_np,aa,av,pv,pq,n_pv,n_pq,dP,dQ,reg)
            if not disablePrint:
                print(f"  J shape={J.shape} nnz={J.nnz} ||J||_inf={abs(J).max():.3e}")
            diag_abs = np.abs(J.diagonal())
            offdiag_sum = np.array(np.abs(J).sum(axis=1)).flatten() - diag_abs
            if not disablePrint:
                print(f"  Diag dom: min={float((diag_abs/np.maximum(offdiag_sum,1e-12)).min()):.3e} max={float(diag_abs.max()):.3e}")
                print(f"  mis_inf={np.abs(mis).max():.3e} worst_idx={np.argmax(np.abs(mis))} n_a={n_a}")
            worst = int(np.argmax(np.abs(mis)))
            if not disablePrint:
                print(f"  worst is {'angle_eq' if worst<n_a else 'volt_eq'} at local_idx={worst if worst<n_a else worst-n_a}")
            dx=_sparse_solve(J,mis)
            # Save the raw Newton correction BEFORE clipping
            dx_raw = dx.copy()
            # fix trust region radius based on the raw step size
            norm_dx = np.linalg.norm(dx_raw)
            dx = dx_raw
            #end fix trust region radius based on the raw step size
            if not disablePrint:
                print(
                    f"After sparse solve NR ITER {_}: "
                    f"mis_inf={np.max(np.abs(mis)):.6e} "
                    f"mis_rms={np.sqrt(np.mean(mis**2)):.6e} "
                    f"dx_inf={np.max(np.abs(dx)):.6e} "
                    f"dx_rms={np.sqrt(np.mean(dx**2)):.6e}"
            )
            if not disablePrint:
                print(
                    f"NR RAW {_}: "
                    f"inf={np.max(np.abs(dx_raw)):.6e} "
                    f"rms={np.sqrt(np.mean(dx_raw**2)):.6e} "
                    f"l2={np.linalg.norm(dx_raw):.6e} "
                    f"trust={_tr:.6e}"
                )
            # skip this dx,norm_dx=clip_step(dx,_tr)
            dx, norm_dx = clip_nr_step_cluster(
                    dx_raw,
                    n_a=n_a,
                    n_v=n_v,
                    aa=aa,
                    av=av,
                    stiff_pairs=gd_ref.get("stiff_pairs", []),
                    g2l=gd_ref.get("global_to_local_bus"),

                    angle_max=0.05,
                    vm_rel_max=0.02,

                    # Stiff branch endpoints
                    stiff_angle_max=0.02,
                    stiff_vm_rel_max=0.01,
                )
            if not disablePrint:
                print(
                    f"NR CLIP {_}: "
                    f"raw_inf={np.max(np.abs(dx_raw)):.6e} "
                    f"clipped_inf={np.max(np.abs(dx)):.6e} "
                    f"norm_dx={norm_dx:.6e} "
                    f"trust={_tr:.6e}"
                )
            if norm_dx>_tr*0.9: _tr=max(_tr*0.5,1e-4)
            elif norm_dx<_tr*0.1: _tr=min(_tr*2.0,10.0)
            V_before = V.copy(); Va[aa]+=damping*dx[:n_a]; Vm[av]*=(1.+damping*dx[n_a:])
            Vm=np.clip(Vm,0.5,1.5); V=Vm*np.exp(1j*Va); J_last=J
            # Project stiff internal pairs onto equality manifold
            for gi, gj in gd_ref.get("stiff_pairs", []):
                    li = gd_ref["global_to_local_bus"].get(int(gi), -1)
                    lj = gd_ref["global_to_local_bus"].get(int(gj), -1)

                    if li >= 0 and lj >= 0:
                        V_avg = 0.5 * (V[li] + V[lj])
                        V[li] = V_avg
                        V[lj] = V_avg
            if not disablePrint:
                print(
                    f"NR UPDATE {_}: "
                    f"|dV|max={np.max(np.abs(V - V_before)):.6e}"
                )
        gd_ref['_tr_radius']=float(_tr)
        V_out=torch.tensor(V,dtype=torch.complex64)
        ctx.save_for_backward(V_in,V_out,Pg,Qg)
        ctx._J=J_last; ctx._aa=aa; ctx._av=av; ctx._n_a=n_a; ctx._n_bus=n_bus
        Yr=gd_ref["Y_real_sp"]; Yi=gd_ref["Y_imag_sp"]
        ic=sparse_complex_mm(Yr,Yi,V_out); Sc2=V_out*torch.conj(ic)
        Sgt=torch.zeros(n_bus,dtype=torch.complex64)
        Sgt.scatter_add_(0,gd_ref["gen_bus"],torch.complex(Pg,Qg))
        Ss2=Sgt-gd_ref["s_demand"]
        ctx._dP=(Ss2.real-Sc2.real).detach(); ctx._dQ=(Ss2.imag-Sc2.imag).detach()
        pq_t=(gd_ref["bus_type"]==1).nonzero(as_tuple=True)[0]
        pv_t=(gd_ref["bus_type"]==2).nonzero(as_tuple=True)[0]
        aa_t=torch.cat([pv_t,pq_t])
        mis_t=torch.cat([ctx._dP[aa_t],ctx._dQ[pq_t]])
        ctx._nr_inf=float(mis_t.abs().max()); ctx._nr_rms=float(torch.sqrt((mis_t**2).mean()))
        return V_out
    @staticmethod
    def backward(ctx,grad_V):
        if ctx._J is None: return grad_V,None,None,None,None,None,None
        JT=ctx._J.T.tocsr()
        rhs=np.concatenate([grad_V.real.numpy().astype(np.float64)[ctx._aa],
                             grad_V.imag.numpy().astype(np.float64)[ctx._av]])
        lam=_sparse_solve(JT,rhs)
        gfr=np.zeros(ctx._n_bus); gfi=np.zeros(ctx._n_bus)
        gfr[ctx._aa]=lam[:ctx._n_a]; gfi[ctx._av]=lam[ctx._n_a:]
        return (torch.complex(torch.tensor(gfr,dtype=torch.float32),
                              torch.tensor(gfi,dtype=torch.float32)),
                None,None,None,None,None,None)

def differentiable_nr_step(state,gd,damping=0.8,regularization=1e-6,nr_steps=1):
    if gd["n_bus"]<SPARSE_LIMIT: return _dense_nr_step(state,gd,damping,regularization)
    V=state["V"]; Pg=state["Pg"]; Qg=state["Qg"]
    Vg=V if V.requires_grad else V.detach().requires_grad_(True)
    Vo=ImplicitNRFunction.apply(Vg,Pg.detach(),Qg.detach(),gd,nr_steps,damping,regularization)
    Yr=gd["Y_real_sp"]; Yi=gd["Y_imag_sp"]
    ic=sparse_complex_mm(Yr,Yi,Vo); Sc=Vo*torch.conj(ic)
    Sg=torch.zeros(gd["n_bus"],dtype=torch.complex64)
    Sg.scatter_add_(0,gd["gen_bus"],torch.complex(Pg,Qg)); Ss=Sg-gd["s_demand"]
    pq=(gd["bus_type"]==1).nonzero(as_tuple=True)[0]
    pv=(gd["bus_type"]==2).nonzero(as_tuple=True)[0]
    aa=torch.cat([pv,pq]); mis=torch.cat([(Ss-Sc).real[aa],(Ss-Sc).imag[pq]])
    return {"V":Vo,"Pg":Pg,"Qg":Qg,
            "deltaP":(Ss.real-Sc.real).detach(),"deltaQ":(Ss.imag-Sc.imag).detach(),
            "nr_mismatch_inf":float(mis.abs().max()),
            "nr_mismatch_rms":float(torch.sqrt((mis**2).mean()))}

def _dense_nr_step(state,gd,damping=0.8,regularization=1e-6):
    V=state["V"]; Pg=state["Pg"]; Qg=state["Qg"]
    n=gd["n_bus"]; dev=V.device
    Sg=torch.zeros(n,dtype=torch.complex64,device=dev)
    Sg.scatter_add_(0,gd["gen_bus"],torch.complex(Pg,Qg))
    ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V)
    Sc=V*torch.conj(ic); Ss=Sg-gd["s_demand"]
    dPa=Ss.real-Sc.real; dQa=Ss.imag-Sc.imag
    pq=(gd["bus_type"]==1).nonzero(as_tuple=True)[0]
    pv=(gd["bus_type"]==2).nonzero(as_tuple=True)[0]
    aa=torch.cat([pv,pq]); av=pq; npv=len(pv); npq=len(pq); na=npv+npq; nv=npq
    if na==0:
        state.update({"V":V,"deltaP":dPa,"deltaQ":dQa,"nr_mismatch_inf":0.,"nr_mismatch_rms":0.})
        return state
    G,B=get_GB(gd); Vm=torch.abs(V).clamp(1e-4); Va=torch.angle(V)
    ai=aa.unsqueeze(1); aj=aa.unsqueeze(0); vi=av.unsqueeze(1); vj=av.unsqueeze(0)
    th=Va[ai]-Va[aj]
    H=Vm[ai]*Vm[aj]*(G[ai,aj]*torch.sin(th)-B[ai,aj]*torch.cos(th))
    ia=torch.arange(na,device=dev); H[ia,ia]=-Sc[aa].imag-B[aa,aa]*Vm[aa]**2
    th=Va[ai]-Va[vj]
    N=Vm[ai]*Vm[vj]*(G[ai,vj]*torch.cos(th)+B[ai,vj]*torch.sin(th))
    ki=torch.arange(npv,na,device=dev); kj=torch.arange(npq,device=dev)
    if len(ki)>0 and len(kj)>0: N[ki,kj]=Sc[pq].real+G[pq,pq]*Vm[pq]**2
    th=Va[vi]-Va[aj]
    M=-Vm[vi]*Vm[aj]*(G[vi,aj]*torch.cos(th)+B[vi,aj]*torch.sin(th))
    if len(ki)>0 and len(kj)>0: M[kj,ki]=Sc[pq].real-G[pq,pq]*Vm[pq]**2
    th=Va[vi]-Va[vj]
    L=Vm[vi]*Vm[vj]*(G[vi,vj]*torch.sin(th)-B[vi,vj]*torch.cos(th))
    iv=torch.arange(nv,device=dev); L[iv,iv]=Sc[av].imag-B[av,av]*Vm[av]**2
    J=torch.cat([torch.cat([H,N],1),torch.cat([M,L],1)],0)
    J=J+regularization*torch.eye(J.shape[0],dtype=J.dtype,device=dev)
    mis=torch.cat([dPa[aa],dQa[av]])
    try: dx=torch.linalg.solve(J,mis)
    except: dx=torch.linalg.lstsq(J,mis.unsqueeze(-1)).solution.squeeze(-1)
    norm_dx=float(dx.norm()); tr_radius=1.0
    if norm_dx>tr_radius: dx=dx*(tr_radius/norm_dx)
    dth=dx[:na]; dVf=dx[na:]
    Van=Va.clone(); Vmn=Vm.clone()
    Van[aa]=Va[aa]+damping*dth; Vmn[av]=Vm[av]*(1.+damping*dVf)
    Vmn=Vmn.clamp(0.5,1.5); Vn=Vmn*torch.exp(1j*Van)
    mi=float(mis.abs().max()); mr=float(torch.sqrt((mis**2).mean()))
    return {"V":Vn,"Pg":Pg,"Qg":Qg,"deltaP":dPa,"deltaQ":dQa,
            "nr_mismatch_inf":mi,"nr_mismatch_rms":mr}

def _jacobian_cond(sg,V_test):
    n=sg["n_bus"]; bt=sg["bus_type"]
    pq=(bt==1).nonzero(as_tuple=True)[0]; pv=(bt==2).nonzero(as_tuple=True)[0]
    aa=torch.cat([pv,pq]); av=pq; na=len(aa); nv=len(av)
    if na==0: return 1.0
    if (na+nv)>1200: return float(1e5*(na+nv)/300.0)
    G,B=get_GB(sg); Vm=torch.abs(V_test).clamp(1e-4); Va=torch.angle(V_test)
    ic=sparse_complex_mm(sg["Y_real_sp"],sg["Y_imag_sp"],V_test); Sc=V_test*torch.conj(ic)
    npv=len(pv); npq=len(pq)
    ai=aa.unsqueeze(1); aj=aa.unsqueeze(0); vi=av.unsqueeze(1); vj=av.unsqueeze(0)
    th=Va[ai]-Va[aj]
    H=Vm[ai]*Vm[aj]*(G[ai,aj]*torch.sin(th)-B[ai,aj]*torch.cos(th))
    ia=torch.arange(na); H[ia,ia]=-Sc[aa].imag-B[aa,aa]*Vm[aa]**2
    th=Va[ai]-Va[vj]
    N=Vm[ai]*Vm[vj]*(G[ai,vj]*torch.cos(th)+B[ai,vj]*torch.sin(th))
    ki=torch.arange(npv,na); kj=torch.arange(npq)
    if len(ki)>0 and len(kj)>0: N[ki,kj]=Sc[pq].real+G[pq,pq]*Vm[pq]**2
    th=Va[vi]-Va[aj]
    M=-Vm[vi]*Vm[aj]*(G[vi,aj]*torch.cos(th)+B[vi,aj]*torch.sin(th))
    if len(ki)>0 and len(kj)>0: M[kj,ki]=Sc[pq].real-G[pq,pq]*Vm[pq]**2
    th=Va[vi]-Va[vj]
    L=Vm[vi]*Vm[vj]*(G[vi,vj]*torch.sin(th)-B[vi,vj]*torch.cos(th))
    iv=torch.arange(nv); L[iv,iv]=Sc[av].imag-B[av,av]*Vm[av]**2
    J=torch.cat([torch.cat([H,N],1),torch.cat([M,L],1)],0)+1e-6*torch.eye(na+nv)
    try:
        sv=torch.linalg.svdvals(J)
        return float(sv.max()/sv.clamp(1e-12).min())
    except: return float('nan')
