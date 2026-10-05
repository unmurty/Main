"""debug_bus.py — diagnose worst residual buses + near-zero impedance branches"""
import torch, numpy as np, sys
sys.path.insert(0,'.')
from utils.sparse_complex import sparse_complex_mm

def diagnose_worst_bus(gd, V=None, Pg=None, Qg=None, top_n=10):
    n=gd["n_bus"]; B=gd["baseMVA"]
    V =(V  if V  is not None else gd["v0"]).detach()
    Pg=(Pg if Pg is not None else gd["pg"]).detach()
    Qg=(Qg if Qg is not None else gd["qg"]).detach()
    ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V)
    Sc=V*torch.conj(ic)
    Sg=torch.zeros(n,dtype=torch.complex64)
    Sg.scatter_add_(0,gd["gen_bus"],torch.complex(Pg,Qg))
    Ss=Sg-gd["s_demand"]; dP=Ss.real-Sc.real; dQ=Ss.imag-Sc.imag; Vm=V.abs()
    print(f"\n{'='*60}\n  Bus residual analysis  baseMVA={B}")
    print(f"  ΔP max={dP.abs().max()*B:.1f}MW  mean={dP.abs().mean()*B:.1f}MW")
    print(f"  ΔQ max={dQ.abs().max()*B:.1f}MVAR mean={dQ.abs().mean()*B:.1f}MVAR")
    print(f"  |V|  [{Vm.min():.4f}, {Vm.max():.4f}] pu")
    combined={}
    for idx in range(n):
        dp=float(dP[idx]*B); dq=float(dQ[idx]*B); w=max(abs(dp),abs(dq))
        if w>0: combined[idx]=(dp,dq,w)
    top=sorted(combined.items(),key=lambda x:-x[1][2])[:top_n]
    bt_names={0:"?",1:"PQ",2:"PV",3:"Slack"}
    print(f"\n  Top {top_n} worst buses:")
    print(f"  {'Bus':>8} {'Type':>6} {'|V|':>7} {'ΔP(MW)':>12} {'ΔQ(MVAR)':>12}  Note")
    for idx,(dp,dq,_) in top:
        bt=int(gd["bus_type"][idx]); vm=float(Vm[idx]); note=""
        if vm<=0.701: note="⚠ lower clamp"
        elif vm>=1.299: note="⚠ upper clamp"
        if max(abs(dp),abs(dq))>10000: note+=" ⚠ stiff branch?"
        print(f"  {idx:>8} {bt_names.get(bt,'?'):>6} {vm:>7.4f} {dp:>12.1f} {dq:>12.1f}  {note}")
    print(f"\n  Near-zero impedance branches (|Z| < 1e-3 pu):")
    branch=gd["branch_arr"]; fr=gd["fbus"].numpy(); tr=gd["tbus"].numpy()
    found=0
    for k in range(len(branch)):
        if int(branch[k,10])==0: continue
        r_=float(branch[k,2]); x_=float(branch[k,3])
        z=complex(r_,x_); absz=abs(z)
        if absz<1e-3:
            y=1./z if absz>1e-12 else 0
            from data.grid_partitioner import branch_edge_weight
            w=branch_edge_weight(r_,x_)
            f=int(fr[k]); t=int(tr[k])
            print(f"    Branch {k:6d}: bus {f}↔{t}  |Z|={absz:.2e}  |Y|={abs(y):.1f}  weight={w}")
            found+=1
            if found>=20: print("    ... (first 20 shown)"); break
    if found==0: print("    None found")
    bus=gd["bus_arr"]; bs_pu=bus[:,5]/gd["baseMVA"]
    extreme=np.where(np.abs(bs_pu)>100)[0]
    if len(extreme)>0:
        print(f"\n  ⚠ Extreme bus shunts (|Bs|>100 pu):")
        for ib in extreme[:5]:
            print(f"    Bus {ib}: Bs_raw={bus[ib,5]:.1f}  Bs_pu={bs_pu[ib]:.1f}")
    return {"dP":dP,"dQ":dQ,"top_buses":top}

if __name__=="__main__":
    from data.matpower_parser import load_matpower_case
    from data.ybus_builder import build_ybus, add_normalised_ybus
    case=sys.argv[1] if len(sys.argv)>1 else "data/cases/case118.m"
    gd=load_matpower_case(casepath=case); gd=build_ybus(gd); gd=add_normalised_ybus(gd)
    diagnose_worst_bus(gd)
