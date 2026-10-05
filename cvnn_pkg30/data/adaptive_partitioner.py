"""adaptive_partitioner.py — power law size=42.06*n^0.2162, returns partition_stats"""
import sys,os,numpy as np,torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.grid_partitioner import partition_grid,_empty_boundary
from physics.differentiable_nr import _dense_nr_step,_jacobian_cond
from utils.sparse_complex import sparse_complex_mm

POWER_C=42.06; POWER_A=0.2162; SIZE_MIN=100; SIZE_MAX=500
PV_MIN=0.08; COND_MAX=5e9

def empirical_target_size(n):
    return int(max(SIZE_MIN,min(SIZE_MAX,round(POWER_C*n**POWER_A))))

def analyse_grid(gd):
    n=gd["n_bus"]; bt=gd["bus_type"].numpy()
    n_pq=int((bt==1).sum()); n_pv=int((bt==2).sum()); n_sl=int((bt==3).sum())
    pv_ratio=n_pv/max(n_pq+n_pv,1); emp=empirical_target_size(n)
    print(f"\n  Grid topology: {n:,} buses (PQ={n_pq} PV={n_pv} slack={n_sl})")
    print(f"  PV ratio={100*pv_ratio:.1f}%  Empirical size=~{emp} ({max(2,round(n/emp))} sub-grids)")
    return {"n_bus":n,"pv_ratio":pv_ratio,"emp_size":emp,"emp_nsubs":max(2,round(n/emp))}

def _check_partition(sgs,bnd,gd):
    V_flat=torch.ones(gd["n_bus"],dtype=torch.complex64)
    pv_ratios=[]; conds=[]
    for sg in sgs:
        bt=sg["bus_type"]; npv=int((bt==2).sum()); npq=int((bt==1).sum())
        pv_ratios.append(npv/max(npv+npq,1))
        conds.append(_jacobian_cond(sg,V_flat[sg["global_bus_idx"]].clone()))
    min_pv=min(pv_ratios)
    max_cond=float(np.nanmax([c for c in conds if np.isfinite(c)] or [0]))
    metrics={"min_pv_ratio":min_pv,"max_cond":max_cond,"n_subs":len(sgs)}
    if min_pv<PV_MIN: return False,f"PV {100*min_pv:.1f}%<{100*PV_MIN:.0f}%",metrics
    if max_cond>COND_MAX: return False,f"cond {max_cond:.2e}",metrics
    return True,"ok",metrics

def _nr_test(sgs,gd,steps=3):
    nr_infs=[]
    for sg in sgs:
        gb=sg["global_bus_idx"]; gg=sg["global_gen_idx"]
        V_init=gd["v0"][gb].clone() if "v0" in gd else torch.ones(sg["n_bus"],dtype=torch.complex64)
        st={"V":V_init,
            "Pg":gd["pg"][gg].clone() if len(gg)>0 else sg["pg"].clone(),
            "Qg":gd["qg"][gg].clone() if len(gg)>0 else sg["qg"].clone()}
        for _ in range(steps): st=_dense_nr_step(st,sg,damping=0.8,regularization=1e-6)
        nr_infs.append(st["nr_mismatch_inf"])
    return float(np.mean(nr_infs)),float(max(nr_infs))

def find_optimal_partition(gd,nr_test_steps=3,verbose=True):
    n=gd["n_bus"]
    if n<=500: return [gd],_empty_boundary(),n,{}
    metrics=analyse_grid(gd)
    emp_size=metrics["emp_size"]; emp_nsubs=metrics["emp_nsubs"]
    candidates=sorted(set(max(2,emp_nsubs+d) for d in [-2,-1,0,1,2,3]))
    sizes=[int(np.ceil(n/p)) for p in candidates]
    if verbose:
        print(f"\n  Testing {len(candidates)} candidates (empirical={emp_size}):")
        print(f"  {'Parts':>6} {'Size':>6} {'MinPV':>7} {'MaxCond':>12}  {'NR_inf':>10}  Result")
        print(f"  {'─'*55}")
    best_sgs=None; best_bnd=None; best_size=emp_size
    best_nr=float('inf'); report={"candidates":[],"chosen":None}
    for n_parts,target in zip(candidates,sizes):
        sgs,bnd,_,_ps=partition_grid(gd,target_size=target)
        actual=len(sgs); sz=int(np.mean([sg["n_bus"] for sg in sgs]))
        passed,reason,pmx=_check_partition(sgs,bnd,gd)
        entry={"n_parts":actual,"size":sz,"target":target,"passed":passed,"reason":reason,**pmx}
        if passed:
            nr_mean,_=_nr_test(sgs,gd,steps=nr_test_steps)
            entry["nr_mean"]=nr_mean
            if verbose:
                print(f"  {actual:6d} {sz:6d}  {100*pmx['min_pv_ratio']:6.1f}%  "
                      f"{pmx['max_cond']:12.2e}  {nr_mean:10.3e}  PASS")
            if nr_mean<best_nr:
                best_nr=nr_mean; best_sgs=sgs; best_bnd=bnd
                best_size=target; report["chosen"]=entry
                report["partition_stats"]=_ps
        else:
            if verbose:
                print(f"  {actual:6d} {sz:6d}  {100*pmx['min_pv_ratio']:6.1f}%  "
                      f"{pmx['max_cond']:12.2e}  {'n/a':>10}  FAIL ({reason})")
        report["candidates"].append(entry)
    if best_sgs is None:
        print(f"\n  WARNING: fallback to empirical size {emp_size}")
        best_sgs,best_bnd,_,_ps=partition_grid(gd,target_size=emp_size)
        best_size=emp_size
        report["chosen"]={"n_parts":len(best_sgs),"size":emp_size,"fallback":True}
        report["partition_stats"]=_ps
    return best_sgs,best_bnd,best_size,report.get("partition_stats",{})

def auto_partition(gd,user_partition_size=None,verbose=True):
    n=gd["n_bus"]
    if n<=500: return [gd],_empty_boundary(),n,{}
    if user_partition_size is not None:
        print(f"\n  Verifying --partition-size {user_partition_size} ...")
        sgs,bnd,_,_ps=partition_grid(gd,target_size=user_partition_size)
        passed,reason,pmx=_check_partition(sgs,bnd,gd)
        emp=empirical_target_size(n)
        if not passed:
            print(f"  WARNING: {reason}. Empirical suggestion: --partition-size {emp}")
        else:
            nr_mean,_=_nr_test(sgs,gd,steps=3)
            print(f"  OK  PV={100*pmx['min_pv_ratio']:.1f}%  "
                  f"cond={pmx['max_cond']:.2e}  NR_inf={nr_mean:.2e}")
        return sgs,bnd,user_partition_size,_ps
    sgs,bnd,rec,_ps=find_optimal_partition(gd,verbose=verbose)
    return sgs,bnd,rec,_ps
