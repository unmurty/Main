"""
train.py — CVNN-GNN + Diakoptics NR (Option B: stiffness-aware METIS)
PRIMARY metric: global_nr_rms
Diagnostics: stiff_cuts, global_nr_rms, bnd_mw, kcl_p99, nan_buses
"""
import os,sys,time,argparse
from turtle import forward
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ.setdefault("OMP_NUM_THREADS","1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

import torch
from data.matpower_parser      import load_matpower_case
from data.ybus_builder         import build_ybus,add_normalised_ybus,SPARSE_LIMIT
from models.cvgnn              import PowerGNN
from physics.slack_balance     import apply_slack_balance
from physics.differentiable_nr import differentiable_nr_step,_dense_nr_step
from physics.branch_flow       import compute_branch_flows
from physics.diagnostics       import compute_all_diagnostics
from losses.loss_function      import (compute_power_flow_loss,compute_voltage_loss,
                                       compute_generator_limit_loss,
                                       compute_thermal_violations,compute_generation_cost)
from utils.sparse_complex      import sparse_complex_mm

DENSE_NR_MAX=SPARSE_LIMIT

def _safe_where(V,fill_real=1.0,fill_imag=0.0):
    bad=torch.isnan(V.real)|torch.isinf(V.real)|torch.isnan(V.imag)|torch.isinf(V.imag)
    if not bad.any(): return V
    fill=torch.complex(torch.full_like(V.real,fill_real),torch.full_like(V.imag,fill_imag))
    return torch.where(bad,fill,V)

def diakoptics_forward(V_gnn,Pg,Qg,subgrids,boundary,gd,nr,damp,reg):
    from data.grid_partitioner import reconcile_boundary
    NR_SKIP=20.0/gd["baseMVA"]; NR_LIGHT=200.0/gd["baseMVA"]; STIFF_Y=500.0
    nr_inf_all=[]; nr_rms_all=[]; V_refined=[]
    for sg in subgrids:
        gb=sg["global_bus_idx"]; gg=sg["global_gen_idx"]
        V_sub=V_gnn[gb]
        Pg_sub=Pg[gg] if len(gg)>0 else sg["pg"].clone()
        Qg_sub=Qg[gg] if len(gg)>0 else sg["qg"].clone()
        with torch.no_grad():
         if 54785 in gb.tolist() and 54787 in gb.tolist():
            lf = (gb == 54785).nonzero(as_tuple=True)[0].item()
            lt = (gb == 54787).nonzero(as_tuple=True)[0].item()

            dv = V_sub[lf] - V_sub[lt]

            print(
                "\n  PRE-NR STIFF CHECK 54785-54787:"
                f" |dV|={float(abs(dv)):.6e}"
                f" dVm={float(abs(V_sub[lf])-abs(V_sub[lt])):.6e}"
                f" dAng={float(torch.angle(V_sub[lf])-torch.angle(V_sub[lt])):.6e}"
                f" |Vf|={float(abs(V_sub[lf])):.6f}"
                f" |Vt|={float(abs(V_sub[lt])):.6f}"
            )
        sg_nr=nr
        if nr>0:
            
            with torch.no_grad():
                ic_pre=sparse_complex_mm(sg["Y_real_sp"],sg["Y_imag_sp"],V_sub.detach())
                Sc_pre=V_sub.detach()*torch.conj(ic_pre)
                Sg_pre=torch.zeros(sg["n_bus"],dtype=torch.complex64)
                if len(gg)>0:
                    Sg_pre.scatter_add_(0,sg["gen_bus"],
                                        torch.complex(Pg_sub.detach(),Qg_sub.detach()))
                mis_inf=float(torch.cat([(Sg_pre-sg["s_demand"]-Sc_pre).real,
                                         (Sg_pre-sg["s_demand"]-Sc_pre).imag]).abs().max())
            if   mis_inf<NR_SKIP:  sg_nr=0
            elif mis_inf<NR_LIGHT: sg_nr=1
        stiff=sg.get("_is_stiff",False)
        if not stiff and sg.get("G") is not None:
            yd=float(sg["G"].diagonal().abs().max()+sg["B"].diagonal().abs().max())
            stiff=(yd>STIFF_Y); sg["_is_stiff"]=stiff
        state={"V":V_sub,"Pg":Pg_sub,"Qg":Qg_sub}
        if sg_nr>0:
            if stiff:
                with torch.no_grad():
                    Vd=V_sub.detach().clone()
                    st2={"V":Vd,"Pg":Pg_sub.detach(),"Qg":Qg_sub.detach()}
                    if sg["n_bus"]<DENSE_NR_MAX:
                        for _ in range(sg_nr): st2=_dense_nr_step(st2,sg,damping=damp,regularization=reg)
                    else:
                        st2=differentiable_nr_step(st2,sg,damping=damp,regularization=reg,nr_steps=sg_nr)
                     # IMPORTANT: freeze NR result before leaving no_grad
                    st2["V"] = st2["V"].detach().clone()
                delta=st2["V"]-Vd; delta=_safe_where(delta,0.,0.)
                # IMPORTANT to detach and clone to avoid autograd issues
                state["V"] = (V_sub+delta).detach().clone()
                
                state["nr_mismatch_inf"]=st2.get("nr_mismatch_inf",0.)
                state["nr_mismatch_rms"]=st2.get("nr_mismatch_rms",0.)
            elif sg["n_bus"]<DENSE_NR_MAX:
                for _ in range(sg_nr): state=_dense_nr_step(state,sg,damping=damp,regularization=reg)
                # IMPORTANT
                state["V"] = state["V"].detach().clone()
            else:
                print(f"  no stiff ⚠ Subgrid {sg['cluster_id']} n_bus={sg['n_bus']} > {DENSE_NR_MAX} — using sparse NR")
                state=differentiable_nr_step(state,sg,damping=damp,regularization=reg,nr_steps=sg_nr)
                # IMPORTANT
                state["V"] = state["V"].detach().clone()
        with torch.no_grad():
            has_nan=(torch.isnan(state["V"].detach().real).any()|
                     torch.isnan(state["V"].detach().imag).any())
# hasNAN diagnostic
            if has_nan:
                    print(
                        f"  ⚠ NR NaN fallback: sg={subgrids.index(sg)} "
                        f"n_bus={sg['n_bus']}"
                    )
# hasNAN diagnostic end
        if has_nan: state["V"]=V_sub; state["nr_mismatch_inf"]=float(1e6)
# pre vrefined diagnostic
        with torch.no_grad():
            if 54785 in gb.tolist() and 54787 in gb.tolist():
                lf = (gb == 54785).nonzero(as_tuple=True)[0].item()
                lt = (gb == 54787).nonzero(as_tuple=True)[0].item()

                print(
                            f"BEFORE APPEND sg={sg['cluster_id']}: "
                            f"|dV|={float(abs(state['V'][lf] - state['V'][lt])):.8e}"
                        )
        with torch.no_grad():
            if 54785 in gb.tolist() and 54787 in gb.tolist():
                lf = (gb == 54785).nonzero(as_tuple=True)[0].item()
                lt = (gb == 54787).nonzero(as_tuple=True)[0].item()

                dv = state["V"][lf] - state["V"][lt]

                print(
                        "  POST-NR STIFF CHECK 54785-54787:"
                        f" |dV|={float(abs(dv)):.6e}"
                        f" dVm={float(abs(state['V'][lf])-abs(state['V'][lt])):.6e}"
                        f" dAng={float(torch.angle(state['V'][lf])-torch.angle(state['V'][lt])):.6e}"
                    )
# pre vrefined diagnostic ends
        V_refined.append(state["V"])

        nr_inf_all.append(state.get("nr_mismatch_inf",0.))
        nr_rms_all.append(state.get("nr_mismatch_rms",0.))
        if boundary["n_bnd"]>0:
            Sg_sub=torch.zeros(sg["n_bus"],dtype=torch.complex64)
            if len(gg)>0:
                Sg_sub.scatter_add_(0,sg["gen_bus"],
                                    torch.complex(Pg_sub.detach(),Qg_sub.detach()))
            S_spec_sub=Sg_sub-sg["s_demand"]
            bnd_g2l=boundary.get("bnd_g2l",{})
            for lb,gb_idx in enumerate(gb.tolist()):
                if gb_idx in bnd_g2l:
                    boundary["S_spec_bnd"][bnd_g2l[gb_idx]]=S_spec_sub[lb].detach()
    V_assembled=torch.zeros(gd["n_bus"],dtype=torch.complex64)
    for sg,Vr in zip(subgrids,V_refined):
        with torch.no_grad():
            has_nan=(torch.isnan(Vr.detach().real).any()|torch.isnan(Vr.detach().imag).any())
        if has_nan: Vr=V_gnn[sg["global_bus_idx"]]
        V_assembled=V_assembled.clone(); V_assembled[sg["global_bus_idx"]]=Vr
            # ------------------------------------------------------------
            # CHECK THE ACTUAL SUBGRID SOLUTION BEING WRITTEN
            # ------------------------------------------------------------
        with torch.no_grad():
                gb_list = sg["global_bus_idx"].tolist()

                if 54785 in gb_list and 54787 in gb_list:

                    lf = gb_list.index(54785)
                    lt = gb_list.index(54787)

                    v_f = Vr[lf]
                    v_t = Vr[lt]

                    print(
                        f"\n  ASSEMBLY WRITE sg{sg['cluster_id']} 54785-54787:"
                        f" |dV|={float(abs(v_f-v_t)):.6e}"
                        f" dAng={float(torch.angle(v_f*torch.conj(v_t))):.6e}"
                        f" |Vf|={float(abs(v_f)):.6f}"
                        f" |Vt|={float(abs(v_t)):.6f}"
                    )
    if boundary["n_bnd"]>0 and nr>0:
        with torch.no_grad():
            V_det=V_assembled.detach().clone()
            # ============================================================
            # STIFF BRANCH VOLTAGE COHERENCE DIAGNOSTIC
            # MUST RUN BEFORE boundary reconciliation
            # ============================================================
            if nr > 0:
                with torch.no_grad():

                    fr = gd["fbus"].detach().cpu().numpy()
                    tr = gd["tbus"].detach().cpu().numpy()
                    branch_arr = gd["branch_arr"]

                    # Build global bus -> subgrid map once
                    bus_to_sg = {}

                    for sg_id, sg in enumerate(subgrids):
                        for gb in sg["global_bus_idx"].detach().cpu().tolist():
                            bus_to_sg[int(gb)] = sg_id

                    stiff_diag = []

                    for k in range(len(branch_arr)):

                        # branch status
                        if int(branch_arr[k, 10]) == 0:
                            continue

                        f = int(fr[k])
                        t = int(tr[k])

                        # branch impedance
                        r = max(float(branch_arr[k, 2]), 0.0)
                        x = float(branch_arr[k, 3])

                        z = complex(r, x)
                        az = abs(z)

                        if az < 1e-12:
                            continue

                        y_mag = 1.0 / az

                        # Only inspect stiff branches
                        if y_mag <= 10000.0:
                            continue

                        Vf = V_assembled[f]
                        Vt = V_assembled[t]

                        dv = Vf - Vt

                        dvm = abs(Vf) - abs(Vt)
                        dang = torch.angle(Vf) - torch.angle(Vt)

                        sf = bus_to_sg.get(f, -1)
                        st = bus_to_sg.get(t, -1)

                        location = "INTERNAL" if sf == st else "BOUNDARY"

                        # Electrical severity
                        ydv = y_mag * float(abs(dv))

                        stiff_diag.append((
                            ydv,
                            k,
                            f,
                            t,
                            y_mag,
                            float(abs(dv)),
                            float(dvm),
                            float(dang),
                            float(abs(Vf)),
                            float(abs(Vt)),
                            sf,
                            st,
                            location
                        ))

                    # Worst first
                    stiff_diag.sort(reverse=True, key=lambda x: x[0])

                    if stiff_diag:

                        print("\n  STIFF BRANCH VOLTAGE CHECK:")

                        for (
                            ydv,
                            k,
                            f,
                            t,
                            y_mag,
                            dv_abs,
                            dvm,
                            dang,
                            vmf,
                            vmt,
                            sf,
                            st,
                            location
                        ) in stiff_diag[:10]:

                            print(
                                f"    br {k:6d}: "
                                f"{f}(sg{sf})-{t}(sg{st}) "
                                f"{location:8s} "
                                f"|Y|={y_mag:9.1f} "
                                f"|dV|={dv_abs:.3e} "
                                f"|Y*dV|={ydv:.3e} "
                                f"dVm={dvm:.3e} "
                                f"dAng={dang:.3e} "
                                f"|Vf|={vmf:.6f} "
                                f"|Vt|={vmt:.6f}"
                            )

                    else:
                        print("\n  STIFF BRANCH VOLTAGE CHECK: none")
            V_rec=reconcile_boundary(V_det,subgrids,boundary,gd,damping=damp,tol=1e-4,max_iter=15)

            bad=(torch.isnan(V_rec.real)|torch.isinf(V_rec.real)|
                 torch.isnan(V_rec.imag)|torch.isinf(V_rec.imag))
            V_rec.real[bad]=V_det.real[bad]; V_rec.imag[bad]=V_det.imag[bad]
            V_assembled=torch.complex(V_rec.real.clone(),V_rec.imag.clone())
# ADD diagnostic 
    # ------------------------------------------------------------
    # STIFF BRANCH VOLTAGE COHERENCE DIAGNOSTIC
    # ------------------------------------------------------------
    with torch.no_grad():

        fr = gd["fbus"].detach().cpu().numpy()
        tr = gd["tbus"].detach().cpu().numpy()
        branch_arr = gd["branch_arr"]

        stiff_diag = []

        for k in range(len(branch_arr)):

            if int(branch_arr[k, 10]) == 0:
                continue

            f = int(fr[k])
            t = int(tr[k])

            r = max(float(branch_arr[k, 2]), 0.0)
            x = float(branch_arr[k, 3])

            z = complex(r, x)
            az = abs(z)

            if az < 1e-12:
                continue

            y_mag = 1.0 / az

            if y_mag <= 10000:
                continue

            dv = V_assembled[f] - V_assembled[t]

            stiff_diag.append((
                y_mag * float(abs(dv)),
                k,
                f,
                t,
                y_mag,
                float(abs(dv)),
                float(abs(V_assembled[f])),
                float(abs(V_assembled[t])),
                float(torch.angle(V_assembled[f]) -
                      torch.angle(V_assembled[t]))
            ))

        stiff_diag.sort(reverse=True)

        if stiff_diag:
            print("\n  STIFF BRANCH VOLTAGE CHECK:")

            for sev,k,f,t,y_mag,dv,vmf,vmt,dang in stiff_diag[:10]:

                print(
                    f"    br {k:6d}: {f}-{t} "
                    f"|Y|={y_mag:9.1f} "
                    f"|dV|={dv:.3e} "
                    f"|Y*dV|={sev:.3e} "
                    f"|Vf|={vmf:.6f} "
                    f"|Vt|={vmt:.6f} "
                    f"dAng={dang:.3e}"
                )
# finsih with diagnostics
    ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V_assembled)
    Sc=V_assembled*torch.conj(ic)
    Sg=torch.zeros(gd["n_bus"],dtype=torch.complex64)
    Sg.scatter_add_(0,gd["gen_bus"],torch.complex(Pg,Qg)); Ss=Sg-gd["s_demand"]
    return {"V":V_assembled,"Pg":Pg,"Qg":Qg,
            "deltaP":Ss.real-Sc.real,"deltaQ":Ss.imag-Sc.imag,
            "nr_mismatch_inf":float(np.mean(nr_inf_all)),
            "nr_mismatch_rms":float(np.mean(nr_rms_all))}

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--case",             required=True)
    pa.add_argument("--epochs",           type=int,   default=200)
    pa.add_argument("--nr-steps",         type=int,   default=3)
    pa.add_argument("--hidden",           type=int,   default=64)
    pa.add_argument("--lr",               type=float, default=1e-3)
    pa.add_argument("--log-every",        type=int,   default=20)
    pa.add_argument("--save-every",       type=int,   default=50)
    pa.add_argument("--save",             type=str,   default="cvgnn.pt")
    pa.add_argument("--resume",           type=str,   default=None)
    pa.add_argument("--load-best",        type=str,   default=None,
                    help="Load best checkpoint weights only (no optimizer/history). "
                         "Use to resume fine-tuning from best model state.")
    pa.add_argument("--partition-size",   type=int,   default=200)
    pa.add_argument("--partition-thresh", type=int,   default=500)
    pa.add_argument("--auto-partition",   action="store_true",default=False)
    pa.add_argument("--nr-start",         type=int,   default=0)
    args=pa.parse_args()

    print(f"\n{'='*65}")
    print(f"  CVNN-GNN + Diakoptics NR  [Option B: stiffness-aware METIS]")
    print(f"  case={args.case}  epochs={args.epochs}  hidden={args.hidden}")
    print(f"  lr={args.lr}  nr-steps={args.nr_steps}  nr-start={args.nr_start}")
    print(f"{'='*65}\n")

    gd=load_matpower_case(casepath=args.case)
    gd=build_ybus(gd); gd=add_normalised_ybus(gd); n=gd["n_bus"]

    from data.grid_partitioner     import partition_grid,assemble_global_voltages
    from data.adaptive_partitioner import auto_partition,analyse_grid

    USE_DIAKOPTICS=n>args.partition_thresh
    USE_SPARSE_NR=SPARSE_LIMIT<=n<=args.partition_thresh

    if USE_DIAKOPTICS:
        if n>10000 and args.nr_start==0:
            print(f"  NOTE: For {n:,}-bus grid recommend --nr-start 100-150")
        subgrids,boundary,rec_size,partition_stats=auto_partition(
            gd,user_partition_size=None if args.auto_partition else args.partition_size,
            verbose=True)
        sc=partition_stats.get('stiff_cuts',0)
        sr=partition_stats.get('boundary_stiff_ratio',0)
        path_name=(f"Diakoptics {'AUTO ' if args.auto_partition else ''}"
                   f"({len(subgrids)} sub-grids x ~{rec_size} buses, "
                   f"{boundary['n_bnd']} boundary buses, "
                   f"stiff_cuts={sc} [{100*sr:.1f}%])")
    else:
        subgrids=boundary=partition_stats=None
        path_name="Sparse NR" if USE_SPARSE_NR else "Dense NR"

    print(f"\n  Grid : {n:,} buses | Path : {path_name}")

    model=PowerGNN(n_buses=n,n_generators=gd["n_gen"],hidden_features=args.hidden)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6)
    start=0
    hist={k:[] for k in ["total","pf","volt","gen","therm","cost",
                           "nr_inf","nr_rms","ms","buses_ok","kcl_mw",
                           "part_nr_inf","bnd_mw","global_nr_inf","global_nr_rms",
                           "kcl_mw_max","kcl_p99"]}

    if args.resume and os.path.exists(args.resume):
        ck=torch.load(args.resume,map_location="cpu")
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
        start=ck["epoch"]+1; hist=ck.get("history",hist)
        for _ in range(start): sched.step()
        print(f"  Resumed from epoch {start}")
    elif args.load_best and os.path.exists(args.load_best):
        # Load best weights only — fresh optimizer and history
        # Use this to fine-tune from best checkpoint with new lr/nr-steps
        ck=torch.load(args.load_best,map_location="cpu")
        model.load_state_dict(ck["model_state"])
        src_ep=ck.get('epoch',0)
        print(f"  Loaded best weights from {args.load_best} (ep {src_ep}) "
              f"— fresh optimizer, training from ep 1")
        # Optionally load history for plotting continuity
        # hist = ck.get('history', hist)  # uncomment to keep history

    if not USE_DIAKOPTICS:
        print("  Warming up NR...",end="",flush=True)
        _s={"V":torch.ones(n,dtype=torch.complex64),"Pg":gd["pg"].clone(),"Qg":gd["qg"].clone()}
        _=differentiable_nr_step(_s,gd,damping=0.3)
        print(" done")

    w_pf=1.; w_v=0.5; w_gen=0.1; w_th=0.1; w_c=0.01
    loss_v=loss_th=torch.tensor(0.)

    best_gnr_so_far   = float('inf')
    best_ep_so_far    = 0
    cumul_ms_to_best  = 0.0   # total wall time to reach best solution
    cumul_ms_total    = 0.0   # total wall time all epochs
    train_start_time  = time.perf_counter()
    best_ep_so_far   = 0
    print(f"\n{'Ep':>5} {'Loss':>11} {'PF':>10} {'Volt':>8} "
          f"{'NR_inf':>10} {'NR_rms':>10} {'KCL':>9} {'ms':>7} {'OK':>9}")
    print("─"*95)

    for ep in range(start,args.epochs):
        t0=time.perf_counter(); model.train()
        ep_nr=max(0,ep-args.nr_start)
        if ep==args.nr_start and args.nr_start>0:
            for pg in opt.param_groups: pg['lr']=pg['lr']/5.0
            print(f"  LR→{opt.param_groups[0]['lr']:.2e} at NR start (ep {ep+1})")
        if ep<args.nr_start:
            nr=0; damp=0.; reg=1e-4; run_nr=False
        elif ep_nr<40:
            run_nr=(ep_nr%13==0); nr=1; damp=0.05+0.04*min(ep_nr,10); reg=1e-3
        elif ep_nr<80:
            run_nr=(ep_nr%11==0); nr=min(2,args.nr_steps); damp=0.4; reg=1e-4
        else:
            run_nr=(ep_nr%3==0); nr=args.nr_steps; damp=0.8; reg=1e-6
        if not run_nr: nr=0
        if ep>0 and ep%10==0:
            if loss_v.item()>1e-3:  w_v=min(w_v*1.05,2.)
            if loss_th.item()>1e-2: w_th=min(w_th*1.05,2.)
       
        gnn_out=model(gd)
        Pg=apply_slack_balance(gnn_out["Pg"],gd)
        Qg=gnn_out["Qg"]; V_gnn=gnn_out["V"]

        if nr==0:
            ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V_gnn)
            Sc=V_gnn*torch.conj(ic)
            Sg=torch.zeros(n,dtype=torch.complex64)
            Sg.scatter_add_(0,gd["gen_bus"],torch.complex(Pg,Qg)); Ss=Sg-gd["s_demand"]
            state={"V":V_gnn,"Pg":Pg,"Qg":Qg,
                   "deltaP":Ss.real-Sc.real,"deltaQ":Ss.imag-Sc.imag,
                   "nr_mismatch_inf":0.,"nr_mismatch_rms":0.}
        elif USE_DIAKOPTICS:
            # Fix 2: after ep_nr>80 GNN is stable — stop NR gradient
            # NR refines only, gradient flows through GNN not NR solve
            if ep_nr > 80:
                with torch.no_grad():
                    state=diakoptics_forward(V_gnn.detach(),Pg.detach(),
                                             Qg.detach(),subgrids,boundary,
                                             gd,nr,damp,reg)
                # Recompute loss on GNN output (not NR output) for grad
                state_for_loss=state; state_for_loss['V']=V_gnn
            else:
                state=diakoptics_forward(V_gnn,Pg,Qg,subgrids,boundary,gd,nr,damp,reg)
                state_for_loss=state
        elif USE_SPARSE_NR:
            if ep_nr > 80:
                with torch.no_grad():
                    state=differentiable_nr_step(
                        {"V":V_gnn.detach(),"Pg":Pg.detach(),"Qg":Qg.detach()},
                        gd,damping=damp,regularization=reg,nr_steps=nr)
                state_for_loss=state; state_for_loss['V']=V_gnn
            else:
                state=differentiable_nr_step({"V":V_gnn,"Pg":Pg,"Qg":Qg},
                                              gd,damping=damp,regularization=reg,nr_steps=nr)
                state_for_loss=state
        else:
            state={"V":V_gnn,"Pg":Pg,"Qg":Qg}
            for _ in range(nr): state=_dense_nr_step(state,gd,damping=damp,regularization=reg)

        # Use state_for_loss for grad (GNN output when NR grad stopped)
        if 'state_for_loss' not in dir(): state_for_loss=state
        
        V=state["V"]; Vm=torch.abs(V)
        loss_pf =compute_power_flow_loss(state_for_loss["deltaP"],state_for_loss["deltaQ"])
        loss_v  =compute_voltage_loss(Vm)
        loss_gen=compute_generator_limit_loss(Pg,Qg,gd)
        sf,st   =compute_branch_flows(V,gd)
        loss_th =compute_thermal_violations(sf,st,gd["rateA"])
        loss_c  =compute_generation_cost(Pg,gd)
        total   =w_pf*loss_pf+w_v*loss_v+w_gen*loss_gen+w_th*loss_th+w_c*loss_c

        if not torch.isnan(total):
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
            opt.step()
        else:
            if ep%args.log_every==0: print(f"{ep+1:5d}  NaN — skipping")
        sched.step()

        # Oscillation detection
        if len(hist["global_nr_rms"])>=3 and nr>0:
            if state["nr_mismatch_rms"]>hist["global_nr_rms"][-3]*2.0:
                damp=max(damp*0.7,0.15)
                if ep%args.log_every==0:
                    print(f"  ⚠ NR oscillation — damping→{damp:.3f}")

        with torch.no_grad():
            ip=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V)
            kcl=float(torch.abs(V*torch.conj(ip)-gd["s_demand"]).mean())*gd["baseMVA"]
            diag=compute_all_diagnostics(state,gd,
                                          boundary=boundary if USE_DIAKOPTICS else None)

        ok=int(((Vm>=0.95)&(Vm<=1.05)).sum()); ms=(time.perf_counter()-t0)*1000
        nr_label=f"nr={nr}" if nr>0 else "nr=0(GNN)"

        for k,v in [("total",total),("pf",loss_pf),("volt",loss_v),("gen",loss_gen),
                    ("therm",loss_th),("cost",loss_c),
                    ("nr_inf",state["nr_mismatch_inf"]),("nr_rms",state["nr_mismatch_rms"]),
                    ("ms",ms),("buses_ok",ok),("kcl_mw",kcl),
                    ("part_nr_inf",diag["part_nr_inf"]),("bnd_mw",diag["bnd_mw"]),
                    ("global_nr_inf",diag["global_nr_inf"]),
                    ("global_nr_rms",diag["global_nr_rms"]),
                    ("kcl_mw_max",diag["kcl_mw_max"]),("kcl_p99",diag["kcl_p99"])]:
            hist[k].append(v.item() if torch.is_tensor(v) else v)

        if ep%args.log_every==0. or run_nr:
            print(f"{ep+1:5d} {total.item():11.4e} {loss_pf.item():10.3e} "
                  f"{loss_v.item():8.5f} {state['nr_mismatch_inf']:10.4e} "
                  f"{state['nr_mismatch_rms']:10.4e} {kcl:9.2f} {ms:7.1f} "
                  f"{ok:4d}/{n}  [{nr_label}]")
            if USE_DIAKOPTICS and nr>0:
                print(f"       partNR={diag['part_nr_inf']:.2e} "
                      f"part_nr_rms={diag['part_nr_rms']:.2e} "
                      f"bnd_inf={diag['bnd_inf']:.2e} "
                      f"global_nr_rms={diag['global_nr_rms']:.2e} "
                      f"worst_bus={diag['worst_bus']} worst_type={diag['worst_type']} "
                      f"bndMW={diag['bnd_mw']:.1f} "
                      f"globNR={diag['global_nr_inf']:.2e} "
                      f"KCL_max={diag['kcl_mw_max']:.1f}MW "
                      f"kcl_p99={diag['kcl_p99']:.1f}MW "
                      f"kcl_p999={diag['kcl_p999']:.1f}MW "
                      f"stiff={diag['stiff']:.0f} "
                      f"n_lowVm7={diag['n_lowVm7']:.0f} "
                      f"n_highVm1dot3={diag['n_highVm1dot3']:.0f} "
                      f"nan_buses={diag['nan_buses']:.0f}")
                if diag['part_nr_inf']>500 and diag['bnd_mw']<100:
                    print(f"       ⚠ Sub-grid NR struggling — try smaller --partition-size")
                if diag['global_nr_inf']>5*diag['part_nr_inf']:
                    print(f"       ⚠ globNR>>partNR — angle discontinuity at boundaries")
                if diag['kcl_mw_max']>100000:
                    print(f"       ⚠ KCL_max overflow — check stiff_cuts in partition_stats")

        # Save best checkpoint whenever global_nr_rms improves
        if diag['global_nr_rms'] < best_gnr_so_far and nr > 0:
            best_gnr_so_far  = diag['global_nr_rms']
            best_ep_so_far   = ep+1
            cumul_ms_to_best = cumul_ms_total  # wall time to reach this best
            torch.save({"epoch":ep,"model_state":model.state_dict(),
                        "optimizer_state":opt.state_dict(),"history":hist},
                       args.save+'.best')
            if ep%args.log_every==0:
                print(f"  ★ Best checkpoint → {args.save}.best  "
                      f"global_nr_rms={best_gnr_so_far:.4e} (ep {best_ep_so_far})")
        if (ep+1)%args.save_every==0:
            torch.save({"epoch":ep,"model_state":model.state_dict(),
                        "optimizer_state":opt.state_dict(),"history":hist},args.save)
            print(f"  ✓ Checkpoint → {args.save}  (ep {ep+1})")

    torch.save({"epoch":args.epochs-1,"model_state":model.state_dict(),
                "optimizer_state":opt.state_dict(),"history":hist},args.save)

    print(f"\n{'='*65}")
    print(f"  FINAL — {n:,} buses  [{path_name}]")
    # All metrics from same epoch as best global_nr_rms
    best_ep       = int(np.argmin(hist['global_nr_rms']))
    best_gnr      = hist['global_nr_rms'][best_ep]
    best_buses_ok = hist['buses_ok'][best_ep]
    best_kcl_mw   = hist['kcl_mw'][best_ep]
    best_kcl_p99  = hist['kcl_p99'][best_ep]
    best_bnd_mw   = hist['bnd_mw'][best_ep]
    best_ep_1idx  = best_ep + 1
    print(f"  PRIMARY: global_nr_rms  = {hist['global_nr_rms'][-1]:.4e} (last epoch)")
    print(f"  Best checkpoint        : {args.save}.best  (ep {best_ep_so_far})")
    print(f"  PRIMARY: global_nr_rms  = {best_gnr:.4e} (best, ep {best_ep_1idx})")
    print(f"  Buses OK               : {best_buses_ok}/{n}  ({100*best_buses_ok/n:.1f}%)")
    print(f"  KCL mean (MW)          : {best_kcl_mw:.2f}")
    print(f"  KCL p99  (MW)          : {best_kcl_p99:.2f}")
    print(f"  bnd_mw                 : {best_bnd_mw:.2f}")
    print(f"  stiff_cuts             : {partition_stats.get('stiff_cuts',0) if partition_stats else 'N/A'}")
    print(f"  Mean infer ms          : {np.mean(hist['ms']):.1f}")
    print(f"  Total wall time        : {cumul_ms_total/1000:.1f}s  "
          f"({cumul_ms_total/1000/60:.1f} min)")
    print(f"  Time to best solution  : {cumul_ms_to_best/1000:.1f}s  "
          f"(ep {best_ep_so_far}, {cumul_ms_to_best/1000/60:.2f} min)")
    print(f"  Checkpoint             : {args.save}")

    ep_ax=range(1,len(hist["total"])+1)
    fig,axes=plt.subplots(2,3,figsize=(18,9))
    for ax,k,lb,c in [(axes[0][0],"total","Total Loss","blue"),
                       (axes[0][1],"global_nr_rms","Global NR RMS (PRIMARY)","red"),
                       (axes[0][2],"kcl_p99","KCL p99 (MW)","orange"),
                       (axes[1][0],"bnd_mw","Boundary MW","purple"),
                       (axes[1][1],"buses_ok","Buses OK","green"),
                       (axes[1][2],"kcl_mw_max","KCL max MW","crimson")]:
        ax.plot(ep_ax,hist[k],color=c,lw=1.5)
        ax.set_title(lb); ax.grid(True,alpha=0.35); ax.set_xlabel("Epoch")
        if k not in ["buses_ok"]: ax.set_yscale("log")
    plt.suptitle(f"Option B stiffness-aware METIS — {n:,} buses  stiff_cuts={partition_stats.get('stiff_cuts',0) if partition_stats else 0}",fontsize=12)
    plt.tight_layout()
    pname=f"training_{n}bus.png"
    plt.savefig(pname,dpi=150); plt.close()
    print(f"  Plot → {pname}")
