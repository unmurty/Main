"""diagnostics.py — 4 convergence metrics. PRIMARY: global_nr_rms"""
import torch, numpy as np
from utils.sparse_complex import sparse_complex_mm
disablePrint=True
def compute_all_diagnostics(state, gd, boundary=None, baseMVA=None):
    disablePrint=True
    V=state["V"].detach(); Pg=state["Pg"].detach(); Qg=state["Qg"].detach()
    n=gd["n_bus"]; B=baseMVA or gd["baseMVA"]
    part_nr_inf=state.get("nr_mismatch_inf",0.)
    part_nr_rms=state.get("nr_mismatch_rms",0.)
    bnd_inf=0.; bnd_mw=0.
    if boundary is not None and boundary.get("n_bnd",0)>0:
        bnd_buses=torch.tensor(boundary["bnd_bus_rows"],dtype=torch.long)
        S_spec_b=boundary["S_spec_bnd"]
        if len(S_spec_b)>0 and len(bnd_buses)>0:
            ic_bnd=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V)
            Sc_bnd=V*torch.conj(ic_bnd)
            dP_bnd=(S_spec_b.real-Sc_bnd[bnd_buses].real).abs()
            dQ_bnd=(S_spec_b.imag-Sc_bnd[bnd_buses].imag).abs()
            bnd_inf=float(torch.cat([dP_bnd,dQ_bnd]).max())
            bnd_mw =float(torch.cat([dP_bnd,dQ_bnd]).mean())*B
    with torch.no_grad():
        Sg=torch.zeros(n,dtype=torch.complex64)
        Sg.scatter_add_(0,gd["gen_bus"],torch.complex(Pg,Qg))
        Ss=Sg-gd["s_demand"]
        ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V)
        Sc=V*torch.conj(ic)
        if not disablePrint:
            print(
                f"baseMVA={B} "
                f"|Pg|max={Pg.abs().max().item():.6e} "
                f"|Qg|max={Qg.abs().max().item():.6e} "
                f"|Sd|max={gd['s_demand'].abs().max().item():.6e} "
                f"|Sc|max={Sc.abs().max().item():.6e}"
            )
        
        dP=Ss.real-Sc.real; dQ=Ss.imag-Sc.imag ; dS=Sc-Ss
        if not disablePrint:
            print(
                f"dP max={dP.abs().max().item():.6e} "
                f"dQ max={dQ.abs().max().item():.6e} "
                f"dS max={torch.abs(Sc-Ss).max().item():.6e}"
            )
        pq=(gd["bus_type"]==1).nonzero(as_tuple=True)[0]
        pv=(gd["bus_type"]==2).nonzero(as_tuple=True)[0]
        aa=torch.cat([pv,pq]); mis=torch.cat([dP[aa],dQ[pq]])
        idx=int(torch.argmax(torch.abs(mis)))
        worst_bus=int(aa[idx]) if idx<len(aa) else int(pq[idx-len(aa)])
        worst_type="ΔP" if idx<len(aa) else "ΔQ"
        global_nr_inf=float(mis.abs().max())
        global_nr_rms=float(torch.sqrt((mis**2).mean()))
        kcl_abs=torch.nan_to_num(torch.abs(Sc-Ss),nan=0.,posinf=0.,neginf=0.)
        kcl_flat=kcl_abs.flatten()
        kcl_mw_mean=float(kcl_flat.mean())*B
        kcl_mw_max =float(kcl_flat.max())*B
        kcl_p99    =float(torch.quantile(kcl_flat,0.99))*B
        kcl_p999   =float(torch.quantile(kcl_flat,0.999))*B
        vm=torch.abs(V)
        n_low =int((vm<=0.701).sum()); n_high=int((vm>=1.299).sum())
        nan_buses=int((torch.isnan(V.real)|torch.isinf(V.real)|
                       torch.isnan(V.imag)|torch.isinf(V.imag)).sum())
        if nan_buses>0: print(f"  ⚠ {nan_buses} NaN/Inf voltages")
        # Vectorised CSR diagonal extraction — O(nnz) numpy, no Python loops
        # For each row i, find the entry where col_indices[pos] == i
        Yr_sp = gd["Y_real_sp"]; Yi_sp = gd["Y_imag_sp"]
        if Yr_sp.layout == torch.sparse_csr:
            import numpy as np
            crow  = Yr_sp.crow_indices().numpy()   # [n+1]
            cols  = Yr_sp.col_indices().numpy()    # [nnz]
            yr_v  = Yr_sp.values().numpy()         # [nnz]
            yi_v  = Yi_sp.values().numpy()         # [nnz]
            # For each row, offset of diagonal = crow[i] + searchsorted(cols[crow[i]:crow[i+1]], i)
            row_idx  = np.arange(n, dtype=np.int64)
            starts   = crow[:-1]; ends = crow[1:]
            diag_r   = np.zeros(n, dtype=np.float32)
            diag_i   = np.zeros(n, dtype=np.float32)
            # Vectorised: for each row find diagonal position using numpy
            for i in range(n):
                s, e = int(starts[i]), int(ends[i])
                if s >= e: continue
                local = cols[s:e]
                pos = np.searchsorted(local, i)
                if pos < len(local) and local[pos] == i:
                    diag_r[i] = yr_v[s + pos]
                    diag_i[i] = yi_v[s + pos]
            y_diag_abs = torch.tensor(np.sqrt(diag_r**2 + diag_i**2))
        else:
            # Dense fallback (small grids only)
            Gd = gd.get("G"); Bd = gd.get("B")
            if Gd is not None:
                y_diag_abs = torch.sqrt(Gd.diagonal()**2 + Bd.diagonal()**2)
            else:
                y_diag_abs = torch.zeros(n)
        stiff = int((y_diag_abs > 10000).sum())
        if stiff > 0:
            stiff_ids = (y_diag_abs > 10000).nonzero(as_tuple=True)[0].tolist()[:5]
            if not disablePrint:
                print(f"  ⚠ stiff={stiff} buses |Y_diag|>10000 pu: {stiff_ids}...")
    return {
        "part_nr_inf":part_nr_inf,"part_nr_rms":part_nr_rms,
        "bnd_inf":bnd_inf,"bnd_mw":bnd_mw,
        "global_nr_inf":global_nr_inf,"global_nr_rms":global_nr_rms,
        "worst_bus":worst_bus,"worst_type":worst_type,
        "kcl_mw_mean":kcl_mw_mean,"kcl_mw_max":kcl_mw_max,
        "kcl_p99":kcl_p99,"kcl_p999":kcl_p999,
        "stiff":stiff,"n_lowVm7":n_low,"n_highVm1dot3":n_high,"nan_buses":nan_buses,
    }
def debug_bus_branches(
    bus_idx,
    V,
    gd,
    Sc,
    baseMVA,
    top_n=20,
):
    i = int(bus_idx)

    branch = gd["branch_arr"]

    fbus = gd["fbus"].detach().cpu().numpy()
    tbus = gd["tbus"].detach().cpu().numpy()

    connected = np.where(
        (fbus == i) | (tbus == i)
    )[0]

    print(f"\n=== BRANCH BREAKDOWN bus={i} ===")

    Vi = V[i]

    print(
        f"Vm={float(torch.abs(Vi)):.6f} "
        f"Va={float(torch.angle(Vi)):.6f} rad"
    )
    if baseMVA is None:
        baseMVA = float(gd.get("baseMVA", 100.0))
    print(
        f"Pcalc={float(Sc.real[i]) * baseMVA:.2f} MW "
        f"Qcalc={float(Sc.imag[i]) * baseMVA:.2f} MVAr"
    )

    print(f"connected branches={len(connected)}")

    rows = []

    for k in connected:
        k = int(k)

        # Branch status
        if float(branch[k, 10]) == 0:
            continue

        f = int(fbus[k])
        t = int(tbus[k])

        r = max(float(branch[k, 2]), 0.0)
        x = float(branch[k, 3])
        b = float(branch[k, 4])

        tap = float(branch[k, 8]) if float(branch[k, 8]) != 0 else 1.0
        shift = np.deg2rad(float(branch[k, 9]))

        z = complex(r, x)

        if abs(z) < 1e-12:
            z = 1e-12 + 0j

        y = 1.0 / z
        ys = 1j * b / 2.0

        tc = tap * np.exp(1j * shift)

        # EXACTLY same branch model as build_ybus
        Yff = (y + ys) / (tc * np.conj(tc))
        Ytt = y + ys
        Yft = -y / np.conj(tc)
        Ytf = -y / tc

        Vf = V[f]
        Vt = V[t]

        # Branch currents
        Ift = (
            torch.as_tensor(Yff, dtype=Vf.dtype, device=Vf.device) * Vf
            + torch.as_tensor(Yft, dtype=Vf.dtype, device=Vf.device) * Vt
        )

        Itf = (
            torch.as_tensor(Ytf, dtype=Vt.dtype, device=Vt.device) * Vf
            + torch.as_tensor(Ytt, dtype=Vt.dtype, device=Vt.device) * Vt
        )

        # Branch complex powers
        Sft = Vf * torch.conj(Ift)
        Stf = Vt * torch.conj(Itf)

        dangle = float(
            torch.angle(Vf) - torch.angle(Vt)
        )

        if f == i:
            P = float(Sft.real) * baseMVA
            Q = float(Sft.imag) * baseMVA
        else:
            P = float(Stf.real) * baseMVA
            Q = float(Stf.imag) * baseMVA

        rows.append(
            (
                abs(P) + abs(Q),
                k,
                f,
                t,
                abs(y),
                dangle,
                P,
                Q,
                float(torch.abs(Vf)),
                float(torch.abs(Vt)),
            )
        )

    # Largest branch power contributions first
    rows.sort(reverse=True)

    print("\nTop incident branches by |P|+|Q|:")

    for (
        _,
        k,
        f,
        t,
        ymag,
        dangle,
        P,
        Q,
        Vmf,
        Vmt,
    ) in rows[:top_n]:

        print(
            f"branch={k:6d} "
            f"{f:6d}->{t:<6d} "
            f"|Y|={ymag:10.2f} "
            f"dAng={dangle: .6f} "
            f"|Vf|={Vmf:.6f} "
            f"|Vt|={Vmt:.6f} "
            f"P={P:10.2f} MW "
            f"Q={Q:10.2f} MVAr"
        )
def compute_best_diagnostics(state, gd, boundary=None, baseMVA=None):
    V = state["V"].detach()
    Pg = state["Pg"].detach()
    Qg = state["Qg"].detach()

    n = gd["n_bus"]
    B = baseMVA or gd["baseMVA"]

    part_nr_inf = state.get("nr_mismatch_inf", 0.)
    part_nr_rms = state.get("nr_mismatch_rms", 0.)

    bnd_inf = 0.
    bnd_mw = 0.

    # ------------------------------------------------------------
    # Boundary diagnostic
    # ------------------------------------------------------------
    if boundary is not None and boundary.get("n_bnd", 0) > 0:
        bnd_buses = torch.tensor(
            boundary["bnd_bus_rows"],
            dtype=torch.long
        )

        S_spec_b = boundary["S_spec_bnd"]

        if len(S_spec_b) > 0 and len(bnd_buses) > 0:
            ic_bnd = sparse_complex_mm(
                gd["Y_real_sp"],
                gd["Y_imag_sp"],
                V
            )

            Sc_bnd = V * torch.conj(ic_bnd)

            dP_bnd = (
                S_spec_b.real -
                Sc_bnd[bnd_buses].real
            ).abs()

            dQ_bnd = (
                S_spec_b.imag -
                Sc_bnd[bnd_buses].imag
            ).abs()

            bnd_inf = float(
                torch.cat([dP_bnd, dQ_bnd]).max()
            )

            bnd_mw = (
                float(torch.cat([dP_bnd, dQ_bnd]).mean())
                * B
            )

    # ------------------------------------------------------------
    # Global diagnostics
    # ------------------------------------------------------------
    with torch.no_grad():

        # --------------------------------------------------------
        # Specified generation
        # --------------------------------------------------------
        Sg = torch.zeros(
            n,
            dtype=torch.complex64
        )

        Sg.scatter_add_(
            0,
            gd["gen_bus"],
            torch.complex(Pg, Qg)
        )

        Ss = Sg - gd["s_demand"]

        # --------------------------------------------------------
        # Network calculated injection
        # --------------------------------------------------------
        ic = sparse_complex_mm(
            gd["Y_real_sp"],
            gd["Y_imag_sp"],
            V
        )

        Sc = V * torch.conj(ic)

        # --------------------------------------------------------
        # Unit / scaling diagnostic
        # --------------------------------------------------------
        
        if not disablePrint:
            print(
                f"baseMVA={B} "
                f"|Pg|max={Pg.abs().max().item():.6e} "
                f"|Qg|max={Qg.abs().max().item():.6e} "
                f"|Sd|max={gd['s_demand'].abs().max().item():.6e} "
                f"|Sc|max={Sc.abs().max().item():.6e}"
        )

        # --------------------------------------------------------
        # Full-bus power mismatch
        # --------------------------------------------------------
        dP = Ss.real - Sc.real
        dQ = Ss.imag - Sc.imag
        dS = Sc - Ss

        if not disablePrint:
            print(
                f"dP max={dP.abs().max().item():.6e} "
                f"dQ max={dQ.abs().max().item():.6e} "
                f"dS max={dS.abs().max().item():.6e}"
            )

        # --------------------------------------------------------
        # Bus types
        # --------------------------------------------------------
        pq = (gd["bus_type"] == 1).nonzero(
            as_tuple=True
        )[0]

        pv = (gd["bus_type"] == 2).nonzero(
            as_tuple=True
        )[0]

        ref = (gd["bus_type"] == 3).nonzero(
            as_tuple=True
        )[0]

        # --------------------------------------------------------
        # Stiff-bus diagnostic
        # --------------------------------------------------------
        Yr_sp = gd["Y_real_sp"]
        Yi_sp = gd["Y_imag_sp"]


        if Yr_sp.layout == torch.sparse_csr:

            import numpy as np

            crow = (
                Yr_sp.crow_indices()
                .numpy()
            )

            cols = (
                Yr_sp.col_indices()
                .numpy()
            )

            yr_v = (
                Yr_sp.values()
                .numpy()
            )

            yi_v = (
                Yi_sp.values()
                .numpy()
            )

            row_idx = np.arange(
                n,
                dtype=np.int64
            )

            starts = crow[:-1]
            ends = crow[1:]

            diag_r = np.zeros(
                n,
                dtype=np.float32
            )

            diag_i = np.zeros(
                n,
                dtype=np.float32
            )

            for i in range(n):
                s = int(starts[i])
                e = int(ends[i])

                if s >= e:
                    continue

                local = cols[s:e]

                pos = np.searchsorted(
                    local,
                    i
                )

                if (
                    pos < len(local)
                    and local[pos] == i
                ):
                    diag_r[i] = yr_v[s + pos]
                    diag_i[i] = yi_v[s + pos]

            y_diag_abs = torch.tensor(
                np.sqrt(
                    diag_r ** 2 +
                    diag_i ** 2
                )
            )

        else:
            # Dense fallback for small grids
            Gd = gd.get("G")
            Bd = gd.get("B")

            if Gd is not None:
                y_diag_abs = torch.sqrt(
                    Gd.diagonal() ** 2 +
                    Bd.diagonal() ** 2
                )
            else:
                y_diag_abs = torch.zeros(n)

        stiff = int(
            (y_diag_abs > 10000).sum()
        )
        stiff_mask = (y_diag_abs > 10000)
        if stiff > 0:
            stiff_ids = (
                (y_diag_abs > 10000)
                .nonzero(as_tuple=True)[0]
                .tolist()[:5]
            )
            if not disablePrint:
                print(
                    f"  ⚠ stiff={stiff} buses "
                    f"|Y_diag|>10000 pu: "
                    f"{stiff_ids}..."
                )
            
        gen_mask = torch.zeros(n, dtype=torch.bool)
        gen_mask[gd["gen_bus"]] = True

        pq = (gd["bus_type"] == 1).nonzero(as_tuple=True)[0]
        pv = (gd["bus_type"] == 2).nonzero(as_tuple=True)[0]
        ref = (gd["bus_type"] == 3).nonzero(as_tuple=True)[0]

        
        # --------------------------------------------------------
        # MATPOWER-style NR equations
        #
        # ΔP : PV + PQ
        # ΔQ : PQ
        # exclude stiff buses from NR metrics
        # --------------------------------------------------------
        aa_all = torch.cat([pv, pq])
        stiff_mask = (y_diag_abs > 10000)
        
        
        
        aa = aa_all[~stiff_mask[aa_all]]
        pq_nr = pq[~stiff_mask[pq]]

        dP_nr = dP[aa]
        dQ_nr = dQ[pq_nr]

        mis = torch.cat([
            dP_nr,
            dQ_nr
        ])

        # --------------------------------------------------------
        # Combined MATPOWER-style mismatch
        # --------------------------------------------------------
        if mis.numel() > 0:
            global_nr_inf = float(mis.abs().max())
            global_nr_rms = float(torch.sqrt((mis ** 2).mean()))
        else:
            global_nr_inf = 0.0
            global_nr_rms = 0.0

        # --------------------------------------------------------
        # Separate P/Q MATPOWER-style metrics
        # --------------------------------------------------------

        if dP_nr.numel() > 0:
            nr_p_inf = float(dP_nr.abs().max())
            nr_p_rms = float(
                torch.sqrt(
                    (dP_nr ** 2).mean()
                )
            )
        else:
            nr_p_inf = 0.0
            nr_p_rms = 0.0

        if dQ_nr.numel() > 0:
            nr_q_inf = float(dQ_nr.abs().max())
            nr_q_rms = float(
                torch.sqrt(
                    (dQ_nr ** 2).mean()
                )
            )
        else:
            nr_q_inf = 0.0
            nr_q_rms = 0.0

        # --------------------------------------------------------
        # Convert MATPOWER-style residuals to MW / MVAr
        # --------------------------------------------------------
        nr_p_inf_mw = nr_p_inf * B
        nr_p_rms_mw = nr_p_rms * B

        nr_q_inf_mvar = nr_q_inf * B
        nr_q_rms_mvar = nr_q_rms * B

        global_nr_inf_mw = global_nr_inf * B
        global_nr_rms_mw = global_nr_rms * B

        # --------------------------------------------------------
        # Worst NR equation
        # --------------------------------------------------------
        idx = int(torch.argmax(mis.abs()))

        nP = len(aa)

        if idx < nP:
            worst_bus = int(aa[idx])
            worst_type = "ΔP"
        else:
            qidx = idx - nP
            worst_bus = int(pq_nr[qidx])
            worst_type = "ΔQ"


        # --------------------------------------------------------
        # Full-bus KCL / complex power balance
        #
        # This is intentionally retained as an independent
        # diagnostic and is NOT the MATPOWER NR metric.
        # --------------------------------------------------------
        kcl_abs = torch.nan_to_num(
            torch.abs(dS),
            nan=0.,
            posinf=0.,
            neginf=0.
        )

        kcl_flat = kcl_abs.flatten()

        kcl_mw_mean = (
            float(kcl_flat.mean()) * B
        )

        kcl_mw_max = (
            float(kcl_flat.max()) * B
        )

        kcl_p99 = (
            float(torch.quantile(kcl_flat, 0.99))
            * B
        )

        kcl_p999 = (
            float(torch.quantile(kcl_flat, 0.999))
            * B
        )
        # --------------------------------------------------------
        # KCL excluding stiff buses
        # --------------------------------------------------------
        stiff_mask = (y_diag_abs > 10000)

        non_stiff_mask = ~stiff_mask

        kcl_nonstiff = kcl_abs[non_stiff_mask]

        if kcl_nonstiff.numel() > 0:
            kcl_nonstiff_mw_mean = (
                float(kcl_nonstiff.mean()) * B
            )

            kcl_nonstiff_mw_max = (
                float(kcl_nonstiff.max()) * B
            )

            kcl_nonstiff_p99 = (
                float(torch.quantile(kcl_nonstiff, 0.99)) * B
            )

            kcl_nonstiff_p999 = (
                float(torch.quantile(kcl_nonstiff, 0.999)) * B
            )
        else:
            kcl_nonstiff_mw_mean = 0.0
            kcl_nonstiff_mw_max = 0.0
            kcl_nonstiff_p99 = 0.0
            kcl_nonstiff_p999 = 0.0
        # --------------------------------------------------------
        # Voltage diagnostics
        # --------------------------------------------------------
        vm = torch.abs(V)

        n_low = int(
            (vm <= 0.701).sum()
        )

        n_high = int(
            (vm >= 1.299).sum()
        )

        nan_buses = int(
            (
                torch.isnan(V.real) |
                torch.isinf(V.real) |
                torch.isnan(V.imag) |
                torch.isinf(V.imag)
            ).sum()
        )

        if nan_buses > 0:
            print(
                f"  ⚠ {nan_buses} NaN/Inf voltages"
            )
    # --------------------------------------------------------
    # Worst KCL buses
    # --------------------------------------------------------
    KCL_TOP_N = 20

    top_n = min(KCL_TOP_N, n)

    kcl_values, kcl_indices = torch.topk(
        kcl_abs,
        k=top_n,
        largest=True
    )
    bt_names = {
    1: "PQ",
    2: "PV",
    3: "Slack",
    }
       
    print("=== WORST KCL BUSES ===")

    gen_bus_np = gd["gen_bus"].detach().cpu().numpy().astype(np.int32)
    # Load is already bus-indexed
    Pd_bus = gd["pd"]
    Qd_bus = gd["qd"]

    for rank, (kcl_val, bus_idx) in enumerate(
    zip(kcl_values, kcl_indices),
    start=1
    ):
       
        i = int(bus_idx)
        bus_type = int(gd["bus_type"][i])
        is_gen = bool(gen_mask[i])
        is_stiff = bool(stiff_mask[i])

        print(
        f"{rank:2d}. "
        f"bus={i:5d} "
        f"type={bt_names.get(bus_type, '?'):5s} "
        f"gen={str(is_gen):5s} "
        f"stiff={str(is_stiff):5s} "
        f"Vm={float(vm[i]):.6f} "
        f"|Ydiag|={float(y_diag_abs[i]):9.2f} "
        f"|dS|={float(kcl_val)*B:10.2f} MW "
        f"dP={float(dP[i])*B:10.2f} MW "
        f"dQ={float(dQ[i])*B:10.2f} MVAr"
        )
        Pd = Pd_bus[i]
        Qd = Qd_bus[i]

        Pspec = -Pd
        Qspec = -Qd

        print(
            f"      "
            f"Pd={float(Pd)*B:10.2f} MW "
            f"Qd={float(Qd)*B:10.2f} MVAr "
            f"Pspec={float(Pspec)*B:10.2f} MW "
            f"Qspec={float(Qspec)*B:10.2f} MVAr "
            f"Pcalc={float(Sc.real[i])*B:10.2f} MW "
            f"Qcalc={float(Sc.imag[i])*B:10.2f} MVAr"
        )
        if not is_gen and rank <= 10:
            debug_bus_branches(bus_idx, V, gd, Sc, baseMVA, top_n=20)
        xxxx=1
        
        if is_gen:
            gen_idx = np.where(gen_bus_np == i)[0]

            if len(gen_idx) > 0:
                Pg_bus = Pg[gen_idx].sum()
                Qg_bus = Qg[gen_idx].sum()
                Pd = Pd_bus[i]
                Qd = Qd_bus[i]

                Pspec = Pg_bus - Pd
                Qspec = Qg_bus - Qd

            print(
            f"      Pg={float(Pg_bus)*B:10.2f} MW "
            f"Qg={float(Qg_bus)*B:10.2f} MVAr "
            f"Pd={float(Pd)*B:10.2f} MW "
            f"Qd={float(Qd)*B:10.2f} MVAr"
        )

            print(
                f"      Pspec={float(Pspec)*B:10.2f} MW "
                f"Qspec={float(Qspec)*B:10.2f} MVAr "
                f"Pcalc={float(Sc.real[i])*B:10.2f} MW "
                f"Qcalc={float(Sc.imag[i])*B:10.2f} MVAr"
            )

            print(
                f"      generator records={gen_idx.tolist()}"
            )
            print(
                f"ngen={len(gen_idx)}"
            )





    # ------------------------------------------------------------
    # Return diagnostics
    # ------------------------------------------------------------
    return {

        # Partitioned NR
        "part_nr_inf": part_nr_inf,
        "part_nr_rms": part_nr_rms,

        # Boundary
        "bnd_inf": bnd_inf,
        "bnd_mw": bnd_mw,

        # --------------------------------------------------------
        # MATPOWER-compatible NR mismatch
        # --------------------------------------------------------
        "global_nr_inf": global_nr_inf,
        "global_nr_rms": global_nr_rms,

        "global_nr_inf_mw": global_nr_inf_mw,
        "global_nr_rms_mw": global_nr_rms_mw,

        # P equation: PV + PQ
        "nr_p_inf": nr_p_inf,
        "nr_p_rms": nr_p_rms,

        "nr_p_inf_mw": nr_p_inf_mw,
        "nr_p_rms_mw": nr_p_rms_mw,

        # Q equation: PQ only
        "nr_q_inf": nr_q_inf,
        "nr_q_rms": nr_q_rms,

        "nr_q_inf_mvar": nr_q_inf_mvar,
        "nr_q_rms_mvar": nr_q_rms_mvar,

        # Worst equation
        "worst_bus": worst_bus,
        "worst_type": worst_type,

        # --------------------------------------------------------
        # Full-bus KCL diagnostic
        # --------------------------------------------------------
        "kcl_mw_mean": kcl_mw_mean,
        "kcl_mw_max": kcl_mw_max,
        "kcl_p99": kcl_p99,
        "kcl_p999": kcl_p999,
        # --------------------------------------------------------
        # Non-stiff-bus KCL diagnostic
        # --------------------------------------------------------
        "kcl_nonstiff_mw_mean": kcl_nonstiff_mw_mean,
        "kcl_nonstiff_mw_max": kcl_nonstiff_mw_max,
        "kcl_nonstiff_p99": kcl_nonstiff_p99,
        "kcl_nonstiff_p999": kcl_nonstiff_p999,

        # --------------------------------------------------------
        # Voltage / numerical diagnostics
        # --------------------------------------------------------
        "stiff": stiff,
        "n_lowVm7": n_low,
        "n_highVm1dot3": n_high,
        "nan_buses": nan_buses,
    }
def compute_kcl_breakdown(state, gd, baseMVA=None, stiff_threshold=10000, top_n=20):
    """
    KCL breakdown:
    1. All buses vs non-stiff buses
    2. Top-N worst KCL buses with bus type, voltage, dP, dQ
    3. Stiff bus mask from CSR diagonal
    """
    import torch, numpy as np
    from utils.sparse_complex import sparse_complex_mm

    V  = state["V"].detach()
    Pg = state["Pg"].detach()
    Qg = state["Qg"].detach()
    n  = gd["n_bus"]
    B  = baseMVA or gd["baseMVA"]

    # ── Network injection ─────────────────────────────────────────
    ic = sparse_complex_mm(gd["Y_real_sp"], gd["Y_imag_sp"], V)
    Sc = V * torch.conj(ic)
    Sg = torch.zeros(n, dtype=torch.complex64)
    Sg.scatter_add_(0, gd["gen_bus"], torch.complex(Pg, Qg))
    Ss = Sg - gd["s_demand"]
    dS = torch.abs(Sc - Ss)
    dP = (Ss.real - Sc.real)
    dQ = (Ss.imag - Sc.imag)

    # ── Stiff bus mask from CSR diagonal ─────────────────────────
    Yr_sp = gd["Y_real_sp"]; Yi_sp = gd["Y_imag_sp"]
    stiff_mask = torch.zeros(n, dtype=torch.bool)
    if Yr_sp.layout == torch.sparse_csr:
        crow = Yr_sp.crow_indices().numpy()
        cols = Yr_sp.col_indices().numpy()
        yr_v = Yr_sp.values().numpy()
        yi_v = Yi_sp.values().numpy()
        diag_abs = np.zeros(n, np.float32)
        for i in range(n):
            s, e = int(crow[i]), int(crow[i+1])
            if s >= e: continue
            p = int(np.searchsorted(cols[s:e], i))
            if p < (e-s) and cols[s+p] == i:
                diag_abs[i] = np.sqrt(yr_v[s+p]**2 + yi_v[s+p]**2)
        stiff_mask = torch.tensor(diag_abs > stiff_threshold, dtype=torch.bool)
    non_stiff_mask = ~stiff_mask

    # ── KCL all vs non-stiff ──────────────────────────────────────
    kcl_all       = dS.flatten()
    kcl_nonstiff  = dS[non_stiff_mask].flatten()

    def stats(t, label):
        t = torch.nan_to_num(t, nan=0., posinf=0., neginf=0.)
        with open("out.txt", "a") as f:
            print(f"  {label}:")
            print(f"    mean={float(t.mean())*B:.2f} MW  "
                f"max={float(t.max())*B:.2f} MW  "
                f"p99={float(torch.quantile(t,0.99))*B:.2f} MW  "
                f"p999={float(torch.quantile(t,0.999))*B:.2f} MW")
    with open("out.txt", "a") as f:
        print(f"\n=== KCL BREAKDOWN (baseMVA={B}) ===")
        print(f"  Stiff buses (|Y_diag|>{stiff_threshold}): {int(stiff_mask.sum())}")
    stats(kcl_all,      "KCL all buses      ")
    stats(kcl_nonstiff, "KCL non-stiff buses")

    # ── Top-N worst KCL buses ─────────────────────────────────────
    bt_names = {1:"PQ", 2:"PV", 3:"Slack"}
    topk = torch.topk(dS, min(top_n, n))
    with open("out.txt", "a") as f:    
        print(f"\n  Top {top_n} worst KCL buses:")
        print(f"  {'Bus':>7} {'Type':>6} {'|V|':>7} "
            f"{'dS(MW)':>10} {'dP(MW)':>10} {'dQ(MVAr)':>10} "
            f"{'Stiff':>6}")
    for val, idx in zip(topk.values, topk.indices):
        i   = int(idx)
        bt  = int(gd["bus_type"][i])
        vm  = float(V[i].abs())
        ds  = float(val) * B
        dp  = float(dP[i]) * B
        dq  = float(dQ[i]) * B
        is_stiff = bool(stiff_mask[i])
        with open("out.txt", "a") as f:
            print(f"  {i:>7} {bt_names.get(bt,'?'):>6} {vm:>7.4f} "
                f"{ds:>10.2f} {dp:>10.2f} {dq:>10.2f} "
                f"{'⚡' if is_stiff else '':>6}")

    return {
        "kcl_mean_all":      float(kcl_all.mean())*B,
        "kcl_p99_all":       float(torch.quantile(kcl_all,0.99))*B,
        "kcl_mean_nonstiff": float(kcl_nonstiff.mean())*B,
        "kcl_p99_nonstiff":  float(torch.quantile(kcl_nonstiff,0.99))*B,
        "n_stiff":           int(stiff_mask.sum()),
        "stiff_mask":        stiff_mask,
    }
def compute_best_diagnosticsNew(state, gd, boundary=None, baseMVA=None):
    V = state["V"].detach()
    Pg = state["Pg"].detach()
    Qg = state["Qg"].detach()

    n = gd["n_bus"]
    B = baseMVA or gd["baseMVA"]

    part_nr_inf = state.get("nr_mismatch_inf", 0.)
    part_nr_rms = state.get("nr_mismatch_rms", 0.)

    bnd_inf = 0.
    bnd_mw = 0.

    # ------------------------------------------------------------
    # Boundary diagnostic
    # ------------------------------------------------------------
    if boundary is not None and boundary.get("n_bnd", 0) > 0:
        bnd_buses = torch.tensor(
            boundary["bnd_bus_rows"],
            dtype=torch.long
        )

        S_spec_b = boundary["S_spec_bnd"]

        if len(S_spec_b) > 0 and len(bnd_buses) > 0:
            ic_bnd = sparse_complex_mm(
                gd["Y_real_sp"],
                gd["Y_imag_sp"],
                V
            )

            Sc_bnd = V * torch.conj(ic_bnd)

            dP_bnd = (
                S_spec_b.real -
                Sc_bnd[bnd_buses].real
            ).abs()

            dQ_bnd = (
                S_spec_b.imag -
                Sc_bnd[bnd_buses].imag
            ).abs()

            bnd_inf = float(
                torch.cat([dP_bnd, dQ_bnd]).max()
            )

            bnd_mw = (
                float(torch.cat([dP_bnd, dQ_bnd]).mean())
                * B
            )

    # ------------------------------------------------------------
    # Global diagnostics
    # ------------------------------------------------------------
    with torch.no_grad():

        # --------------------------------------------------------
        # Specified generation
        # --------------------------------------------------------
        Sg = torch.zeros(
            n,
            dtype=torch.complex64
        )

        Sg.scatter_add_(
            0,
            gd["gen_bus"],
            torch.complex(Pg, Qg)
        )

        Ss = Sg - gd["s_demand"]

        # --------------------------------------------------------
        # Network calculated injection
        # --------------------------------------------------------
        ic = sparse_complex_mm(
            gd["Y_real_sp"],
            gd["Y_imag_sp"],
            V
        )

        Sc = V * torch.conj(ic)

        # --------------------------------------------------------
        # Unit / scaling diagnostic
        # --------------------------------------------------------
        
        if not disablePrint:
            print(
                f"baseMVA={B} "
                f"|Pg|max={Pg.abs().max().item():.6e} "
                f"|Qg|max={Qg.abs().max().item():.6e} "
                f"|Sd|max={gd['s_demand'].abs().max().item():.6e} "
                f"|Sc|max={Sc.abs().max().item():.6e}"
        )

        # --------------------------------------------------------
        # Full-bus power mismatch
        # --------------------------------------------------------
        dP = Ss.real - Sc.real
        dQ = Ss.imag - Sc.imag
        dS = Sc - Ss

        if not disablePrint:
            print(
                f"dP max={dP.abs().max().item():.6e} "
                f"dQ max={dQ.abs().max().item():.6e} "
                f"dS max={dS.abs().max().item():.6e}"
            )

        # --------------------------------------------------------
        # Bus types
        # --------------------------------------------------------
        pq = (gd["bus_type"] == 1).nonzero(
            as_tuple=True
        )[0]

        pv = (gd["bus_type"] == 2).nonzero(
            as_tuple=True
        )[0]

        ref = (gd["bus_type"] == 3).nonzero(
            as_tuple=True
        )[0]

        # --------------------------------------------------------
        # MATPOWER-style NR equations
        #
        # ΔP : PV + PQ
        # ΔQ : PQ
        # --------------------------------------------------------
        aa = torch.cat([pv, pq])

        dP_nr = dP[aa]
        dQ_nr = dQ[pq]

        mis = torch.cat([
            dP_nr,
            dQ_nr
        ])

        # --------------------------------------------------------
        # Combined MATPOWER-style mismatch
        # --------------------------------------------------------
        global_nr_inf = float(mis.abs().max())

        global_nr_rms = float(
            torch.sqrt(
                (mis ** 2).mean()
            )
        )

        # --------------------------------------------------------
        # Separate P/Q MATPOWER-style metrics
        # --------------------------------------------------------

        if dP_nr.numel() > 0:
            nr_p_inf = float(dP_nr.abs().max())
            nr_p_rms = float(
                torch.sqrt(
                    (dP_nr ** 2).mean()
                )
            )
        else:
            nr_p_inf = 0.0
            nr_p_rms = 0.0

        if dQ_nr.numel() > 0:
            nr_q_inf = float(dQ_nr.abs().max())
            nr_q_rms = float(
                torch.sqrt(
                    (dQ_nr ** 2).mean()
                )
            )
        else:
            nr_q_inf = 0.0
            nr_q_rms = 0.0

        # --------------------------------------------------------
        # Convert MATPOWER-style residuals to MW / MVAr
        # --------------------------------------------------------
        nr_p_inf_mw = nr_p_inf * B
        nr_p_rms_mw = nr_p_rms * B

        nr_q_inf_mvar = nr_q_inf * B
        nr_q_rms_mvar = nr_q_rms * B

        global_nr_inf_mw = global_nr_inf * B
        global_nr_rms_mw = global_nr_rms * B

        # --------------------------------------------------------
        # Worst NR equation
        # --------------------------------------------------------
        idx = int(torch.argmax(mis.abs()))

        nP = len(aa)

        if idx < nP:
            worst_bus = int(aa[idx])
            worst_type = "ΔP"
        else:
            qidx = idx - nP
            worst_bus = int(pq[qidx])
            worst_type = "ΔQ"

        # --------------------------------------------------------
        # Stiff-bus diagnostic
        # --------------------------------------------------------
        Yr_sp = gd["Y_real_sp"]
        Yi_sp = gd["Y_imag_sp"]


        if Yr_sp.layout == torch.sparse_csr:

            import numpy as np

            crow = (
                Yr_sp.crow_indices()
                .numpy()
            )

            cols = (
                Yr_sp.col_indices()
                .numpy()
            )

            yr_v = (
                Yr_sp.values()
                .numpy()
            )

            yi_v = (
                Yi_sp.values()
                .numpy()
            )

            row_idx = np.arange(
                n,
                dtype=np.int64
            )

            starts = crow[:-1]
            ends = crow[1:]

            diag_r = np.zeros(
                n,
                dtype=np.float32
            )

            diag_i = np.zeros(
                n,
                dtype=np.float32
            )

            for i in range(n):
                s = int(starts[i])
                e = int(ends[i])

                if s >= e:
                    continue

                local = cols[s:e]

                pos = np.searchsorted(
                    local,
                    i
                )

                if (
                    pos < len(local)
                    and local[pos] == i
                ):
                    diag_r[i] = yr_v[s + pos]
                    diag_i[i] = yi_v[s + pos]

            y_diag_abs = torch.tensor(
                np.sqrt(
                    diag_r ** 2 +
                    diag_i ** 2
                )
            )

        else:
            # Dense fallback for small grids
            Gd = gd.get("G")
            Bd = gd.get("B")

            if Gd is not None:
                y_diag_abs = torch.sqrt(
                    Gd.diagonal() ** 2 +
                    Bd.diagonal() ** 2
                )
            else:
                y_diag_abs = torch.zeros(n)

        stiff = int(
            (y_diag_abs > 10000).sum()
        )
        
        if stiff > 0:
            stiff_ids = (
                (y_diag_abs > 10000)
                .nonzero(as_tuple=True)[0]
                .tolist()[:5]
            )
            if not disablePrint:
                print(
                    f"  ⚠ stiff={stiff} buses "
                    f"|Y_diag|>10000 pu: "
                    f"{stiff_ids}..."
                )
            if not disablePrint:
                print(
                    f"  ⚠ stiff={stiff} buses "
                    f"|Y_diag|>10000 pu: "
                    f"{stiff_ids}..."
                )
            if not disablePrint:
                print(
                    f"  ⚠ stiff={stiff} buses "
                    f"|Y_diag|>10000 pu: "
                    f"{stiff_ids}..."
                )
        # --------------------------------------------------------
        # Full-bus KCL / complex power balance
        #
        # This is intentionally retained as an independent
        # diagnostic and is NOT the MATPOWER NR metric.
        # --------------------------------------------------------
        kcl_abs = torch.nan_to_num(
            torch.abs(dS),
            nan=0.,
            posinf=0.,
            neginf=0.
        )

        kcl_flat = kcl_abs.flatten()

        kcl_mw_mean = (
            float(kcl_flat.mean()) * B
        )

        kcl_mw_max = (
            float(kcl_flat.max()) * B
        )

        kcl_p99 = (
            float(torch.quantile(kcl_flat, 0.99))
            * B
        )

        kcl_p999 = (
            float(torch.quantile(kcl_flat, 0.999))
            * B
        )
        # --------------------------------------------------------
        # KCL excluding stiff buses
        # --------------------------------------------------------
        stiff_mask = (y_diag_abs > 10000)

        non_stiff_mask = ~stiff_mask

        kcl_nonstiff = kcl_abs[non_stiff_mask]

        if kcl_nonstiff.numel() > 0:
            kcl_nonstiff_mw_mean = (
                float(kcl_nonstiff.mean()) * B
            )

            kcl_nonstiff_mw_max = (
                float(kcl_nonstiff.max()) * B
            )

            kcl_nonstiff_p99 = (
                float(torch.quantile(kcl_nonstiff, 0.99)) * B
            )

            kcl_nonstiff_p999 = (
                float(torch.quantile(kcl_nonstiff, 0.999)) * B
            )
        else:
            kcl_nonstiff_mw_mean = 0.0
            kcl_nonstiff_mw_max = 0.0
            kcl_nonstiff_p99 = 0.0
            kcl_nonstiff_p999 = 0.0
        # --------------------------------------------------------
        # Voltage diagnostics
        # --------------------------------------------------------
        vm = torch.abs(V)

        n_low = int(
            (vm <= 0.701).sum()
        )

        n_high = int(
            (vm >= 1.299).sum()
        )

        nan_buses = int(
            (
                torch.isnan(V.real) |
                torch.isinf(V.real) |
                torch.isnan(V.imag) |
                torch.isinf(V.imag)
            ).sum()
        )

        if nan_buses > 0:
            print(
                f"  ⚠ {nan_buses} NaN/Inf voltages"
            )



    # ------------------------------------------------------------
    # Return diagnostics
    # ------------------------------------------------------------
    return {

        # Partitioned NR
        "part_nr_inf": part_nr_inf,
        "part_nr_rms": part_nr_rms,

        # Boundary
        "bnd_inf": bnd_inf,
        "bnd_mw": bnd_mw,

        # --------------------------------------------------------
        # MATPOWER-compatible NR mismatch
        # --------------------------------------------------------
        "global_nr_inf": global_nr_inf,
        "global_nr_rms": global_nr_rms,

        "global_nr_inf_mw": global_nr_inf_mw,
        "global_nr_rms_mw": global_nr_rms_mw,

        # P equation: PV + PQ
        "nr_p_inf": nr_p_inf,
        "nr_p_rms": nr_p_rms,

        "nr_p_inf_mw": nr_p_inf_mw,
        "nr_p_rms_mw": nr_p_rms_mw,

        # Q equation: PQ only
        "nr_q_inf": nr_q_inf,
        "nr_q_rms": nr_q_rms,

        "nr_q_inf_mvar": nr_q_inf_mvar,
        "nr_q_rms_mvar": nr_q_rms_mvar,

        # Worst equation
        "worst_bus": worst_bus,
        "worst_type": worst_type,

        # --------------------------------------------------------
        # Full-bus KCL diagnostic
        # --------------------------------------------------------
        "kcl_mw_mean": kcl_mw_mean,
        "kcl_mw_max": kcl_mw_max,
        "kcl_p99": kcl_p99,
        "kcl_p999": kcl_p999,
        # --------------------------------------------------------
        # Non-stiff-bus KCL diagnostic
        # --------------------------------------------------------
        "kcl_nonstiff_mw_mean": kcl_nonstiff_mw_mean,
        "kcl_nonstiff_mw_max": kcl_nonstiff_mw_max,
        "kcl_nonstiff_p99": kcl_nonstiff_p99,
        "kcl_nonstiff_p999": kcl_nonstiff_p999,

        # --------------------------------------------------------
        # Voltage / numerical diagnostics
        # --------------------------------------------------------
        "stiff": stiff,
        "n_lowVm7": n_low,
        "n_highVm1dot3": n_high,
        "nan_buses": nan_buses,
    }
