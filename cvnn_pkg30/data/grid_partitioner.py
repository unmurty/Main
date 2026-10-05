"""
grid_partitioner.py — Stiffness-aware METIS partitioning (Option B)
- No Z clamp in Ybus; weighted METIS keeps stiff branches internal
- branch_edge_weight() assigns high cost to stiff branches
- _build_weighted_graph() builds adjacency + weight dict
- _adj_to_csr() converts to xadj/adjncy/eweights for pymetis CSR form
- partition_stats tracks stiff_cuts for falsifiable experiment
PRIMARY metric: global_nr_rms
"""
import os, numpy as np, torch
import scipy.sparse as sp, scipy.sparse.linalg as spla
from data.ybus_builder import build_ybus, add_normalised_ybus, SPARSE_LIMIT

STIFF_Y_THRESHOLD = 500.0
Y_REF_DEFAULT     = 10.0
disablePrint=True
# ── Partitioner selection ─────────────────────────────────────────────────────
_PART = None
def _init():
    global _PART
    try:
        import pymetis; pymetis.part_graph(2,adjacency=[[1],[0]])
        _PART=("pymetis",pymetis); return
    except: pass
    try:
        for p in ["/usr/lib/x86_64-linux-gnu/libmetis.so",
                  "/usr/lib/x86_64-linux-gnu/libmetis.so.5"]:
            if os.path.exists(p): os.environ["METIS_DLL"]=p; break
        import metis as m; m.adjlist_to_metis([[1],[0]])
        _PART=("metis",m); return
    except: pass
    try:
        import networkx as nx; _PART=("networkx",nx); return
    except: pass
    _PART=("fallback",None)
_init()
if not disablePrint:
    print(f"  Grid partitioner : {_PART[0]}")


# ── Edge weight function ──────────────────────────────────────────────────────
def branch_edge_weight(r, x):
    """
    Tiered stiffness weight for METIS.
    Normal (|Y|<=500) → 1, Stiff (|Y|>500) → 20-1000.
    High weight = expensive to cut = METIS avoids cutting stiff branches.
    """
    z  = complex(float(r), float(x))
    az = abs(z)
    if az < 1e-12: return 10000
    y = 1.0 / az
    # Weight = int(max(1, min(10000, y_abs/10)))
    # |Y|=83333 → weight=8333, |Y|=50000 → weight=5000, |Y|=10000 → weight=1000
    # Ensures stiff pairs (54785↔54787 etc) are virtually never cut by METIS
    if   y > 50000: return 10000
    elif y > 10000: return 5000
    elif y > 5000:  return 1000
    elif y > 1000:  return 100
    elif y > 500:   return 20
    return 1


def _build_weighted_graph(gd):
    """Undirected adjacency list + edge weight dict keyed by (i,j)."""
    n=gd["n_bus"]; branch=gd["branch_arr"]
    fbus=gd["fbus"].numpy(); tbus=gd["tbus"].numpy()
    adjacency=[set() for _ in range(n)]; edge_weights={}
    for k in range(len(branch)):
        if int(branch[k,10])==0: continue
        f=int(fbus[k]); t=int(tbus[k])
        if f==t: continue
        r=max(float(branch[k,2]),0.0); x=float(branch[k,3])
        w=branch_edge_weight(r,x)
        adjacency[f].add(t); adjacency[t].add(f)
        edge_weights[(f,t)]=w; edge_weights[(t,f)]=w
    adjacency=[sorted(list(s)) for s in adjacency]
    return adjacency, edge_weights


def _adj_to_csr(adj, edge_weights=None):
    """Convert sorted adjacency list to pymetis CSR arrays."""
    xadj=[0]; adjncy=[]; ew=[]
    for i,nbrs in enumerate(adj):
        for j in nbrs:
            adjncy.append(int(j))
            if edge_weights is not None:
                ew.append(max(1,int(edge_weights.get((i,j),1))))
        xadj.append(len(adjncy))
    xadj_np  =np.asarray(xadj,  dtype=np.int32)
    adjncy_np=np.asarray(adjncy,dtype=np.int32)
    ew_np    =np.asarray(ew,    dtype=np.int32) if edge_weights is not None else None
    return xadj_np, adjncy_np, ew_np


def _do_partition(adj, n_parts, edge_weights=None):
    """Partition using pymetis CSR form with optional stiffness weights."""
    n=len(adj)
    if n_parts<=1 or n<=n_parts:
        return list(range(n)),0,{"stiff_cuts":0,"boundary_stiff_ratio":0.,"cutcount":0}
    xadj_np,adjncy_np,ew_np=_adj_to_csr(adj,edge_weights)
    if ew_np is not None and (np.any(ew_np<=0) or len(ew_np)!=len(adjncy_np)):
        print("  WARNING: invalid edge weights — falling back to unweighted")
        ew_np=None
    name,lib=_PART
    if name=="pymetis":
        cutcount,membership=lib.part_graph(
            n_parts,xadj=xadj_np,adjncy=adjncy_np,eweights=ew_np)
    elif name=="metis":
        g=lib.adjlist_to_metis(adj)
        cutcount,membership=lib.part_graph(g,nparts=n_parts,recursive=(n_parts<8))
    elif name=="networkx":
        import networkx as nx
        G=nx.Graph()
        for i,nbrs in enumerate(adj):
            for j in nbrs: G.add_edge(i,j)
        parts=[set(G.nodes())]
        while len(parts)<n_parts:
            parts.sort(key=len,reverse=True); lg=parts.pop(0)
            try: A,B=nx.community.kernighan_lin_bisection(G.subgraph(lg),seed=42)
            except:
                nd=sorted(lg); A,B=set(nd[:len(nd)//2]),set(nd[len(nd)//2:])
            parts+=[A,B] if A and B else [lg]
            while len(parts)>n_parts:
                parts.sort(key=len); parts[1]|=parts.pop(0)
        membership=[0]*n
        for pid,pt in enumerate(parts):
            for nd in pt: membership[nd]=pid
        cutcount=0
    else:
        membership=(np.arange(n)*n_parts//n).tolist(); cutcount=0
    membership=list(map(int,membership))
    # Count stiff cuts
    stiff_cuts=0; stiff_wt=0; total_wt=0
    if ew_np is not None:
        for i in range(n):
            for pos in range(int(xadj_np[i]),int(xadj_np[i+1])):
                j=int(adjncy_np[pos])
                if i>=j: continue
                if membership[i]!=membership[j]:
                    w=int(ew_np[pos]); total_wt+=w
                    if w>=20: stiff_cuts+=1; stiff_wt+=w
    stats={"cutcount":int(cutcount),"stiff_cuts":stiff_cuts,
           "stiff_cut_weight":stiff_wt,"total_cut_weight":total_wt,
           "boundary_stiff_ratio":stiff_wt/max(total_wt,1)}
    return membership,int(cutcount),stats


def partition_grid(gd, target_size=200):
    n=gd["n_bus"]; n_parts=max(2,int(np.ceil(n/target_size)))
    if not disablePrint:
        print(f"\n  Partitioning {n:,} buses → {n_parts} sub-grids (target ~{target_size})")
    if n_parts==1: return [gd],_empty_boundary(),np.zeros(n,dtype=np.int32),{}

    # Build stiffness-weighted graph
    adjacency, edge_weights = _build_weighted_graph(gd)
    membership, cutcount, do_stats = _do_partition(adjacency, n_parts, edge_weights)
    membership = np.array(membership, dtype=np.int32)

    if not disablePrint:
        print(f"  METIS: {cutcount} cuts  stiff_cuts={do_stats['stiff_cuts']} "
              f"({100*do_stats['boundary_stiff_ratio']:.1f}% of cut weight is stiff)")

    # Precompute lists
    fr_list   = gd["fbus"].numpy().tolist()
    tr_list   = gd["tbus"].numpy().tolist()
    bstat_list= gd["branch_status"].numpy().tolist()
    gen_bus_np= gd["gen_bus"].numpy().astype(np.int32)
    btype_np  = gd["bus_type"].numpy()
    branch_arr= gd["branch_arr"]

    # gens_per_bus O(n_gen)
    gens_per_bus=[[] for _ in range(n)]
    for gi,bus in enumerate(gen_bus_np.tolist()): gens_per_bus[int(bus)].append(gi)

    # One-pass branch classification O(n_branches)
    branch_lists  =[[] for _ in range(n_parts)]
    boundary_lists=[[] for _ in range(n_parts)]
    all_ties=[]; stiff_cut_ids=[]; stiff_cut_buses=set()

    for br,(f,t) in enumerate(zip(fr_list,tr_list)):
        if bstat_list[br]==0: continue
        cf=int(membership[f]); ct=int(membership[t])
        if cf==ct:
            branch_lists[cf].append(br)
        else:
            boundary_lists[cf].append(br); boundary_lists[ct].append(br)
            all_ties.append({"br_idx":br,"f":f,"t":t,"cf":cf,"ct":ct})
            # Track stiff boundary branches
            r_=max(float(branch_arr[br,2]),0.0); x_=float(branch_arr[br,3])
            w=branch_edge_weight(r_,x_)
            if w>=20:
                stiff_cut_ids.append(br)
                stiff_cut_buses.add((min(f,t),max(f,t)))

    gen_lists=[[] for _ in range(n_parts)]
    for bus_row,cid_bus in enumerate(membership.tolist()):
        for gi in gens_per_bus[bus_row]: gen_lists[cid_bus].append(gi)

    subgrids=[]
    for cid in range(n_parts):
        bidx=np.where(membership==cid)[0].astype(np.int64)
        bidx_list=bidx.tolist(); ns=len(bidx)
        g2l={gb:lb for lb,gb in enumerate(bidx_list)}
        gen_idx=np.array(gen_lists[cid],dtype=np.int64); ng=len(gen_idx)
        lgb=np.array([g2l[int(gen_bus_np[g])] for g in gen_idx],np.int64)
        int_br=np.asarray(branch_lists[cid],dtype=np.int64)
        stiff_pairs = []
        for br in int_br:
            r_ = max(float(branch_arr[br, 2]), 0.0)
            x_ = float(branch_arr[br, 3])
            y_ = 1.0 / max(abs(complex(r_, x_)), 1e-12)

            if y_ > STIFF_Y_THRESHOLD:
                f = int(fr_list[br])
                t = int(tr_list[br])

                stiff_pairs.append((f, t))
# IMPORTANT: this is the flag consumed by differentiable_nr.py
        _is_stiff = len(stiff_pairs) > 0
        bnd_br=np.asarray(boundary_lists[cid],dtype=np.int64)
        nb_int=len(int_br)
        lbt=btype_np[bidx].copy()
        lsl=np.where(lbt==3)[0]
        if len(lsl)==0:
            lpv=np.where(lbt==2)[0]
            pr=lpv[:1] if len(lpv)>0 else np.array([0],np.int64)
            if len(pr)>0 and pr[0]<len(lbt): lbt[pr[0]]=3; lsl=pr
            else: lsl=np.array([0],np.int64); lbt[0]=3
        if nb_int>0:
            sbr=gd["branch_arr"][int_br]
            sfb=np.array([g2l[fr_list[b]] for b in int_br],np.int64)
            stb=np.array([g2l[tr_list[b]] for b in int_br],np.int64)
        else:
            sbr=np.zeros((0,gd["branch_arr"].shape[1]),np.float32)
            sfb=np.zeros(0,np.int64); stb=np.zeros(0,np.int64)
        Bv=gd["baseMVA"]
        def _b(k): return gd[k][bidx]
        def _g(k): return gd[k][gen_idx] if ng>0 else torch.zeros(0,dtype=gd[k].dtype)
        sg_sparse=(ns>=SPARSE_LIMIT)
        sg=dict(cluster_id=cid,baseMVA=Bv,n_bus=ns,n_gen=ng,n_branch=nb_int,
                use_sparse=sg_sparse,
                bus_arr=gd["bus_arr"][bidx],branch_arr=sbr.astype(np.float32),
                gen_arr=gd["gen_arr"][gen_idx] if ng>0 else np.zeros((0,gd["gen_arr"].shape[1]),np.float32),
                bus_type=torch.tensor(lbt,dtype=torch.long),
                slack_idx=torch.tensor(lsl,dtype=torch.long),
                bus_id=_b("bus_id"),
                fbus=torch.tensor(sfb,dtype=torch.long),
                tbus=torch.tensor(stb,dtype=torch.long),
                branch_status=torch.ones(nb_int,dtype=torch.long),
                r=torch.tensor(sbr[:,2] if nb_int>0 else [],dtype=torch.float32),
                x=torch.tensor(sbr[:,3] if nb_int>0 else [],dtype=torch.float32),
                b=torch.tensor(sbr[:,4] if nb_int>0 else [],dtype=torch.float32),
                rateA=torch.tensor(sbr[:,5]/Bv if nb_int>0 else [],dtype=torch.float32),
                tap=torch.tensor(sbr[:,8] if nb_int>0 else [],dtype=torch.float32),
                shift=torch.tensor(np.deg2rad(sbr[:,9]) if nb_int>0 else [],dtype=torch.float32),
                angmin=torch.tensor(sbr[:,11] if nb_int>0 and sbr.shape[1]>11 else np.full(nb_int,-360.),dtype=torch.float32),
                angmax=torch.tensor(sbr[:,12] if nb_int>0 and sbr.shape[1]>12 else np.full(nb_int,360.),dtype=torch.float32),
                pd=_b("pd"),qd=_b("qd"),gs=_b("gs"),bs=_b("bs"),
                vm0=_b("vm0"),va0=_b("va0"),v0=_b("v0"),vmax=_b("vmax"),vmin=_b("vmin"),
                s_demand=_b("s_demand"),gen_bus=torch.tensor(lgb,dtype=torch.long),
                gen_incidence=torch.zeros(ns,ng),
                pg=_g("pg"),qg=_g("qg"),Pg_max=_g("Pg_max"),Pg_min=_g("Pg_min"),
                Qg_max=_g("Qg_max"),Qg_min=_g("Qg_min"),vg_set=_g("vg_set"),
                gen_status=_g("gen_status") if ng>0 else torch.zeros(0,dtype=torch.long),
                gencost_c2=_g("gencost_c2"),gencost_c1=_g("gencost_c1"),gencost_c0=_g("gencost_c0"),
                cost2=_g("cost2"),cost1=_g("cost1"),cost0=_g("cost0"),gencost=None,
                global_bus_idx=torch.tensor(bidx,dtype=torch.long),
                global_gen_idx=torch.tensor(gen_idx,dtype=torch.long),
                global_branch_int_idx=torch.tensor(int_br,dtype=torch.long),
                global_branch_bnd_idx=torch.tensor(bnd_br,dtype=torch.long),
                global_to_local_bus=g2l,
                stiff_pairs=stiff_pairs,_is_stiff=_is_stiff)
        sg=build_ybus(sg); sg=add_normalised_ybus(sg)
        subgrids.append(sg)
        print(f"    Sub-grid {cid:3d}: {ns:5d} buses|{nb_int:5d} int|"
              f"{len(bnd_br):4d} tie|{ng:4d} gens|sparse={sg_sparse}")

    bnd=_build_boundary_data(gd,all_ties)
    total_cuts=len(all_ties)
    stiff_cuts_final=len(stiff_cut_ids)
    # Report ALL stiff branches — internal (good) vs boundary (problem)
    all_stiff = []
    for br in range(len(fr_list)):
        if bstat_list[br] == 0: continue
        r_ = max(float(branch_arr[br,2]), 0.0); x_ = float(branch_arr[br,3])
        z  = complex(r_,x_); absz = abs(z) if abs(z)>1e-12 else 1e-12
        y_mag = 1.0/absz
        if y_mag > STIFF_Y_THRESHOLD:
            f_ = fr_list[br]; t_ = tr_list[br]
            cf_ = int(membership[f_]); ct_ = int(membership[t_])
            loc = "BOUNDARY ← problem!" if cf_!=ct_ else "internal ✓"
            all_stiff.append((br,f_,t_,y_mag,cf_,ct_,loc))
    print(f"\n  Stiff branches in grid (|Y|>{STIFF_Y_THRESHOLD} pu):")
    for br,f_,t_,y_mag,cf_,ct_,loc in all_stiff:
        w = branch_edge_weight(max(float(branch_arr[br,2]),0.0), float(branch_arr[br,3]))
        if (y_mag > 10000) :
            print(f"    Branch {br:6d}: bus {f_}(sg{cf_}) ↔ {t_}(sg{ct_})  "
              f"|Y|={y_mag:.1f} pu  weight={w}  [{loc}]")
    if not all_stiff:
        print(f"    None found (all |Y| <= {STIFF_Y_THRESHOLD} pu)")
    print(f"\n  Tie-lines : {bnd['n_tie']}  Boundary buses : {bnd['n_bnd']}")
    print(f"  Stiff boundary branches: {stiff_cuts_final} "
          f"(target=0 for Option B success)")
    if stiff_cut_ids:
        print(f"  Stiff cuts detail (|Y|>500, weight>=20 branches crossing boundaries):")
        for br_id in stiff_cut_ids[:10]:
            r_=max(float(branch_arr[br_id,2]),0.0); x_=float(branch_arr[br_id,3])
            z=complex(r_,x_); absz=abs(z) if abs(z)>1e-12 else 1e-12
            y_mag=1.0/absz; w=branch_edge_weight(r_,x_)
            f_=fr_list[br_id]; t_=tr_list[br_id]
            cf_=int(membership[f_]); ct_=int(membership[t_])
            print(f"    Branch {br_id:6d}: bus {f_}(sg{cf_}) ↔ {t_}(sg{ct_})  "
                  f"|Y|={y_mag:.1f} pu  weight={w}  ← should be internal!")
        if len(stiff_cut_ids)>10:
            print(f"    ... and {len(stiff_cut_ids)-10} more")
    else:
        print(f"  ✓ All stiff branches kept internal — Option B working correctly")

    partition_stats={
        "stiff_cuts":           stiff_cuts_final,
        "stiff_cut_ids":        stiff_cut_ids,
        "stiff_cut_buses":      stiff_cut_buses,
        "total_cuts":           total_cuts,
        "boundary_stiff_ratio": stiff_cuts_final/max(total_cuts,1),
        "metis_stats":          do_stats,
    }
    return subgrids, bnd, membership, partition_stats


def reconcile_boundary(V_global,subgrids,boundary,gd,damping=0.3,tol=1e-4,max_iter=15):
    from utils.sparse_complex import sparse_complex_mm
    if boundary["n_tie"]==0: return V_global
    fill=torch.complex(torch.ones_like(V_global.real),torch.zeros_like(V_global.imag))
    bad=torch.isnan(V_global.real)|torch.isinf(V_global.real)|torch.isnan(V_global.imag)|torch.isinf(V_global.imag)
    V_global=torch.where(bad,fill,V_global)
    bnd_buses=boundary["bnd_bus_rows"]; n_bnd=len(bnd_buses)
    for _ in range(max_iter):
        ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V_global)
        Sc=V_global*torch.conj(ic)
        dP=(boundary["S_spec_bnd"].real-Sc[bnd_buses].real).numpy()
        dQ=(boundary["S_spec_bnd"].imag-Sc[bnd_buses].imag).numpy()
        mis=np.concatenate([dP,dQ])
        if np.abs(mis).max()<tol: break
        J=_boundary_jacobian(V_global,bnd_buses,gd)
        try:
            ilu=spla.spilu(J.tocsc(),fill_factor=10)
            Mpc=spla.LinearOperator(J.shape,ilu.solve)
            dx,info=spla.bicgstab(J,mis,M=Mpc,atol=1e-8,maxiter=300)
            if info!=0: raise RuntimeError()
        except: dx=spla.spsolve(J.tocsc(),mis)
        Vm=torch.abs(V_global).numpy(); Va=torch.angle(V_global).numpy()
        Va[bnd_buses]+=damping*dx[:n_bnd]
        Vm[bnd_buses]*=(1.+damping*dx[n_bnd:])
        Vm=np.clip(Vm,0.5,1.3)
        V_new=torch.tensor(Vm*np.exp(1j*Va),dtype=torch.complex64)
        bad2=(torch.isnan(V_new.real)|torch.isinf(V_new.real)|
              torch.isnan(V_new.imag)|torch.isinf(V_new.imag))
        V_global=torch.where(bad2,V_global,V_new)
    return V_global


def _boundary_jacobian(V,bnd_buses,gd):
    """Full 4-block [H N;M L] — corrects angle AND magnitude discontinuities."""
    from utils.sparse_complex import sparse_complex_mm
    from data.ybus_builder import get_GB
    G,B=get_GB(gd)
    ic=sparse_complex_mm(gd["Y_real_sp"],gd["Y_imag_sp"],V)
    Sc=V*torch.conj(ic)
    Vm=torch.abs(V).numpy(); Va=torch.angle(V).numpy()
    Gn=G.numpy(); Bn=B.numpy(); n=len(bnd_buses)
    rH,cH,vH=[],[],[]; rN,cN,vN=[],[],[]; rM,cM,vM=[],[],[]; rL,cL,vL=[],[],[]
    for ki,bi in enumerate(bnd_buses.tolist()):
        for kj,bj in enumerate(bnd_buses.tolist()):
            th=Va[bi]-Va[bj]; Vij=Vm[bi]*Vm[bj]
            if ki==kj:
                hv=float(-Sc[bi].imag)-Bn[bi,bi]*Vm[bi]**2
                nv=float( Sc[bi].real)+Gn[bi,bi]*Vm[bi]**2
                mv=float( Sc[bi].real)-Gn[bi,bi]*Vm[bi]**2
                lv=float( Sc[bi].imag)-Bn[bi,bi]*Vm[bi]**2
            else:
                hv= Vij*(Gn[bi,bj]*np.sin(th)-Bn[bi,bj]*np.cos(th))
                nv= Vij*(Gn[bi,bj]*np.cos(th)+Bn[bi,bj]*np.sin(th))
                mv=-Vij*(Gn[bi,bj]*np.cos(th)+Bn[bi,bj]*np.sin(th))
                lv= Vij*(Gn[bi,bj]*np.sin(th)-Bn[bi,bj]*np.cos(th))
            rH.append(ki);cH.append(kj);vH.append(hv)
            rN.append(ki);cN.append(kj);vN.append(nv)
            rM.append(ki);cM.append(kj);vM.append(mv)
            rL.append(ki);cL.append(kj);vL.append(lv)
    H=sp.csr_matrix((vH,(rH,cH)),shape=(n,n)); N=sp.csr_matrix((vN,(rN,cN)),shape=(n,n))
    M=sp.csr_matrix((vM,(rM,cM)),shape=(n,n)); L=sp.csr_matrix((vL,(rL,cL)),shape=(n,n))
    J=sp.bmat([[H,N],[M,L]],format='csr')
    return J+1e-6*sp.eye(2*n,format='csr')


def _build_boundary_data(gd,all_ties):
    if not all_ties: return _empty_boundary()
    bset=set()
    for tl in all_ties: bset.add(tl["f"]); bset.add(tl["t"])
    bnd=np.array(sorted(bset),np.int64)
    return dict(n_tie=len(all_ties),n_bnd=len(bnd),bnd_bus_rows=bnd,
                bnd_g2l={int(b):i for i,b in enumerate(bnd.tolist())},
                tie_lines=all_ties,
                S_spec_bnd=torch.zeros(len(bnd),dtype=torch.complex64),
                J_bnd_sp=sp.eye(2*len(bnd),format='csr'))

def _empty_boundary():
    return dict(n_tie=0,n_bnd=0,bnd_bus_rows=np.array([]),bnd_g2l={},
                tie_lines=[],S_spec_bnd=torch.zeros(0),J_bnd_sp=None)

def assemble_global_voltages(subgrids,gd):
    V=torch.ones(gd["n_bus"],dtype=torch.complex64)
    for sg in subgrids:
        if "V" in sg: V[sg["global_bus_idx"]]=sg["V"]
    return V
