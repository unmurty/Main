"""
ybus_builder.py — Option B: NO Z clamp in Ybus.
Stiff branches kept at full admittance.
Weighted METIS in grid_partitioner ensures stiff branches never cross boundaries.
G/B dense only for sub-grids < SPARSE_LIMIT (256) buses.
"""
import numpy as np, torch, warnings
SPARSE_LIMIT = 256
disablePrint=True
def build_ybus(gd):
    branch=gd["branch_arr"]; bus=gd["bus_arr"]
    nb=gd["n_bus"]; B=gd["baseMVA"]
    fr=gd["fbus"].numpy(); tr=gd["tbus"].numpy()
    usp = gd["use_sparse"] or (nb >= SPARSE_LIMIT)
    Yff_l,Yft_l,Ytf_l,Ytt_l=[],[],[],[]
    if usp:
        import scipy.sparse as sp
        rl,cl,vl=[],[],[]
    else:
        Yn=np.zeros((nb,nb),dtype=np.complex64)
    for k in range(len(branch)):
        f=int(fr[k]); t=int(tr[k])
        if int(branch[k,10])==0:
            for lst in [Yff_l,Yft_l,Ytf_l,Ytt_l]: lst.append(0j)
            continue
        # Option B: clamp only negative resistance (physically impossible)
        # No Z-magnitude clamp — weighted METIS keeps stiff branches internal
        r_=max(float(branch[k,2]), 0.0)   # R >= 0 always
        x_=float(branch[k,3]); b_=float(branch[k,4])
        tap=float(branch[k,8]) if branch[k,8]!=0 else 1.
        sh=np.deg2rad(float(branch[k,9]))
        z=complex(r_,x_)
        if abs(z)<1e-12: z=1e-12+0j   # only guard true zero
        y=1./z; ys=1j*b_/2.; tc=tap*np.exp(1j*sh)
        Yff=(y+ys)/(tc*np.conj(tc)); Ytt=y+ys; Yft=-y/np.conj(tc); Ytf=-y/tc
        for lst,v in zip([Yff_l,Yft_l,Ytf_l,Ytt_l],[Yff,Yft,Ytf,Ytt]): lst.append(v)
        if usp: rl+=[f,t,f,t]; cl+=[f,t,t,f]; vl+=[Yff,Ytt,Yft,Ytf]
        else:   Yn[f,f]+=Yff; Yn[t,t]+=Ytt; Yn[f,t]+=Yft; Yn[t,f]+=Ytf
    # Bus shunt — clamp to ±500 pu (catches unit errors in case files)
    MAX_SHUNT=500.0; n_clamped=0
    for i in range(nb):
        gs_pu=float(np.clip(bus[i,4]/B,-MAX_SHUNT,MAX_SHUNT))
        bs_pu=float(np.clip(bus[i,5]/B,-MAX_SHUNT,MAX_SHUNT))
        if abs(bus[i,4]/B)>MAX_SHUNT or abs(bus[i,5]/B)>MAX_SHUNT: n_clamped+=1
        g=complex(gs_pu,bs_pu)
        if usp: rl.append(i); cl.append(i); vl.append(g)
        else:   Yn[i,i]+=g
    if n_clamped>0:
        print(f"  WARNING: {n_clamped} bus shunts clamped to ±{MAX_SHUNT} pu")
    if usp:
        import scipy.sparse as sp
        Yc=sp.csr_matrix((np.array(vl,dtype=np.complex64),
                          (np.array(rl,np.int64),np.array(cl,np.int64))),
                         shape=(nb,nb)).tocsr()
        Yc.sort_indices(); Yc.sum_duplicates()
        def csr(d):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return torch.sparse_csr_tensor(
                    torch.from_numpy(Yc.indptr.astype(np.int64)),
                    torch.from_numpy(Yc.indices.astype(np.int64)),
                    torch.from_numpy(d.astype(np.float32)),size=(nb,nb))
        sc=float(np.abs(Yc.diagonal()).max())
        Yr=csr(Yc.data.real); Yi=csr(Yc.data.imag)
        Yrn=csr(Yc.data.real/sc); Yin=csr(Yc.data.imag/sc)
        # Dense G/B only for small sub-grids — avoids OOM on large grids
        if nb<SPARSE_LIMIT:
            arr=Yc.toarray()
            Gd=torch.tensor(arr.real,dtype=torch.float32)
            Bd=torch.tensor(arr.imag,dtype=torch.float32)
        else:
            Gd=None; Bd=None
    else:
        Yt=torch.tensor(Yn,dtype=torch.complex64)
        sc=float(torch.abs(Yt).max().clamp(min=1.))
        Yr=Yt.real.to_sparse(); Yi=Yt.imag.to_sparse()
        Yrn=(Yt.real/sc).to_sparse(); Yin=(Yt.imag/sc).to_sparse()
        Gd=Yt.real; Bd=Yt.imag
    
    gd.update(Y_real_sp=Yr,Y_imag_sp=Yi,Y_real_norm=Yrn,Y_imag_norm=Yin,
              ybus_scale=sc,G=Gd,B=Bd,use_sparse=usp,
              Yff=torch.tensor(Yff_l,dtype=torch.complex64),
              Yft=torch.tensor(Yft_l,dtype=torch.complex64),
              Ytf=torch.tensor(Ytf_l,dtype=torch.complex64),
              Ytt=torch.tensor(Ytt_l,dtype=torch.complex64),
              f_bus=gd["fbus"],t_bus=gd["tbus"])
    if not disablePrint:
        print(
        "Yreal max =",
        float(Yr.values().abs().max())
            )

        print(
        "Yimag max =",
        float(Yi.values().abs().max())
        )

        print(
        "|Y| max =",
        float(torch.sqrt(
            Yr.values()**2 + Yi.values()**2
        ).max())
        )
    if not disablePrint:
        print(
        "Yreal max =",
        float(Yr.values().abs().max())
            )

        print(
        "Yimag max =",
        float(Yi.values().abs().max())
        )

        print(
        "|Y| max =",
        float(torch.sqrt(
            Yr.values()**2 + Yi.values()**2
        ).max())
        )
    return gd

def add_normalised_ybus(gd): return gd

def get_GB(gd):
    if gd["G"] is not None:
        return gd["G"], gd["B"]
    Yr=gd["Y_real_sp"]; Yi=gd["Y_imag_sp"]
    if not disablePrint:
        print(
        "Yreal max =",
        float(Yr.values().abs().max())
            )

        print(
        "Yimag max =",
        float(Yi.values().abs().max())
        )

        print(
        "|Y| max =",
        float(torch.sqrt(
            Yr.values()**2 + Yi.values()**2
        ).max())
        )
    if not disablePrint:
        print(
        "Yreal max =",
        float(Yr.values().abs().max())
            )

    if not disablePrint:
        print(
        "Yimag max =",
        float(Yi.values().abs().max())
        )

    if not disablePrint:
        print(
        "|Y| max =",
        float(torch.sqrt(
            Yr.values()**2 + Yi.values()**2
        ).max())
        )

    G=Yr.to_dense(); B=Yi.to_dense()
    gd["G"]=G; gd["B"]=B
    return G,B
