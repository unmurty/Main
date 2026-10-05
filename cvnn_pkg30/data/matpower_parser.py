import re, numpy as np, torch
def load_matpower_case(casepath, builtin=None, device="cpu"):
    print(f"  Parsing: {casepath}")
    txt=open(casepath).read()
    def ps(name):
        m=re.search(rf'mpc\.{name}\s*=\s*\[(.*?)\];',txt,re.DOTALL)
        if not m: return None
        rows=[]
        for ln in m.group(1).split('\n'):
            ln=re.sub(r'%.*','',ln).strip().rstrip(';').strip()
            if not ln: continue
            try:
                v=[float(x) for x in ln.split()]
                if v: rows.append(v)
            except: pass
        return np.array(rows,dtype=np.float64)
    bm=re.search(r'mpc\.baseMVA\s*=\s*([\d.]+)',txt)
    B=float(bm.group(1)) if bm else 100.
    bus=ps('bus'); branch=ps('branch'); gen=ps('gen'); gc=ps('gencost')
    nb=len(bus); nbr=len(branch); ng=len(gen)
    bids=bus[:,0].astype(int); i2r={b:i for i,b in enumerate(bids)}
    btype=bus[:,1].astype(int)
    pd=bus[:,2]/B; qd=bus[:,3]/B; gs=bus[:,4]/B; bs=bus[:,5]/B
    vm0=bus[:,7]; va0=np.deg2rad(bus[:,8]); vmax=bus[:,11]; vmin=bus[:,12]
    v0=(vm0*np.exp(1j*va0)).astype(np.complex64)
    sl=np.where(btype==3)[0]
    gbids=gen[:,0].astype(int); gbr=np.array([i2r[b] for b in gbids],int)
    pg=gen[:,1]/B; qg=gen[:,2]/B
    pgmx=np.clip(gen[:,8]/B,None,1e4); pgmn=np.clip(gen[:,9]/B,-1e4,None)
    qgmx=np.clip(gen[:,3]/B,None,1e4); qgmn=np.clip(gen[:,4]/B,-1e4,None)
    vgs=gen[:,5]; gst=gen[:,7].astype(int)
    gi=np.zeros((nb,ng),np.float32)
    for g,br in enumerate(gbr): gi[br,g]=1.
    slbid=bids[sl[0]] if len(sl)>0 else bids[0]
    sgm=np.where(gbids==slbid)[0]; sgi=int(sgm[0]) if len(sgm)>0 else 0
    if gc is not None:
        c2,c1,c0=[],[],[]
        for row in gc:
            nc=int(row[3]); co=row[4:]
            if nc==3: c2.append(co[0]);c1.append(co[1]);c0.append(co[2])
            elif nc==2: c2.append(0.);c1.append(co[0]);c0.append(co[1])
            else: c2.append(0.);c1.append(0.);c0.append(0.)
        c2,c1,c0=np.array(c2,np.float32),np.array(c1,np.float32),np.array(c0,np.float32)
    else:
        c2=np.ones(ng,np.float32); c1=np.zeros(ng,np.float32); c0=np.zeros(ng,np.float32)
    fbids=branch[:,0].astype(int); tbids=branch[:,1].astype(int)
    fbr=np.array([i2r[b] for b in fbids],int); tbr=np.array([i2r[b] for b in tbids],int)
    r=branch[:,2]; x=branch[:,3]; b=branch[:,4]; rA=branch[:,5]/B
    tap=branch[:,8].copy(); tap[tap==0]=1.; sh=np.deg2rad(branch[:,9])
    bst=branch[:,10].astype(int)
    amn=branch[:,11] if branch.shape[1]>11 else np.full(nbr,-360.)
    amx=branch[:,12] if branch.shape[1]>12 else np.full(nbr,360.)
    sd=(pd+1j*qd).astype(np.complex64); usp=nb>500
    def T(a,dt=torch.float32):
        a=np.array(a,dtype=np.float32 if dt==torch.float32 else np.int64 if dt==torch.long else np.complex64)
        return torch.tensor(a,dtype=dt)
    gd=dict(baseMVA=B,n_bus=nb,n_gen=ng,n_branch=nbr,use_sparse=usp,
            bus_arr=bus.astype(np.float32),branch_arr=branch.astype(np.float32),
            gen_arr=gen.astype(np.float32),
            bus_id=T(bids,torch.long),bus_type=T(btype,torch.long),
            pd=T(pd),qd=T(qd),gs=T(gs),bs=T(bs),vmax=T(vmax),vmin=T(vmin),
            vm0=T(vm0),va0=T(va0),v0=T(v0,torch.complex64),
            slack_idx=T(sl,torch.long),
            slack_vm=T(vm0[sl] if len(sl) else vm0[:1]),
            slack_va=T(va0[sl] if len(sl) else va0[:1]),
            s_demand=T(sd,torch.complex64),gen_bus=T(gbr,torch.long),
            slack_gen_idx=sgi,pg=T(pg),qg=T(qg),
            Pg_max=T(pgmx),Pg_min=T(pgmn),Qg_max=T(qgmx),Qg_min=T(qgmn),
            vg_set=T(vgs),gen_status=T(gst,torch.long),gen_incidence=T(gi),
            cost2=T(c2),cost1=T(c1),cost0=T(c0),
            gencost_c2=T(c2),gencost_c1=T(c1),gencost_c0=T(c0),gencost=gc,
            fbus=T(fbr,torch.long),tbus=T(tbr,torch.long),
            r=T(r),x=T(x),b=T(b),rateA=T(rA),tap=T(tap),shift=T(sh),
            branch_status=T(bst,torch.long),angmin=T(amn),angmax=T(amx))
    print(f"  {nb} buses|{nbr} branches|{ng} gens|baseMVA={B}|sparse={usp}")
    return gd
