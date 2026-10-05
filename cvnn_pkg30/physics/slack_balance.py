import torch
def apply_slack_balance(Pg,gd):
    bt=gd["bus_type"]; gb=gd["gen_bus"]; pdt=gd["pd"].sum()
    sl=(bt==3).nonzero(as_tuple=True)[0]
    sr=sl[0] if len(sl)>0 else torch.tensor(0,dtype=torch.long)
    m=(gb==sr).nonzero(as_tuple=True)[0]
    if len(m)==0:
        pv=(bt==2).nonzero(as_tuple=True)[0]; idx=torch.tensor(0,dtype=torch.long)
        for pr in pv:
            pm=(gb==pr).nonzero(as_tuple=True)[0]
            if len(pm)>0: idx=pm[0]; break
    else: idx=m[0]
    mask=torch.ones(len(Pg),dtype=torch.bool); mask[idx]=False
    ps=(pdt-Pg[mask].sum()).clamp(min=gd["Pg_min"][idx],max=gd["Pg_max"][idx])
    P=Pg.clone(); P[idx]=ps; return P
