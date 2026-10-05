import torch
def compute_branch_flows(V,gd):
    f=gd["f_bus"]; t=gd["t_bus"]; Vf=V[f]; Vt=V[t]
    return Vf*torch.conj(gd["Yff"]*Vf+gd["Yft"]*Vt), Vt*torch.conj(gd["Ytf"]*Vf+gd["Ytt"]*Vt)
