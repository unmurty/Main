import torch
def compute_power_flow_loss(dP,dQ):
    return torch.mean(dP.abs())+torch.mean(dQ.abs())
def compute_voltage_loss(Vm,vmin=0.95,vmax=1.05):
    return torch.mean(torch.relu(vmin-Vm)**2+torch.relu(Vm-vmax)**2)
def compute_generator_limit_loss(Pg,Qg,gd):
    pl=torch.relu(gd["Pg_min"]-Pg)+torch.relu(Pg-gd["Pg_max"])
    ql=torch.relu(gd["Qg_min"]-Qg)+torch.relu(Qg-gd["Qg_max"])
    return torch.mean(pl**2+ql**2)
def compute_thermal_violations(Sf,St,rA):
    return torch.mean(torch.relu(Sf.abs()-rA)**2+torch.relu(St.abs()-rA)**2)
def compute_generation_cost(Pg,gd):
    Pmw=Pg*gd["baseMVA"]
    return torch.sum(gd["gencost_c2"]*Pmw**2+gd["gencost_c1"]*Pmw+gd["gencost_c0"])
