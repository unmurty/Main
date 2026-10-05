import torch,torch.nn as nn
from models.complex_layers import ComplexGCNConv,ComplexLayerNorm,StableModReLU,ComplexResidualBlock
class PowerGNN(nn.Module):
    def __init__(self,n_buses,n_generators,hidden_features=64):
        super().__init__(); h=hidden_features
        self.c1=ComplexGCNConv(1,h); self.n1=ComplexLayerNorm(h); self.a1=StableModReLU(h)
        self.r1=ComplexResidualBlock(h); self.r2=ComplexResidualBlock(h)
        self.c2=ComplexGCNConv(h,h); self.n2=ComplexLayerNorm(h); self.a2=StableModReLU(h)
        self.vh=ComplexGCNConv(h,1)
        self.vb=nn.Parameter(torch.complex(torch.ones(n_buses,1),torch.zeros(n_buses,1)))
        self.pg=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
        self.qg=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(self,gd):
        Yr=gd["Y_real_norm"]; Yi=gd["Y_imag_norm"]; gb=gd["gen_bus"]
        x=gd["s_demand"].unsqueeze(-1)
        x=self.a1(self.n1(self.c1(x,Yr,Yi))); x=self.r1(x,Yr,Yi); x=self.r2(x,Yr,Yi)
        x=self.a2(self.n2(self.c2(x,Yr,Yi)))
        vr=(self.vh(x,Yr,Yi)+self.vb).squeeze(-1)
        # GNN clamp [0.85,1.15] — tight physical constraint
        # NR clamp [0.5,1.5] in differentiable_nr — wide numerical guardrail
        vm=0.85+0.30*torch.sigmoid(torch.abs(vr)-1.0)
        V=vm*torch.exp(1j*torch.angle(vr))
        xg=x[gb].real
        pg=gd["Pg_min"]+torch.sigmoid(self.pg(xg).squeeze(-1))*(gd["Pg_max"]-gd["Pg_min"])
        qm=(gd["Qg_max"]+gd["Qg_min"])/2.; qr=(gd["Qg_max"]-gd["Qg_min"]).clamp(min=0.)/2.
        qg=qm+torch.tanh(self.qg(xg).squeeze(-1))*qr
        return {"V":V,"Pg":pg,"Qg":qg}
