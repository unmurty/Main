import torch,torch.nn as nn,torch.nn.functional as F,math
from utils.sparse_complex import sparse_complex_mm
class StableModReLU(nn.Module):
    def __init__(self,f,eps=1e-6):
        super().__init__(); self.bias=nn.Parameter(torch.zeros(1,f)); self.eps=eps
    def forward(self,z): m=torch.abs(z); return F.relu(m+self.bias)/(m+self.eps)*z
class ComplexLayerNorm(nn.Module):
    def __init__(self,f,eps=1e-6):
        super().__init__(); self.gamma=nn.Parameter(torch.ones(1,f)); self.eps=eps
    def forward(self,z):
        mu=z.mean(0,keepdim=True)
        std=torch.sqrt((torch.abs(z-mu)**2).mean(0,keepdim=True)+self.eps)
        return self.gamma*(z-mu)/std
class ComplexGCNConv(nn.Module):
    def __init__(self,inf,outf,bias=True):
        super().__init__(); sc=math.sqrt(2./(inf+outf))
        self.Wr=nn.Parameter(torch.randn(inf,outf)*sc)
        self.Wi=nn.Parameter(torch.randn(inf,outf)*sc)
        self.bias=nn.Parameter(torch.zeros(outf,dtype=torch.complex64)) if bias else None
    def forward(self,x,Yr,Yi):
        h=torch.complex(x.real@self.Wr-x.imag@self.Wi,x.real@self.Wi+x.imag@self.Wr)
        out=sparse_complex_mm(Yr,Yi,h)
        return out+(self.bias if self.bias is not None else 0)
class ComplexResidualBlock(nn.Module):
    def __init__(self,f):
        super().__init__()
        self.conv=ComplexGCNConv(f,f); self.norm=ComplexLayerNorm(f); self.act=StableModReLU(f)
    def forward(self,x,Yr,Yi): return self.act(self.norm(self.conv(x,Yr,Yi)))+x
