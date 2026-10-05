import torch
def sparse_complex_mm(Ar, Ai, x):
    sq = x.dim()==1
    xr = x.real.unsqueeze(-1) if sq else x.real
    xi = x.imag.unsqueeze(-1) if sq else x.imag
    r  = torch.sparse.mm(Ar,xr) - torch.sparse.mm(Ai,xi)
    i  = torch.sparse.mm(Ar,xi) + torch.sparse.mm(Ai,xr)
    return torch.complex(r.squeeze(-1),i.squeeze(-1)) if sq else torch.complex(r,i)
sparse_complex_mv = sparse_complex_mm
