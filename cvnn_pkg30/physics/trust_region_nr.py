import numpy as np
def clip_step(dx, trust_radius):
    norm_dx=np.linalg.norm(dx)
    if norm_dx>trust_radius: dx=dx*(trust_radius/norm_dx)
    return dx, norm_dx
def clip_nr_step_cluster(
    dx,
    n_a,
    n_v,
    aa,
    av,
    stiff_pairs=None,
    g2l=None,
    angle_max=0.05,
    vm_rel_max=0.02,
    stiff_angle_max=0.02,
    stiff_vm_rel_max=0.01,
):
    dx = np.asarray(dx, dtype=np.float64).copy()

    da = dx[:n_a]
    dv = dx[n_a:n_a + n_v]

    # Normal buses
    da[:] = np.clip(da, -angle_max, angle_max)
    dv[:] = np.clip(dv, -vm_rel_max, vm_rel_max)

    # ------------------------------------------------------------
    # Tighten only buses participating in stiff internal branches
    # ------------------------------------------------------------
    if stiff_pairs and g2l is not None:

        stiff_global = set()

        for gi, gj in stiff_pairs:
            stiff_global.add(int(gi))
            stiff_global.add(int(gj))

        for k, gi in enumerate(aa):
            if int(gi) in stiff_global:
                da[k] = np.clip(
                    da[k],
                    -stiff_angle_max,
                    stiff_angle_max,
                )

        for k, gi in enumerate(av):
            if int(gi) in stiff_global:
                dv[k] = np.clip(
                    dv[k],
                    -stiff_vm_rel_max,
                    stiff_vm_rel_max,
                )

    dx[:n_a] = da
    dx[n_a:n_a + n_v] = dv

    return dx, np.linalg.norm(dx)
def update_trust_radius(rho,norm_dx,trust_radius,eta1=0.25,eta2=0.75,
                        gamma1=0.25,gamma2=2.0,delta_min=1e-6,delta_max=10.0):
    if rho<eta1: return max(gamma1*norm_dx,delta_min),False
    elif rho>=eta2: return min(gamma2*trust_radius,delta_max),True
    return trust_radius,True
class TrustRegionState:
    def __init__(self,trust_radius=1.0):
        self.radius=trust_radius; self.n_accept=0; self.n_reject=0
