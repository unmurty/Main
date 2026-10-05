cvnn_final/
  train.py                   ← unified trainer, all grids
  data/
    matpower_parser.py       ← non-sequential bus IDs, inf-limit clamp
    ybus_builder.py          ← CSR sparse, correct bus shunt /baseMVA
    cases/  case118.m  case1354pegase.m  case13659pegase.m  case_ACTIVSg10k.m
  models/
    complex_layers.py        ← Wirtinger ComplexGCNConv, CSR-native
    cvgnn.py                 ← PowerGNN, bounded |V| [0.7,1.3]
  physics/
    differentiable_nr.py     ← fractional polar NR, vectorised Jacobian
    slack_balance.py         ← handles no-gen-at-slack (PEGASE/ACTIVSg)
    branch_flow.py           ← per-branch Yff/Yft tensors
  losses/
    loss_function.py         ← MAE for PF (avoids float32 overflow)
  utils/
    sparse_complex.py        ← CSR-native sparse_complex_mm
	
	train.py
  │
  ├─ imports at module level:
  │    from data.grid_partitioner    import partition_grid, assemble_global_voltages, get_stiff
  │    from data.adaptive_partitioner import auto_partition, analyse_grid
  │
  ├─ if USE_DIAKOPTICS:  (n > partition_thresh, default 500)
  │    │
  │    ├─ --auto-partition flag set:
  │    │    auto_partition(gd, user_partition_size=None)
  │    │    → analyse_grid() prints topology metrics
  │    │    → tests candidates 2,3,4...N parts analytically
  │    │    → runs NR verification on each passing candidate
  │    │    → returns optimal subgrids + recommended size
  │    │
  │    └─ --partition-size 500 (default):
  │         auto_partition(gd, user_partition_size=500)
  │         → verifies your choice against 3 analytical checks
  │         → warns if suboptimal and prints better alternative
  │         → still uses your size (you stay in control)

    cd cvnn_final
python train.py --case data/cases/case118.m  --epochs 500 --nr-steps 1 --hidden 64 --lr 1e-3 --log-every 20 --save-every 100
python train.py --case data/cases/case1354pegase.m --epochs 200 --nr-steps 2 --hidden 64 --lr 5e-4

 --case data/cases/case118.m --epochs 300 --nr-steps 1 --resume cvgnn.pt --save cvgnn_500ep.pt
 --case data/cases/case118.m --epochs 200 --nr-steps 1 --save cvgnn.pt
 --case data/cases/case118.m --epochs 300 --nr-steps 5 --resume cvgnn.pt --save cvgnn_300ep.pt
 --case data/cases/case118.m --epochs 400 --nr-steps 10 --resume cvgnn300ep.pt --save cvgnn_400ep.pt
 --case data/cases/case_ACTIVSg70k.m --epochs 300 --nr-steps 2 --partition-size 10000 --lr 1e-4 --nr-start 100
python physics/debug_bus.py data/cases/case_ACTIVSg10k.m
METIS Usage
  pip install metis
C:\Users\sarva\Documents\METIS_fixed\home\claude\metis_src\METIS\build
rmdir /s /q build 
cmake .. -DSHARED=ON -DCMAKE_BUILD_TYPE=Release ^  -DCMAKE_INSTALL_PREFIX=C:\metis_install ^  -G "MinGW Makefiles"
cmake --build . --parallel 
cmake --install .  
creates metis.dll
set METIS_DLL=C:\metis_install\bin\metis.dll

# Auto-select optimal partition (recommended for new grids)
python train.py --case data/cases/case1354pegase.m 
  --epochs 200 --nr-steps 1 --hidden 64 --lr 5e-4  --partition-thresh 10000
  --partition-size 700 --nr-start 100

# Manual with verification warning

  --partition-size 500
# → prints "User partition-size=500 passed checks" or warns with better suggestion
--case data/cases/case13659pegase.m 
  --epochs 200 --nr-steps 1 --hidden 64 --lr 2e-4 
   --partition-size 700
# 13659-bus auto
python train.py --case data/cases/case13659pegase.m \
  --epochs 200 --nr-steps 1 --hidden 64 --lr 5e-4  --partition-thresh 10000 --nr-start 100
 ./.venv/bin/python3 -m pip install
 
  Install .venv 
 ./.venv/bin/python3 -m pip install --upgrade pip
  ./.venv/bin/python3 -m pip install torch pymetis scipy numpy matplotlib pandapower
   source ./.venv/bin/activate 


./.venv/bin/python3 -m pip install pymetis  pandapower

 
 python3 -m pip install --upgrade numpy
 ./venv/bin/python -m pip install --force-reinstall "numpy==1.26.4"


 ./.venv/bin/python -m pip install --force-reinstall "numpy==1.26.4"./


 /usr/bin/env /Users/murtyuppuluri/Documents/PythonProjects/cvnn_pkg15/.venv/bin/python3 /Users/murtyuppuluri/.vscode/extensions/ms-python.debugpy-2026.6.0-darwin-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 63399 -- /Users/murtyuppuluri/Documents/PythonProjects/cvnn_pkg13/train.py --case data/cases/case_ACTIVSg10k.m --epochs 200 --nr-steps 2 --partition-size 200 --lr 1e-4 

 .

    The input graph is given in either a Pythonic way as the *adjacency* parameter
    or in the direct C-like way that Metis likes as *xadj* and *adjncy*. It
    is an error to specify both graph inputs.

    The Pythonic graph specifier *adjacency* is required to have the following
    properties:

    - len(adjacency) needs to return the number of vertices
    - ``adjacency[i]`` needs to be an iterable of vertices adjacent to vertex i.
      Both directions of an undirected graph edge are required to be stored.

    If you would like to use *eweights* (edge weights), you need to use the
    xadj/adjncy way of specifying graph connectivity. This works as follows:

        The adjacency structure of the graph is stored as follows: The
        adjacency list of vertex *i* is stored in array *adjncy* starting at
        index ``xadj[i]`` and ending at (but not including) index ``xadj[i +
        1]``. That is, for each vertex i, its adjacency list is stored in
        consecutive locations in the array *adjncy*, and the array *xadj* is
        used to point to where it begins and where it ends.

        The weights of the edges (if any) are stored in an additional array
        called *eweights*. This array contains *2m* elements (where *m* is the
        number of edges, taking into account the undirected nature of the
        graph), and the weight of edge ``adjncy[j]`` is stored at location
        ``eweights[j]``. The edge-weights must be integers greater than zero. If
        all the edges of the graph have the same weight (i.e., the graph is
        unweighted), then the eweight can be set to ``None``.
1. buses_ok
2. bndMW
3. KCL p99 / max
4. voltage validity
5. global_nr_rms
ep
globalNR
bndMW
KCL_p99
KCL_max
lowVm
busesOK
worstBus
    METIS runtime options can be specified by supplying an :class:`Options` object in
    the input.

    (quoted with slight adaptations from the Metis docs)
    global_nr_rms Range	Interpretation
< 1.0	Excellent — NR fully converged, solution is AC-feasible
1.0 – 10	Good — minor residuals, acceptable for most OPF applications
10 – 50	Moderate — GNN warm-start is reasonable but NR needs more steps
50 – 200	Poor — NR barely correcting, likely curriculum or gradient issue
> 200	Bad — NR diverging or stiff branch dominance
For the current stage — yes, keep `kcl_mw_max` in diagnostics but remove it from the per-epoch log line. It's useful for debugging specific problem buses but noisy as a training metric since a single stiff bus can make it look catastrophic even when 99.9% of buses are fine.

**Keep in diagnostics.py** — compute it, store in hist, report in FINAL summary.

**Remove from per-epoch print** — it's distracting and `kcl_p99`/`kcl_p999` tell a cleaner story.

**Benchmark comparison metrics — what actually matters:**

| Metric | Keep in log | Why |
|---|---|---|
| `global_nr_rms` | ✓ PRIMARY | Direct convergence measure |
| `kcl_p99` | ✓ | 99% of buses — meaningful system health |
| `partNR` | ✓ | Local convergence quality |
| `bnd_mw` | ✓ | Boundary reconciliation health |
| `buses_ok` | ✓ | Voltage compliance count |
| `kcl_mw_max` | ✗ from log | Single bus outlier — misleads |
| `kcl_p999` | optional | Only needed when chasing tail buses |
| `nan_buses` | ✓ SAFETY | Must always show |
| `n_lowVm7` | ✓ SAFETY | Must always show |
| `stiff` | once at startup | Doesn't change during training |

**For benchmarking against other OPF solvers** (MATPOWER, IPOPT), the standard comparison metrics are:
- `global_nr_rms` vs solver KKT residual
- `buses_ok` (V in [0.95, 1.05]) vs solver feasibility
- `KCL_p99` vs solver constraint violation
- Inference time (ms) vs solver wall time

`KCL_max` is not a standard benchmark metric — it's a debugging tool. Remove it from the log line, keep it in FINAL summary with a note that it's dominated by stiff buses.

cluster-based step clipping is already controlling Δθ and ΔV. Damping and clipping then have distinct roles:

Damping → controls how much of the Newton correction is applied globally.
Cluster clipping → limits individual angle/voltage corrections.
Stiff-pair projection → enforces local voltage consistency for pathological high-|Y| branches.
No global trust radius → avoids the previous 10k-dimensional L2 suppression problem.

IEEE PES General Meeting 2027 — typically abstract due October, full paper January. Ideal for the 70k scalability result and Diakoptics architecture.

IEEE Transactions on Power Systems — rolling submission, 3-6 month review. The right venue for the full methodology paper with the KCL residual analysis and zero-injection hub finding.

NeurIPS 2026 Machine Learning for Physical Sciences Workshop — typically due September. Good fit for the differentiable NR + implicit differentiation angle.

Near-term after fixes:

IEEE PESGM 2027 / PowerTech 2027 — after implementing zero-injection bus Kron reduction and angle-consistency regularization, the KCL floor drops from 30 MW to near-zero, making the result much stronger for publication.

ICLR 2027 (Energy Track) — if you add topology transfer results across multiple grids
Conference priority

For our work I would currently rank:

Priority	Venue	Deadline	Fit
1	IEEE PES General Meeting 2027	Nov 10	⭐⭐⭐⭐⭐
2	ACDC Global 2027	Oct 2	⭐⭐⭐⭐½
3	IEEE PES International Meeting 2027	Sep 15	⭐⭐⭐⭐⭐ but extremely tight
4	PEEE 2027	Check CFP	⭐⭐⭐
5	other generic AI/energy conferences	varies	⭐⭐–⭐⭐⭐