import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import trange

# 1) Load parameter names and parameter file (first 100 rows for testing)
pt = "/Users/hiteshkandarpa/Desktop/IISC/Summer'25/Code/initial_sims/Toggle_tetrahedron/Hypothesis_test"
params_file = os.path.join(pt, "TS_parameters.dat") # RACIPE parameter values
names_file  = os.path.join(pt, "TS.prs") # RACIPE parameter names

# Read TS.prs, ignore first header line, take only the first token each line
with open(names_file, 'r') as f:
    lines       = [ln.strip() for ln in f if ln.strip()]
param_names = [ln.split()[0] for ln in lines[1:]]

# Read parameters: col0=S_no, col1=Reported_states, cols2+ = kinetic values
pars = pd.read_csv(
    params_file,
    delim_whitespace=True,
    header=None,
    names=["S_no", "Reported_states"] + param_names,
    nrows=10 #change number of rows - number of parameter sets
)

# Extract gene list ['A','B','C','D'] from "Prod_of_<Gene>"
genes = ['A', 'B', 'C', 'D']
ng   = len(genes)
gene_index = {g:i for i, g in enumerate(genes)}

# 2) Precompute parameter‐arrays (Prod, Deg, lam, K, n) for one row
def extract_param_arrays(row):
    p = row[param_names]
    Prod = p[[f"Prod_of_{G}" for G in genes]].astype(float).values
    Deg  = p[[f"Deg_of_{G}"  for G in genes]].astype(float).values
    lam = np.zeros((ng,ng))
    K   = np.zeros((ng,ng))
    nco = np.zeros((ng,ng))
    np.fill_diagonal(K, 1.0)
    np.fill_diagonal(nco, 0.0)
    for H in genes:
        for G in genes:
            if H == G:
                continue
            i, j = gene_index[H], gene_index[G]
            lam[i,j] = float(p[f"Inh_of_{H}To{G}"])
            K[i,j]   = float(p[f"Trd_of_{H}To{G}"])
            nco[i,j] = float(p[f"Num_of_{H}To{G}"])
    return Prod, Deg, lam, K, nco

# 3) Vectorized RHS for all ICs at once
def ode_rhs_vector(Xs, Prod, Deg, lam, K, nco):
    ratio = Xs[:,:,None] / K[None,:,:]
    # print(K)  # shape (n_ics, ng, ng)
    hill  = 1.0 / (1.0 + ratio**(nco[None,:,:]))
    factors = (1 - lam)[None,:,:] * hill + lam[None,:,:]
    prod    = Prod[None,:] * np.prod(factors, axis=1)
    dXs     = prod - Deg[None,:] * Xs
    return dXs

# 4) Vectorized Euler integrator
def integrate_euler_vectorized(Prod, Deg, lam, K, nco,
                               n_ics=100, dt=0.1, n_steps=20000):
    """
    Xs initial conditions: each X0_{i,G} ~ Uniform(0, Prod[G]/Deg[G])
    """
    # uniform on [0,1), then scale by Prod/Deg
    scale = (Prod[None, :] / Deg[None, :])  # shape (1, ng)
    Xs = np.random.uniform(0, 1, size=(n_ics, ng)) * scale

    for _ in range(n_steps):
        Xs += dt * ode_rhs_vector(Xs, Prod, Deg, lam, K, nco)
        Xs[Xs < 0] = 0
    return Xs


# 5) Find unique steady‐states by clustering final rows
def find_steady_states_vectorized(row,
                                  n_ics=100,
                                  dt=0.1,
                                  n_steps=20000,
                                  tol_cluster=1.0):
    Prod, Deg, lam, K, nco = extract_param_arrays(row)
    Xs = integrate_euler_vectorized(Prod, Deg, lam, K, nco,
                                     n_ics=n_ics, dt=dt, n_steps=n_steps)
    finals = []
    for xf in Xs:
        if not any(np.allclose(xf, f, atol=tol_cluster) for f in finals):
            finals.append(xf)
    return finals

# 6) Simulate all parameter sets and write RACIPE‐style output
output_file = os.path.join(pt, "simulated_steady_states_vectorized.dat")
with open(output_file, 'w') as fout:
    for idx in trange(len(pars), desc="Vectorized sim"):
        row = pars.iloc[idx]
        prodA = float(row["Prod_of_A"])      
        # Use the exact S_no from the parameter file
        s_no = int(row["S_no"])
        ss   = find_steady_states_vectorized(row,
                                              n_ics=100,
                                              dt=0.1,
                                              n_steps=20000,
                                              tol_cluster=1.0)
        # Log2 transform, handle non-positive
        flattened = [np.log2(v) if v>0 else -np.inf for st in ss for v in st]
        # Compose line: [S_no, number of states, log2 steady‐states...]
        line = [s_no, prodA, len(ss)] + flattened
        fout.write("\t".join(f"{x:.6g}" for x in line) + "\n")

