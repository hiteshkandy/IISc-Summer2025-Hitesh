
# ────────────────────────────────────────────────────────────────────────────────
# B) Lambdify for numerical use
# ────────────────────────────────────────────────────────────────────────────────
f_num = sp.lambdify(
    (A, B, C, D) + tuple(params),
    odes_simpl,
    'numpy'
)

# ────────────────────────────────────────────────────────────────────────────────
# C) Numeric simulation to verify (first 10 parameter sets)
# ────────────────────────────────────────────────────────────────────────────────
# Load first 10 rows of TS_parameters.dat
pt = "/Users/hiteshkandarpa/Desktop/IISC/Summer'25/Code/initial_sims/Toggle_tetrahedron/Hypothesis_test"
names_file  = os.path.join(pt, "TS.prs")
with open(names_file, 'r') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
param_names = [ln.split()[0] for ln in lines[1:]]
pars = pd.read_csv(
    os.path.join(pt, "TS_parameters.dat"),
    sep=r'\s+',
    header=None,
    names=["S_no","Reported_states"] + param_names,
    nrows=5
)

dt = 0.1
n_steps = 20000
tol = 1.0
n_ics = 100

results = []
for idx in trange(len(pars), desc="Sympy‐verified sim"):
    row = pars.iloc[idx]
    s_no = int(row["S_no"])
    param_vals = [float(row[str(p)]) for p in params]
    finals = []
    for _ in range(n_ics):
        # uniform IC on [0, Prod/Deg]
        X = np.array([row[f"Prod_of_{G}"]/row[f"Deg_of_{G}"] * np.random.uniform(0,1) for G in ['A','B','C','D']])
        for _ in range(n_steps):
            dX = np.array(f_num(*X, *param_vals), float)
            X += dt*dX
            X[X<0] = 0
        if not any(np.allclose(X, F, atol=tol) for F in finals):
            finals.append(X.copy())
    flat = [np.log2(v) if v>0 else -np.inf for state in finals for v in state]
    results.append([s_no, len(finals)] + flat)

max_states = max((len(r) - 2)//4 for r in results)

# build column names
cols = ["S_no","n_states"] + [
    f"{g}_ss{s+1}"
    for s in range(max_states)
    for g in genes
]

# create the DataFrame — pandas will fill missing cells with NaN for rows
# that have fewer than max_states
df_verify = pd.DataFrame(results, columns=cols)

df_verify