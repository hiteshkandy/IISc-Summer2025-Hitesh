# epigenetic_SATS_toggleswitch.py
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from collections import Counter
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
#  (1) FIXED PARAMETERS (from your “SATS with epigenetic A⊣B” table)
# =============================================================================

# Production / Degradation:
gA = 5.0       # production rate of A
gB = 5.0       # production rate of B
kA = 0.1       # degradation rate of A
kB = 0.1       # degradation rate of B

# Inhibitory fold-changes (λ < 1) for A⊣B and B⊣A:
lam_AB_const = 0.1   # λ_{A,B} = 0.1  (A inhibits B)
lam_BA_const = 0.1   # λ_{B,A} = 0.1  (B inhibits A)

# # of binding sites (Hill exponents) for each link:
n_AB = 1      # # of A⊣B sites
n_BA = 1      # # of B⊣A sites

# Basal thresholds A⁰_B (for B’s promoter) and B⁰_A (for A’s promoter), in molecule units:
#   These were given as 120 in your table.  (Hence 120 molecules is the “half‐point.”)
X0_AB_basal = 120.0   # threshold for A⊣B (so that A=120 yields 50% inhibition of B)
X0_BA_const = 120.0   # threshold for B⊣A

# Self‐activation fold‐changes (λ > 1):
lam_AA = 10.0    # strong self‐activation of A
lam_BB = 10.0    # strong self‐activation of B

# # of self‐activation binding sites:
n_AA = 4         # Hill exponent for A⟶A
n_BB = 4         # Hill exponent for B⟶B

# Basal thresholds for self‐activation (A⁰_A, B⁰_B):
X0_AA_basal = 80.0    # threshold for A⟶A
X0_BB_basal = 80.0    # threshold for B⟶B

# External signal S is **not** used here (we assume none).

# Epigenetic feedback parameters (will be varied):
BETA = 0.01    # relaxation rate β (same as “100 hr” in your supplementary)
#   We will sweep α₍AB₎ from 0 → 0.5, while α₍BA₎=0  (only the A⊣B link is epigenetically regulated).

# =============================================================================
#  (2) SHIFTED‐HILL DEFINITIONS
# =============================================================================
# We use the “shifted‐Hill” form exactly as in RACIPE/papers:
#
#    For an inhibitory link X --| Y  with fold‐change λ∈(0,1):
#        H_inhib(X; X0, n, λ) = λ + (1−λ)/(1 + (X/X0)^n).
#
#    For an activating link X → Y with λ > 1:
#        H_act(X; X0, n, λ)   = λ + (1−λ)/(1 + (X/X0)^n).
#
#    Note that when X=0, H_inhib(0)=1   (no inhibition, “basal=1”).
#                   H_act(0)=1          (no activation, “basal=1”).
#    When X » X0, H_inhib → λ (strong inhibition),
#                  H_act → λ   (strong activation).
#
def shifted_hill_inhib(x, X0, n, lam):
    """ Inhibitory shifted‐Hill: λ + (1−λ)/(1 + (x/X0)^n )   with λ<1. """
    return lam + (1.0 - lam) / (1.0 + (x / X0)**n)

def shifted_hill_act(x, X0, n, lam):
    """ Activating shifted‐Hill:  λ + (1−λ)/(1 + (x/X0)^n )   with λ>1. """
    # (Same algebraic form; but interpret λ>1 so H increases with x.)
    return lam + (1.0 - lam) / (1.0 + (x / X0)**n)

# =============================================================================
#  (3) ODE INTEGRATION FOR A ⊣ B TOGGLE‐SWITCH WITH EPIGENETIC FEEDBACK ON A⊣B ONLY
# =============================================================================

def simulate_one_parameter_set(args):
    """
    Simulate one “epigenetically‐regulated toggle‐switch” model, for N_INIT random
    initial conditions (A(0),B(0)), and return:
       - final_states: list of length N_INIT of (A_ss, B_ss)
       - phenotypes:   list of length N_INIT of strings "A", "B", or "A/B"
    """
    alpha_AB, M = args
    np.random.seed()   # ensure independence in each process

    final_states = []
    phenotypes   = []

    # Pre‐set the constant threshold X0_BA (B⊣A) and X0_AA, X0_BB (self‐activations):
    X0_BA = X0_BA_const
    X0_AA = X0_AA_basal
    X0_BB = X0_BB_basal

    # The **dynamic** threshold for A⊣B will be updated epigenetically each time‐step:
    #    X0_AB(t+dt) = X0_AB(t) + dt*[( A⁰_B − X0_AB(t) ) – α_AB * A(t) ] / BETA
    # where A⁰_B = X0_AB_basal is the basal threshold.

    # For each of N_INIT independent initial conditions:
    for _ in range(M["N_INIT"]):
        # 1) Sample a uniform random initial (A(0), B(0)) in [0, A_max] × [0, B_max].
        #    We choose A_max and B_max slightly above the “max possible steady‐state”
        #    We know gA/kA = 5/0.1 = 50; gB/kB = 50. We therefore sample uniformly in [0,100].
        A = np.zeros(M["TIME_STEPS"])
        B = np.zeros(M["TIME_STEPS"])
        A[0] = np.random.uniform(0.0, 100.0)
        B[0] = np.random.uniform(0.0, 100.0)

        # 2) Set the starting epigenetic threshold for A⊣B to its basal value:
        X0_AB = X0_AB_basal

        # 3) Integrate forward in time by Euler‐Maruyama (with noise):
        for t in range(M["TIME_STEPS"] - 1):
            # 3a) Epigenetic threshold update for A⊣B only:
            #       dX0_AB/dt = [ (A0B − X0_AB)  –  α_AB * A(t) ] / BETA
            X0_AB += M["DT"] * ((X0_AB_basal - X0_AB) - alpha_AB * A[t]) / BETA

            # 3b) Evaluate all six shifted‐Hill terms (two inhibitory links, two self‐activations):
            #       B⊣A  uses shifted_hill_inhib( B[t],  X0_BA,  n_BA,  lam_BA_const )
            #       A⊣B  uses shifted_hill_inhib( A[t],  X0_AB,  n_AB,  lam_AB_const )
            #       A⟶A  uses shifted_hill_act(   A[t],  X0_AA,  n_AA,  lam_AA )
            #       B⟶B  uses shifted_hill_act(   B[t],  X0_BB,  n_BB,  lam_BB )
            H_BA = shifted_hill_inhib(B[t], X0_BA, n_BA, lam_BA_const)
            H_AB = shifted_hill_inhib(A[t], X0_AB, n_AB, lam_AB_const)
            H_AA = shifted_hill_act  (A[t], X0_AA, n_AA, lam_AA)
            H_BB = shifted_hill_act  (B[t], X0_BB, n_BB, lam_BB)

            # 3c) Compute instantaneous noise amplitude σ ≃ 0.1 · mean(gA/kA, gB/kB)
            #     Since gA/kA = 50, gB/kB = 50, we have noise amplitude ~ 0.1·50 = 5.
            N_noise = 0.1 * ((gA/kA) + (gB/kB))/2.0
            noiseA  = np.random.normal(0.0, 1.0) * N_noise
            noiseB  = np.random.normal(0.0, 1.0) * N_noise

            # 3d) Finally, update A, B by Euler‐Maruyama:
            #      dA/dt = gA·H_BA·H_AA   –   kA·A
            #      dB/dt = gB·H_AB·H_BB   –   kB·B
            A[t+1] = max(
                A[t] + M["DT"] * (gA * H_BA * H_AA - kA * A[t]) + noiseA,
                0.0
            )
            B[t+1] = max(
                B[t] + M["DT"] * (gB * H_AB * H_BB - kB * B[t]) + noiseB,
                0.0
            )

        # 4) At t = T, record the final steady‐state (rounded to 3 decimal places):
        valA = np.round(A[-1], 3)
        valB = np.round(B[-1], 3)
        final_states.append((valA, valB))

        # 5) Classify phenotype “A”, “B”, or “A/B (hybrid)” based on 2/3 rule:
        if valA < (2.0/3.0) * valB:
            phenotypes.append("B")
        elif valB < (2.0/3.0) * valA:
            phenotypes.append("A")
        else:
            phenotypes.append("A/B")

    return final_states, phenotypes

# =============================================================================
#  (4) MAIN: sweep α₍AB₎ ∈ {0, 0.05, 0.10, …, 0.50}, collect “difference%” curves
# =============================================================================

if __name__ == "__main__":
    #  (a) Hyper‐parameters controlling the simulation
    N_MODELS      = 1              # We do not randomize parameters here (all fixed); so just 1 “model” 
    N_INIT        = 1000           # number of random initial‐states per α
    TIME_STEPS    = 7000           # ~70 τ = 7000 time‐steps (dt=0.01 → T=70)
    DT            = 0.01           # time‐step size
    BETA          = 0.01           # epigenetic relaxation (fixed)
    ALPHA_AB_list = np.linspace(0.0, 0.50, 11)    # 11 points from 0→0.5

    # Pack into dictionary M for convenience:
    M = {
        "N_INIT":    N_INIT,
        "TIME_STEPS": TIME_STEPS,
        "DT":        DT
    }

    # (b) For each α₍AB₎, run the single “parameter set” simulation
    results_by_alpha = {}
    for alpha_AB in ALPHA_AB_list:
        # We only need to run one “parameter set” because all other parameters are fixed
        args = (alpha_AB, M)
        # simulate 1×N_INIT(=1000) initial conditions, get back lists of length=1000
        final_states, phenotypes = simulate_one_parameter_set(args)

        # (c) Count how many ended in “A”, “B”, “A/B” categories:
        cnt = Counter(phenotypes)
        nA      = cnt["A"]
        nB      = cnt["B"]
        nHybrid = cnt["A/B"]
        total   = float(N_INIT)

        # Compute "difference%" = 100·(nA − nB)/N_INIT
        diff_percent = 100.0 * (nA - nB) / total

        # Store for plotting:
        results_by_alpha[alpha_AB] = (nA, nB, nHybrid, diff_percent)

        print(f" α={alpha_AB:.2f}  →  A={nA:4d},  B={nB:4d},  A/B={nHybrid:4d},  diff%={diff_percent:6.1f}%")

    # =============================================================================
    #  (5) PLOT  I: difference% vs α₍AB₎ (solid‐line), with a red guide‐line at 0%.
    # =============================================================================
    x_vals = ALPHA_AB_list
    y_diff = [results_by_alpha[a][3] for a in x_vals]

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, y_diff, "o-", color="tab:blue", lw=2, label="(nA − nB)/1000 × 100%")
    plt.axhline(0.0, color="gray", ls="--", lw=1.5)
    plt.title("Epigenetic SATS: % (A−B) vs α₍AB₎")
    plt.xlabel(r"Epigenetic strength $\alpha_{AB}$")
    plt.ylabel("Difference % = 100·(n_A − n_B)/1000")
    plt.ylim(-100, +100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =============================================================================
    #  (6) PLOT  II: Bar‐plot of (n_A, n_B, n_Hybrid) vs α₍AB₎ as three stacked bars
    # =============================================================================
    width = 0.04
    idxs  = np.arange(len(x_vals))

    nA_list      = [results_by_alpha[a][0] for a in x_vals]
    nB_list      = [results_by_alpha[a][1] for a in x_vals]
    nHyb_list    = [results_by_alpha[a][2] for a in x_vals]

    plt.figure(figsize=(8,5))
    plt.bar(idxs, nA_list,      width, color="tab:green", label="A‐dominant")
    plt.bar(idxs, nHyb_list,    width, bottom=nA_list, color="tab:gray",  label="Hybrid")
    bottom_for_B = np.array(nA_list) + np.array(nHyb_list)
    plt.bar(idxs, nB_list,      width, bottom=bottom_for_B, color="tab:red",   label="B‐dominant")

    plt.xticks(idxs, [f"{a:.2f}" for a in x_vals])
    plt.xlabel(r"$\alpha_{AB}$")
    plt.ylabel("Count (out of 1000 ICs)")
    plt.title("Epigenetic SATS: A/B/Hybrid counts vs α₍AB₎")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # =============================================================================
    #  (7) PLOT  III: PCA of all 2,000 steady‐states at two selected α‐values (e.g. 0.00 vs 0.50)
    # =============================================================================
    #    We’ll pick α=0.00 and α=0.50 as examples, collect their (A_ss, B_ss), do PCA→2D.
    # =============================================================================
    for alpha_AB in [0.00, 0.50]:
        # rerun the simulation just for collecting “flat_states”
        final_states, phenotypes = simulate_one_parameter_set((alpha_AB, M))
        flat = np.vstack(final_states)   # shape=(1000,2)

        coords = PCA(n_components=2).fit_transform(flat)
        colmap = np.array([ "tab:green" if ph=="A"
                              else "tab:red" if ph=="B"
                              else "tab:gray"
                              for ph in phenotypes ])

        plt.figure(figsize=(5,5))
        plt.scatter(coords[:,0], coords[:,1], c=colmap, s=20, alpha=0.6)
        plt.title(f"PCA of Steady‐States at α={alpha_AB:.2f}")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

    # =============================================================================
    #  (8) PLOT  IV: Heatmap of steady‐states clustered into three clusters for α=0.50
    # =============================================================================
    alpha_plot = 0.50
    final_states, phenotypes = simulate_one_parameter_set((alpha_plot, M))
    flat_states = np.vstack(final_states)   # shape=(1000,2)

    # Cluster into 3 (A, B, hybrid)
    labels = KMeans(n_clusters=3, random_state=0).fit_predict(flat_states)
    order  = np.argsort(labels)

    plt.figure(figsize=(5,7))
    plt.imshow(flat_states[order], aspect="auto", cmap="viridis")
    plt.colorbar(label="Expression Level")
    plt.title(f"Heatmap @ α={alpha_plot:.2f} (clustered)")
    plt.xlabel("Gene Index (0=A, 1=B)"); plt.ylabel("State Index (sorted)")
    plt.tight_layout()
    plt.show()
