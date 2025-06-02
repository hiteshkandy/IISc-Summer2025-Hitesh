import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from collections import Counter
import itertools

# Import PCA and KMeans only if needed later; for Figures 2 and 3 they are not required
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# Import your RACIPE class definition (adjust path as needed)
import sys
sys.path.append("/Users/hiteshkandarpa/Desktop/IISC/Summer'25/Code/initial_sims")
from RACIPE_definition import RACIPE

# ===================== Parameter Definitions =====================
# Number of random parameter sets and initial conditions per set
N_MODELS      = 100    # Number of random parameter sets
N_INIT_SWITCH = 20    # Initial conditions per parameter set

# Integration parameters
TIME_STEPS  = 40000  # Number of integration steps (per trajectory)
DT_SWITCH   = 0.01   # Time step size (τ)

# Epigenetic feedback strengths (we will sweep α_AB)
ALPHA_BA    = 0.0    # No epigenetic feedback on B -> A
BETA        = 0.01   # Epigenetic relaxation rate (1/β = 100 hours)

# RACIPE sampling parameters (for threshold‐median estimation)
M_SAMPLES   = 200   # Samples for estimating median thresholds

# Two‐gene toggle‐switch topology (self‐activation loops omitted here)
GENES    = ['A', 'B']
TOPOLOGY = {
    'genes': GENES,
    'regulators': {
        'A': [('A', 'act'), ('B', 'inh')],
        'B': [('B', 'act'), ('A', 'inh')],
    }
}


# ==============================================================================
# 4) Estimate each gene’s median (G/k) via RACIPE so that we can sample X₀ properly
# ==============================================================================
# Create a tiny RACIPE instance (1 initial condition, 1 time unit) just to estimate medians
racipe_tmp = RACIPE(TOPOLOGY, n_init=1, t_max=1.0, dt=0.1)
M = {
    g: racipe_tmp._estimate_X0_median(n_samples=M_SAMPLES)
    for g in GENES
}
# Now M['A'] ≈ median(G_A / k_A) and M['B'] ≈ median(G_B / k_B).


# ==============================================================================
# 5) Define the “simulate_toggle” function for one SATS parameter set + epigenetic α_AB
#
#   - Strong self‐activation on A & B (fixed λ_self = 10, n_self = 4, X₀_self = M[g])
#   - Random mutual inhibition parameters (λ_inh ∈ [1/100, 1/1], n_inh ∈ [1..6], 
#     X₀_inh ∈ Uniform(0.02 M, 1.98 M))
#   - Epigenetic feedback only on A⟶B (   dX₀^{AB}/dt = [ (X₀^{AB,basal} - X₀^{AB}) - α_AB·A ] / β  )
# ==============================================================================
def simulate_toggle(args):
    """
    Simulates N_INIT_SWITCH trajectories for one random‐parameter SATS toggle,
    with a given α_AB. Returns a list of final steady‐states (A_ss, B_ss) and a
    list of phenotype labels ('A', 'B', 'A/B') for each trajectory.
    """
    idx, M, alpha_AB = args
    np.random.seed()  # ensures each worker is different

    final_states = []
    phenotypes   = []

    for _ in range(N_INIT_SWITCH):
        # --------------------------------------------------------------
        # 5.1) Sample **intrinsic** kinetics for A & B (RACIPE style)
        # --------------------------------------------------------------
        # Sample basal production gA, gB ∈ Uniform(1, 100)
        gA, gB = np.random.uniform(1, 100, 2)

        # Sample degradation kA, kB = 1 / Uniform(1, 100)
        kA = 1.0 / np.random.uniform(1, 100)
        kB = 1.0 / np.random.uniform(1, 100)

        # --------------------------------------------------------------
        # 5.2) Sample **basal thresholds** X₀^(basal) by “half‐functional” rule
        # (Uniform(0.02 M, 1.98 M)) for each link that is not self‐activation
        # --------------------------------------------------------------
        x0_AA = M['A']   # We fix self‐activation X₀_self = M['A']
        x0_BB = M['B']   # We fix self‐activation X₀_self = M['B']

        x0_BA = np.random.uniform(0.02 * M['B'], 1.98 * M['B'])  # B⟶A (inhibitory) basal
        x0_AB = np.random.uniform(0.02 * M['A'], 1.98 * M['A'])  # A⟶B (inhibitory) basal

        # We'll use “x0_AB_dynamic” below to update epigenetically;
        # but store the “basal” value separately to reference:
        x0_AB_basal = x0_AB

        # --------------------------------------------------------------
        # 5.3) Sample Hill exponents (n) & fold‐changes (λ) for each link
        #        - For **inhibition**: λ_inh = 1/Uniform(1,100)
        #        - For **self‐activation**: we force λ_self = 10  (strong)
        # --------------------------------------------------------------
        n_AA = 4
        lam_AA = 10.0

        n_BB = 4
        lam_BB = 10.0

        # Inhibitory exponents and lambdas (random)
        n_BA = np.random.randint(1, 7)
        lam_BA = 1.0 / np.random.uniform(1, 100)

        n_AB = np.random.randint(1, 7)
        lam_AB = 1.0 / np.random.uniform(1, 100)

        # --------------------------------------------------------------
        # 5.4) Initialize A(t), B(t), and dynamic threshold A0B
        # --------------------------------------------------------------
        A = np.zeros(TIME_STEPS)
        B = np.zeros(TIME_STEPS)
        A[0], B[0] = np.random.rand(), np.random.rand()

        # Start dynamic threshold A0B at its basal
        A0B = x0_AB_basal

        # --------------------------------------------------------------
        # 5.5) Time integration (Euler–Maruyama) for t = 0 … TIME_STEPS−2
        #    dA/dt = gA·H_AA(A)·H_BA(B) − kA·A
        #    dB/dt = gB·H_BB(B)·H_AB(A; A0B) − kB·B
        #  where:
        #    H_AA(A) = λ_AA + (1 − λ_AA) / [1 + (A / X0_AA)^n_AA]   (self‐act on A)
        #    H_BB(B) = λ_BB + (1 − λ_BB) / [1 + (B / X0_BB)^n_BB]   (self‐act on B)
        #    H_BA(B) = λ_BA + (1 − λ_BA) / [1 + (B / X0_BA)^n_BA]   (B ⟶ A inh)
        #    H_AB(A) = λ_AB + (1 − λ_AB) / [1 + (A / A0B)^n_AB]     (A ⟶ B inh with dynamic A0B)
        #
        #  Epigenetic update on A0B: dA0B/dt = [ (x0_AB_basal − A0B) − α_AB·A ] / β
        #
        #  Additive Gaussian noise: σ = 0.05 · mean(G/k).  (SATS used ≈0.05×(G/k).)
        # --------------------------------------------------------------
        for t in range(TIME_STEPS - 1):
            # 1) Epigenetic threshold update (on A⟶B only)
            dA0B_dt = ((x0_AB_basal - A0B) - alpha_AB * A[t]) / BETA
            A0B += DT_SWITCH * dA0B_dt

            # 2) Compute shifted‐Hill for each regulatory link
            H_AA = lam_AA + (1.0 - lam_AA) / (1.0 + (A[t] / x0_AA) ** n_AA)
            H_BB = lam_BB + (1.0 - lam_BB) / (1.0 + (B[t] / x0_BB) ** n_BB)

            H_BA = lam_BA + (1.0 - lam_BA) / (1.0 + (B[t] / x0_BA) ** n_BA)
            H_AB = lam_AB + (1.0 - lam_AB) / (1.0 + (A[t] / A0B) ** n_AB)

            # 3) Compute noise scale ≈ 0.05·mean(G/k)
            noise_scale = 0.05 * np.mean([gA / kA, gB / kB])
            noiseA = np.random.normal(0, 1) * noise_scale
            noiseB = np.random.normal(0, 1) * noise_scale

            # 4) Euler–Maruyama updates (A, B)
            dA = gA * H_AA * H_BA - kA * A[t]
            dB = gB * H_BB * H_AB - kB * B[t]

            A[t+1] = max(A[t] + DT_SWITCH * dA + noiseA, 0.0)
            B[t+1] = max(B[t] + DT_SWITCH * dB + noiseB, 0.0)

        # --------------------------------------------------------------
        # 5.6) Record final steady‐states (rounded for clarity) and phenotype
        # --------------------------------------------------------------
        valA = round(A[-1], 3)
        valB = round(B[-1], 3)
        final_states.append((valA, valB))

        # Classify “dominance”: if one gene < (4/5)·other → that other is dominant;
        # otherwise call it hybrid “A/B”.
        if valA < (4.0 / 5.0) * valB:
            phenotypes.append('B')
        elif valB < (4.0 / 5.0) * valA:
            phenotypes.append('A')
        else:
            phenotypes.append('A/B')

    return final_states, phenotypes



# ========================== Main Execution ==========================
if __name__ == '__main__':
    # 1) Choose a list of alpha_AB values to sweep over
    alpha_list = [0.00, 0.15, 0.35, 0.5]  # e.g., [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Pre-allocate lists to store summary statistics at each alpha
    fracA_list  = []    # fraction of runs that end A-dominant (excludes hybrids if desired)
    meanA_list  = []    # mean of all A_ss across the entire ensemble
    stdA_list   = []    # std of all A_ss across the entire ensemble

    # To build histograms later, store all A_ss for each alpha in a dict
    full_A_ss = {}  # key = alpha, value = list of all A_ss (length = N_MODELS * N_INIT_SWITCH)

    # 2) Parallel epigenetic‐toggle‐switch simulations for each alpha
    #    Prepare argument list for Pool.map
    #    Each tuple = (model_index, M, alpha_value)
    for alpha_val in alpha_list:
        args = [(i, M, alpha_val) for i in range(N_MODELS)]
        with Pool(cpu_count()) as pool:
            results = pool.map(simulate_toggle, args)

        # 3) Unpack results for this alpha
        #    results is a list of length N_MODELS, each entry = (final_states_list, phenotypes_list)
        all_final_states = list(itertools.chain.from_iterable(r[0] for r in results))
        all_phenotypes   = list(itertools.chain.from_iterable(r[1] for r in results))

        # Extract just the A_ss values for histogram and statistics
        A_values = [a_ss for (a_ss, b_ss) in all_final_states]
        full_A_ss[alpha_val] = A_values

        # Compute fraction of “A-dominant” phenotypes (exclude hybrids if you wish; here we count only strict 'A')
        count_A     = sum(1 for ph in all_phenotypes if ph == 'A')
        count_total = sum(1 for ph in all_phenotypes if ph in ['A', 'B'])  # exclude 'A/B'
        fracA = (count_A / count_total) if (count_total > 0) else 0.0
        fracA_list.append(fracA)

        # Compute mean and std of all A_ss (including hybrids; you may filter if desired)
        meanA_list.append(np.mean(A_values))
        stdA_list.append(np.std(A_values))

    # Convert lists to numpy arrays for plotting
    alpha_arr = np.array(alpha_list)
    fracA_arr = np.array(fracA_list)
    meanA_arr = np.array(meanA_list)
    stdA_arr  = np.array(stdA_list)

    # ===================== Figure 1: Fraction A‐Dominant vs α₍AB₎ =====================
    plt.figure(figsize=(6, 4))
    plt.plot(alpha_arr, fracA_arr, marker='o', color='tab:blue', linestyle='-')
    plt.title("Fraction of A‐Dominant vs α₍AB₎")
    plt.xlabel("α₍AB₎")
    plt.ylabel("Fraction A‐Dominant")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===================== Figure 2: ⟨A_ss⟩ ± σ vs α₍AB₎ (Bifurcation‐style) =====================
    plt.figure(figsize=(6, 4))
    plt.errorbar(alpha_arr, meanA_arr, yerr=stdA_arr, marker='s',
                 color='tab:red', ecolor='lightcoral', capsize=4, linestyle='-')
    plt.title("Mean A_ss ± SD vs α₍AB₎")
    plt.xlabel("α₍AB₎")
    plt.ylabel("Mean Steady‐State A")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===================== Figure 3: Histograms of A_ss at Selected α₍AB₎ =====================
    # We choose four representative α values for the 2×2 grid
    selected_alphas = [0.00, 0.10, 0.20, 0.30]
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

    for ax, alpha_val in zip(axs.flat, selected_alphas):
        A_vals = full_A_ss[alpha_val]
        ax.hist(A_vals, bins=30, color='tab:purple', alpha=0.75, edgecolor='k')
        ax.set_title(f"α₍AB₎ = {alpha_val:.2f}")
        ax.set_xlabel("A_ss")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Histogram of A_ss at Selected α₍AB₎", y=1.02)
    plt.tight_layout()
    plt.show()

    countA_list = np.array([ results_by_alpha[a][0] for a in alpha_vals ])
    total_list  = np.array([ results_by_alpha[a][1] for a in alpha_vals ])

    plt.figure(figsize=(6,4))
    plt.plot(alpha_vals, countA_list,
            marker='s', linestyle='--', color='tab:green')
    plt.title('Epigenetic SATS: Number of A‐Dominant vs α₍AB₎')
    plt.xlabel('α₍AB₎')
    plt.ylabel('Raw # of A‐Dominant Steady States')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ===================== End of Script =====================
