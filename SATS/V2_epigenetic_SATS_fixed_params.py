# epigenetic_STAT_noRACIPE_modified.py

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
#  (1) GLOBAL PARAMETERS
# =============================================================================

# Number of independent simulations (trajectories) per α_AB
N_INIT     = 100    # run 1000 replicates from the same initial condition

# Integration parameters
TIME_STEPS = 20000    # number of time‐steps
DT         = 0.01    # Δt for Euler–Maruyama

# Production / Degradation rates
gA = 5.0   # production rate of A
gB = 5.0   # production rate of B
kA = 0.1   # degradation rate of A
kB = 0.1   # degradation rate of B

# Hill‐function parameters
#   Inhibitory fold‐changes (λ < 1)
lam_AB_const = 0.1   # λ_{A→B}  (A inhibits B)
lam_BA_const = 0.1   # λ_{B→A}  (B inhibits A)

#   Hill exponents (number of binding sites)
n_AB = 1  # Hill exponent for A⊣B
n_BA = 1  # Hill exponent for B⊣A

# Basal thresholds (half‐points) in molecule units
X0_AB_basal = 120.0   # basal threshold for A⊣B
X0_BA_const = 120.0   # threshold for B⊣A

# Self‐activation (λ > 1) parameters
lam_AA = 10.0   # A⟶A fold‐change
lam_BB = 10.0   # B⟶B fold‐change

n_AA = 4        # Hill exponent for A⟶A
n_BB = 4        # Hill exponent for B⟶B

X0_AA_basal = 80.0   # threshold for A⟶A
X0_BB_basal = 80.0   # threshold for B⟶B    

# Epigenetic feedback parameters (on A→B only)
#   dX0_AB/dt = [ (X0_AB_basal – X0_AB)  –  α_AB * A ] / BETA
BETA = 10.0       # β (relaxation factor ⇒ 100 hr)


SIGMA = 70.0 # Gaussian white noise 

# =============================================================================
#  (2) FIXED INITIAL CONDITION (“B high, A low”)
# =============================================================================

A_INIT = 1    # A(0) = 1 (very low)
B_INIT = 999  # B(0) = 999 (fairly high)

# =============================================================================
#  (3) SHIFTED‐HILL DEFINITIONS
# =============================================================================

def shifted_hill_inhib(x, X0, n, lam):
    """
    Inhibitory shifted‐Hill:
      H_inhib(x; X0, n, λ) = λ + (1−λ)/(1 + (x/X0)^n ),  0 < λ < 1.
    """
    return lam + (1.0 - lam) / (1.0 + (x / X0)**n)

def shifted_hill_act(x, X0, n, lam):
    """
    Activating shifted‐Hill:
      H_act(x; X0, n, λ) = λ + (1−λ)/(1 + (x/X0)^n ),  λ > 1.
    """
    return (lam + (1.0 - lam) / (1.0 + (x / X0)**n))/lam # dividing by lambda as in RACIPE 

# =============================================================================
#  (4) SINGLE‐PARAMETER‐SET SIMULATION FUNCTION
#     (N_INIT replicates, all starting from the same “B high, A low” state)
# =============================================================================

def simulate_fixed_initial(args):
    """
    Simulate N_INIT trajectories for one α_AB value, always starting from
    the fixed initial state A(0)=A_INIT, B(0)=B_INIT.
    Returns:
      - final_states:    length N_INIT, each is (A_ss, B_ss)
      - phenotypes:      length N_INIT, each is 'A', 'B', or 'A/B'
      - A_traj:          array shape (N_INIT, TIME_STEPS)
      - B_traj:          array shape (N_INIT, TIME_STEPS)
    """
    alpha_AB, _ = args   # second element unused
    np.random.seed()     # ensure different RNG per call

    # Pre‐allocate arrays for all trajectories
    A_traj = np.zeros((N_INIT, TIME_STEPS))
    B_traj = np.zeros((N_INIT, TIME_STEPS))

    final_states = []
    phenotypes   = []

    # Constant thresholds for links that do not change:
    X0_BA = X0_BA_const
    X0_AA = X0_AA_basal
    X0_BB = X0_BB_basal

    for iteration in range(N_INIT):
        # 1) Initialize A and B for this replicate
        A = np.zeros(TIME_STEPS)
        B = np.zeros(TIME_STEPS)
        A[0] = A_INIT
        B[0] = B_INIT

        # 2) Initial epigenetic threshold for A⊣B
        X0_AB = X0_AB_basal

        # Store initial values
        A_traj[iteration, 0] = A[0]
        B_traj[iteration, 0] = B[0]

        # 3) Time‐stepping via Euler–Maruyama
        for t in range(TIME_STEPS - 1):
            # 3a) Epigenetic update for A⊣B:
            #     dX0_AB/dt = [(X0_AB_basal – X0_AB) – α_AB·A[t]] / BETA
            X0_AB_new = X0_AB
            X0_AB_new += DT * ((X0_AB_basal - X0_AB) - alpha_AB * A[t]) / BETA
            if X0_AB_new > 0:
                X0_AB=X0_AB_new

            # 3b) Compute all four shifted‐Hill terms:
            H_BA = shifted_hill_inhib(B[t], X0_BA, n_BA, lam_BA_const)
            H_AB = shifted_hill_inhib(A[t], X0_AB, n_AB, lam_AB_const)
            H_AA = shifted_hill_act  (A[t], X0_AA, n_AA, lam_AA)
            H_BB = shifted_hill_act  (B[t], X0_BB, n_BB, lam_BB)

            # 3c) Sample Gaussian noise
            noiseA = np.random.normal(0.0, 1.0) * SIGMA * np.sqrt(DT)
            noiseB = np.random.normal(0.0, 1.0) * SIGMA * np.sqrt(DT)

            # 3d) Deterministic increments
            dA = gA * H_BA * H_AA - kA * A[t]
            dB = gB * H_AB * H_BB - kB * B[t]

            # 3e) Euler–Maruyama update (non‐negative)
            A[t+1] = max(A[t] + DT * dA + noiseA, 0.0)
            B[t+1] = max(B[t] + DT * dB + noiseB, 0.0)

            A_traj[iteration, t+1] = A[t+1]
            B_traj[iteration, t+1] = B[t+1]

        # 4) After TIME_STEPS, record final steady‐state (round to 3 decimals)
        valA = np.round(A[-1], 3)
        valB = np.round(B[-1], 3)
        final_states.append((valA, valB))

        # 5) Classify phenotype using “4/5 rule”:
        #    if A_ss < (4/5)·B_ss → “B‐dominant”
        #    if B_ss < (4/5)·A_ss → “A‐dominant”
        #    else              → “A/B” (hybrid)
        if valA < (4.0 / 5.0) * valB:
            phenotypes.append('B')
        elif valB < (4.0 / 5.0) * valA:
            phenotypes.append('A')
        else:
            phenotypes.append('A/B')

    return final_states, phenotypes, A_traj, B_traj

# =============================================================================
#  (5) MAIN EXECUTION: SWEEP OVER α_AB, SIMULATE, AND PLOT
# =============================================================================

if __name__ == '__main__':
    # α_AB values to sweep
    alpha_list = [0.00, 0.15, 0.35, 0.50]

    # Pre‐allocate for summary statistics
    fracA_list = []    # fraction of “A‐dominant” phenotypes
    meanA_list = []    # mean of A_ss across replicates
    stdA_list  = []    # std of A_ss across replicates

    # For storing all A_ss per α_AB
    full_A_ss = {}     # { alpha_AB : [list of 1000 A_ss values] }

    M_dummy = None
    A_alpha_traj = {}
    B_alpha_traj = {}

    # Run simulations for each alpha_val
    for alpha_val in alpha_list:
        final_states, phenotypes, A_traj, B_traj = simulate_fixed_initial((alpha_val, M_dummy))
        A_alpha_traj[alpha_val] = A_traj
        B_alpha_traj[alpha_val] = B_traj

        # Unravel steady states
        A_values = [a_ss for (a_ss, b_ss) in final_states]
        full_A_ss[alpha_val] = A_values

        # Compute fraction of “A‐dominant” (count only 'A' vs 'B')
        count_A     = sum(1 for ph in phenotypes if ph == 'A')
        count_total = sum(1 for ph in phenotypes if ph in ['A', 'B'])
        fracA = (count_A / count_total) if (count_total > 0) else 0.0
        fracA_list.append(fracA)

        meanA_list.append(np.mean(A_values))
        stdA_list.append(np.std(A_values))

    alpha_arr = np.array(alpha_list)
    fracA_arr = np.array(fracA_list)
    meanA_arr = np.array(meanA_list)
    stdA_arr  = np.array(stdA_list)

    # --------------------------- FIGURE 1 ---------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(alpha_arr, fracA_arr, marker='o', color='tab:blue', linestyle='-')
    plt.title("Fraction of A‐Dominant vs α₍AB₎")
    plt.xlabel("α₍AB₎")
    plt.ylabel("Fraction A‐Dominant")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------- FIGURE 2 ---------------------------
    # plt.figure(figsize=(6, 4))
    # plt.errorbar(alpha_arr, meanA_arr, yerr=stdA_arr,
    #              marker='s', color='tab:red', ecolor='lightcoral',
    #              capsize=4, linestyle='-')
    # plt.title("Mean Steady‐State A ± SD vs α₍AB₎")
    # plt.xlabel("α₍AB₎")
    # plt.ylabel("Mean A_ss")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    # --------------------------- FIGURE 3 ---------------------------
    # --------------------------- FIGURE 3 (all subplots in one 2×2 grid) ---------------------------
    selected_alphas = [0.00, 0.15, 0.35, 0.50]
    time_steps = np.arange(TIME_STEPS)
    xticks = np.arange(0, TIME_STEPS + 1, 2000)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    for ax, alpha_val in zip(axs.flat, selected_alphas):
        A_trajs = A_alpha_traj[alpha_val]  # shape (1000, TIME_STEPS)
        B_trajs = B_alpha_traj[alpha_val]

        # Classify each trajectory’s final state
        states = []
        for i in range(N_INIT):
            end_A = A_trajs[i, -1]
            end_B = B_trajs[i, -1]
            if end_A > end_B:
                states.append('A')
            elif end_B > end_A:
                states.append('B')
            # hybrids are ignored

        # Separate trajectories by final state
        A_dom_A = [A_trajs[i] for i, s in enumerate(states) if s == 'A']  # A values for A-dominants
        A_dom_B = [B_trajs[i] for i, s in enumerate(states) if s == 'A']  # B values for A-dominants
        B_dom_A = [A_trajs[i] for i, s in enumerate(states) if s == 'B']  # A values for B-dominants
        B_dom_B = [B_trajs[i] for i, s in enumerate(states) if s == 'B']  # B values for B-dominants

        n_A_dom = len(A_dom_A)
        n_B_dom = len(B_dom_A)

        # Convert to arrays (or single-row zero arrays if none)
        if A_dom_A:
            A_dom_A_arr = np.vstack(A_dom_A)  # shape (n_A_dom, TIME_STEPS)
            A_dom_B_arr = np.vstack(A_dom_B)
        else:
            A_dom_A_arr = np.zeros((1, TIME_STEPS))
            A_dom_B_arr = np.zeros((1, TIME_STEPS))

        if B_dom_A:
            B_dom_A_arr = np.vstack(B_dom_A)
            B_dom_B_arr = np.vstack(B_dom_B)
        else:
            B_dom_A_arr = np.zeros((1, TIME_STEPS))
            B_dom_B_arr = np.zeros((1, TIME_STEPS))

        # Compute means and stds
        mean_A_A = A_dom_A_arr.mean(axis=0)
        std_A_A  = A_dom_A_arr.std(axis=0)
        mean_B_A = A_dom_B_arr.mean(axis=0)
        std_B_A  = A_dom_B_arr.std(axis=0)

        mean_A_B = B_dom_A_arr.mean(axis=0)
        std_A_B  = B_dom_A_arr.std(axis=0)
        mean_B_B = B_dom_B_arr.mean(axis=0)
        std_B_B  = B_dom_B_arr.std(axis=0)

        # ---------- Plot trajectories + means on this subplot ----------
        # (i) Faint individual trajectories
        for row in range(A_dom_A_arr.shape[0]):
            ax.plot(time_steps, A_dom_A_arr[row], color='red', alpha=0.05, lw=0.5)
            ax.plot(time_steps, A_dom_B_arr[row], color='coral', alpha=0.05, lw=0.5, linestyle='dashed')
        for row in range(B_dom_A_arr.shape[0]):
            ax.plot(time_steps, B_dom_A_arr[row], color='blue', alpha=0.05, lw=0.5)
            ax.plot(time_steps, B_dom_B_arr[row], color='dodgerblue', alpha=0.05, lw=0.5, linestyle='dashed')

        # (ii) Plot mean trajectories (solid for A, dashed for B)
        ax.plot(time_steps, mean_A_A, color='red', lw=2, label="A-dom: mean A(t)")
        ax.plot(time_steps, mean_B_A, color='coral', lw=2, linestyle='dashed', label="A-dom: mean B(t)")
        ax.plot(time_steps, mean_A_B, color='blue', lw=2, label="B-dom: mean A(t)")
        ax.plot(time_steps, mean_B_B, color='dodgerblue', lw=2, linestyle='dashed', label="B-dom: mean B(t)")

        # (iii) Variance bars at each 2000-step mark
        for t_idx in xticks:
            if t_idx < TIME_STEPS:
                # A-dominant group bars
                ax.errorbar(t_idx, mean_A_A[t_idx], yerr=std_A_A[t_idx],
                            fmt='o', color='red', alpha=0.8, capsize=2)
                ax.errorbar(t_idx, mean_B_A[t_idx], yerr=std_B_A[t_idx],
                            fmt='o', color='coral', alpha=0.8, capsize=2)
                # B-dominant group bars
                ax.errorbar(t_idx, mean_A_B[t_idx], yerr=std_A_B[t_idx],
                            fmt='o', color='blue', alpha=0.8, capsize=2)
                ax.errorbar(t_idx, mean_B_B[t_idx], yerr=std_B_B[t_idx],
                            fmt='o', color='dodgerblue', alpha=0.8, capsize=2)

        # Subplot title with counts
        ax.set_title(f"α₍AB₎={alpha_val:.2f} (n_A-dom={n_A_dom}, n_B-dom={n_B_dom})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Gene expression")
        ax.set_xticks(xticks)
        ax.grid(alpha=0.2)
        ax.legend()

    plt.tight_layout()
    plt.show()


    # # --------------------------- FIGURE 4 (optional PCA) ---------------------------
    # # Uncomment if you want PCA on steady‐states for the maximum α:
    # # alpha_plot = max(alpha_list)
    # # final_states_plot, phenotypes_plot, _, _ = simulate_fixed_initial((alpha_plot, M_dummy))
    # # flat_states = np.vstack(final_states_plot)   # shape = (1000, 2)
    # #
    # # coords = PCA(n_components=2).fit_transform(flat_states)
    # # colmap = np.array([
    # #     "tab:green" if ph == "A"
    # #     else "tab:red" if ph == "B"
    # #     else "tab:gray"
    # #     for ph in phenotypes_plot
    # # ])
    # #
    # # plt.figure(figsize=(5, 5))
    # # plt.scatter(coords[:, 0], coords[:, 1], c=colmap, s=20, alpha=0.6)
    # # plt.title(f"PCA of Steady‐States at α = {alpha_plot:.2f}")
    # # plt.xlabel("PC1")
    # # plt.ylabel("PC2")
    # # plt.tight_layout()
    # # plt.show()

    # # --------------------------- FIGURE 5 (optional Heatmap + clustering) ---------------------------
    # # Uncomment if you want a heatmap + KMeans clustering at α = max(alpha_list):
    # # labels = KMeans(n_clusters=3, random_state=0).fit_predict(flat_states)
    # # order  = np.argsort(labels)
    # #
    # # plt.figure(figsize=(5, 7))
    # # plt.imshow(flat_states[order], aspect="auto", cmap="viridis")
    # # plt.colorbar(label="Expression Level")
    # # plt.title(f"Heatmap @ α={alpha_plot:.2f} (clustered)")
    # # plt.xlabel("Gene Index (0=A, 1=B)")
    # # plt.ylabel("State Index (sorted)")
    # # plt.tight_layout()
    # # plt.show()
