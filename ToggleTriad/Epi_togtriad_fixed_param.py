import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# ---- Shifted Hill Inhibition ----
def shifted_hill_inhib(x, X0, n, lam):
    return lam + (1.0 - lam) / (1.0 + (x / X0) ** n)

# ---- Tri-Stable Parameter sets as used in the paper(mentioned in Table S1)----
param_sets = {
    "P1": {
        "g_A": 74.661273, "g_B": 88.839696, "g_C": 33.342705,
        "k_A": 0.7806, "k_B": 0.913511, "k_C": 0.678669,
        "a0_BA": 11.243532, "n_BA": 6, "lam_BA": 0.056011,
        "a0_CA": 4.745698, "n_CA": 5, "lam_CA": 0.012852,
        "a0_AB": 5.979105, "n_AB": 6, "lam_AB": 0.020298,
        "a0_CB": 8.414342, "n_CB": 2, "lam_CB": 0.016738,
        "a0_BC": 11.509482, "n_BC": 4, "lam_BC": 0.022222,
        "a0_AC": 9.969809, "n_AC": 5, "lam_AC": 0.010311
    },
    "P2": {
        "g_A": 12.756414, "g_B": 24.611005, "g_C": 21.892383,
        "k_A": 0.49127, "k_B": 0.87707, "k_C": 0.668234,
        "a0_BA": 9.454558, "n_BA": 5, "lam_BA": 0.134716,
        "a0_CA": 8.071632, "n_CA": 2, "lam_CA": 0.019995,
        "a0_AB": 13.404026, "n_AB": 4, "lam_AB": 0.012008,
        "a0_CB": 5.068572, "n_CB": 3, "lam_CB": 0.017363,
        "a0_BC": 8.145014, "n_BC": 5, "lam_BC": 0.028832,
        "a0_AC": 9.122129, "n_AC": 4, "lam_AC": 0.01217
    },
    "P3": {
        "g_A": 45.389182, "g_B": 45.473476, "g_C": 69.274178,
        "k_A": 0.79649, "k_B": 0.660882, "k_C": 0.567332,
        "a0_BA": 11.248542, "n_BA": 5, "lam_BA": 0.028717,
        "a0_CA": 11.806646, "n_CA": 5, "lam_CA": 0.010077,
        "a0_AB": 9.089216, "n_AB": 2, "lam_AB": 0.027849,
        "a0_CB": 7.735465, "n_CB": 2, "lam_CB": 0.016295,
        "a0_BC": 12.262296, "n_BC": 6, "lam_BC": 0.02782,
        "a0_AC": 3.430087, "n_AC": 5, "lam_AC": 0.053563
    },
    "P4": {
        "g_A": 43.125467, "g_B": 42.705912, "g_C": 22.316457,
        "k_A": 0.631683, "k_B": 0.758171, "k_C": 0.429697,
        "a0_BA": 9.723269, "n_BA": 6, "lam_BA": 0.024531,
        "a0_CA": 9.151694, "n_CA": 5, "lam_CA": 0.017554,
        "a0_AB": 7.893864, "n_AB": 4, "lam_AB": 0.013122,
        "a0_CB": 12.18131, "n_CB": 6, "lam_CB": 0.069397,
        "a0_BC": 9.875833, "n_BC": 4, "lam_BC": 0.013806,
        "a0_AC": 6.194162, "n_AC": 3, "lam_AC": 0.011343
    },
    "P5": {
        "g_A": 42.683186, "g_B": 7.390051, "g_C": 19.679016,
        "k_A": 0.666992, "k_B": 0.248021, "k_C": 0.616533,
        "a0_BA": 9.225716, "n_BA": 4, "lam_BA": 0.026475,
        "a0_CA": 7.928571, "n_CA": 6, "lam_CA": 0.027664,
        "a0_AB": 9.793446, "n_AB": 2, "lam_AB": 0.011958,
        "a0_CB": 9.087973, "n_CB": 4, "lam_CB": 0.013192,
        "a0_BC": 10.199934, "n_BC": 6, "lam_BC": 0.011774,
        "a0_AC": 11.074304, "n_AC": 2, "lam_AC": 0.011618
    },
    "P6": {
        "g_A": 50.540607, "g_B": 31.447113, "g_C": 26.023803,
        "k_A": 0.4229, "k_B": 0.691099, "k_C": 0.77937,
        "a0_BA": 5.094179, "n_BA": 1, "lam_BA": 0.011036,
        "a0_CA": 1.804967, "n_CA": 3, "lam_CA": 0.031085,
        "a0_AB": 5.269278, "n_AB": 4, "lam_AB": 0.104037,
        "a0_CB": 8.599533, "n_CB": 6, "lam_CB": 0.011182,
        "a0_BC": 5.288305, "n_BC": 1, "lam_BC": 0.018682,
        "a0_AC": 10.552204, "n_AC": 2, "lam_AC": 0.011434
    }
}

# Choose parameter set here!
param_set_choice = "P6"   # <-- change this as needed ("P1"..."P6")

# Simulation parameters 
N_INIT = 50 #no. of random initial conditions
TIME_STEPS = 30000
dt = 0.1
noise_factor = 3 #noise strength
beta = 100.0  # threshold relaxation timescale(as used in the paper(methods section))

# Epigenetic feedback strengths (user input or leave at zero)
alpha_BA = 0.5
alpha_CA = 0
alpha_AB = 0
alpha_CB = 0
alpha_AC = 0
alpha_BC = 0.5

# Load selected set into local variables
params = param_sets[param_set_choice]
globals().update(params)  # Makes variables like g_A, k_A, a0_BA, etc., available

# ---- Main simulation Definition----
def simulate_triad(N_INIT, TIME_STEPS, dt):
    A_traj = np.zeros((N_INIT, TIME_STEPS))
    B_traj = np.zeros((N_INIT, TIME_STEPS))
    C_traj = np.zeros((N_INIT, TIME_STEPS))
    B0A_traj = np.zeros((N_INIT, TIME_STEPS))
    C0A_traj = np.zeros((N_INIT, TIME_STEPS))
    A0B_traj = np.zeros((N_INIT, TIME_STEPS))
    C0B_traj = np.zeros((N_INIT, TIME_STEPS))
    A0C_traj = np.zeros((N_INIT, TIME_STEPS))
    B0C_traj = np.zeros((N_INIT, TIME_STEPS))
    phenotypes = []

    for it in range(N_INIT):
        # Gene concentrations
        A = np.zeros(TIME_STEPS)
        B = np.zeros(TIME_STEPS)
        C = np.zeros(TIME_STEPS)
        # Thresholds (one for each edge)
        B0A = np.zeros(TIME_STEPS)
        C0A = np.zeros(TIME_STEPS)
        A0B = np.zeros(TIME_STEPS)
        C0B = np.zeros(TIME_STEPS)
        A0C = np.zeros(TIME_STEPS)
        B0C = np.zeros(TIME_STEPS)

        # Initial conditions (randomized)
        A[0] = (g_A / k_A) * np.random.rand()
        B[0] = (g_B / k_B) * np.random.rand()
        C[0] = (g_C / k_C) * np.random.rand()
        # Initial thresholds (basal values)
        B0A[0] = a0_BA; C0A[0] = a0_CA
        A0B[0] = a0_AB; C0B[0] = a0_CB
        A0C[0] = a0_AC; B0C[0] = a0_BC

        for t in range(TIME_STEPS-1):
            # Update thresholds with epigenetic feedback, ensuring non negative values using max(0.001, ...)using 0.001 to avoid division by zero
            B0A[t+1] = max(0.001, B0A[t] + dt * (a0_BA - B0A[t] - alpha_BA * A[t]) / beta)
            C0A[t+1] = max(0.001,C0A[t] + dt * (a0_CA - C0A[t] - alpha_CA * A[t]) / beta)
            A0B[t+1] = max(0.001,A0B[t] + dt * (a0_AB - A0B[t] - alpha_AB * B[t]) / beta)
            C0B[t+1] = max(0.001,C0B[t] + dt * (a0_CB - C0B[t] - alpha_CB * B[t]) / beta)
            A0C[t+1] = max(0.001,A0C[t] + dt * (a0_AC - A0C[t] - alpha_AC * C[t]) / beta)
            B0C[t+1] = max(0.001,B0C[t] + dt * (a0_BC - B0C[t] - alpha_BC * C[t]) / beta)

            # Update gene concentrations
            inhib_A = shifted_hill_inhib(B[t], B0A[t], n_BA, lam_BA) * shifted_hill_inhib(C[t], C0A[t], n_CA, lam_CA)
            inhib_B = shifted_hill_inhib(A[t], A0B[t], n_AB, lam_AB) * shifted_hill_inhib(C[t], C0B[t], n_CB, lam_CB)
            inhib_C = shifted_hill_inhib(A[t], A0C[t], n_AC, lam_AC) * shifted_hill_inhib(B[t], B0C[t], n_BC, lam_BC)

            # Multiplicative noise (instead of additive, added at each step instead of at intervals as in the paper), ensuring non negative values using max(0, ...)
            A[t+1] = max(0, A[t] + dt * (g_A * inhib_A - k_A * A[t]) + noise_factor * np.sqrt(dt) * np.sqrt(A[t]) * np.random.randn())
            B[t+1] = max(0, B[t] + dt * (g_B * inhib_B - k_B * B[t]) + noise_factor * np.sqrt(dt) * np.sqrt(B[t]) * np.random.randn())
            C[t+1] = max(0, C[t] + dt * (g_C * inhib_C - k_C * C[t]) + noise_factor * np.sqrt(dt) * np.sqrt(C[t]) * np.random.randn())

        A_traj[it] = A
        B_traj[it] = B
        C_traj[it] = C
        B0A_traj[it] = B0A
        C0A_traj[it] = C0A
        A0B_traj[it] = A0B
        C0B_traj[it] = C0B
        A0C_traj[it] = A0C
        B0C_traj[it] = B0C

        # Phenotype based on final values
        final_val = np.array([A[-1], B[-1], C[-1]])
        max_idx = np.argmax(final_val)
        if (final_val == final_val[max_idx]).sum() == 1:  # Only one winner, no tie, no hybrid states classified by paper's code
            if max_idx == 0:
                phenotypes.append('A')
            elif max_idx == 1:
                phenotypes.append('B')
            else:
                phenotypes.append('C')
        else:
            phenotypes.append('None')  # In case of an exact tie


    return (A_traj, B_traj, C_traj, B0A_traj, C0A_traj, A0B_traj, C0B_traj, A0C_traj, B0C_traj, phenotypes)

# ---- Run and plots ----
if __name__ == "__main__":
    results = simulate_triad(N_INIT, TIME_STEPS, dt)
    (A_traj, B_traj, C_traj,
     B0A_traj, C0A_traj, A0B_traj, C0B_traj, A0C_traj, B0C_traj, phenotypes) = results


     # --- PLOT 1 - Time-resolved population distribution ---

    n_cells, n_time = A_traj.shape
    a_state = np.zeros(n_time)
    b_state = np.zeros(n_time)
    c_state = np.zeros(n_time)

    for t in range(n_time):
        for j in range(n_cells):
            a = A_traj[j, t]
            b = B_traj[j, t]
            c = C_traj[j, t]
            if a > b and a > c:
                a_state[t] += 1
            elif b > a and b > c:
                b_state[t] += 1
            elif c > a and c > b:
                c_state[t] += 1
            # Ties are ignored

    # Convert to percentages
    a_perc = 100 * a_state / n_cells
    b_perc = 100 * b_state / n_cells
    c_perc = 100 * c_state / n_cells

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(a_perc, color='blue', label='A', linewidth=0.5)
    plt.plot(b_perc, color='red', label='B', linewidth=0.5)
    plt.plot(c_perc, color='orange', label='C', linewidth=0.5)
    plt.xlabel('Time step')
    plt.ylabel('Population percentage (%)')
    plt.title(rf'$(\alpha_{{BA}}, \alpha_{{BC}}) = ({alpha_BA},{alpha_BC})$')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ----- PLOT 2 - Mean Trajectory Plot -----
    mean_A = np.mean(A_traj, axis=0)
    mean_B = np.mean(B_traj, axis=0)
    mean_C = np.mean(C_traj, axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(mean_A, label='Mean A', linewidth=2)
    plt.plot(mean_B, label='Mean B', linewidth=2)
    plt.plot(mean_C, label='Mean C', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Mean concentration')
    plt.legend()
    plt.title(f'Mean Trajectory ({param_set_choice})')
    plt.tight_layout()
    plt.show()

    # ----- PLOT 3 - Two Random Trajectories -----
    N = A_traj.shape[0]
    idxs = random.sample(range(N), 2)

    plt.figure(figsize=(12,6))
    for idx in idxs:
        plt.plot(A_traj[idx], label=f'A, Run {idx+1}', linestyle='-')
        plt.plot(B_traj[idx], label=f'B, Run {idx+1}', linestyle='--')
        plt.plot(C_traj[idx], label=f'C, Run {idx+1}', linestyle=':')
    plt.xlabel('Time step')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title(f'Two Random Trajectories ({param_set_choice})')
    plt.tight_layout()
    plt.show()

    # # Phenotype histogram
    # print("Phenotype counts:", Counter(phenotypes))
