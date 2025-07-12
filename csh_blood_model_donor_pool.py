
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model Parameters ---
N = 5  # number of nodes
T_max = 24 * 30  # total time in hours (30 days)
dt = 0.1
time = np.arange(0, T_max, dt)

# Initial deployment quantities per node (CSH)
INIT_RBC = 300
INIT_FFP = 100
INIT_PLT = 24
INIT_CRYO = 100
INIT_WBB = 0

# Total deployment and resupply ratios
PROPORTIONS = {"RBC": INIT_RBC, "FFP": INIT_FFP, "PLT": INIT_PLT, "CRYO": INIT_CRYO}
TOTAL_DEPLOY = sum(PROPORTIONS.values())
RESUPPLY_RATIO = {k: v / TOTAL_DEPLOY for k, v in PROPORTIONS.items()}

# Blood supply constraints
B_max = 500
WBB_RATE = 10.0  # units/hour per node
tau = 6.0 * 24  # donor cooldown (in hours, e.g., 6 days)
setup_delay = 4.0  # hours to activate WBB
N_total = 400  # donors per node
p_resupply = np.full(N, 0.5)

# WBB setup delay
t_setup = np.random.uniform(0, setup_delay, N)

# Initialize stockpiles
B_init = {
    "RBC": np.full(N, INIT_RBC, dtype=float),
    "FFP": np.full(N, INIT_FFP, dtype=float),
    "PLT": np.full(N, INIT_PLT, dtype=float),
    "CRYO": np.full(N, INIT_CRYO, dtype=float),
    "WBB": np.zeros(N, dtype=float),
}
NR_init = np.full(N, N_total)  # Ready donors
NU_init = np.zeros(N)          # Donors in cooldown

# Generate casualties: assume 100/day per node = ~4.17/hr
def generate_casualties(N, t):
    return np.full(N, 4.17)

# Blood demand: 1/3 need transfusion
def casualty_blood_demand(casualties):
    need_transfusion = (casualties / 3).astype(int)
    return {
        "RBC": need_transfusion,
        "FFP": need_transfusion,
        "PLT": (need_transfusion + 3) // 4,
        "CRYO": (need_transfusion + 4) // 5,
        "WBB": np.zeros_like(need_transfusion),
    }

# System ODE
def system(t, y):
    B = {k: y[i*N:(i+1)*N] for i, k in enumerate(["RBC", "FFP", "PLT", "CRYO", "WBB"])}
    NR = y[5*N:6*N]
    NU = y[6*N:7*N]

    dB = {k: np.zeros(N) for k in B}
    dNR = np.zeros(N)
    dNU = np.zeros(N)

    casualties = generate_casualties(N, t)
    demand = casualty_blood_demand(casualties)

    for i in range(N):
        for k in ["RBC", "FFP", "PLT", "CRYO"]:
            available = B[k][i]
            use = min(demand[k][i], available)
            dB[k][i] -= use
            demand[k][i] -= use

        # WBB production constrained by donor pool
        if t > t_setup[i] and NR[i] > 0 and B["WBB"][i] < B_max:
            max_production = min(WBB_RATE * dt, NR[i], B_max - B["WBB"][i])
            dB["WBB"][i] += max_production
            dNR[i] -= max_production
            dNU[i] += max_production

        # Resupply
        if np.random.rand() < p_resupply[i]:
            for k in ["RBC", "FFP", "PLT", "CRYO"]:
                dB[k][i] += 30 * RESUPPLY_RATIO[k]

    # WBB donor recovery
    dNR += NU / tau
    dNU -= NU / tau

    return np.concatenate([dB[k] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]] + [dNR, dNU])

# Initial state
y0 = np.concatenate([B_init[k] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]] + [NR_init, NU_init])

# Solve ODE
sol = solve_ivp(system, [0, T_max], y0, t_eval=time, method='RK45')

# Plot results
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
products = ["RBC", "FFP", "PLT", "CRYO", "WBB"]
for idx, k in enumerate(products):
    for i in range(N):
        axs[idx].plot(time, sol.y[idx*N + i], label=f'FOB {i}')
    axs[idx].set_ylabel(f'{k} Units')
    axs[idx].legend()
axs[-1].set_xlabel('Time (hours)')
plt.tight_layout()
plt.show()
