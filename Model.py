import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model Parameters ---
N = 5  # number of nodes (FOBs)
T_max = 24*30  # total time (hours)
dt = 0.1
time = np.arange(0, T_max, dt)

# Blood supply parameters
B_max = 10
B_crit = 1
r0 = 0.1  # max resupply rate
beta = 0.01  # WBB blood per donor per hour
u_base = 0.1  # baseline usage
sigma_u = 0.01  # utilization noise
tau = 6.0  # WBB donor cooldown (hours)
gamma = 0.05  # blood redistribution coefficient
setup_delay = 4.0  # hours to activate WBB
N_total = 40  # total donors per node
p_resupply = np.full(N, 0.5)  # resupply success chance

# Initial conditions
B_init = np.full(N, 5.0)
NR_init = np.full(N, N_total)
NU_init = np.zeros(N)

"""
Explanation of 1D topology I really wanna change this,
Essentially this is saying that if I am at state 2, my neighbor at state 1 and
state 3 are essentially same distance away as I am to them. 
"""
# Adjacency matrix (1D chain topology)
A = np.eye(N, k=1) + np.eye(N, k=-1)

# Random WBB setup delays-This will have to be updated based on CPG
t_setup = np.random.uniform(0, setup_delay, N)

# Flattened state vector: [B0...BN, NR0...NRN, NU0...NUN]
def system(t, y):
    B = y[0:N]
    NR = y[N:2*N]
    NU = y[2*N:3*N]

    dB = np.zeros(N)
    dNR = np.zeros(N)
    dNU = np.zeros(N)

    # Stochastic inputs
    mass_casualty_triggered = np.random.rand(N)<0.3# 40% chance of mass call
    chi_resupply = np.random.rand(N) < p_resupply #Convoy success-could make this a constant tbh

    for i in range(N):
        Si = 1 if B[i] >= B_crit else 0  # node is alive

        Ui = Si * (u_base*(10 if mass_casualty_triggered[i] else 1))  # utilization goes up 10 if mass call
        Ri = Si * r0 * chi_resupply[i]  # stochastic resupply
        '''
        Not really a fan of this Wi, honestly I feel like we need
        to make it more probabilty based based off of data in Ukraine
        (only 1 of 4 times did they utilize WBB, don't really know the
        reasons need to find that out)
        '''
        Wi = Si * beta * NR[i] * (1 - B[i]/B_max) if t > t_setup[i] else 0  # WBB with delay

        # Redistribution
        Ti = 0
        for j in range(N):
            if A[i, j] and B[j] > B[i]:
                Sj = 1 if B[j] >= B_crit else 0
                Ti += gamma * A[i, j] * (B[j] - B[i]) * Si * Sj  # Weighted by A[i, j]
        '''
        Above the Transfer rate is a stochastic process
        essentially there is some baseline transfer rate gamma. 
        But there will only be resupply if the neighboring node
        is #1 still there and #2 have a deficit compared to itself.
        Obviously this will take a crap ton of calcualtions and will be
        innefectient coding
        '''
        # Update equations
        dB[i] = Ti + Ri + Wi - Ui
        dNR[i] = -Wi + NU[i]/tau
        dNU[i] = Wi - NU[i]/tau

    return np.concatenate([dB, dNR, dNU])

# Initial state
y0 = np.concatenate([B_init, NR_init, NU_init])

# Integrate ODEs
sol = solve_ivp(system, [0, T_max], y0, t_eval=time, method='RK45')

# Extract results
B_vals = sol.y[0:N]
NR_vals = sol.y[N:2*N]
NU_vals = sol.y[2*N:3*N]

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i in range(N):
    axs[0].plot(time, B_vals[i], label=f'FOB {i}')
axs[0].set_ylabel('Blood Units')
axs[0].set_title('Blood Supply at Each FOB')
axs[0].legend()

for i in range(N):
    axs[1].plot(time, NR_vals[i], label=f'FOB {i}')
axs[1].set_ylabel('Ready Donors')
axs[1].set_title('WBB: Ready Donors')

for i in range(N):
    axs[2].plot(time, NU_vals[i], label=f'FOB {i}')
axs[2].set_ylabel('Unavailable Donors')
axs[2].set_xlabel('Time (hours)')
axs[2].set_title('WBB: Recovering Donors')

plt.tight_layout()
plt.show()

# --- Optional: Custom Weighted Adjacency Matrix ---
# Example: Node 1 is twice as far from Node 2 as from Node 0
# A = np.zeros((3, 3))
# A[0, 1] = A[1, 0] = 1.0   # Normal connection
# A[1, 2] = A[2, 1] = 0.5   # Half as efficient (twice the distance)

