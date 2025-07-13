
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
INIT_CAS = 0

THRESHOLD_FRACTION = 0.10
INIT_STOCK = {
    "RBC": INIT_RBC,
    "FFP": INIT_FFP,
    "PLT": INIT_PLT,
    "CRYO": INIT_CRYO,
    "CAS": INIT_CAS
}

low_supply_events = {k: [[] for _ in range(N)] for k in INIT_STOCK}  # list of times per product per node
wbb_overwhelmed_events = [[] for _ in range(N)]


# Deployment and resupply ratios
PROPORTIONS = {"RBC": INIT_RBC, "FFP": INIT_FFP, "PLT": INIT_PLT, "CRYO": INIT_CRYO}
TOTAL_DEPLOY = sum(PROPORTIONS.values())
RESUPPLY_RATIO = {k: v / TOTAL_DEPLOY for k, v in PROPORTIONS.items()}

# Blood constraints
B_max = 500 # IDEK if I need this but I figure we only have so many freezers
WBB_RATE = 10.0  # units/hr per node
tau = 6.0 * 24   # donor cooldown (hours)
setup_delay = 4.0
N_total = 400
#p_resupply = np.full(N, 0.5)  # use this for stochastic resupplies, could be useful
#gamma = 0.05  # redistribution coefficient in conjuction with above line

# Adjacency matrix
A = np.eye(N, k=1) + np.eye(N, k=-1)
t_setup = np.random.uniform(0, setup_delay, N)

# Travel time matrix (in hours) for Blackhawk redistribution (based on ~270 km/h)
# Nodes: 0=Kherson, 1=Crimea, 2=Tokmak, 3=E105, 4=Rivnopil

# Lat/lon (deg) for 5 CSH nodes
lat_lon = np.array([
    [49.9935, 36.2304],  # Kharkiv
    [48.5862, 38.0000],  # Bakhmut
    [47.8388, 35.1396],  # Zaporizhzhia
    [48.4647, 35.0462],  # Dnipro
    [46.6354, 32.6169]   # Kherson
])



def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Create travel time matrix (in hours)
speed_kmph = 278
buffer_time = 0.167  # ~10 minutes

N = len(lat_lon)
travel_time_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i != j:
            dist = haversine(*lat_lon[i], *lat_lon[j])
            travel_time_matrix[i, j] = dist / speed_kmph + buffer_time

print("Travel Time Matrix (hrs):")
print(np.round(travel_time_matrix, 2))

# travel_times = np.array([
#     [ 0.00, 0.56, 0.48, 0.85, 1.11],
#     [0.56, 0.00, 0.63, 1.00,  np.inf],
#     [0.48, 0.63, 0.00, 0.33, 0.59],
#     [0.85, 1.00, 0.33, 0.00, 0.26],
#     [1.11,  np.inf, 0.59, 0.26, 0.00],]
# )

def get_flight_delay(i, j):
    base_time = travel_time_matrix[i, j]
    if not np.isfinite(base_time):
        return np.inf
    jitter = np.random.uniform(0.25, 1.0)  # delay due to terrain, enemy contact, etc.
    return base_time + jitter


# --- Generate Blackout Windows (shared across all nodes) ---
blackout_windows = []
t = 0
while t < T_max:
    if np.random.rand() < 0.15:  # ~15% chance every 6 hrs → blackout (loss of air superiority once a day)
        start = t
        duration = np.random.uniform(24, 48)  # blackout lasts 1–2 days
        blackout_windows.append((start, start + duration))
        t += duration
    else:
        t += 6  # check for new blackout every 6 hrs

def in_blackout(t, blackout_windows):
    return any(start <= t <= end for start, end in blackout_windows)

# --- Scheduled resupply arrivals ---
resupply_schedule = [[] for _ in range(N)]
for i in range(N):
    t = 0
    while t < T_max:
        interval = np.random.uniform(48, 72)
        delay = np.random.uniform(4, 8)
        arrival = t + delay
        if in_blackout(arrival, blackout_windows):
            for window in blackout_windows:
                if window[0] <= arrival <= window[1]:
                    arrival = window[1] + 0.1  # push just past blackout
                    break
        # Now always append the (possibly delayed) arrival
        resupply_schedule[i].append(arrival)

        t += interval

#--Redistribution between individual CSHs--
redistribution_events = []  # (from_node, to_node, product, qty, arrival_time)
redistribution_check_interval = 6.0  # hours
redistribution_delay_range = (2, 4)  # hours


def schedule_redistribution(t, B):
    for i in range(N):
        for j in range(N):
            if A[i, j] and np.isfinite(travel_time_matrix[i, j]):
                for k in ["RBC", "FFP", "PLT", "CRYO"]:
                    if B[k][i] > B[k][j] + 50 and (B["RBC"][j] + B["WBB"][j]) >= 1:
                        qty = min(30, B[k][i] - B[k][j])
                        delay = get_flight_delay(i, j)
                        arrival_time = t + delay
                        redistribution_events.append((i, j, k, qty, arrival_time))


# Initial stock
B_init = {
    "RBC": np.full(N, INIT_RBC),
    "FFP": np.full(N, INIT_FFP),
    "PLT": np.full(N, INIT_PLT),
    "CRYO": np.full(N, INIT_CRYO),
    "WBB": np.zeros(N), "CAS": np.zeros(N),
}
NR_init = np.full(N, N_total)
NU_init = np.zeros(N)

# cum_casualties=[[]for _ in range(N)]
# running_total=[0]*N

def generate_casualties(N, t):
        base = 3
        if 24*5 < t < 24*10:  # simulate a 5-day spike
            base = 8
        casualties = np.random.poisson(base, size=N)
        return {"CAS" : casualties }



def casualty_blood_demand(casualties):
    need = (casualties / 3).astype(int)
    return {
        "RBC": need,
        "FFP": need,
        "PLT": (need + 3) // 4,
        "CRYO": (need + 4) // 5,
        "WBB": np.zeros_like(need),
    }

def withdraw_from_queue(queue, needed):
    withdrawn = 0
    new_queue = []
    for age, qty in queue:
        if withdrawn >= needed:
            new_queue.append([age, qty])
        else:
            take = min(qty, needed - withdrawn)
            withdrawn += take
            if qty > take:
                new_queue.append([age, qty - take])
    return new_queue, withdrawn


#--Establish a dynamic queue for checking last redistributions between CSH--
last_redistribution_check = -12.0  # forces initial scheduling at t = 0
unmet_demand_log = np.zeros((N, len(time)))  # nodes × time
# Expiration time constants
EXPIRY_WBB = 24   # hours
EXPIRY_PLT = 120  # hours

# Per-node age-tracking queues (list of [age, quantity] pairs)
wbb_queues = [[] for _ in range(N)]
plt_queues = [[] for _ in range(N)]


def system(t, y):
    global last_redistribution_check

    # Convert flat y back into inventory dictionary for live snapshot
    B_live = {k: y[i * N:(i + 1) * N] for i, k in enumerate(["RBC", "FFP", "PLT", "CRYO", "WBB"])}

    # Schedule redistribution every 6 hours
    if t - last_redistribution_check >= redistribution_check_interval:
        schedule_redistribution(t, B_live)
        last_redistribution_check = t

    B = {k: y[i*N:(i+1)*N] for i, k in enumerate(["RBC", "FFP", "PLT", "CRYO", "WBB","CAS"])}
    NR = y[5*N:6*N]
    NU = y[6*N:7*N]
    dB = {k: np.zeros(N) for k in B}
    dNR = np.zeros(N)
    dNU = np.zeros(N)

    casualties = generate_casualties(N, t)  # returns a dict now
    demand = casualty_blood_demand(casualties["CAS"])  # pass the actual array
    unmet=False

    # Update WBB and PLT age queues
    for i in range(N):
        # Age all units by dt
        wbb_queues[i] = [[age + dt, qty] for age, qty in wbb_queues[i] if age + dt <= EXPIRY_WBB]
        plt_queues[i] = [[age + dt, qty] for age, qty in plt_queues[i] if age + dt <= EXPIRY_PLT]

        # 🔍 Debug: Print oldest platelet age (optional)
        #if plt_queues[i]:
         #   max_age = max(age for age, _ in plt_queues[i])
          #  print(f"t={t:.1f} hrs | Node {i} oldest PLT age: {max_age:.1f} hrs")


        # Total up valid units
        B["WBB"][i] = sum(qty for age, qty in wbb_queues[i])
        B["PLT"][i] = sum(qty for age, qty in plt_queues[i])
        B["CAS"][i] += casualties["CAS"][i]

    # Process redistribution arrivals (global delivery queue)
    for event in redistribution_events:
        from_node, to_node, k, qty, arrival_time = event
        if abs(t - arrival_time) < dt:
            dB[k][to_node] += qty
            dB[k][from_node] -= qty


    for i in range(N):
        # Node alive status (based on RBC + WBB)
        S = 1 if demand["RBC"][i] > 0 and (B["RBC"][i] + B["WBB"][i]) >= 4 else 0

        # Transfusion prioritization logic
        n_needed = demand["RBC"][i]  # assume demand["RBC"] encodes # of patients

        for _ in range(n_needed):
            unmet = True  # assume failure unless we succeed

            # 1. Try massive transfusion: 4 RBC + 4 FFP + 1 PLT
            if B["RBC"][i] >= 4 and B["FFP"][i] >= 4 and B["PLT"][i] >= 1:
                dB["RBC"][i] -= 4 * S
                dB["FFP"][i] -= 4 * S
                plt_queues[i], used = withdraw_from_queue(plt_queues[i], 1 * S)
                dB["PLT"][i] -= used
                unmet=False
                continue

            # 2. Try 1 unit of whole blood (WBB)
            elif B["WBB"][i] >= 1:
                wbb_queues[i], used = withdraw_from_queue(wbb_queues[i], 1 * S)
                dB["WBB"][i] -= used
                unmet=False
                continue

            # 3. Try 4 RBC + 4 FFP (no PLT available)
            elif B["RBC"][i] >= 4 and B["FFP"][i] >= 4:
                dB["RBC"][i] -= 4 * S
                dB["FFP"][i] -= 4 * S
                unmet=False
                continue

            # 4. Try 1 FFP + 1 PLT
            elif B["FFP"][i] >= 1 and B["PLT"][i] >= 1:
                dB["FFP"][i] -= 1 * S
                plt_queues[i], used = withdraw_from_queue(plt_queues[i], 1 * S)
                dB["PLT"][i] -= used
                unmet=False
                continue

            # 5. Final fallback: 1 CRYO
            elif B["CRYO"][i] >= 1:
                dB["CRYO"][i] -= 1 * S
                unmet=False
                continue

        # 🔴 Nothing worked — log unmet need
        if unmet:
            t_idx = int(t / dt) if t < T_max else len(time) - 1
            unmet_demand_log[i, t_idx] += 1


        # WBB generation if alive and setup delay passed
        if S and t > t_setup[i] and NR[i] > 0 and B["WBB"][i] < B_max:
            max_draw = min(WBB_RATE * dt, NR[i], B_max - B["WBB"][i])
            wbb_queues[i].append([0.0, max_draw])
            dNR[i] -= max_draw
            dNU[i] += max_draw

        # Scheduled depot resupply
        for arrival_time in resupply_schedule[i]:
            if abs(t - arrival_time) < dt:
                delivery = np.random.randint(180, 241)
                # --- NEW DYNAMIC LOGIC ---
                baseline = {"RBC": INIT_RBC, "FFP": INIT_FFP, "PLT": INIT_PLT, "CRYO": INIT_CRYO}
                depletion = {k: max(0, 1.0 - B[k][i] / baseline[k]) for k in ["RBC", "FFP", "PLT", "CRYO"]}
                total_depletion = sum(depletion.values()) + 1e-6  # avoid div by zero

                for k in ["RBC", "FFP", "PLT", "CRYO"]:
                    share = depletion[k] / total_depletion
                    units = delivery * share * S
                    if k == "PLT":
                        plt_queues[i].append([0.0, units])
                    else:
                        dB[k][i] += units
                break

    # Donor recovery
    dNR += NU / tau
    dNU -= NU / tau

    return np.concatenate([dB[k] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]] + [dNR, dNU]), B

def system_with_logging(t, y):
    global low_supply_events, wbb_overwhelmed_events
    # Call the original system function
    dydt, B = system(t, y)


    # Capture live WBB and PLT for this time index
    t_idx = min(int(t / dt), len(time) - 1)

    for i in range(N):
        live_WBB[i, t_idx] = sum(qty for age, qty in wbb_queues[i])
        live_PLT[i, t_idx] = sum(qty for age, qty in plt_queues[i])
        live_CAS[i,t_idx]  = B["CAS"][i]

    # Low supply alerts
    for k in INIT_STOCK:
        if k == "CAS":
            continue
        idx = list(PROPORTIONS.keys()).index(k)
        for i in range(N):
            val = y[idx * N + i]
            if val < THRESHOLD_FRACTION * INIT_STOCK[k]:
                low_supply_events[k][i].append(t)

    # WBB overwhelmed alert
    if (t > t_setup[i] and
            NR_init[i] > 0 and
            B_init["WBB"][i] < 1.0 and
            sum(qty for age, qty in wbb_queues[i]) < 1.0):
        wbb_overwhelmed_events[i].append(t)

    return dydt

# Initial state
y0 = np.concatenate([B_init[k] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]] + [NR_init, NU_init])
scheduled_times = np.arange(0, T_max, redistribution_check_interval)

#Live results
live_WBB = np.zeros((N, len(time)))
live_PLT = np.zeros((N, len(time)))
live_CAS = np.zeros((N, len(time)))


sol = solve_ivp(system_with_logging, [0, T_max], y0, t_eval=time, method='RK45')


# Plotting

# Total unmet demand per node
for i in range(N):
    print(f"Node {i} total unmet patients: {int(np.sum(unmet_demand_log[i]))}")

# Optional: plot unmet demand over time
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(time, unmet_demand_log[i], label=f"Node {i}")
plt.xlabel("Time (hours)")
plt.ylabel("Unmet Patients")
plt.title("Unmet Transfusion Demand Over Time")
plt.legend()
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# RBC
for i in range(N):
    axs[0].plot(time, sol.y[0*N + i], label=f'FOB {i}')
axs[0].set_ylabel('RBC Units')
axs[0].legend()

# FFP
for i in range(N):
    axs[1].plot(time, sol.y[1*N + i], label=f'FOB {i}')
axs[1].set_ylabel('FFP Units')

# PLT (live from queue)
for i in range(N):
    axs[2].plot(time, live_PLT[i], label=f'FOB {i}')
axs[2].set_ylabel('PLT Units (live)')

# CRYO
for i in range(N):
    axs[3].plot(time, sol.y[3*N + i], label=f'FOB {i}')
axs[3].set_ylabel('CRYO Units')

# WBB (live from queue)
for i in range(N):
    axs[4].plot(time, live_WBB[i], label=f'FOB {i}')
axs[4].set_ylabel('WBB Units (live)')

# #Casualties cummulative
# for i in range(N):
#         axs[5].plot(time, live_CAS[i], label=f'FOB {i}')
# axs[5].set_ylabel('Casualties')

axs[-1].set_xlabel('Time (hours)')
plt.tight_layout()
plt.show()

def evaluate_plt_coverage(live_PLT, threshold=0.1):
    durations = np.sum(live_PLT < (threshold * INIT_PLT), axis=1) * dt
    coverage = 100 * (1 - durations / T_max)
    return coverage, durations

def run_simulation(seed):
    np.random.seed(seed)

    # Reset global vars
    global wbb_queues, plt_queues, redistribution_events, last_redistribution_check, unmet_demand_log
    wbb_queues = [[] for _ in range(N)]
    plt_queues = [[] for _ in range(N)]
    redistribution_events = []
    last_redistribution_check = -6.0
    unmet_demand_log = np.zeros((N, len(time)))
    failure_counts = np.zeros(N, dtype=int)

    # Reset queues
    live_WBB.fill(0)
    live_PLT.fill(0)

    # Rebuild blackout/resupply schedule
    blackout_windows.clear()
    t = 0
    while t < T_max:
        if np.random.rand() < 0.05:
            start = t
            duration = np.random.uniform(24, 48)
            blackout_windows.append((start, start + duration))
            t += duration
        else:
            t += 6

    for i in range(N):
        resupply_schedule[i] = []
        t = 0
        while t < T_max:
            interval = np.random.uniform(6, 12)
            delay = np.random.uniform(2, 4)
            arrival = t + delay
            if not in_blackout(arrival, blackout_windows):
                resupply_schedule[i].append(arrival)
            t += interval

    # Reset state
    y0 = np.concatenate([B_init[k] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]] + [NR_init, NU_init])
    sol = solve_ivp(system_with_logging, [0, T_max], y0, t_eval=time, method='RK45')

    # --- Enhanced failure detection ---
    # Unpack each product across time: shape = (N, len(time))
    rbc_t = sol.y[0 * N:1 * N]
    ffp_t = sol.y[1 * N:2 * N]
    plt_t = sol.y[2 * N:3 * N]
    cryo_t = sol.y[3 * N:4 * N]
    wbb_t = sol.y[4 * N:5 * N]

    # Track failures per node
    for i in range(N):
        # Extract this node's inventory time series
        r = rbc_t[i]
        f = ffp_t[i]
        p = live_PLT[i]
        c = cryo_t[i]
        w = live_WBB[i]

        # Check at each timestep: is node i unable to transfuse anyone?
        # Define failure as being unable to do any of:
        #   - 4 RBC + 4 FFP
        #   - 1 WBB
        #   - 1 CRYO (even fallback fails)
        unable_all = (
                ((r < 4) | (f < 4)) &  # can't do massive
                (w < 1) &  # no WBB
                (c < 1)  # no CRYO
        )

        if np.any(unable_all):
            failure_counts[i] += 1

    coverage, durations = evaluate_plt_coverage(live_PLT)

    return failure_counts

## Run the simulation
num_trials = 10
failures_per_node = np.zeros(N)

for seed in range(num_trials):
    failed_nodes = run_simulation(seed)
    failures_per_node += failed_nodes

print("Node failure counts over", num_trials, "trials:", failures_per_node)
print("\n🔻 Low Supply Events (Below 10% of initial deployment):")
for k in INIT_STOCK:
    for i in range(N):
        if low_supply_events[k][i]:
            print(f"  Node {i} | {k}: {len(low_supply_events[k][i])} low events (e.g., first at t={low_supply_events[k][i][0]:.1f} hrs)")

print("\n🚨 WBB Overwhelmed Events:")
for i in range(N):
    if wbb_overwhelmed_events[i]:
        print(f"  Node {i}: {len(wbb_overwhelmed_events[i])} times (e.g., first at t={wbb_overwhelmed_events[i][0]:.1f} hrs)")

plt.figure(figsize=(12, 6))

fob_labels = ["Kharkiv", "Bakhmut", "Zaporizhzhia", "Donetsk", "Kherson"]

# WBB overwhelmed
for i in range(N):
    plt.eventplot(wbb_overwhelmed_events[i], lineoffsets=i + 0.2, colors='red', linewidths=2, label='WBB Overwhelmed' if i == 0 else "")

# RBC < 10%
for i in range(N):
    plt.eventplot(low_supply_events["RBC"][i], lineoffsets=i, colors='blue', linewidths=2, label='RBC <10%' if i == 0 else "")

# FFP < 10%
for i in range(N):
    plt.eventplot(low_supply_events["FFP"][i], lineoffsets=i - 0.2, colors='darkgreen', linewidths=2, label='FFP <10%' if i == 0 else "")

# Spike highlight
plt.axvspan(120, 240, color='orange', alpha=0.2, label='5-Day Casualty Spike')

plt.yticks(range(N), fob_labels)
plt.xlabel("Time (hours)")
plt.ylabel("Node (CSH)")
plt.title("⚠️ Blood Supply Vulnerability Events Over Time")
plt.legend(loc='upper right')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

def generate_interval_graphs(intervals=[24, 36, 48, 72, 120]):
    """
    Replaces the stress test function with a graph-focused simulation loop.
    For each interval in `intervals`, run a single simulation and generate plots.
    """
    for interval in intervals:
        print(f"\n🌀 Simulating with resupply interval: {interval} hours...")

        # Seeded randomness for repeatability
        np.random.seed(0)

        # Generate blackout windows
        blackout_windows.clear()
        t = 0
        while t < T_max:
            if np.random.rand() < 0.05:
                start = t
                duration = np.random.uniform(24, 48)
                blackout_windows.append((start, start + duration))
                t += duration
            else:
                t += 6

        # Build resupply schedule with fixed interval + delivery delay
        for i in range(N):
            resupply_schedule[i] = []
            t = 0
            while t < T_max:
                delay = np.random.uniform(2, 4)
                arrival = t + delay
                if not in_blackout(arrival, blackout_windows):
                    resupply_schedule[i].append(arrival)
                t += interval

        # Reset simulation state
        global wbb_queues, plt_queues, redistribution_events, last_redistribution_check, unmet_demand_log,cum_casualties, running_total
        wbb_queues = [[] for _ in range(N)]
        plt_queues = [[] for _ in range(N)]
        cum_casualties=[[] for _ in range(N)]
        running_total=[[] for _ in range(N)]
        redistribution_events = []
        last_redistribution_check = -6.0
        unmet_demand_log = np.zeros((N, len(time)))
        live_WBB.fill(0)
        live_PLT.fill(0)

        y0 = np.concatenate([B_init[k] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]] + [NR_init, NU_init])
        sol = solve_ivp(system_with_logging, [0, T_max], y0, t_eval=time, method='RK45')

        # Plot blood product levels for each interval
        fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        labels = [f"FOB {i}" for i in range(N)]

        for i in range(N):
            axs[0].plot(time, sol.y[0*N + i], label=labels[i])
        axs[0].set_ylabel('RBC Units')
        axs[0].legend()

        for i in range(N):
            axs[1].plot(time, sol.y[1*N + i], label=labels[i])
        axs[1].set_ylabel('FFP Units')

        for i in range(N):
            axs[2].plot(time, live_PLT[i], label=labels[i])
        axs[2].set_ylabel('PLT Units (Live)')

        for i in range(N):
            axs[3].plot(time, sol.y[3*N + i], label=labels[i])
        axs[3].set_ylabel('CRYO Units')

        for i in range(N):
            axs[4].plot(time, live_WBB[i], label=labels[i])
        axs[4].set_ylabel('WBB Units (Live)')
        axs[4].set_xlabel('Time (hours)')

        plt.suptitle(f"Blood Product Dynamics — Resupply Interval = {interval} hrs")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        print(f"time shape: {len(time)}")
        for i in range(N):
            print(f"FOB {i} casualty length: {len(cum_casualties[i])}")

        plt.figure(figsize=(12, 6))
        for i in range(N):
            plt.plot(time, live_CAS[i], label=f"FOB {i}")
        plt.title("📉 Casualties Over Time")
        plt.xlabel("Time (hours)")
        plt.ylabel("Casualties per Hour")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        #print(cum_casualties)


generate_interval_graphs()
