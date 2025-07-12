# ───────────────────────────────────────────────────────────────
# ROLE-2 BLOOD-SUPPLY SIMULATION
# ────Assumptions────
# Uses refrigerated Whole-Blood / WB-equivalents (WB) first
# Falls back to Walking Blood Bank (WBB) if WB exhausted
# If demand still remains and WB is empty → collect fresh WBB
# from Walking-Blood-Bank donors (subject to daily cap & 56 day donor deferral).
# Unused fresh units enter the WBB inventory for 21 days.
# Daily casualties are an INT drawn randomly between CAS_MIN and CAS_MAX.
# ───────────────────────────────────────────────────────────────

import numpy as np, matplotlib.pyplot as plt, pandas as pd

# ╭─ USER-TUNABLE VARIABLES ─╮
DAYS                = 45
CAS_MIN, CAS_MAX    = 0, 200              # casualties / day
PCT_NEED_BLOOD      = 0.20
UNITS_PER_CAS       = 4.0

START_WB_STOCK      = 360.0               # Initial units on day-0
RESUPPLY_EVERY      = 3                   # Day interval to resupply
RESUPPLY_QTY        = 120.0

INIT_DONORS         = 100.0               # ready on day-0
COLLECT_CAP_DAY     = 100.0               # donors/day
UNITS_PER_DONOR     = 1.0
DEFERRAL_DAYS       = 56
NEW_DONORS_BATCH    = 20.0
NEW_DONOR_EVERY     = 14

WB_SHELF            = 35.0
WBB_SHELF           = 21.0
RANDOM_SEED         = 42                  # None → new draw each run
# ╰───────────────────────────╯

rng = np.random.default_rng(RANDOM_SEED)

# ───── INVENTORY HELPERS ───────────────────────────────────────
def add(inv, qty):
    """Add a new age-0 segment of ‘qty’ units to an inventory list."""
    if qty > 0:
        inv.append([0.0, qty])

def age_and_spoil(inv, shelf):
    """Age all segments 1 d and discard/return qty of spoiled units."""
    spoiled, kept = 0.0, []
    for age, qty in inv:
        age += 1
        (kept if age < shelf else [])[0:0] = ([[age, qty]] if age < shelf else [])
        spoiled += 0 if age < shelf else qty
    return kept, spoiled

def withdraw_fifo(inv, need):
    """Withdraw ‘need’ units (oldest first). Return new_inv, need_left."""
    taken, new_inv = 0.0, []
    inv.sort(key=lambda x: -x[0])          # descending age
    for age, qty in inv:
        if taken >= need:
            new_inv.append([age, qty]); continue
        take = min(qty, need - taken); taken += take
        if qty - take > 0: new_inv.append([age, qty - take])
    return new_inv, need - taken

# ────── STATE ARRAYS ───────────────────────────────────────────
t            = np.arange(DAYS)
cas_arr      = np.zeros(DAYS, int)
wb_stock     = np.zeros(DAYS)
wbb_stock    = np.zeros(DAYS)
total_stock  = np.zeros(DAYS)
don_ready    = np.zeros(DAYS)
demand_u     = np.zeros(DAYS); cum_dem = np.zeros(DAYS)
spoil_u      = np.zeros(DAYS)
balance      = np.zeros(DAYS)

wb_inv, wbb_inv = [], []; add(wb_inv, START_WB_STOCK)
donors_ready = INIT_DONORS
deferral_q   = {}                    # day → donors returning
cum_units    = START_WB_STOCK
stockout_day = None

# ────── MAIN DAILY LOOP ────────────────────────────────────────
# ───── SHELF-STABLE BLOOD SURROGATE (SSBS) SETUP ─────────────────────────────
SSBS_SHELF = 180
SSBS_INIT_STOCK = 300.0
SSBS_RESUPPLY_EVERY = 7
SSBS_RESUPPLY_QTY = 100.0

ssbs_inv = []
add(ssbs_inv, SSBS_INIT_STOCK)
ssbs_stock = np.zeros(DAYS)

# ───── MODIFIED DAILY LOOP WITH SSBS ─────────────────────────────────────────
for d in range(DAYS):
    # new donor rotation (if scheduled)
    if NEW_DONORS_BATCH and d % NEW_DONOR_EVERY == 0 and d:
        donors_ready += NEW_DONORS_BATCH

    # deferred donors re-enter
    donors_ready += deferral_q.pop(d, 0)
    don_ready[d]  = donors_ready            # snapshot before collection

    # scheduled WB pallet
    if d % RESUPPLY_EVERY == 0 and d:
        add(wb_inv, RESUPPLY_QTY); cum_units += RESUPPLY_QTY
        add(ssbs_inv,SSBS_RESUPPLY_QTY); cum_units += SSBS_RESUPPLY_QTY

    # age & spoil
    wb_inv, sp1 = age_and_spoil(wb_inv,  WB_SHELF)
    wbb_inv,sp2 = age_and_spoil(wbb_inv, WBB_SHELF)
    spoil_u[d]  = sp1 + sp2; cum_units -= spoil_u[d]

    # casualties & demand
    cas_today      = int(rng.integers(CAS_MIN, CAS_MAX + 1))
    cas_arr[d]     = cas_today
    need           = cas_today * PCT_NEED_BLOOD * UNITS_PER_CAS
    demand_u[d]    = need
    cum_dem[d]     = need if d == 0 else cum_dem[d-1] + need
    cum_units     -= need

    # withdraw WB then stored WBB
    wb_inv,  need = withdraw_fifo(wb_inv,  need)
    wbb_inv, need = withdraw_fifo(wbb_inv, need)

    # if WB empty & need persists, collect fresh WBB
    if not wb_inv and need > 0 and donors_ready > 0:
        donors_to_collect = min(COLLECT_CAP_DAY, donors_ready)
        donors_ready     -= donors_to_collect
        units_collected   = donors_to_collect * UNITS_PER_DONOR
        add(wbb_inv, units_collected); cum_units += units_collected
        wbb_inv, need = withdraw_fifo(wbb_inv, need)
        deferral_q[d+DEFERRAL_DAYS] = deferral_q.get(d+DEFERRAL_DAYS,0)+donors_to_collect

    # fallback to SSBS if demand still unmet
    if need > 0:
        ssbs_inv, need = withdraw_fifo(ssbs_inv, need)
        ssbs_stock[d] = sum(qty for age, qty in ssbs_inv)
        demand_u[d] = need

    # final unmet demand (true deficit)
    demand_u[d] = need
    if need > 0:
        cum_units -= need  # only subtract unmet demand once

    wb_stock[d]  = sum(q for _,q in wb_inv)
    wbb_stock[d] = sum(q for _,q in wbb_inv)
    ssbs_stock[d] = sum(q for _, q in ssbs_inv)  # NEW LINE
    balance[d]   = cum_units
    total_stock[d] = wb_stock[d] + wbb_stock[d] + ssbs_stock[d]
    if stockout_day is None and wb_stock[d]+wbb_stock[d]+ ssbs_stock[d] <= 0:
        stockout_day = d

print(f"Final balance: {cum_units:+.1f} units")
if stockout_day is not None:
    print(f"*** STOCKOUT day {stockout_day} ***")

# ────── PLOT ───────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 7))

# primary axis ─ inventories & donors
ax1.step(t, wb_stock,  where='mid', lw=2, label='WB / WBE')
ax1.step(t, wbb_stock, where='mid', lw=2, label='Stored WBB')
ax1.step(t, ssbs_stock, where='mid', lw=2, label='Stored SSBS')
ax1.step(t, total_stock, where='mid', lw=2, label='Total Stock')
ax1.step(t, don_ready, where='mid', lw=1.5, color='green', label='Donors ready')
ax1.set_xlabel('Day')
ax1.set_ylabel('Units/WBB Donors Available')

# secondary axis ─ cumulative curves
ax2 = ax1.twinx()
ax2.plot(t, balance, color='black', lw=2, label='Running balance')
ax2.plot(t, cum_dem, color='blue', ls='--', lw=2, label='Cumulative demand')
ax2.set_ylabel('Cumulative units')
if stockout_day is not None:
    ax2.axvline(stockout_day, color='purple', ls=':', lw=2,
                label=f'Stock-out day {stockout_day}')

# third axis ─ daily casualties & demand  (shifted outward)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))   # ← key line
ax3.bar(t, cas_arr,  alpha=.15, color='gray', label='Casualties/day')
ax3.bar(t, demand_u, alpha=.25, color='red',  label='Units needed/day')
ax3.set_ylabel('Units in demand', labelpad=8)

# merge legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(handles1 + handles2 + handles3,
           labels1  + labels2  + labels3,
           loc='best')

# tidy layout (leave room for the outer spine)
fig.subplots_adjust(right=0.83)
fig.tight_layout()
plt.title('Role-2 Blood Simulation')
plt.show()
# ────── END PLOT ──────────────────────────────────────────────

# ────── TABLE ──────────────────────────────────────────────────
df = pd.DataFrame({
    'Casualties': cas_arr,
    'Demand': demand_u,
    'Cumulative_Demand': cum_dem,
    'WB_stock': wb_stock,
    'WBB_stock': wbb_stock,
    'Spoiled': spoil_u,
    'Donors_ready': don_ready,
    'Balance': balance})
display(df.head(DAYS))