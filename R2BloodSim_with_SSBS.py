
# ───────────────────────────────────────────────────────────────
# ROLE-2 BLOOD-SUPPLY SIMULATION with Shelf-Stable Blood Surrogate
# ───────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

# ╭─ USER-TUNABLE VARIABLES ─╮
DAYS = 30
CAS_MIN, CAS_MAX = 0, 150
PCT_NEED_BLOOD = 0.2
UNITS_PER_CAS = 4.0
START_WB_STOCK = 300.0
WB_SHELF = 35
RANDOM_SEED = 42
# ╰───────────────────────────╯

# Shelf-Stable Blood Surrogate Parameters
SSBS_SHELF = 180
SSBS_INIT_STOCK = 300.0
SSBS_RESUPPLY_EVERY = 7
SSBS_RESUPPLY_QTY = 100.0

rng = np.random.default_rng(RANDOM_SEED)

# ─── INVENTORY FUNCTIONS ───────────────────────────────────────
def add(inv, qty):
    inv[0] += qty
    return inv

def advance(inv):
    return inv[1:] + [0.0]

# Initialize inventories
WB_inventory = [0.0] * WB_SHELF
WB_inventory = add(WB_inventory, START_WB_STOCK)

SSBS_inventory = [0.0] * SSBS_SHELF
SSBS_inventory = add(SSBS_inventory, SSBS_INIT_STOCK)

WB_used = []
SSBS_used = []

# ─── DAILY SIMULATION LOOP ─────────────────────────────────────
for day in range(DAYS):
    casualties_today = rng.integers(CAS_MIN, CAS_MAX + 1)
    units_needed = casualties_today * PCT_NEED_BLOOD * UNITS_PER_CAS

    # Resupply logic
    if day % SSBS_RESUPPLY_EVERY == 0:
        SSBS_inventory = add(SSBS_inventory, SSBS_RESUPPLY_QTY)

    # Consume WB inventory first
    wb_available = sum(WB_inventory)
    used_wb = min(units_needed, wb_available)
    units_needed -= used_wb

    for i in range(len(WB_inventory)):
        if used_wb == 0: break
        take = min(WB_inventory[i], used_wb)
        WB_inventory[i] -= take
        used_wb -= take

    # If demand remains, consume SSBS inventory
    ssbs_available = sum(SSBS_inventory)
    used_ssbs = min(units_needed, ssbs_available)
    units_needed -= used_ssbs

    for i in range(len(SSBS_inventory)):
        if used_ssbs == 0: break
        take = min(SSBS_inventory[i], used_ssbs)
        SSBS_inventory[i] -= take
        used_ssbs -= take

    # Advance aging
    WB_inventory = advance(WB_inventory)
    SSBS_inventory = advance(SSBS_inventory)

    # Log daily usage
    WB_used.append(sum(WB_inventory))
    SSBS_used.append(sum(SSBS_inventory))

# ─── VISUALIZATION ─────────────────────────────────────────────
plt.plot(WB_used, label="WB Remaining")
plt.plot(SSBS_used, label="SSBS Remaining")
plt.xlabel("Day")
plt.ylabel("Units")
plt.title("Blood Inventory with SSBS Support")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
