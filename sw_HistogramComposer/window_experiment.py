"""
window_experiment.py  --  adaptive-FlexiHist motivation experiment

Uses the ORIGINAL FlexiHist composer (histogramcomposer.py) as a library.
Produces two figures:
  1. distribution_shift.png  -- ground-truth frequency vector per window (the data moving)
  2. degradation.png         -- a config optimised on window 0, frozen and replayed
                                on every later window -> per-value RMSE climbs

Run from inside sw_HistogramComposer/ :
    python window_experiment.py
"""
import glob, os
import matplotlib
matplotlib.use("Agg")            # save to file, no GUI needed
import matplotlib.pyplot as plt

import histogramcomposer as fh   # the real FlexiHist code
def _fast_pvh(data):
    mn = min(data)
    gt = [0] * (max(data) - mn + 1)
    for d in data: gt[d - mn] += 1
    return gt

fh.compute_per_value_histogram = _fast_pvh
# ---- settings -----------------------------------------------------------
WINDOW_GLOB   = "Windows/window_*.csv"
BUDGET        = (20800, 41600, 50, 90)   # Basys 3: LUTs, FFs, BRAM, DSPs
NUM_BUCKETS   = 16                       # same simplification as Experiment 0
SPLIT         = "depth-count"
ALGORITHM     = fh.greedy_min_error_histogram
OUTDIR        = "Figures"
# -------------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)
windows = sorted(glob.glob(WINDOW_GLOB))
assert windows, f"no files matched {WINDOW_GLOB}"

# load every window once: list of int values + its ground-truth frequency vector
data   = [fh.read_csv_file(w) for w in windows]
gts    = [fh.compute_per_value_histogram(d) for d in data]
mins   = [min(d) for d in data]

# ========== PLOT 1 : distribution shift =================================
# ========== PLOT 1 : distribution shift (zoomed, color gradient) =========
import matplotlib.cm as cm
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
colors = cm.plasma(np.linspace(0.1, 0.9, len(data)))

# find the zoom window: where 99% of all orders actually are
all_prices = [p for d in data for p in d]
p1, p99 = int(np.percentile(all_prices, 0.5)), int(np.percentile(all_prices, 99.5))

for i, (d, gt) in enumerate(zip(data, gts)):
    xs = list(range(mins[i], mins[i] + len(gt)))
    label = f"w{i}" if i in (0, 3, 6, 9, 12) else None
    ax.plot(xs, gt, color=colors[i], alpha=0.8, linewidth=1.2, label=label)

ax.set_xlim(280,295)
ax.set_title("AAPL order price distribution — each line is one time window\n"
             "(purple = market open, yellow = market close)", fontsize=13)
ax.set_xlabel("Price (USD price)")
ax.set_ylabel("Order count at that price")
ax.legend(title="window", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/distribution_shift.png", dpi=150)
print("saved", f"{OUTDIR}/distribution_shift.png")

# ========== build the window-0 optimal config (FlexiHist composer) =======
cands = fh.construct_candidates(data[0], gts[0], BUDGET, NUM_BUCKETS, SPLIT, False)
cands.sort(key=lambda c: (c[0][0].high - c[0][0].low))
selected = ALGORITHM(cands, BUDGET)
assert selected is not None, "window 0 config did not fit the budget"
selected.sort(key=lambda s: s[0].low)

SIZES = ["S", "M", "L", "XL"]
def recover_size(cls, res):
    for s in SIZES:
        try:
            if tuple(cls.get_resources_of_bucket(s)) == tuple(res):
                return s
        except Exception:
            pass
    return "L"

# freeze the config as (class, size, low, high) specs -> rebuildable with zero counts
specs = [(type(b[0]), recover_size(type(b[0]), b[0].get_resource_consumption()),
          b[0].low, b[0].high) for b in selected]
luts, ffs, bram, dsp = 0, 0, 0, 0
for b in selected:
    L, F, B, D = b[0].get_resource_consumption(); luts += L; ffs += F; bram += B; dsp += D
print(f"window-0 config: {len(specs)} buckets, "
      f"{luts} LUTs / {ffs} FFs / {bram} BRAM / {dsp} DSP  (Basys3 budget {BUDGET})")

def build(specs, mn, mx):
    h = fh.HybridHistogram(mn, mx + 1)
    for cls, size, low, high in specs:
        bk = cls.create_default_bucket(size); bk.config(low, high); h.add_bucket(bk)
    return h

# ========== PLOT 2 : degradation =========================================
errors = []

print("\nStarting window degradation simulation (this is the heavy part)...")

for i, d in enumerate(data):
    # 1. Print when a new window starts
    print(f"  -> Processing window {i+1} of {len(data)} ({len(d)} items)...")
    
    histo = build(specs, min(d), max(d))     # frozen window-0 bounds, fresh counts
    
    for j, v in enumerate(d):
        histo.update(v)
        
        # 2. Print a progress update every 20,000 items so you know it's moving
        if (j + 1) % 20000 == 0:
            print(f"     ... {j + 1} items updated")

    window_rmse = fh.rmse(histo, gts[i])
    errors.append(window_rmse)    # per-value RMSE -- the correct metric
    
    # 3. Print when the window finishes and show its result
    print(f"     [Done] Window {i+1} RMSE: {window_rmse:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(errors)), errors, marker="o")
plt.title("Window-0 config frozen, replayed on each window  (FlexiHist rmse)")
plt.xlabel("window"); plt.ylabel("per-value RMSE")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/degradation.png", dpi=130)
print(f"\nsaved {OUTDIR}/degradation.png")
print("RMSE per window:", [round(e, 2) for e in errors])
print(f"spike w0 -> w1: {errors[1]/errors[0]:.1f}x" if errors[0] else "w0 RMSE is 0")