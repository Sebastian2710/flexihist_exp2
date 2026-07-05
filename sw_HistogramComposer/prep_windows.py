import os, csv
import pandas as pd

PARQUET     = "aapl_orders.parquet"
PRICE_COL   = "price"
TIME_COL    = "timestamp_ns"          # <-- confirm this matches your parquet's actual column name
N_WINDOWS   = 13                   # 13 x 30min = 6.5h = 9:30 -> 16:00 regular session

START_NS    = 34200022603012       # 9:30:00.022603012 (your data's actual open timestamp)
WINDOW_NS   = 30 * 60 * 1_000_000_000   # 30 minutes in ns
END_NS      = START_NS + N_WINDOWS * WINDOW_NS   # 16:00:00.022603012

df = pd.read_parquet(PARQUET)
df = df.sort_values(TIME_COL)      # ITCH should already be in order, but enforce it

# Convert prices to integer cents while preserving two decimal places.
# Multiply by 100, round to nearest cent, then cast to int.
prices = df[PRICE_COL].astype(int).to_numpy()
times  = df[TIME_COL].to_numpy()

# keep only regular session, drop anything before open / after close
mask = (times >= START_NS) & (times < END_NS)
prices, times = prices[mask], times[mask]

window_idx = ((times - START_NS) // WINDOW_NS).astype(int)

os.makedirs("Windows", exist_ok=True)
for w in range(N_WINDOWS):
    chunk = prices[window_idx == w]
    with open(f"Windows/window_{w:02d}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        for i, v in enumerate(chunk):
            wr.writerow([i, int(v)])
    if len(chunk):
        print(f"window_{w:02d}.csv  n={len(chunk)}  range=[{chunk.min()},{chunk.max()}]")
    else:
        print(f"window_{w:02d}.csv  n=0  (EMPTY WINDOW)")