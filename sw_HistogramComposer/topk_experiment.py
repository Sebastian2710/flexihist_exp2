# topk_experiment.py
#
# Implements Space-Saving (Metwally, Agrawal, El Abbadi, 2005) and validates
# it against EXACT ground truth computed on the same data, per 30-minute
# window, across a sweep of sketch sizes.
#
# Run: python topk_experiment.py

import numpy as np
import pandas as pd

PARQUET = r"C:\faculta\LICENTA\FlexiHist\sw_HistogramComposer\aapl_book_events_regular.parquet"

N_WINDOWS       = 13
NS_PER_MINUTE   = 60_000_000_000
WINDOW_START_NS = 9 * 3_600_000_000_000 + 30 * NS_PER_MINUTE  # 09:30 ET

SLOT_SIZES_SUBMITTED = [20, 50, 100, 200, 500]  # wider sweep for submitted interest
SLOT_SIZES_TRADED     = [20, 50, 100, 150]       # wider sweep for traded interest
TOP_K_REPORT          = 3                        # how many top entries to compare/print
PRICE_DECIMALS        = 2                        # prices rounded to the cent (exact "items")


class SpaceSaving:
    """
    Space-Saving heavy-hitters sketch. Fixed number of (item, count) slots.
    On a new, untracked item with no free slot: evict the minimum-count
    slot, and start the new item's count at (evicted_count + weight) --
    the standard guarantee that bounds overestimation by what was evicted
    at the moment of eviction.
    """
    def __init__(self, m):
        self.m = m
        self.counts = {}   # item -> count

    def update(self, item, weight=1):
        if item in self.counts:
            self.counts[item] += weight
            return
        if len(self.counts) < self.m:
            self.counts[item] = weight
            return
        min_item = min(self.counts, key=self.counts.get)
        min_count = self.counts.pop(min_item)
        self.counts[item] = min_count + weight

    def top_k(self, k):
        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:k]


def build_windows():
    windows = []
    for i in range(N_WINDOWS):
        t0 = WINDOW_START_NS + i * 30 * NS_PER_MINUTE
        t1 = t0 + 30 * NS_PER_MINUTE
        hh = (9 * 60 + 30 + i * 30) // 60
        mm = (9 * 60 + 30 + i * 30) % 60
        windows.append({"index": i, "t0": t0, "t1": t1, "label": f"{hh:02d}:{mm:02d}"})
    return windows


def build_full_day_window():
    t0 = WINDOW_START_NS
    t1 = WINDOW_START_NS + N_WINDOWS * 30 * NS_PER_MINUTE
    return [{"index": 0, "t0": t0, "t1": t1, "label": "ALL DAY"}]


def true_top_k(prices, weights, k):
    """Exact ground truth: sum weights per exact price, sort descending.
    Only possible here because this is an offline validation script -- the
    whole point of the sketch is that hardware could never afford this."""
    s = pd.Series(weights, index=prices).groupby(level=0).sum()
    return list(s.sort_values(ascending=False).head(k).items())


def run_experiment(df_subset, weight_col, label, windows, slot_sizes):
    prices = df_subset["price"].round(PRICE_DECIMALS).to_numpy()
    weights = df_subset[weight_col].to_numpy() if weight_col else np.ones(len(df_subset))
    timestamps = df_subset["timestamp_ns"].to_numpy()

    print(f"\n{'='*72}\n{label}  (n={len(df_subset):,} rows)\n{'='*72}")

    for w in windows:
        mask = (timestamps >= w["t0"]) & (timestamps < w["t1"])
        wp, ww = prices[mask], weights[mask]
        if len(wp) == 0:
            print(f"Window {w['index']:2d} ({w['label']}): no data, skipping")
            continue

        true_top = true_top_k(wp, ww, TOP_K_REPORT)
        true_prices = [p for p, _ in true_top]
        true_lookup = dict(true_top)

        print(f"\nWindow {w['index']:2d} ({w['label']})  n={len(wp):,}")
        print(f"  TRUE  top-{TOP_K_REPORT}: " +
              ", ".join(f"${p:.2f}={c:,.0f}" for p, c in true_top))

        for m in slot_sizes:
            sketch = SpaceSaving(m)
            for p, wt in zip(wp, ww):
                sketch.update(p, wt)

            sk_top = sketch.top_k(TOP_K_REPORT)
            sk_prices = [p for p, _ in sk_top]
            matched = len(set(sk_prices) & set(true_prices))

            errs = [abs(est - true_lookup[p]) / true_lookup[p] * 100
                    for p, est in sk_top if p in true_lookup]
            avg_err = np.mean(errs) if errs else float("nan")

            print(f"    m={m:3d}: match {matched}/{TOP_K_REPORT}  "
                  f"avg err={avg_err:5.1f}%   sketch: " +
                  ", ".join(f"${p:.2f}={c:,.0f}" for p, c in sk_top))


def run_cumulative_experiment(df_subset, weight_col, label, windows, slot_sizes):
    """Run cumulative top-k over full day, snapshotting every 30 minutes.
    Compare rolling/cumulative counts with windowed counts."""
    prices = df_subset["price"].round(PRICE_DECIMALS).to_numpy()
    weights = df_subset[weight_col].to_numpy() if weight_col else np.ones(len(df_subset))
    timestamps = df_subset["timestamp_ns"].to_numpy()

    print(f"\n{'='*72}\n{label} [CUMULATIVE]  (n={len(df_subset):,} rows)\n{'='*72}")

    # Sort by timestamp to process in order
    sort_idx = np.argsort(timestamps)
    prices_sorted = prices[sort_idx]
    weights_sorted = weights[sort_idx]
    timestamps_sorted = timestamps[sort_idx]

    # Process windows with cumulative state
    cumulative_sketches = {}  # m -> SpaceSaving sketch
    for m in slot_sizes:
        cumulative_sketches[m] = SpaceSaving(m)

    for w in windows:
        mask = (timestamps_sorted >= w["t0"]) & (timestamps_sorted < w["t1"])
        window_prices = prices_sorted[mask]
        window_weights = weights_sorted[mask]

        if len(window_prices) == 0:
            print(f"Window {w['index']:2d} ({w['label']}): no data, skipping")
            continue

        # Compute true top-k for this window
        true_top_window = true_top_k(window_prices, window_weights, TOP_K_REPORT)
        true_prices_window = [p for p, _ in true_top_window]
        true_lookup_window = dict(true_top_window)

        print(f"\nWindow {w['index']:2d} ({w['label']})  n={len(window_prices):,}")
        print(f"  WINDOWED   top-{TOP_K_REPORT}: " +
              ", ".join(f"${p:.2f}={c:,.0f}" for p, c in true_top_window))

        for m in slot_sizes:
            # Update cumulative sketch with this window's data
            for p, wt in zip(window_prices, window_weights):
                cumulative_sketches[m].update(p, wt)

            # Get cumulative snapshot
            cumul_top = cumulative_sketches[m].top_k(TOP_K_REPORT)
            cumul_prices = [p for p, _ in cumul_top]
            cumul_lookup = dict(cumul_top)

            # Get windowed sketch for this window only
            window_sketch = SpaceSaving(m)
            for p, wt in zip(window_prices, window_weights):
                window_sketch.update(p, wt)
            window_top = window_sketch.top_k(TOP_K_REPORT)

            # Compare
            cumul_matched = len(set(cumul_prices) & set(true_prices_window))
            window_matched = len(set([p for p, _ in window_top]) & set(true_prices_window))

            print(f"    m={m:3d}: window match {window_matched}/{TOP_K_REPORT} | "
                  f"cumulative match {cumul_matched}/{TOP_K_REPORT}")
            print(f"            cumulative: " +
                  ", ".join(f"${p:.2f}={c:,.0f}" for p, c in cumul_top))


def main():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)
    windows = build_windows()

    add_df = df[df["event"] == "ADD"]
    run_experiment(add_df, weight_col=None,
                   label="SUBMITTED INTEREST (Add orders, unweighted count)",
                   windows=windows,
                   slot_sizes=SLOT_SIZES_SUBMITTED)

    exec_df = df[df["event"].isin(["EXECUTE", "EXECUTE_PRICE"])]
    run_experiment(exec_df, weight_col="shares",
                   label="TRADED INTEREST (Executions, weighted by shares)",
                   windows=windows,
                   slot_sizes=SLOT_SIZES_TRADED)

    # Cumulative analyses (full day rolling count, snapshotted every 30 min)
    run_cumulative_experiment(add_df, weight_col=None,
                              label="SUBMITTED INTEREST (Add orders, unweighted count)",
                              windows=windows,
                              slot_sizes=SLOT_SIZES_SUBMITTED)

    run_cumulative_experiment(exec_df, weight_col="shares",
                              label="TRADED INTEREST (Executions, weighted by shares)",
                              windows=windows,
                              slot_sizes=SLOT_SIZES_TRADED)


if __name__ == "__main__":
    main()