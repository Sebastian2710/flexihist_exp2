# window_frequency_plots.py
#
# Standalone research script — NOT part of the FlexiHist pipeline.
#
# Builds a per-window price frequency vector (bar chart) directly from the
# raw AAPL order data. Produces 13 separate PNGs, one per 30-minute window
# (09:30-16:00 ET), using a FIXED, shared set of equi-width price bins so the
# windows are visually comparable when opened side by side.
#
# Run:  python window_frequency_plots.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths — edit if your layout differs ──────────────────────────────────────
PARQUET = r"C:\faculta\LICENTA\flexihist\sw_histogramcomposer\aapl_book_events_regular.parquet"
OUT_DIR = r"C:\faculta\LICENTA\flexihist\sw_histogramcomposer\window_freq_plots"

# ── Parameters ────────────────────────────────────────────────────────────────
BIN_WIDTH      = 1   # $0.10 equi-width buckets (~10 ticks per bucket)
N_WINDOWS      = 13
PRICE_DECIMALS = 2      # round prices to 2 decimals before binning/plotting

NS_PER_MINUTE   = 60_000_000_000
WINDOW_START_NS = 9 * 3_600_000_000_000 + 30 * NS_PER_MINUTE  # 09:30 ET

os.makedirs(OUT_DIR, exist_ok=True)


def build_windows():
    windows = []
    for i in range(N_WINDOWS):
        t0 = WINDOW_START_NS + i * 30 * NS_PER_MINUTE
        t1 = t0 + 30 * NS_PER_MINUTE
        hh = (9 * 60 + 30 + i * 30) // 60
        mm = (9 * 60 + 30 + i * 30) % 60
        windows.append({"index": i, "t0": t0, "t1": t1, "label": f"{hh:02d}:{mm:02d}"})
    return windows


def main():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)

    # Round to 2 decimals (cents). The raw 1/10000-dollar precision from the
    # ITCH parser isn't meaningful for a visual frequency plot.
    df["price"] = df["price"].round(PRICE_DECIMALS)

    windows = build_windows()

    # ── Fixed, shared bin edges so all 13 plots are directly comparable ──────
    # Use the 0.5-99.5 percentile of the FULL day to exclude stub-quote
    # outliers, then round out to the nearest BIN_WIDTH so bins align cleanly.
    all_prices = df["price"].to_numpy()
    p_lo = np.floor(np.percentile(all_prices, 0.5) / BIN_WIDTH) * BIN_WIDTH
    p_hi = np.ceil(np.percentile(all_prices, 99.5) / BIN_WIDTH) * BIN_WIDTH
    bin_edges = np.arange(p_lo, p_hi + BIN_WIDTH, BIN_WIDTH)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    print(f"Shared price range for all plots: ${p_lo:.2f} - ${p_hi:.2f}  "
          f"({len(bin_edges) - 1} bins of ${BIN_WIDTH:.2f})")

    # ── Compute counts per window first, so we can share the y-axis too ──────
    window_data = []
    max_count_seen = 0
    for w in windows:
        wp = df[(df["timestamp_ns"] >= w["t0"]) &
                (df["timestamp_ns"] < w["t1"])]["price"].to_numpy()
        counts, _ = np.histogram(wp, bins=bin_edges)
        window_data.append((w, wp, counts))
        if len(counts):
            max_count_seen = max(max_count_seen, counts.max())

    # ── Plot each window ──────────────────────────────────────────────────────
    for w, wp, counts in window_data:
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.bar(bin_centers, counts, width=BIN_WIDTH * 0.9,
               color="#2166ac", edgecolor="none")

        ax.set_title(
            f"AAPL Order Price Frequency — Window {w['index']} ({w['label']} ET)\n"
            f"n={len(wp):,} orders, bin width=${BIN_WIDTH:.2f}",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Price (USD)", fontsize=12)
        ax.set_ylabel("Order count", fontsize=12)
        ax.set_xlim(p_lo, p_hi)
        ax.set_ylim(0, max_count_seen * 1.05 if max_count_seen else 1)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fname = f"window_{w['index']:02d}_{w['label'].replace(':', '')}_freq.png"
        out_path = os.path.join(OUT_DIR, fname)
        plt.savefig(out_path, dpi=130)
        plt.close(fig)

        print(f"  saved {out_path}  (n={len(wp):,}, "
              f"max_count_in_window={counts.max() if len(counts) else 0})")

    # ── Plot full-day accumulated counts ─────────────────────────────────────
    full_day_prices = df["price"].to_numpy()
    day_counts, _ = np.histogram(full_day_prices, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(bin_centers, day_counts, width=BIN_WIDTH * 0.9,
           color="#b2182b", edgecolor="none")

    ax.set_title(
        f"AAPL Order Price Frequency — Full Day Accumulated (09:30-16:00 ET)\n"
        f"n={len(full_day_prices):,} orders, bin width=${BIN_WIDTH:.2f}",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Price (USD)", fontsize=12)
    ax.set_ylabel("Order count", fontsize=12)
    ax.set_xlim(p_lo, p_hi)
    ax.set_ylim(0, day_counts.max() * 1.05 if len(day_counts) else 1)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    full_day_fname = "window_full_day_accumulated_freq.png"
    full_day_path = os.path.join(OUT_DIR, full_day_fname)
    plt.savefig(full_day_path, dpi=130)
    plt.close(fig)
    print(f"  saved {full_day_path}  (n={len(full_day_prices):,}, max_count={day_counts.max() if len(day_counts) else 0})")

    print(f"\nDone. {N_WINDOWS} window plots + 1 full-day plot saved to {OUT_DIR}")


if __name__ == "__main__":
    main()