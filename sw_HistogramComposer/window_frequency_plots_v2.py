# window_frequency_plots_v2.py
#
# Rebuilds the per-window price frequency plots, but from
# aapl_book_events_regular.parquet instead of aapl_orders.parquet, splitting
# into TWO separate ground truths:
#
#   1. SUBMITTED interest  -- ADD order prices, unweighted count.
#      Same concept as the original window_frequency_plots.py, just sourced
#      from the session-restricted book-events file instead of the
#      unrestricted Add-only file.
#
#   2. TRADED interest     -- EXECUTE / EXECUTE_PRICE prices, weighted by
#      shares. This is the real Volume Profile / Point of Control concept:
#      where actual trading volume concentrated, not just where orders were
#      submitted.
#
# Both are needed because your Top-K sketch will eventually be validated
# against ONE of these (or both) -- feeding execution-weighted data into the
# sketch and comparing against the submitted-interest histogram would compare
# two different questions, not test the sketch's correctness.
#
# Run: python window_frequency_plots_v2.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths — edit if your layout differs ──────────────────────────────────────
PARQUET      = r"C:\faculta\LICENTA\flexihist\sw_HistogramComposer\aapl_book_events_regular.parquet"
OUT_DIR_SUB  = r"C:\faculta\LICENTA\flexihist\out\window_freq_plots_submitted"
OUT_DIR_TRD  = r"C:\faculta\LICENTA\flexihist\out\window_freq_plots_traded"

# ── Parameters (same as the original script) ─────────────────────────────────
BIN_WIDTH      = 0.10
N_WINDOWS      = 13
PRICE_DECIMALS = 2

NS_PER_MINUTE   = 60_000_000_000
WINDOW_START_NS = 9 * 3_600_000_000_000 + 30 * NS_PER_MINUTE  # 09:30 ET

os.makedirs(OUT_DIR_SUB, exist_ok=True)
os.makedirs(OUT_DIR_TRD, exist_ok=True)


def build_windows():
    windows = []
    for i in range(N_WINDOWS):
        t0 = WINDOW_START_NS + i * 30 * NS_PER_MINUTE
        t1 = t0 + 30 * NS_PER_MINUTE
        hh = (9 * 60 + 30 + i * 30) // 60
        mm = (9 * 60 + 30 + i * 30) % 60
        windows.append({"index": i, "t0": t0, "t1": t1, "label": f"{hh:02d}:{mm:02d}"})
    return windows


def plot_windowed_histograms(df_subset, windows, weight_col, out_dir, subset_label):
    """
    df_subset  : the rows to histogram (already filtered by event type)
    weight_col : None -> count rows; or a column name -> sum that column
                 (used for shares-weighted execution volume)
    """
    if len(df_subset) == 0:
        print(f"  WARNING: {subset_label} subset is empty, skipping.")
        return

    prices = df_subset["price"].round(PRICE_DECIMALS).to_numpy()
    weights_all = df_subset[weight_col].to_numpy() if weight_col else None

    p_lo = np.floor(np.percentile(prices, 0.5) / BIN_WIDTH) * BIN_WIDTH
    p_hi = np.ceil(np.percentile(prices, 99.5) / BIN_WIDTH) * BIN_WIDTH
    bin_edges = np.arange(p_lo, p_hi + BIN_WIDTH, BIN_WIDTH)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    print(f"\n[{subset_label}] rows={len(df_subset):,}  "
          f"price range=${p_lo:.2f}-${p_hi:.2f}  bins={len(bin_edges)-1}")
    if weight_col:
        print(f"[{subset_label}] total {weight_col} = {df_subset[weight_col].sum():,}")

    window_data = []
    max_count_seen = 0
    for w in windows:
        mask = (df_subset["timestamp_ns"] >= w["t0"]) & (df_subset["timestamp_ns"] < w["t1"])
        wp = prices[mask.to_numpy()]
        ww = weights_all[mask.to_numpy()] if weights_all is not None else None
        counts, _ = np.histogram(wp, bins=bin_edges, weights=ww)
        window_data.append((w, wp, counts))
        if len(counts):
            max_count_seen = max(max_count_seen, counts.max())

    for w, wp, counts in window_data:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(bin_centers, counts, width=BIN_WIDTH * 0.9, color="#2166ac", edgecolor="none")

        ylabel = "Shares (weighted)" if weight_col else "Order count"
        n_label = f"n={len(wp):,} events"
        if weight_col:
            n_label += f", total {weight_col}={counts.sum():,.0f}"

        ax.set_title(
            f"AAPL {subset_label} — Window {w['index']} ({w['label']} ET)\n{n_label}",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Price (USD)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(p_lo, p_hi)
        ax.set_ylim(0, max_count_seen * 1.05 if max_count_seen else 1)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fname = f"window_{w['index']:02d}_{w['label'].replace(':', '')}_freq.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

    print(f"[{subset_label}] saved {N_WINDOWS} plots to {out_dir}")


def main():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)
    windows = build_windows()

    print("\nEvent type breakdown (whole session):")
    print(df["event"].value_counts().to_string())

    # ── Subset 1: submitted interest (Add orders, unweighted) ────────────────
    add_df = df[df["event"] == "ADD"].copy()
    plot_windowed_histograms(add_df, windows, weight_col=None,
                              out_dir=OUT_DIR_SUB, subset_label="Submitted Interest (Add Orders)")

    # ── Subset 2: traded interest (executions, weighted by shares) ───────────
    exec_df = df[df["event"].isin(["EXECUTE", "EXECUTE_PRICE"])].copy()
    plot_windowed_histograms(exec_df, windows, weight_col="shares",
                              out_dir=OUT_DIR_TRD, subset_label="Traded Interest (Executed Volume)")

    print("\nDone.")


if __name__ == "__main__":
    main()