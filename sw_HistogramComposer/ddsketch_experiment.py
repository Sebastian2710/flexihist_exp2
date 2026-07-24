# ddsketch_experiment.py
#
# Implements DDSketch (Masson, Rim, Lee, 2019) and validates it against
# exact percentiles, per 30-minute window, across a sweep of the sketch's
# relative-accuracy target (alpha).
#
# BUGFIX (v2): the first version used numpy's default linear-interpolation
# percentile as ground truth. For discrete, clustered data like share sizes
# (which come in round lots -- 100, 200, 300...), interpolation between two
# real, far-apart order statistics manufactures FRACTIONAL values that were
# never actually observed (e.g. "360.68 shares" -- shares are always whole
# numbers). DDSketch's guarantee only covers values that were actually
# inserted; testing it against a fabricated in-between value that was never
# fed to the sketch produced spurious "guarantee violations" that were
# never a real bug in the sketch itself.
#
# FIX: ground truth is now computed with the same RANK-BASED definition the
# sketch itself uses internally (the value at rank ceil(q*n) in the sorted
# data) -- so both sides are answering the identical question, and every
# ground-truth value is guaranteed to be something the sketch actually saw.
#
# TWO INPUTS are run, deliberately, because they tell different stories:
#
#   SPREAD  -- narrow, penny-quantized range (~$0.01-$0.45 across the whole
#              day). Bucket count came back completely FLAT across every
#              alpha tested -- DDSketch is correct here, but this quantity's
#              actual scale doesn't create the problem the sketch exists to
#              solve. Kept in as an honest, documented negative result.
#
#   SHARES  -- wide range (1 to 50,000+), the real demonstration of
#              DDSketch's actual value: bucket count genuinely grows as the
#              accuracy target tightens, a real memory-for-accuracy trade.
#
# Run: python ddsketch_experiment.py

import math
import numpy as np
import pandas as pd

PARQUET = r"C:\faculta\LICENTA\FlexiHist\sw_HistogramComposer\aapl_book_events_regular.parquet"

N_WINDOWS       = 13
NS_PER_MINUTE   = 60_000_000_000
WINDOW_START_NS = 9 * 3_600_000_000_000 + 30 * NS_PER_MINUTE  # 09:30 ET

ALPHA_SWEEP = [0.02, 0.01, 0.005, 0.002]   # relative accuracy targets: 2%, 1%, 0.5%, 0.2%
QUANTILES   = [0.50, 0.90, 0.95, 0.99]      # typical -> tail


class DDSketch:
    """
    DDSketch (Masson, Rim, Lee, 2019). Logarithmic bucketing: bucket
    boundaries are gamma^(i-1) < v <= gamma^i, where gamma = (1+alpha)/(1-alpha).
    ANY value falling in bucket i is guaranteed to be within +/- alpha
    relative error of that bucket's returned estimate -- a hard per-value
    bound, not a statistical average -- PROVIDED the value was actually
    inserted (see bugfix note above for why this matters for validation).
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.gamma = (1 + alpha) / (1 - alpha)
        self.log_gamma = math.log(self.gamma)
        self.counts = {}   # bucket index -> count
        self.total = 0

    def _bucket_index(self, value):
        return math.ceil(math.log(value) / self.log_gamma)

    def update(self, value, weight=1):
        idx = self._bucket_index(value)
        self.counts[idx] = self.counts.get(idx, 0) + weight
        self.total += weight

    def _bucket_estimate(self, idx):
        return 2 * (self.gamma ** idx) / (self.gamma + 1)

    def query(self, q):
        """Return the estimated q-th quantile (q in [0, 1])."""
        if self.total == 0:
            return None
        target = math.ceil(q * self.total)
        cumulative = 0
        for idx in sorted(self.counts):
            cumulative += self.counts[idx]
            if cumulative >= target:
                return self._bucket_estimate(idx)
        return self._bucket_estimate(max(self.counts))

    def num_buckets(self):
        return len(self.counts)


def true_rank_value(sorted_values, q):
    """
    Ground truth using the SAME rank definition DDSketch itself uses
    internally (target = ceil(q * n)) -- guarantees the "true" value being
    compared against is always one that was actually observed/inserted,
    unlike a linear-interpolation percentile.
    """
    n = len(sorted_values)
    target = math.ceil(q * n)
    idx = min(target - 1, n - 1)
    return sorted_values[idx]


def build_windows():
    windows = []
    for i in range(N_WINDOWS):
        t0 = WINDOW_START_NS + i * 30 * NS_PER_MINUTE
        t1 = t0 + 30 * NS_PER_MINUTE
        hh = (9 * 60 + 30 + i * 30) // 60
        mm = (9 * 60 + 30 + i * 30) % 60
        windows.append({"index": i, "t0": t0, "t1": t1, "label": f"{hh:02d}:{mm:02d}"})
    return windows


def run_experiment(values, timestamps, label, windows, value_fmt="{:.4f}"):
    print(f"\n{'='*72}\n{label} (n={len(values):,} events)\n{'='*72}")

    for w in windows:
        mask = (timestamps >= w["t0"]) & (timestamps < w["t1"])
        wv = values[mask]
        if len(wv) == 0:
            print(f"Window {w['index']:2d} ({w['label']}): no data, skipping")
            continue

        sorted_wv = np.sort(wv)
        true_q = {q: true_rank_value(sorted_wv, q) for q in QUANTILES}

        print(f"\nWindow {w['index']:2d} ({w['label']})  n={len(wv):,}  "
              f"min={value_fmt.format(wv.min())}  max={value_fmt.format(wv.max())}")
        print("  TRUE   : " + "  ".join(
            f"p{int(q*100)}=" + value_fmt.format(v) for q, v in true_q.items()))

        for alpha in ALPHA_SWEEP:
            sketch = DDSketch(alpha)
            for v in wv:
                sketch.update(v)

            worst_rel_err = 0.0
            violations = 0
            est_strs = []
            for q in QUANTILES:
                est = sketch.query(q)
                true_v = true_q[q]
                rel_err = abs(est - true_v) / true_v
                worst_rel_err = max(worst_rel_err, rel_err)
                if rel_err > alpha * 1.0001:   # tiny float tolerance
                    violations += 1
                est_strs.append(f"p{int(q*100)}=" + value_fmt.format(est))

            flag = "  <-- EXCEEDED GUARANTEE, BUG!" if violations else ""
            print(f"    alpha={alpha:5.3f}  buckets={sketch.num_buckets():3d}: "
                  f"worst rel err={worst_rel_err*100:5.2f}%  "
                  f"({violations} guarantee violations)  " +
                  "  ".join(est_strs) + flag)


def main():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)
    windows = build_windows()

    # ── SPREAD: documented negative result (kept for the record) ──────────
    spread_df = df.dropna(subset=["spread"])
    run_experiment(spread_df["spread"].to_numpy(), spread_df["timestamp_ns"].to_numpy(),
                   label="SPREAD DISTRIBUTION (narrow range -- expect flat bucket count)",
                   windows=windows, value_fmt="${:.4f}")

    # ── SHARES: the real demonstration of DDSketch's value ─────────────────
    shares_df = df.dropna(subset=["shares"])
    run_experiment(shares_df["shares"].to_numpy(), shares_df["timestamp_ns"].to_numpy(),
                   label="SHARES DISTRIBUTION (wide range -- expect real compression tradeoff)",
                   windows=windows, value_fmt="{:.0f}")


if __name__ == "__main__":
    main()