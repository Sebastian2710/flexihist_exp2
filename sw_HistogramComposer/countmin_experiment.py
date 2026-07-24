# countmin_experiment.py
#
# Implements Count-Min Sketch (Cormode & Muthukrishnan, 2005) and validates
# it against exact ground truth, per 30-minute window, across a sweep of
# sketch widths.
#
# BUGFIX (v2): the first version silently overflowed on Windows, where
# numpy's default integer width from a plain Python list is 32-bit (not
# 64-bit, as on Linux/macOS). Hash coefficients like 3,221,225,473 exceed
# 32-bit range, so `a * item_int` wrapped around inconsistently depending
# on which numpy dtype happened to be in play at that call site -- update()
# and query() ended up hashing the SAME price to DIFFERENT table columns,
# which is exactly what produced the "impossible" underestimates (Count-Min
# can only ever overestimate; underestimation is proof the hash function
# wasn't actually consistent between the two calls).
#
# FIX: every price is forced into a genuine, arbitrary-precision Python int
# right at the point it enters the hash function (_hash), so no numpy
# fixed-width dtype can ever leak into the arithmetic, regardless of what
# dtype the array happens to be upstream.
#
# UNLIKE Top-K, Count-Min Sketch cannot enumerate what it's tracking -- it
# only stores counters, never item identities. You must already know which
# price to ask about; the sketch can only answer, never list its contents.
# That's why this script tests TWO different query sets, not one:
#
#   Query Set A: the TRUE top-3 prices (from the Top-K experiment) -- does
#                Count-Min agree with Top-K on the heavy hitters, even
#                though it wasn't built to specifically track them?
#
#   Query Set B: EVERY distinct price seen in the window -- this is Count-
#                Min's actual selling point, an arbitrary point query, not
#                just the busy ones. Testing only Set A would understate
#                what this sketch is actually for.
#
# Run: python countmin_experiment.py

import numpy as np
import pandas as pd

PARQUET = r"C:\faculta\LICENTA\FlexiHist\sw_HistogramComposer\aapl_book_events_regular.parquet"

N_WINDOWS       = 13
NS_PER_MINUTE   = 60_000_000_000
WINDOW_START_NS = 9 * 3_600_000_000_000 + 30 * NS_PER_MINUTE  # 09:30 ET

WIDTH_SIZES    = [16, 32, 64, 128, 256]   # counters per row -- main sweep, adjust after seeing results
DEPTH          = 2                         # independent hash rows -- fixed for now, rarely needs > 4-5
TOP_K_REPORT   = 3
PRICE_DECIMALS = 2

# Large prime > any possible hashed integer -- prices in integer cents are
# well under this, so no wraparound concerns AS LONG AS arithmetic stays in
# genuine Python ints (see _hash below -- this is the part that was broken).
HASH_PRIME = 2_147_483_647  # 2^31 - 1 (Mersenne prime)

# Carter-Wegman universal hashing: h(x) = ((a*x + b) mod p) mod width.
# Citation: Carter, J.L., Wegman, M.N. (1979). Universal Classes of Hash
# Functions. Journal of Computer and System Sciences, 18(2), 143-154.
# All coefficients kept comfortably under HASH_PRIME for clarity (not
# required for correctness once _hash forces Python-int arithmetic, but
# avoids any confusion about what value is "actually" being used mod p).
# Fixed pairs, one per depth row -- reproducible across runs, not
# re-randomized each time.
HASH_COEFFS = [
    (1000003, 5000011),
    (2000003, 7000003),
    (3000017, 9000011),
    (4000037, 11000003),
    (500009, 13000003),
]
assert all(0 < a < HASH_PRIME and 0 <= b < HASH_PRIME for a, b in HASH_COEFFS), \
    "hash coefficients out of range"


class CountMinSketch:
    """
    Count-Min Sketch. `depth` independent counter rows of `width` counters.
    update() increments one counter per row. query() returns the MINIMUM
    across all rows -- this is what gives Count-Min its one-sided
    guarantee: a hash collision can only ever inflate a counter, never
    deflate one, so the true count can never be underestimated, only
    overestimated -- PROVIDED the same item always hashes to the same
    position. See _hash() for why that required an explicit fix.

    Stores counters ONLY -- no item identities. Cannot enumerate what it
    has seen; can only answer a query about an item you already specify.
    """
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.coeffs = HASH_COEFFS[:depth]

    def _hash(self, item_int, row):
        # ── THE FIX ──────────────────────────────────────────────────────
        # Force genuine, arbitrary-precision Python int arithmetic here,
        # no matter what numpy dtype item_int arrived as (int32, int64,
        # numpy scalar, whatever). This is what guarantees update() and
        # query() always compute the SAME column for the SAME price.
        item_int = int(item_int)
        # ─────────────────────────────────────────────────────────────────
        a, b = self.coeffs[row]
        return ((a * item_int + b) % HASH_PRIME) % self.width

    def update(self, item_int, weight=1):
        for row in range(self.depth):
            col = self._hash(item_int, row)
            self.table[row, col] += weight

    def query(self, item_int):
        return min(self.table[row, self._hash(item_int, row)] for row in range(self.depth))


def price_to_int(price):
    """
    ── EXPLICIT CONVERSION POINT ──────────────────────────────────────────
    Convert a dollar price to an integer number of CENTS before it ever
    touches the hash function. Count-Min hashes integers; hashing floats
    directly is bad practice (tiny representation differences break
    equality) and inconsistent with the "data is integer" framing this
    whole thesis is built on.
    ────────────────────────────────────────────────────────────────────────
    """
    return int(round(price * 100))


def build_windows():
    windows = []
    for i in range(N_WINDOWS):
        t0 = WINDOW_START_NS + i * 30 * NS_PER_MINUTE
        t1 = t0 + 30 * NS_PER_MINUTE
        hh = (9 * 60 + 30 + i * 30) // 60
        mm = (9 * 60 + 30 + i * 30) % 60
        windows.append({"index": i, "t0": t0, "t1": t1, "label": f"{hh:02d}:{mm:02d}"})
    return windows


def true_counts(prices_int, weights):
    """Exact ground truth: sum weights per exact integer-cent price."""
    return pd.Series(weights, index=prices_int).groupby(level=0).sum()


def run_experiment(df_subset, weight_col, label, windows):
    prices_dollars = df_subset["price"].round(PRICE_DECIMALS).to_numpy()
    # Explicit int64, belt-and-suspenders alongside the _hash fix above --
    # this alone would NOT have been enough without the fix, since the
    # bug was about update()/query() disagreeing, not just one bad dtype.
    prices_int = np.array([price_to_int(p) for p in prices_dollars], dtype=np.int64)
    weights = df_subset[weight_col].to_numpy() if weight_col else np.ones(len(df_subset))
    timestamps = df_subset["timestamp_ns"].to_numpy()

    print(f"\n{'='*72}\n{label}  (n={len(df_subset):,} rows)\n{'='*72}")

    for w in windows:
        mask = (timestamps >= w["t0"]) & (timestamps < w["t1"])
        wp, ww = prices_int[mask], weights[mask]
        if len(wp) == 0:
            print(f"Window {w['index']:2d} ({w['label']}): no data, skipping")
            continue

        true_c = true_counts(wp, ww)
        true_sorted = true_c.sort_values(ascending=False)
        top3_items = [int(p) for p in true_sorted.head(TOP_K_REPORT).index]
        all_distinct_items = [int(p) for p in true_c.index]  # Query Set B

        print(f"\nWindow {w['index']:2d} ({w['label']})  n={len(wp):,}  "
              f"distinct prices={len(all_distinct_items):,}")
        print(f"  TRUE top-3 (cents): " +
              ", ".join(f"{p}={c:,.0f}" for p, c in true_sorted.head(TOP_K_REPORT).items()))

        for width in WIDTH_SIZES:
            sketch = CountMinSketch(width, DEPTH)
            for p, wt in zip(wp, ww):
                sketch.update(p, wt)

            violations = 0

            # ── Query Set A: known heavy hitters ──────────────────────────
            errsA = []
            for item in top3_items:
                est = sketch.query(item)
                true_v = true_c[item]
                if est < true_v:
                    violations += 1
                errsA.append((est - true_v) / true_v * 100)

            # ── Query Set B: every distinct price -- the real use case ────
            errsB = []
            for item in all_distinct_items:
                est = sketch.query(item)
                true_v = true_c[item]
                if est < true_v:
                    violations += 1
                errsB.append((est - true_v) / true_v * 100)

            avgA, avgB, maxB = np.mean(errsA), np.mean(errsB), np.max(errsB)
            flag = "  <-- UNDERESTIMATE, BUG!" if violations else ""

            print(f"    width={width:4d} depth={DEPTH}: "
                  f"top-3 avg over-err={avgA:6.1f}%   "
                  f"all-distinct avg over-err={avgB:6.1f}%  max={maxB:7.1f}%"
                  f"  ({violations} underestimate violations){flag}")


def main():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)
    windows = build_windows()

    add_df = df[df["event"] == "ADD"]
    run_experiment(add_df, weight_col=None,
                   label="SUBMITTED INTEREST (Add orders, unweighted count)",
                   windows=windows)

    exec_df = df[df["event"].isin(["EXECUTE", "EXECUTE_PRICE"])]
    run_experiment(exec_df, weight_col="shares",
                   label="TRADED INTEREST (Executions, weighted by shares)",
                   windows=windows)


if __name__ == "__main__":
    main()