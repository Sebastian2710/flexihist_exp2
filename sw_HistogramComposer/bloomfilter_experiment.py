# bloomfilter_experiment.py
#
# Implements a Bloom Filter and validates it against exact ground truth,
# processed as ONE continuous filter across the ENTIRE regular session --
# unlike the earlier sketches, this one is NOT reset per window.
#
# WHY NOT PER-WINDOW: the real use case here is duplicate/replay-message
# detection on order reference numbers -- "has this exact order reference
# ever been seen" -- a question that only ever grows, never needs
# forgetting. A plain Bloom Filter can never remove an item once inserted
# (that's exactly why Quotient Filter, which supports deletion, was
# flagged earlier as the better fit for a DIFFERENT question -- "is this
# order CURRENTLY open" -- which needs to forget closed orders). Resetting
# this filter every 30 minutes would misrepresent what it's actually for.
#
# INPUT: order_ref values from ADD events in aapl_book_events_regular.parquet.
# Sized ONCE, upfront, for the day's total expected volume -- exactly the
# real decision a hardware implementation faces (you don't know the exact
# final count in advance). The experiment then checks, every 30 minutes,
# how false-positive rate evolves as the fixed-size filter genuinely fills.
#
# NEGATIVE TEST SET: window 12's order_ref values, held out and NEVER
# inserted until the very last checkpoint. Querying them earlier in the
# day is a genuine "has this actually-unseen thing been mistaken for seen"
# test, not a synthetic random-number stand-in -- and using the SAME
# held-out set at every checkpoint keeps the comparison apples-to-apples
# as the filter fills.
#
# CRITICAL: order_ref is a large (up to 8-byte) integer -- an even more
# important place to apply the Count-Min overflow lesson than price was.
# Every hashed value is forced into a genuine Python int, and the hash
# modulus prime is chosen comfortably larger than any realistic order
# reference, more conservative than Count-Min's prime on purpose.
#
# Run: python bloomfilter_experiment.py

import math
import numpy as np
import pandas as pd

PARQUET = r"C:\faculta\LICENTA\FlexiHist\sw_HistogramComposer\aapl_book_events_regular.parquet"

N_WINDOWS       = 13
NS_PER_MINUTE   = 60_000_000_000
WINDOW_START_NS = 9 * 3_600_000_000_000 + 30 * NS_PER_MINUTE  # 09:30 ET

BITS_PER_ITEM_SWEEP = [1, 2, 4, 8, 12, 16]   # bits allocated per expected item -- the main sweep
K = 4                                    # number of hash functions, fixed -- mirrors Count-Min's depth

# Prime comfortably larger than any realistic order_ref (up to 8 bytes) --
# more conservative than Count-Min's prime, deliberately.
HASH_PRIME = 2**61 - 1

# BUGFIX: the first version used small coefficients (~10^6). Order
# references here are modest integers (tens of millions at most), so
# a*item_int never reached anywhere near HASH_PRIME (~2.3*10^18) -- the
# "mod p" step was a silent no-op, degenerating the hash into a plain
# LINEAR function of the input. Order references are assigned roughly
# sequentially, so a linear map produces evenly-spaced, deterministic
# collisions instead of a genuine scatter -- exactly what caused empirical
# false-positive rates thousands of times above the theoretical prediction.
#
# FIX: coefficients must be large enough, relative to the prime, that
# a*item_int genuinely wraps around many times regardless of how small or
# structured item_int is. These are now comparable in magnitude to
# HASH_PRIME itself.
HASH_COEFFS = [
    (1152921504606846883, 998877665544332211),
    (2019283746574839201, 123456789987654321),
    (761234567890123457, 555555555555555555),
    (1889999999999999937, 111111111111111111),
]
assert all(0 < a < HASH_PRIME and 0 <= b < HASH_PRIME for a, b in HASH_COEFFS)
assert all(a > HASH_PRIME // 1000 for a, b in HASH_COEFFS), \
    "coefficient too small relative to prime -- risks the degenerate near-linear hash bug"


class BloomFilter:
    """
    Plain Bloom Filter. `k` independent hash functions set `k` bits per
    inserted item. Membership: an item is "possibly present" only if ALL
    k of its bits are set; if even one bit is unset, the item is
    DEFINITELY absent -- this is what makes false negatives structurally
    impossible, and it's the hard guarantee checked below.
    """
    def __init__(self, size_bits, k):
        self.size_bits = size_bits
        self.k = k
        self.coeffs = HASH_COEFFS[:k]
        self.bits = np.zeros(size_bits, dtype=bool)

    def _hash(self, item_int, row):
        item_int = int(item_int)   # force genuine Python int -- see header note
        a, b = self.coeffs[row]
        return ((a * item_int + b) % HASH_PRIME) % self.size_bits

    def add(self, item_int):
        for row in range(self.k):
            self.bits[self._hash(item_int, row)] = True

    def contains(self, item_int):
        return all(self.bits[self._hash(item_int, row)] for row in range(self.k))

    def fill_ratio(self):
        return self.bits.mean()


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
    add_df = df[df["event"] == "ADD"].sort_values("timestamp_ns")
    windows = build_windows()

    # Distinct order_ref per window, in time order
    window_refs = []
    for w in windows:
        mask = (add_df["timestamp_ns"] >= w["t0"]) & (add_df["timestamp_ns"] < w["t1"])
        refs = add_df.loc[mask, "order_ref"].unique().tolist()
        window_refs.append(refs)

    n_total = sum(len(r) for r in window_refs)
    held_out_negatives = window_refs[-1]   # window 12's refs -- untouched until the very end

    print(f"\nTotal distinct order_ref (ADD events, whole day): {n_total:,}")
    print(f"Held-out negative test set (window 12 refs, untouched until the end): "
          f"{len(held_out_negatives):,}")

    for bits_per_item in BITS_PER_ITEM_SWEEP:
        m = round(bits_per_item * n_total)
        print(f"\n{'='*72}\nbits_per_item={bits_per_item}  k={K}  "
              f"total bits m={m:,}  ({m/8/1024:.1f} KB)\n{'='*72}")

        bf = BloomFilter(m, K)
        false_negatives = 0
        inserted_so_far = 0

        for i, w in enumerate(windows):
            for ref in window_refs[i]:
                bf.add(ref)
                if not bf.contains(ref):
                    false_negatives += 1   # should NEVER happen
            inserted_so_far += len(window_refs[i])

            if i < N_WINDOWS - 1:   # skip the last window -- no held-out data left
                fp_count = sum(1 for ref in held_out_negatives if bf.contains(ref))
                empirical_fpr = fp_count / len(held_out_negatives)
            else:
                empirical_fpr = None

            theoretical_fpr = (1 - math.exp(-K * inserted_so_far / m)) ** K

            fpr_str = f"{empirical_fpr*100:6.3f}%" if empirical_fpr is not None else "   n/a"
            flag = "  <-- FALSE NEGATIVE, BUG!" if false_negatives else ""
            print(f"  Window {w['index']:2d} ({w['label']})  "
                  f"inserted so far={inserted_so_far:7,}  "
                  f"fill={bf.fill_ratio()*100:5.1f}%  "
                  f"empirical FPR={fpr_str}  theoretical FPR={theoretical_fpr*100:6.3f}%  "
                  f"false negatives={false_negatives}{flag}")


if __name__ == "__main__":
    main()