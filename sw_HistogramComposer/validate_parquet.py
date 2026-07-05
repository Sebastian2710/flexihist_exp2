"""
validate_parquet.py -- sanity-check an AI-generated ITCH parser's output
before trusting it in the thesis pipeline.

Run:  py validate_parquet.py
Edit PARQUET below if your filename differs.
"""
import pandas as pd

PARQUET = "aapl_orders.parquet"   # <-- change if needed

df = pd.read_parquet(PARQUET)

print("="*60)
print("STEP 0: structure")
print("="*60)
print(f"rows: {len(df):,}")
print(f"columns: {list(df.columns)}")
print(df.dtypes)
print()
print(df.head(5).to_string())
print()

# --- try to auto-detect the relevant columns by common names ---------------
def find_col(candidates):
    for c in candidates:
        for col in df.columns:
            if c.lower() == col.lower():
                return col
    return None

price_col   = find_col(["price"])
time_col    = find_col(["timestamp", "time", "ts"])
locate_col  = find_col(["stock_locate", "locate"])
symbol_col  = find_col(["symbol", "stock"])
type_col    = find_col(["msg_type", "message_type", "type"])

print("="*60)
print("STEP 1: row count sanity")
print("="*60)
print(f"Total rows: {len(df):,}")
print("Expected order of magnitude for one day of AAPL Add Orders: ~10^5-10^6")
print("-> OK" if 1e4 < len(df) < 5e6 else "-> SUSPICIOUS, check filtering logic")
print()

if price_col:
    print("="*60)
    print(f"STEP 2: price range sanity (column: '{price_col}')")
    print("="*60)
    pmin, pmax, pmean = df[price_col].min(), df[price_col].max(), df[price_col].mean()
    print(f"min={pmin}  max={pmax}  mean={pmean:.1f}")
    # ITCH prices are integers in units of 1/10000 $.
    # AAPL traded roughly $265-$295 in Dec 2019 -> expect ~2,650,000-2,950,000
    print("If ITCH-scaled (1/10000 $): expect roughly 2,600,000-3,000,000 for AAPL Dec 2019.")
    print("If already converted to dollars: expect roughly 260-300.")
    implied_dollars_raw   = pmean / 10000
    print(f"mean / 10000 = {implied_dollars_raw:.2f}  (does this look like a real AAPL price?)")
else:
    print("STEP 2: SKIPPED -- no 'price' column found. Columns are:", list(df.columns))
print()

if time_col:
    print("="*60)
    print(f"STEP 3: timestamp monotonicity (column: '{time_col}')")
    print("="*60)
    is_sorted = df[time_col].is_monotonic_increasing
    n_decreases = (df[time_col].diff() < 0).sum()
    print(f"strictly non-decreasing: {is_sorted}")
    print(f"number of backward jumps: {n_decreases}")
    print("-> OK" if is_sorted else "-> SUSPICIOUS: ITCH messages should arrive in time order")
else:
    print("STEP 3: SKIPPED -- no timestamp column found. Columns are:", list(df.columns))
print()

if locate_col:
    print("="*60)
    print(f"STEP 4: stock_locate consistency (column: '{locate_col}')")
    print("="*60)
    uniq = df[locate_col].unique()
    print(f"unique values: {uniq}")
    print("-> OK, single symbol" if len(uniq) == 1 else "-> SUSPICIOUS: multiple symbols mixed in")
elif symbol_col:
    print(f"STEP 4: symbol consistency (column: '{symbol_col}')")
    uniq = df[symbol_col].unique()
    print(f"unique values: {uniq}")
    print("-> OK, single symbol" if len(uniq) == 1 else "-> SUSPICIOUS: multiple symbols mixed in")
else:
    print("STEP 4: SKIPPED -- no locate/symbol column found. Columns are:", list(df.columns))
print()

if type_col:
    print("="*60)
    print(f"STEP 5: message type check (column: '{type_col}')")
    print("="*60)
    uniq = df[type_col].unique()
    print(f"unique values: {uniq}")
    print("-> OK, single message type" if len(uniq) == 1 else "-> Check: should be Add Orders only ('A')")
else:
    print("STEP 5: SKIPPED -- no message-type column found (may have been filtered before saving).")
print()

print("="*60)
print("DONE -- review each '-> OK' / '-> SUSPICIOUS' line above.")
print("="*60)