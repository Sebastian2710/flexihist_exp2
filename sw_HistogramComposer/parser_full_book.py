# parser_full_book.py
#
# Order-book reconstruction: Add (A/F), Order Executed (E), Order Executed
# With Price (C), Order Cancel (X), Order Delete (D), Order Replace (U).
# Output restricted to the regular trading session (09:30:00-16:00:00 ET).
# The book itself is still built from the FIRST AAPL message in the file
# (including pre-market), so best_bid/best_ask at 09:30:00.000 correctly
# reflects resting liquidity carried over from pre-market -- only the
# OUTPUT rows are filtered, not the book state.
#
# Two more fields per event beyond a basic book reconstruction:
#   - best_bid_size / best_ask_size  -> needed for Order Flow Imbalance
#     (Cont, Kukanov, Stoikov 2014 define OFI using the change in
#     resting size at the best level, not just the price)
#   - side / price / shares of the event itself -> needed for VWAP
#     (execution price x executed shares) and for splitting buy-side vs
#     sell-side executed volume
#
# PRICE RANGE: kept RAW/unclipped on purpose, same reasoning as parser.py --
# this lets the traded-interest and submitted-interest comparisons include
# the full price range (including stub-quote-only prices that never trade),
# which is exactly the kind of clutter a Top-K/heavy-hitters sketch needs to
# be tested against. PRICE_MIN / PRICE_MAX below are an explicit,
# off-by-default toggle if you ever want a clipped range instead.
#
# Run: python parser_full_book.py

import gzip
import struct
import heapq
import pandas as pd

# ── EDIT THESE TWO LINES ──────────────────────────────────────────────────────
ITCH_FILE   = r"C:\faculta\LICENTA\FlexiHist\sw_HistogramComposer\12302019.NASDAQ_ITCH50.gz"
OUTPUT_FILE = r"C:\faculta\LICENTA\FlexiHist\sw_HistogramComposer\aapl_book_events_regular.parquet"
# ─────────────────────────────────────────────────────────────────────────────

AAPL_LOCATE = 13

# Regular session bounds, ns since midnight ET (same convention as experiment.py)
SESSION_START_NS = 9 * 3_600_000_000_000 + 30 * 60_000_000_000   # 09:30:00.000
SESSION_END_NS   = 16 * 3_600_000_000_000                          # 16:00:00.000

# Optional price clip -- OFF by default (None, None) to keep raw prices.
# Applies only to which rows get EMITTED, not to what the book tracks
# internally -- an out-of-range resting order still correctly affects
# best_bid/best_ask for in-range orders if it's genuinely the best price.
PRICE_MIN = None
PRICE_MAX = None

MSG_LEN = {
    ord('A'): 36, ord('F'): 40, ord('E'): 31, ord('C'): 36,
    ord('X'): 23, ord('D'): 19, ord('U'): 35,
}
BOOK_MSGS = set(MSG_LEN.keys())


def decode_ts(body: bytes) -> int:
    return struct.unpack_from(">Q", b"\x00\x00" + body[5:11])[0]


class OrderBook:
    def __init__(self):
        self.orders = {}        # order_ref -> [side, price, shares, mpid]
        self.bid_levels = {}
        self.ask_levels = {}
        self.bid_heap = []
        self.ask_heap = []

    def _bump_level(self, side, price, delta):
        book = self.bid_levels if side == 'B' else self.ask_levels
        book[price] = book.get(price, 0) + delta
        if side == 'B':
            heapq.heappush(self.bid_heap, -price)
        else:
            heapq.heappush(self.ask_heap, price)

    def add_order(self, ref, side, price, shares, mpid):
        self.orders[ref] = [side, price, shares, mpid]
        self._bump_level(side, price, shares)

    def reduce_order(self, ref, shares_delta):
        o = self.orders.get(ref)
        if o is None:
            return
        side, price, shares, mpid = o
        book = self.bid_levels if side == 'B' else self.ask_levels
        book[price] = book.get(price, 0) - shares_delta
        remaining = shares - shares_delta
        if remaining <= 0:
            del self.orders[ref]
        else:
            o[2] = remaining

    def delete_order(self, ref):
        o = self.orders.pop(ref, None)
        if o is None:
            return
        side, price, shares, mpid = o
        book = self.bid_levels if side == 'B' else self.ask_levels
        book[price] = book.get(price, 0) - shares

    def best_bid(self):
        while self.bid_heap:
            price = -self.bid_heap[0]
            if self.bid_levels.get(price, 0) > 0:
                return price
            heapq.heappop(self.bid_heap)
        return None

    def best_ask(self):
        while self.ask_heap:
            price = self.ask_heap[0]
            if self.ask_levels.get(price, 0) > 0:
                return price
            heapq.heappop(self.ask_heap)
        return None

    def best_bid_size(self):
        p = self.best_bid()
        return self.bid_levels.get(p, 0) if p is not None else None

    def best_ask_size(self):
        p = self.best_ask()
        return self.ask_levels.get(p, 0) if p is not None else None


def parse(itch_path: str, output_path: str) -> pd.DataFrame:
    book = OrderBook()
    records = []
    msg_count = 0
    aapl_count = 0
    kept_count = 0
    clipped_count = 0

    print(f"Opening {itch_path} ...")
    with gzip.open(itch_path, "rb") as f:
        while True:
            len_bytes = f.read(2)
            if len(len_bytes) < 2:
                break

            body_len = struct.unpack(">H", len_bytes)[0]
            body = f.read(body_len)
            if len(body) < body_len:
                print("WARNING: truncated message at end of file — stopping.")
                break

            msg_count += 1
            if msg_count % 20_000_000 == 0:
                print(f"  {msg_count:,} messages processed, "
                      f"{kept_count:,} regular-session AAPL events kept so far ...")

            mtype = body[0]
            if mtype not in BOOK_MSGS:
                continue

            locate = struct.unpack_from(">H", body, 1)[0]
            if locate != AAPL_LOCATE:
                continue

            ts_ns = decode_ts(body)
            ref = struct.unpack_from(">Q", body, 11)[0]

            side = price = shares = mpid = None

            if mtype in (ord('A'), ord('F')):
                side = chr(body[19])
                shares = struct.unpack_from(">I", body, 20)[0]
                price = struct.unpack_from(">I", body, 32)[0] / 10_000.0
                mpid = body[36:40].decode('ascii').strip() if mtype == ord('F') else None
                book.add_order(ref, side, struct.unpack_from(">I", body, 32)[0], shares, mpid)
                event = "ADD"

            elif mtype == ord('E'):
                exec_shares = struct.unpack_from(">I", body, 19)[0]
                o = book.orders.get(ref)
                if o is not None:
                    side, raw_price, mpid = o[0], o[1], o[3]
                    price = raw_price / 10_000.0
                shares = exec_shares
                book.reduce_order(ref, exec_shares)
                event = "EXECUTE"

            elif mtype == ord('C'):
                exec_shares = struct.unpack_from(">I", body, 19)[0]
                raw_price = struct.unpack_from(">I", body, 32)[0]
                price = raw_price / 10_000.0
                o = book.orders.get(ref)
                if o is not None:
                    side = o[0]
                    mpid = o[3]
                shares = exec_shares
                book.reduce_order(ref, exec_shares)
                event = "EXECUTE_PRICE"

            elif mtype == ord('X'):
                cancel_shares = struct.unpack_from(">I", body, 19)[0]
                o = book.orders.get(ref)
                if o is not None:
                    side, raw_price, mpid = o[0], o[1], o[3]
                    price = raw_price / 10_000.0
                shares = cancel_shares
                book.reduce_order(ref, cancel_shares)
                event = "CANCEL"

            elif mtype == ord('D'):
                o = book.orders.get(ref)
                if o is not None:
                    side, raw_price, raw_shares, mpid = o[0], o[1], o[2], o[3]
                    price = raw_price / 10_000.0
                    shares = raw_shares
                book.delete_order(ref)
                event = "DELETE"

            elif mtype == ord('U'):
                old = book.orders.get(ref)
                side = old[0] if old else None
                mpid = old[3] if old else None
                book.delete_order(ref)
                new_ref = struct.unpack_from(">Q", body, 19)[0]
                new_shares = struct.unpack_from(">I", body, 27)[0]
                new_price_raw = struct.unpack_from(">I", body, 31)[0]
                if side is not None:
                    book.add_order(new_ref, side, new_price_raw, new_shares, mpid)
                price = new_price_raw / 10_000.0
                shares = new_shares
                event = "REPLACE"

            aapl_count += 1

            if not (SESSION_START_NS <= ts_ns < SESSION_END_NS):
                continue  # book state still updated above; just not emitted

            # ── Optional price clip on the EMITTED row only (book state above ──
            # already reflects this event regardless -- see class docstring note)
            if price is not None:
                if PRICE_MIN is not None and price < PRICE_MIN:
                    clipped_count += 1
                    continue
                if PRICE_MAX is not None and price > PRICE_MAX:
                    clipped_count += 1
                    continue

            bb, ba = book.best_bid(), book.best_ask()
            bb_d = bb / 10_000.0 if bb is not None else None
            ba_d = ba / 10_000.0 if ba is not None else None
            spread = (ba_d - bb_d) if (bb_d is not None and ba_d is not None) else None
            bb_size = book.best_bid_size()
            ba_size = book.best_ask_size()

            records.append((
                ts_ns, event, ref, side, mpid, price, shares,
                bb_d, ba_d, spread, bb_size, ba_size
            ))
            kept_count += 1

    print("\nParsing complete.")
    print(f"  Total messages read       : {msg_count:,}")
    print(f"  AAPL book events (all)    : {aapl_count:,}")
    print(f"  Regular-session events    : {kept_count:,}")
    if PRICE_MIN is not None or PRICE_MAX is not None:
        print(f"  Price clip applied        : [{PRICE_MIN}, {PRICE_MAX}]  "
              f"({clipped_count:,} rows excluded)")
    else:
        print(f"  Price clip applied        : none (raw)")

    df = pd.DataFrame(records, columns=[
        "timestamp_ns", "event", "order_ref", "side", "mpid", "price", "shares",
        "best_bid", "best_ask", "spread", "best_bid_size", "best_ask_size"
    ])
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to                  : {output_path}")

    return df


if __name__ == "__main__":
    df = parse(ITCH_FILE, OUTPUT_FILE)

    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    print("\nSpread sanity check (regular session only):")
    valid_spread = df["spread"].dropna()
    print(f"  n with valid spread : {len(valid_spread):,} / {len(df):,}")
    print(f"  min spread          : {valid_spread.min():.4f}")
    print(f"  median spread       : {valid_spread.median():.4f}")
    print(f"  max spread          : {valid_spread.max():.4f}")
    print(f"  negative spreads    : {(valid_spread < 0).sum():,}")

    print("\nEvent type breakdown:")
    print(df["event"].value_counts().to_string())

    print("\nExecuted-volume sanity check (needed for VWAP):")
    exec_df = df[df["event"].isin(["EXECUTE", "EXECUTE_PRICE"])]
    print(f"  n executions        : {len(exec_df):,}")
    print(f"  total executed shares: {exec_df['shares'].sum():,}")
    print(f"  by side:\n{exec_df.groupby('side')['shares'].sum().to_string()}")