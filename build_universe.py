"""
build_universe.py — rebuild the tradeable universe from LIQUIDITY, not index
membership.

WHY THIS EXISTS
    The scan used to screen the S&P 500. But the S&P 500 is not the 500
    most-traded US stocks, and the measured edge lives in the MOST liquid names
    (holdout by dollar-volume quartile: least liquid +$65, most liquid +$478).
    Screening by index membership therefore excludes names the strategy most
    wants — liquid non-index stocks, big ADRs, recent large listings — while
    including S&P names that fail the liquidity gate anyway.

    It also starves the signal. The call rule holds roughly 3-4 names in the
    triggered state at any moment out of 500. During earnings season almost all
    of them are blocked, so the screen produces nothing for weeks at a time.
    Doubling the pool roughly doubles the flow.

HOW IT WORKS
    Stage 1 (this file, run MONTHLY): take every active US common stock, pull one
    year of daily bars, and keep the names that are genuinely liquid and have
    enough history to score. Writes universe.txt.

    Stage 2 (scan.py, run NIGHTLY): read universe.txt and do the full two-year
    pull and scoring on that list only. Keeping the heavy liquidity sweep out of
    the nightly job is what makes this affordable — the set of names trading
    $100M+/day changes slowly, so monthly is often enough.

HEADROOM, ON PURPOSE
    This file screens at $40 / $75M while scan.py gates at $50 / $100M. The
    looser thresholds here leave a buffer so a name that drifts up in price or
    volume between monthly rebuilds is already in the list when it qualifies.

CANDIDATE LIST
    Fetched at runtime, not committed: every active US common stock from the
    Alpha Vantage listing feed, UNIONED with the S&P 500 constituents. The union
    matters — that feed is missing real names (BXP and DOC were both absent on
    2026-10-12, and CBOE is filed under BATS), so taking it alone would silently
    drop coverage.

    Needs ALPHAVANTAGE_API_KEY as a repository secret. Without it the script
    still runs, on the S&P 500 list alone, and says so loudly — you get the old
    behaviour rather than a crash or a silently narrow universe.

    No symbol-pattern filtering is applied. SPAC units, warrants and rights are
    illiquid and die on the dollar-volume screen anyway, whereas any suffix
    heuristic that strips them also kills LOW, SNOW, GLW and CRWV.
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SESSION = None
try:
    from curl_cffi import requests as _cr
    SESSION = _cr.Session(impersonate="chrome")
    print("curl_cffi session active")
except Exception:
    print("!! curl_cffi NOT installed — expect heavy throttling.")

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()

OUT = HERE / "universe.txt"
AV_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
AV_LISTING = ("https://www.alphavantage.co/query"
              "?function=LISTING_STATUS&state=active&apikey={key}")

YEARS = 1
CHUNK = 120                # short window, so big batches are fine
PAUSE = 1.0
PASSES = 3
MIN_BARS = 240             # a year of trading days, less a little slack
MIN_PRICE = 40             # headroom under scan.py's $50
MIN_DOLLAR_VOL = 75e6      # headroom under scan.py's $100M
TARGET = 1500              # cap, by dollar volume
FLOOR = 400                # below this the run is broken, not selective

SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")


def die(msg):
    print("\n" + "=" * 70)
    print("UNIVERSE BUILD FAILED — universe.txt not written")
    print("=" * 70)
    print(msg)
    sys.exit(1)


def norm_index(df):
    if df is None or not len(df):
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download(batch):
    kw = dict(period=f"{YEARS}y", interval="1d", auto_adjust=False,
              progress=False, group_by="column", threads=False)
    if SESSION is not None:
        kw["session"] = SESSION
    return yf.download(batch, **kw)


def pull(symbols):
    """Chunked download with retry passes. Returns close and volume frames."""
    closes, vols = {}, {}
    todo = list(symbols)
    for p in range(1, PASSES + 1):
        if not todo:
            break
        print(f"\npass {p}: {len(todo)} symbols to fetch")
        failed = []
        for i in range(0, len(todo), CHUNK):
            batch = todo[i:i + CHUNK]
            try:
                raw = _download(batch)
            except Exception as e:
                print(f"   chunk {i//CHUNK + 1}: {type(e).__name__}")
                failed += batch
                time.sleep(PAUSE * 2)
                continue
            if raw is None or not len(raw):
                failed += batch
                time.sleep(PAUSE)
                continue
            try:
                c = raw["Close"] if "Close" in raw else None
                v = raw["Volume"] if "Volume" in raw else None
            except Exception:
                c = v = None
            if c is None or v is None:
                failed += batch
                time.sleep(PAUSE)
                continue
            if isinstance(c, pd.Series):          # single-ticker shape
                c = c.to_frame(batch[0])
                v = v.to_frame(batch[0])
            c, v = norm_index(c), norm_index(v)
            got = 0
            for t in batch:
                if t in c.columns and c[t].notna().sum() >= MIN_BARS:
                    closes[t] = c[t]
                    vols[t] = v[t]
                    got += 1
            # a symbol that returned nothing usable is not retried forever;
            # most of these are genuinely dead tickers, not throttling
            if got == 0:
                failed += batch
            done = min(i + CHUNK, len(todo))
            if (i // CHUNK) % 10 == 0:
                print(f"   {done}/{len(todo)}  kept so far: {len(closes)}")
            time.sleep(PAUSE)
        todo = [t for t in failed if t not in closes]
        if todo:
            print(f"   {len(todo)} to retry")
            time.sleep(PAUSE * 3)
    return pd.DataFrame(closes), pd.DataFrame(vols)


def candidate_symbols():
    """Active US common stocks, unioned with the S&P 500 constituents."""
    import re
    sp = set()
    try:
        df = pd.read_csv(SP500_CSV)
        sp = set(df["Symbol"].str.replace(".", "-", regex=False))
        print(f"S&P 500 constituents: {len(sp):,}")
    except Exception as e:
        print(f"!! could not fetch the S&P 500 list: {e}")

    listed = set()
    if not AV_KEY:
        print("!! ALPHAVANTAGE_API_KEY is not set — falling back to the S&P 500")
        print("!! alone. Add it as a repository secret to widen the universe.")
    else:
        try:
            import urllib.request
            url = AV_LISTING.format(key=AV_KEY)
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read().decode("utf-8", "replace")
            import csv as _csv
            rows = list(_csv.DictReader(io.StringIO(body)))
            if not rows or "symbol" not in (rows[0] or {}):
                raise ValueError(f"unexpected response: {body[:200]}")
            listed = {r["symbol"] for r in rows if r.get("assetType") == "Stock"}
            # yfinance cannot address anything outside this shape
            listed = {s for s in listed if re.fullmatch(r"[A-Z][A-Z0-9\-]{0,7}", s)}
            print(f"active US common stock listings: {len(listed):,}")
        except Exception as e:
            print(f"!! listing fetch failed ({type(e).__name__}: {e})")
            print("!! continuing on the S&P 500 list alone.")

    syms = sorted(sp | listed)
    if len(syms) < 400:
        die(f"only {len(syms)} candidate symbols assembled. Both the S&P 500 "
            f"list and the listing feed appear to have failed — do not rebuild "
            f"the universe from this, the existing universe.txt is better.")
    only_sp = len(sp - listed)
    if listed and only_sp:
        print(f"   {only_sp} name(s) present in the S&P list but NOT in the "
              f"listing feed — this is why the two are unioned")
    return syms


def main():
    syms = candidate_symbols()
    print(f"candidate symbols: {len(syms):,}")

    close, vol = pull(syms)
    print(f"\nsymbols with {MIN_BARS}+ bars of history: {close.shape[1]:,}")
    if close.shape[1] < FLOOR * 2:
        die(f"only {close.shape[1]} symbols returned usable history out of "
            f"{len(syms):,}. Expected several thousand — this is a download "
            f"failure, not a selective screen. Re-run rather than trusting it.")

    dv = (close * vol).rolling(60).median().iloc[-1]
    px = close.ffill(limit=3).iloc[-1]

    tab = pd.DataFrame({"price": px, "dollar_vol": dv}).dropna()
    print(f"with a computable 60-day dollar volume: {len(tab):,}")

    keep = tab[(tab.price >= MIN_PRICE) & (tab.dollar_vol >= MIN_DOLLAR_VOL)]
    print(f"price >= ${MIN_PRICE} and >= ${MIN_DOLLAR_VOL/1e6:.0f}M/day: {len(keep):,}")

    keep = keep.sort_values("dollar_vol", ascending=False).head(TARGET)
    print(f"capped at {TARGET:,} by dollar volume: {len(keep):,}")

    if len(keep) < FLOOR:
        die(f"only {len(keep)} names passed the liquidity screen. The floor is "
            f"{FLOOR}. Either the download was partial or the thresholds are "
            f"wrong — do not write a universe this thin, the scan's coverage "
            f"gates are calibrated against it.")

    # keep SPY for the beta calculation downstream
    names = sorted(keep.index)
    if "SPY" not in names:
        names.append("SPY")

    prev = set()
    if OUT.exists():
        prev = {s.strip() for s in OUT.read_text().split() if s.strip()}
    OUT.write_text("\n".join(names) + "\n")

    print("\n" + "=" * 70)
    print(f"wrote {OUT.name}: {len(names):,} names")
    print("=" * 70)
    print(f"  dollar volume range: ${keep.dollar_vol.min()/1e6:,.0f}M "
          f"to ${keep.dollar_vol.max()/1e6:,.0f}M per day")
    print(f"  price range:         ${keep.price.min():,.2f} to ${keep.price.max():,.2f}")
    if prev:
        print(f"  added vs last build:   {len(set(names) - prev):>4}")
        print(f"  dropped vs last build: {len(prev - set(names)):>4}")
        drop = sorted(prev - set(names))[:15]
        if drop:
            print(f"    dropped sample: {drop}")

    # sector labels, where we have them — the wider universe has non-index names
    try:
        sp = pd.read_csv(SP500_CSV)
        sym = sp["Symbol"].str.replace(".", "-", regex=False)
        known = dict(zip(sym, sp["GICS Sector"]))
        have = sum(1 for n in names if n in known)
        print(f"\n  GICS sector known for {have:,} of {len(names):,}; the rest "
              f"will be labelled 'Other' in the scan output.")
    except Exception as e:
        print(f"\n  (could not fetch GICS sectors for labelling: {e})")


if __name__ == "__main__":
    main()
