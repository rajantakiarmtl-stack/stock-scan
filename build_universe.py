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
    Stage 1 (this file, run MONTHLY): take every active US common stock, discard
    the ones that obviously cannot be liquid enough, pull one year of daily bars
    for the rest, and keep the names that are genuinely liquid and have enough
    history to score. Writes universe.txt.

    THE PRE-SCREEN EXISTS BECAUSE OF A REAL FAILURE. The first live run
    (2026-10-12) assembled 8,604 candidates and asked Yahoo for a year of bars
    for all of them. Yahoo rate-limited it after roughly 1,320 symbols;
    945 came back usable and the run correctly refused to write. Yahoo throttles
    on request count, and yfinance makes about one request per symbol even
    inside a batch, so the only real fix is to ask for fewer symbols. One
    Nasdaq screener request gives last sale and volume for every listed stock,
    which discards most of the list for free. The S&P 500 is protected from the
    trim so a screener problem can never shrink the core universe.

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

    Symbol filtering is NARROW and deliberately so. The first attempt used a
    blanket dash regex and was wrong in both directions: it let SPAC units,
    warrants and rights through (AACBR, AACIU, AACOW) while dropping real names
    (BF-B and BRK-B on the dash, CBOE on an exchange filter). What is dropped
    now is only unambiguous non-common-stock SUFFIXES — -W/-WS warrant, -R
    right, -U unit, -P-x preferred series — which leaves class shares like
    BRK-B, BF-B, HEI-A and LEN-B untouched, and leaves LOW, SNOW, GLW and CRWV
    untouched because they have no dash at all. Suffix-less SPAC leftovers
    (AACOW, SDSTW) still slip through on purpose: catching them needs a pattern
    that also kills real tickers, and the dollar-volume screen removes them for
    free.
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
# Retuned 2026-10-12 after the first real run died on YFRateLimitError. Yahoo
# throttles on REQUEST COUNT, not payload size, and yfinance issues roughly one
# request per symbol even inside a batch. The first attempt swept 8,604 symbols
# at CHUNK=120 / PAUSE=1.0 and was cut off after ~1,320. Smaller batches, longer
# pauses, and more patient retries; the heavy lifting is now avoided entirely by
# the pre-screen below, which is what actually fixes this.
CHUNK = 60
PAUSE = 2.0
PASSES = 4

# Pre-screen. One request to Nasdaq's screener returns every listed US stock
# with last sale and volume, which is enough to discard the ~75% of the listing
# feed that could never pass a $75M/day test. Only the survivors get a year of
# bars pulled from Yahoo. If this request fails the build still works — it falls
# back to sweeping the whole list at the conservative pacing above — so this is
# an optimisation, never a dependency.
NASDAQ_SCREENER = ("https://api.nasdaq.com/api/screener/stocks"
                   "?tableonly=true&limit=25000&download=true")
PRESCREEN_KEEP = 2600      # how many to carry into the heavy pull
PRESCREEN_MIN_DV = 20e6    # loose: the real $75M test happens on 60-day medians
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
            # TARGETED junk filter, added 2026-10-12. The earlier blanket dash
            # regex was wrong in both directions and was removed; this is the
            # narrow version. Only unambiguous non-common-stock suffixes:
            #   -W / -WS  warrant      -R   right        -U  unit
            #   -P-x      preferred series
            # Class shares keep a bare single letter (BRK-B, BF-B), so they are
            # untouched — that is the whole point of matching the suffix
            # exactly rather than matching any dash.
            junk = re.compile(r"-(W|WS|R|U|RW|RT|P-[A-Z])$")
            before = len(listed)
            listed = {s for s in listed if not junk.search(s)}
            print(f"   dropped {before - len(listed):,} warrants/rights/units/"
                  f"preferreds by suffix -> {len(listed):,}")
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


def prescreen(syms, protect):
    """Cut the candidate list down before touching Yahoo.

    One Nasdaq screener request returns last sale and volume for every listed
    US stock. That is enough to throw away the three-quarters of the listing
    feed that trades nowhere near $75M/day, which is what made the first run
    hammer Yahoo into a rate limit.

    `protect` names that are kept regardless (the S&P 500 plus SPY) so a
    screener hiccup can never silently shrink the core universe.

    Returns the trimmed list, or the ORIGINAL list unchanged if anything about
    the screener looks wrong. Failing open matters: a bad trim would quietly
    produce a narrow universe, which is the exact failure mode this whole file
    is built to avoid.
    """
    import json
    import urllib.request
    try:
        req = urllib.request.Request(
            NASDAQ_SCREENER,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        table = (payload.get("data") or {}).get("table") or {}
        rows = table.get("rows") or []
        if len(rows) < 2000:
            print(f"!! screener returned only {len(rows)} rows — not trimming")
            return syms

        def num(s):
            try:
                return float(str(s).replace("$", "").replace(",", "").replace("%", ""))
            except Exception:
                return 0.0

        dv = {}
        for row in rows:
            sym = (row.get("symbol") or "").strip().upper().replace(".", "-")
            if not sym:
                continue
            dv[sym] = num(row.get("lastsale")) * num(row.get("volume"))
        print(f"screener: dollar volume for {len(dv):,} symbols")

        cand = [s for s in syms if dv.get(s, 0.0) >= PRESCREEN_MIN_DV]
        cand.sort(key=lambda s: dv.get(s, 0.0), reverse=True)
        cand = cand[:PRESCREEN_KEEP]
        keep = sorted(set(cand) | set(protect))
        # A symbol the screener simply doesn't know (recent listing, odd
        # ticker) must not be discarded on that basis alone, but we also
        # cannot afford to pull all of them. The protect list covers the ones
        # that matter; the rest wait for next month's rebuild.
        unknown = sum(1 for s in syms if s not in dv)
        print(f"pre-screen: {len(syms):,} -> {len(keep):,} "
              f"(>= ${PRESCREEN_MIN_DV/1e6:.0f}M/day, top {PRESCREEN_KEEP:,}, "
              f"plus {len(protect):,} protected; {unknown:,} unknown to screener)")
        if len(keep) < FLOOR * 2:
            print(f"!! trim left only {len(keep)} — not trimming after all")
            return syms
        return keep
    except Exception as e:
        print(f"!! screener unavailable ({type(e).__name__}: {e})")
        print(f"!! sweeping the full list at CHUNK={CHUNK}/PAUSE={PAUSE} instead; "
              f"this is slower and may hit Yahoo's rate limit.")
        return syms


def main():
    syms = candidate_symbols()
    print(f"candidate symbols: {len(syms):,}")

    core = set()
    try:
        core = set(pd.read_csv(SP500_CSV)["Symbol"].str.replace(".", "-", regex=False))
    except Exception:
        pass
    core.add("SPY")
    syms = prescreen(syms, core)

    close, vol = pull(syms)
    got = close.shape[1]
    frac = got / max(len(syms), 1)
    print(f"\nsymbols with {MIN_BARS}+ bars of history: {got:,} "
          f"({frac:.0%} of the list asked for)")
    # PROPORTIONAL, not absolute. The old test was `got < FLOOR * 2`, i.e. 800.
    # That made sense against an 8,600-symbol sweep and became wrong the moment
    # the pre-screen cut the input to ~2,600: the run that died on 2026-10-12
    # returned 945 usable symbols, which would now SAIL PAST an 800 floor while
    # being a plainly throttled run. What matters is the share that came back,
    # because the names Yahoo refused are not random — the sweep is alphabetical,
    # so a partial run is a biased run, not a smaller one.
    if frac < 0.55 or got < FLOOR:
        die(f"only {got:,} of {len(syms):,} symbols ({frac:.0%}) returned "
            f"{MIN_BARS}+ bars.\n"
            f"Yahoo throttled this run. The sweep is alphabetical, so what came\n"
            f"back is skewed toward the start of the alphabet — writing it would\n"
            f"produce a universe that looks fine and silently excludes most of\n"
            f"the market. Wait a while and re-run; the existing universe.txt is\n"
            f"better than a biased one.")

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
