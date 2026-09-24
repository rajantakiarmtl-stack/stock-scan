"""
scan.py — daily universe scan for the mean-reversion debit-spread system.

Colab:
    !pip install -q yfinance pyarrow curl_cffi
    then paste this whole file into a cell and run it.

Outputs, next to the script (in Colab: /content/):
    scan_universe.csv    every name with all metrics   <-- attach this one
    scan_candidates.csv  only the names at a trigger
    scan_prices.parquet  price cache

HOW THE CACHE WORKS (this is what broke the last two runs):
Yahoo throttles Colab, so a run may only return part of the universe. Every
run now checks, per ticker, whether the cache holds enough history. Names with
enough get a cheap 1-month top-up; names without get a full multi-year pull.
So a throttled run is not wasted — it banks whatever it got, and the next run
picks up only the remainder. Two or three runs and you are at full coverage.

The old version cached the 60 names that survived the first throttle, then on
every later run asked Yahoo for only 1 month. New names came back with 20 days
of history, could not form a 200-day average, and were silently dropped. That
is why you got the identical 60 tickers twice.

CORRECTNESS NOTE: auto_adjust=False and 'Close' is deliberate. That is
split-adjusted but NOT dividend-adjusted, which is what option strikes
reference. 'Adj Close' would silently shift every moving average.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Run:  pip install yfinance pyarrow curl_cffi")

# curl_cffi lets yfinance impersonate a real browser. This is the single
# biggest factor in getting past Yahoo's throttling on Colab.
SESSION = None
try:
    from curl_cffi import requests as _cr
    SESSION = _cr.Session(impersonate="chrome")
    print("curl_cffi session active")
except Exception:
    print("!! curl_cffi NOT installed — expect heavy throttling.")
    print("!! run:  !pip install curl_cffi   then restart and try again")

try:                       # works as a .py file
    HERE = Path(__file__).parent
except NameError:          # ...and in a Colab / Jupyter cell
    HERE = Path.cwd()

YEARS = 2
MIN_HIST = 260             # trading days a ticker needs before it can be scored
CACHE = HERE / "scan_prices.parquet"
CHUNK = 15
PAUSE = 1.5
PASSES = 4                 # retry sweeps over whatever is still missing
MIN_COVERAGE = 0.80        # refuse to write the CSV below this
MIN_DOLLAR_VOL = 100e6
MIN_PRICE = 50
VOL_RANK_MAX = 0.33
MIN_RVOL = 0.10            # data-quality floor: below this the series is broken
CALL_TRIGGER = -0.07       # 7% below the 50-day mean (raised 2026-09-24)
PUT_TRIGGER = 0.15         # 15% above the 200-day mean
EXTRA_TICKERS = []

SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")


def get_universe():
    try:
        df = pd.read_csv(SP500_CSV)
        sym = df["Symbol"].str.replace(".", "-", regex=False)
        tickers = sym.tolist()
        sectors = dict(zip(sym, df["GICS Sector"]))
    except Exception as e:
        sys.exit(f"could not fetch the ticker list: {e}")
    for t in EXTRA_TICKERS:
        if t not in tickers:
            tickers.append(t)
            sectors[t] = "Added"
    if "SPY" not in tickers:
        tickers.append("SPY")
        sectors["SPY"] = "ETF"
    print(f"universe: {len(tickers)} names")
    return tickers, sectors


def _download(batch, period):
    kw = dict(period=period, interval="1d", auto_adjust=False,
              progress=False, group_by="column", threads=False)
    global SESSION
    if SESSION is not None:
        try:
            return yf.download(batch, session=SESSION, **kw)
        except TypeError:          # yfinance version handles curl_cffi itself
            SESSION = None
    return yf.download(batch, **kw)


def _fetch(batch, period, min_days):
    d = _download(batch, period)
    if d is None or d.empty:
        return None, None
    if not isinstance(d.columns, pd.MultiIndex):     # single ticker comes back flat
        d.columns = pd.MultiIndex.from_product([d.columns, batch])
    if "Close" not in d.columns.get_level_values(0):
        return None, None
    cl, vl = d["Close"], d["Volume"]
    keep = [c for c in cl.columns if cl[c].notna().sum() >= min_days]
    if not keep:
        return None, None
    return cl[keep], vl[keep]


def download(names, period, label, min_days):
    """Sweep `names` repeatedly until they are all in, or PASSES is spent."""
    got_c, got_v = {}, {}
    todo = list(names)
    for p in range(PASSES):
        if not todo:
            break
        if p:
            wait = 20 * p
            print(f"  {len(todo)} still missing — waiting {wait}s before pass {p+1}")
            time.sleep(wait)
        for i in range(0, len(todo), CHUNK):
            batch = todo[i:i + CHUNK]
            try:
                cl, vl = _fetch(batch, period, min_days)
                if cl is not None:
                    for c in cl.columns:
                        got_c[c], got_v[c] = cl[c], vl[c]
            except Exception:
                time.sleep(5)
            print(f"  {label} pass {p+1}: {min(i+CHUNK, len(todo))}/{len(todo)} "
                  f"asked, {len(got_c)}/{len(names)} in", end="\r", flush=True)
            time.sleep(PAUSE)
        print()
        todo = [t for t in todo if t not in got_c]
    if not got_c:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(got_c).sort_index(), pd.DataFrame(got_v).sort_index()


def rsi14(s):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    tickers, sectors = get_universe()

    old_close = old_vol = None
    if CACHE.exists():
        try:
            cached = pd.read_parquet(CACHE)
            old_close = cached.xs("close", axis=1, level=1)
            old_vol = cached.xs("volume", axis=1, level=1)
            print(f"cache: {old_close.shape[1]} tickers through "
                  f"{old_close.index[-1].date()}")
        except Exception as e:
            print(f"cache unreadable ({e}) — starting fresh")
            old_close = old_vol = None

    # THE FIX: decide per ticker whether the cache already has enough history.
    if old_close is not None:
        deep = {t for t in old_close.columns
                if t in tickers and old_close[t].notna().sum() >= MIN_HIST}
    else:
        deep = set()
    need_full = [t for t in tickers if t not in deep]
    need_topup = [t for t in tickers if t in deep]
    print(f"need full history: {len(need_full)}   only a top-up: {len(need_topup)}\n")

    frames = []
    if need_topup:
        c, v = download(need_topup, "1mo", "top-up", 5)
        if len(c):
            frames.append((c, v))
    if need_full:
        c, v = download(need_full, f"{YEARS}y", "full", MIN_HIST)
        if len(c):
            frames.append((c, v))

    close = old_close if old_close is not None else pd.DataFrame()
    vol = old_vol if old_vol is not None else pd.DataFrame()
    for c, v in frames:                      # new values win, union of both
        close = c.combine_first(close) if len(close) else c
        vol = v.combine_first(vol) if len(vol) else v

    if not len(close):
        sys.exit("nothing came back at all — Yahoo is hard-blocking. "
                 "Wait 30 minutes and run again.")

    close = close.replace(0, np.nan)
    vol = vol.reindex(columns=close.columns)
    cutoff = close.index[-1] - pd.Timedelta(days=int(YEARS * 372))
    close, vol = close[close.index >= cutoff], vol[vol.index >= cutoff]

    # bank progress BEFORE any exit, so a throttled run is never wasted
    pd.concat({"close": close, "volume": vol}, axis=1).swaplevel(
        0, 1, axis=1).sort_index(axis=1).to_parquet(CACHE)

    scorable = [t for t in tickers
                if t in close.columns and close[t].notna().sum() >= MIN_HIST]
    coverage = len(scorable) / len(tickers)
    print(f"\nscorable: {len(scorable)}/{len(tickers)} = {coverage:.0%}")

    if coverage < MIN_COVERAGE:
        missing = [t for t in tickers if t not in scorable]
        sys.exit(
            f"\nSTOPPING — only {coverage:.0%} of the universe has enough history.\n"
            f"No CSV written, because volatility rank is measured ACROSS the\n"
            f"universe: scoring 60 mega-caps against each other gives a wrong\n"
            f"answer, not a partial one.\n\n"
            f"Progress IS saved. Just run this cell again — it will only fetch\n"
            f"the {len(missing)} names still missing, so each run gets closer.\n"
            f"Still stuck after 3 runs? Yahoo has your Colab IP throttled:\n"
            f"  Runtime > Disconnect and delete runtime, then reconnect.\n"
            f"missing e.g. {missing[:12]}")

    close = close[[c for c in close.columns if c in scorable]]
    vol = vol.reindex(columns=close.columns)
    asof = close.index[-1]
    print(f"prices through {asof.date()}\n")

    spy = close["SPY"] if "SPY" in close else None
    px = close.drop(columns=["SPY"], errors="ignore")
    vl = vol.drop(columns=["SPY"], errors="ignore")

    ret = px.pct_change()
    rv60 = ret.rolling(60).std(ddof=1) * np.sqrt(252)
    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    liq = (px * vl).rolling(60).median().shift(1)
    hi252 = px.rolling(252).max()
    run5 = px / px.shift(5) - 1
    rsi = px.apply(rsi14)
    volrank = rv60.rank(axis=1, pct=True)

    if spy is not None:
        sr = spy.pct_change()
        beta = ret.rolling(250).cov(sr).div(sr.rolling(250).var(), axis=0)
    else:
        beta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)

    last = asof
    out = pd.DataFrame({
        "ticker": px.columns,
        "sector": [sectors.get(t, "") for t in px.columns],
        "price": px.loc[last].values,
        "sma50": sma50.loc[last].values,
        "sma200": sma200.loc[last].values,
        "pct_vs_sma50": (100 * (px.loc[last] / sma50.loc[last] - 1)).values,
        "pct_vs_sma200": (100 * (px.loc[last] / sma200.loc[last] - 1)).values,
        "pct_off_52w_high": (100 * (px.loc[last] / hi252.loc[last] - 1)).values,
        "realized_vol": (100 * rv60.loc[last]).values,
        "vol_rank": volrank.loc[last].values,
        "rsi14": rsi.loc[last].values,
        "beta": beta.loc[last].values,
        "run_5d": (100 * run5.loc[last]).values,
        "dollar_vol_musd": (liq.loc[last] / 1e6).values,
    }).dropna(subset=["price", "sma50", "sma200", "realized_vol"])

    out["sigma_21d"] = out.realized_vol / 100 * np.sqrt(21 / 365) * out.price
    out["suggested_width"] = (1.5 * out.sigma_21d).round(1)
    out["eligible"] = ((out.dollar_vol_musd >= MIN_DOLLAR_VOL / 1e6)
                       & (out.price >= MIN_PRICE)
                       & (out.realized_vol >= 100 * MIN_RVOL)
                       & (out.vol_rank <= VOL_RANK_MAX))
    out["call_trigger"] = (out.eligible & (out.price > out.sma200)
                           & (out.pct_vs_sma50 <= CALL_TRIGGER * 100))
    out["put_trigger"] = out.eligible & (out.pct_vs_sma200 >= PUT_TRIGGER * 100)
    out["asof_date"] = last.date()

    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(3)
    out = out.sort_values("pct_vs_sma50")
    out.to_csv(HERE / "scan_universe.csv", index=False)

    cand = out[out.call_trigger | out.put_trigger].copy()
    if len(cand):
        print(f"fetching earnings dates for {len(cand)} candidates...")
        dates = []
        for t in cand.ticker:
            try:
                e = yf.Ticker(t, session=SESSION).get_earnings_dates(limit=8)
                fut = e[e.index > pd.Timestamp.now(tz=e.index.tz)]
                dates.append(fut.index.min().date() if len(fut) else "")
            except Exception:
                dates.append("")
            time.sleep(0.3)
        cand["next_earnings"] = dates
    cand.to_csv(HERE / "scan_candidates.csv", index=False)

    print(f"\n{'='*70}\nas of {last.date()}   "
          f"eligible universe: {int(out.eligible.sum())} of {len(out)}")
    print("="*70)
    print(f"\nCALL triggers ({int(out.call_trigger.sum())}):")
    c = out[out.call_trigger]
    if len(c):
        print(c[["ticker", "sector", "price", "pct_vs_sma50", "rsi14", "beta",
                 "suggested_width", "dollar_vol_musd"]].to_string(index=False))
    print(f"\nPUT triggers ({int(out.put_trigger.sum())}):")
    p = out[out.put_trigger].sort_values("pct_vs_sma200", ascending=False)
    if len(p):
        print(p[["ticker", "sector", "price", "pct_vs_sma200", "rsi14", "beta",
                 "suggested_width", "dollar_vol_musd"]].to_string(index=False))
    if spy is not None:
        s = spy.loc[last]
        print(f"\nSPY {s:.2f}   vs 50d "
              f"{100*(s/spy.rolling(50).mean().loc[last]-1):+.2f}%"
              f"   off 52w high {100*(s/spy.rolling(252).max().loc[last]-1):+.2f}%")
    print(f"\nwrote scan_universe.csv ({len(out)} rows), "
          f"scan_candidates.csv ({len(cand)} rows)")
    print("attach scan_universe.csv to the chat.")


if __name__ == "__main__":
    main()
