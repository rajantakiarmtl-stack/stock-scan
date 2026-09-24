"""
scan.py — daily universe scan for the mean-reversion debit-spread system.

Runs in GitHub Actions (see .github/workflows/scan.yml) or in Colab:
    !pip install -q yfinance pyarrow curl_cffi

Outputs:
    scan_universe.csv    every name with all metrics
    scan_candidates.csv  only the names at a trigger
    scan_prices.parquet  price cache

TWO BUGS THIS FILE EXISTS TO PREVENT, both of which produced a plausible-looking
CSV with 60 rows instead of 500 and raised no error:

1. Cache poisoning. A throttled run cached only the names that came back, and
   every later run then asked for just 1 month, so new names had too little
   history for a 200-day average and were dropped. Fixed by classifying per
   ticker: names already deep in the cache get a top-up, names that are not get
   a full multi-year pull. A throttled run banks its progress.

2. Single-timestamp scoring. Metrics were read at close.index[-1]. Names fetched
   minutes apart, or mid-session, ended one bar short of each other, and dropna
   silently removed every name missing that final bar. Coverage reported 99% and
   the CSV still had 60 rows. Fixed by normalising the index, dropping any date
   where fewer than MIN_ROW_COVERAGE of names report, then forward-filling at
   most STALE_LIMIT bars.

Why either mattered rather than merely annoyed: vol_rank is a CROSS-SECTIONAL
rank. Sixty mega-caps ranked against each other is a different and wrong answer,
not a partial one — it produced 0 put triggers where the full universe produced 8.

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

SESSION = None
try:
    from curl_cffi import requests as _cr
    SESSION = _cr.Session(impersonate="chrome")
    print("curl_cffi session active")
except Exception:
    print("!! curl_cffi NOT installed — expect heavy throttling.")

try:
    HERE = Path(__file__).parent
except NameError:                      # Colab / Jupyter cell
    HERE = Path.cwd()

YEARS = 2
MIN_HIST = 260             # trading days a ticker needs before it can be scored
MIN_ROW_COVERAGE = 0.60    # drop a date where fewer than this share of names report
STALE_LIMIT = 3            # forward-fill a name at most this many bars
CACHE = HERE / "scan_prices.parquet"
CHUNK = 15
PAUSE = 1.5
PASSES = 4
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


def norm_index(df):
    """Tz-naive, midnight-normalised, no duplicate dates. Without this a
    tz-aware fetch and a tz-naive cache produce two rows for the same day."""
    if df is None or not len(df):
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


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
        except TypeError:              # yfinance version handles curl_cffi itself
            SESSION = None
    return yf.download(batch, **kw)


def _fetch(batch, period, min_days):
    d = _download(batch, period)
    if d is None or d.empty:
        return None, None
    if not isinstance(d.columns, pd.MultiIndex):
        d.columns = pd.MultiIndex.from_product([d.columns, batch])
    if "Close" not in d.columns.get_level_values(0):
        return None, None
    cl, vl = norm_index(d["Close"]), norm_index(d["Volume"])
    keep = [c for c in cl.columns if cl[c].notna().sum() >= min_days]
    if not keep:
        return None, None
    return cl[keep], vl[keep]


def download(names, period, label, min_days):
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
            old_close = norm_index(cached.xs("close", axis=1, level=1))
            old_vol = norm_index(cached.xs("volume", axis=1, level=1))
            print(f"cache: {old_close.shape[1]} tickers through "
                  f"{old_close.index[-1].date()}")
        except Exception as e:
            print(f"cache unreadable ({e}) — starting fresh")
            old_close = old_vol = None

    # FIX 1: decide per ticker whether the cache already has enough history.
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
        c, v = download(need_topup, "3mo", "top-up", 5)
        if len(c):
            frames.append((c, v))
    if need_full:
        c, v = download(need_full, f"{YEARS}y", "full", MIN_HIST)
        if len(c):
            frames.append((c, v))

    close = old_close if old_close is not None else pd.DataFrame()
    vol = old_vol if old_vol is not None else pd.DataFrame()
    for c, v in frames:                # new values win, union of both
        close = c.combine_first(close) if len(close) else c
        vol = v.combine_first(vol) if len(vol) else v

    if not len(close):
        sys.exit("nothing came back at all — Yahoo is hard-blocking. "
                 "Wait 30 minutes and run again.")

    close = norm_index(close.replace(0, np.nan))
    vol = norm_index(vol).reindex(index=close.index, columns=close.columns)
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
            f"No CSV written: vol_rank is measured ACROSS the universe, so a\n"
            f"partial run gives a wrong answer, not a partial one.\n"
            f"Progress IS saved — run again, it only fetches the {len(missing)}\n"
            f"still missing. e.g. {missing[:12]}")

    close = close[[c for c in close.columns if c in scorable]]
    vol = vol.reindex(index=close.index, columns=close.columns)

    # FIX 2: a date where only some names report (a partial bar mid-session, or
    # batches fetched minutes apart) used to silently drop every name that was
    # one bar short. Drop those dates, then allow a short forward-fill.
    print("\nreporting names by date (last 6):")
    for d in close.index[-6:]:
        print(f"   {d.date()}  {close.loc[d].notna().sum():>4}/{close.shape[1]}")
    thin = close.notna().sum(axis=1) < MIN_ROW_COVERAGE * close.shape[1]
    if thin.any():
        print(f"   dropping {int(thin.sum())} thin date(s): "
              f"{[str(d.date()) for d in close.index[thin]][-4:]}")
        close, vol = close[~thin], vol[~thin]
    last_seen = close.apply(lambda s: s.last_valid_index())
    close = close.ffill(limit=STALE_LIMIT)
    vol = vol.ffill(limit=STALE_LIMIT)

    asof = close.index[-1]
    stale = (asof - last_seen).dt.days
    print(f"\nprices through {asof.date()}   "
          f"{int((stale > 0).sum())} name(s) carried forward\n")

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
        "last_bar": [last_seen.get(t) for t in px.columns],
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
    })
    before = len(out)
    out = out.dropna(subset=["price", "sma50", "sma200", "realized_vol"])
    if len(out) < before:
        print(f"note: {before - len(out)} name(s) still lack a full metric set")

    out["last_bar"] = pd.to_datetime(out.last_bar).dt.date
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

    # final assertion: never write a universe too small to rank across
    if len(out) < MIN_COVERAGE * len(tickers):
        sys.exit(f"\nSTOPPING — only {len(out)} of {len(tickers)} names scored. "
                 f"vol_rank would be measured across the wrong universe.")

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
    print("=" * 70)
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
    try:
        from google.colab import files
        files.download(str(HERE / "scan_universe.csv"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
