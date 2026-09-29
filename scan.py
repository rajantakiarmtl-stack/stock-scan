"""
scan.py — daily universe scan for the mean-reversion debit-spread system.

Runs in GitHub Actions (see .github/workflows/scan.yml). Also works in Colab:
    !pip install -q yfinance pyarrow curl_cffi

Outputs:
    scan_universe.csv    every name with all metrics
    scan_candidates.csv  only the names at a trigger, with earnings dates

DESIGN NOTE, 2026-09-28. This file does a FULL multi-year pull on every run and
keeps no cache. That is slower and deliberately dumber than the incremental
version it replaces, because the incremental path failed silently three separate
times and every failure produced a well-formed, confident, WRONG file:

  1. A throttled first run cached 60 names; later runs asked for only 1 month,
     so new names had too little history for a 200-day average and were dropped.
     The same 60 names came back forever.
  2. Metrics were read at a single last timestamp, so names fetched minutes
     apart ended one bar short of each other and were silently dropped. Coverage
     reported 99% and the file still had 60 rows.
  3. The thin-row guard added to fix (2) worked exactly as written — it dropped
     an incomplete Friday bar — and then the script happily scored Thursday
     instead and wrote a file that looked perfect. Three runs stayed stuck on
     stale data with no error.

Each fix addressed a symptom. The defect underneath was that nothing asserted
the output was actually current and actually complete. A full pull removes the
class of bug; the asserts below catch what is left.

WHY IT MATTERS: vol_rank is a CROSS-SECTIONAL rank. A screen that ranks names
against each other does not degrade gracefully when the universe shrinks or goes
stale — it returns a different and wrong answer, not a partial one. The 60-name
universe produced 0 put triggers where the full 500 produced 8.

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
MIN_COVERAGE = 0.80        # share of the universe that must have enough history
MIN_LAST_BAR_COVERAGE = 0.80   # share that must report on the FINAL bar
MAX_STALE_BDAYS = 1        # how many business days behind the run date is allowed
STALE_LIMIT = 2            # forward-fill an individual laggard at most this far
CHUNK = 15
PAUSE = 1.5
PASSES = 4
MIN_DOLLAR_VOL = 100e6
MIN_PRICE = 50
VOL_RANK_MAX = 0.33
MIN_RVOL = 0.10            # data-quality floor: below this the series is broken
CALL_TRIGGER = -0.07       # 7% below the 50-day mean
PUT_TRIGGER = 0.15         # 15% above the 200-day mean
EXTRA_TICKERS = []

SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")


def die(msg):
    """Exit non-zero so the Actions run goes red and emails you."""
    print("\n" + "=" * 70)
    print("SCAN FAILED — no CSV written")
    print("=" * 70)
    print(msg)
    sys.exit(1)


def norm_index(df):
    """Tz-naive, midnight-normalised, no duplicate dates."""
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
        die(f"could not fetch the S&P 500 ticker list: {e}")
    for t in EXTRA_TICKERS:
        if t not in tickers:
            tickers.append(t)
            sectors[t] = "Added"
    if "SPY" not in tickers:
        tickers.append("SPY")
        sectors["SPY"] = "ETF"
    print(f"universe: {len(tickers)} names")
    return tickers, sectors


def _download(batch):
    kw = dict(period=f"{YEARS}y", interval="1d", auto_adjust=False,
              progress=False, group_by="column", threads=False)
    global SESSION
    if SESSION is not None:
        try:
            return yf.download(batch, session=SESSION, **kw)
        except TypeError:              # yfinance version handles curl_cffi itself
            SESSION = None
    return yf.download(batch, **kw)


def _fetch(batch):
    d = _download(batch)
    if d is None or d.empty:
        return None, None
    if not isinstance(d.columns, pd.MultiIndex):
        d.columns = pd.MultiIndex.from_product([d.columns, batch])
    if "Close" not in d.columns.get_level_values(0):
        return None, None
    cl, vl = norm_index(d["Close"]), norm_index(d["Volume"])
    keep = [c for c in cl.columns if cl[c].notna().sum() >= MIN_HIST]
    if not keep:
        return None, None
    return cl[keep], vl[keep]


def download(names):
    """Full multi-year pull, swept until everything is in or PASSES is spent."""
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
                cl, vl = _fetch(batch)
                if cl is not None:
                    for c in cl.columns:
                        got_c[c], got_v[c] = cl[c], vl[c]
            except Exception:
                time.sleep(5)
            print(f"  pass {p+1}: {min(i+CHUNK, len(todo))}/{len(todo)} asked, "
                  f"{len(got_c)}/{len(names)} in", end="\r", flush=True)
            time.sleep(PAUSE)
        print()
        todo = [t for t in todo if t not in got_c]
    if not got_c:
        die("nothing came back at all — Yahoo is hard-blocking this runner.")
    return pd.DataFrame(got_c).sort_index(), pd.DataFrame(got_v).sort_index()


def rsi14(s):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    tickers, sectors = get_universe()
    print(f"full {YEARS}-year pull, no cache — this takes a few minutes\n")
    close, vol = download(tickers)

    close = norm_index(close.replace(0, np.nan))
    vol = norm_index(vol).reindex(index=close.index, columns=close.columns)

    # ---- GATE 1: enough names with enough history to rank across ----------
    scorable = [t for t in tickers
                if t in close.columns and close[t].notna().sum() >= MIN_HIST]
    coverage = len(scorable) / len(tickers)
    print(f"scorable: {len(scorable)}/{len(tickers)} = {coverage:.0%}")
    if coverage < MIN_COVERAGE:
        missing = [t for t in tickers if t not in scorable][:12]
        die(f"only {coverage:.0%} of the universe has {MIN_HIST}+ days of history.\n"
            f"vol_rank is ranked ACROSS the universe, so a partial run is wrong,\n"
            f"not partial. Re-run; if it keeps failing Yahoo is throttling.\n"
            f"missing e.g. {missing}")

    close = close[[c for c in close.columns if c in scorable]]
    vol = vol.reindex(index=close.index, columns=close.columns)

    # ---- GATE 2: the FINAL bar must be broadly reported ------------------
    print("\nreporting names by date (last 6):")
    for d in close.index[-6:]:
        print(f"   {d.date()}  {close.loc[d].notna().sum():>4}/{close.shape[1]}")
    asof = close.index[-1]
    last_cov = close.loc[asof].notna().sum() / close.shape[1]
    if last_cov < MIN_LAST_BAR_COVERAGE:
        die(f"the final bar ({asof.date()}) has only {last_cov:.0%} of names\n"
            f"reporting. Earlier versions silently dropped this bar and scored\n"
            f"the day before instead, which is how three runs went stale without\n"
            f"an error. Re-run rather than trust a partial bar.")

    # ---- GATE 3: the data must actually be current -----------------------
    run_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    behind = int(np.busday_count(asof.date(), run_date.date()))
    print(f"\nlast bar {asof.date()}, run date {run_date.date()} "
          f"({behind} business day(s) behind)")
    if behind > MAX_STALE_BDAYS:
        die(f"the newest bar is {asof.date()}, which is {behind} business days\n"
            f"behind today ({run_date.date()}). The scan is STALE.\n\n"
            f"If yesterday was a market holiday this is expected — re-run\n"
            f"tomorrow. Otherwise Yahoo has not published the latest bar yet;\n"
            f"wait an hour and re-run rather than trading off old data.")

    # individual laggards only — the bar as a whole has already been checked
    last_seen = close.apply(lambda s: s.last_valid_index())
    close = close.ffill(limit=STALE_LIMIT)
    vol = vol.ffill(limit=STALE_LIMIT)
    carried = int((asof - last_seen).dt.days.gt(0).sum())
    print(f"prices through {asof.date()}   {carried} name(s) carried forward\n")

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
        print(f"note: {before - len(out)} name(s) lack a full metric set")

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

    # ---- GATE 4: last line of defence before anything is written ---------
    if len(out) < MIN_COVERAGE * len(tickers):
        die(f"only {len(out)} of {len(tickers)} names scored. vol_rank would be\n"
            f"measured across the wrong universe.")

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
