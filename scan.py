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

WHY IT MATTERED: the volatility filter used to be vol_rank, a CROSS-SECTIONAL
rank. A screen that ranks names against each other does not degrade gracefully
when the universe shrinks or goes stale — it returns a different and wrong
answer, not a partial one. The 60-name universe produced 0 put triggers where
the full 500 produced 8. That was briefly fixed by swapping in an absolute
ceiling, and REVERTED on 2026-10-29: the rank is what the research validated
(+0.885 sigma holdout vs +0.308), and a stale universe is the coverage gates'
job to catch, not the volatility filter's. The coverage gates below stay,
because the moving averages still need complete history.

UNIVERSE, 2026-10-12. The scan reads universe.txt, built by build_universe.py
from dollar-volume rankings across every active US common stock — not from index
membership. The S&P 500 is not the 500 most-traded US names, and the measured
edge is concentrated in the most liquid ones (holdout by dollar-volume quartile:
least liquid +$65, most liquid +$478). The old 500-name universe also held only
3-4 triggered names at any moment, so a single earnings season could produce
weeks with nothing tradeable.

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
PHANTOM_BAR = 0.20         # below this share, a date is not a trading day at all
MAX_STALE_BDAYS = 1        # how many business days behind the run date is allowed
STALE_LIMIT = 2            # forward-fill an individual laggard at most this far
CHUNK = 15
PAUSE = 1.5
PASSES = 4
MIN_DOLLAR_VOL = 100e6
MIN_PRICE = 50
# Volatility filter: CROSS-SECTIONAL RANK. Reverted 2026-10-29 from the
# absolute 25% ceiling that replaced it on 2026-10-12.
#
# The 10-12 reasoning was that a percentile silently changes meaning when the
# universe changes, which is true, and that the edge lives in low ABSOLUTE
# volatility, which is NOT what the data says. Measured head to head on the
# same panel: rank <= 0.33 gives +0.885 sigma holdout at t = 5.00; the 25%
# absolute ceiling gives +0.308 sigma at t = 2.31. The edge is REGIME-RELATIVE
# — "calm" means calm against today's market, not against a fixed number.
#
# The silent-failure worry the ceiling was meant to fix is real but belongs to
# a different layer: it is the coverage gates below that must catch a shrunken
# universe, not the volatility filter. A rank computed over a universe that
# failed its coverage check is wrong either way.
#
# 2026-10-29 note: with the current universe the two happen to agree (the 33rd
# percentile sits at rv60 26.2%), so this change moves nothing today. It
# matters in the next regime, which is the point.
MAX_VOL_RANK = 0.33        # cross-sectional percentile of 60-day realized vol
MIN_RVOL = 0.10            # data-quality floor: below this the series is broken
CALL_TRIGGER = -0.07       # 7% below the 50-day mean
PUT_TRIGGER = 0.15         # 15% above the 200-day mean
EXTRA_TICKERS = []

UNIVERSE_FILE = "universe.txt"      # written by build_universe.py

# Expiry window for pricing the candidate spread. 30 is the target: measured on
# dollars per day of capital, 30 DTE beat 21 (+$27.35/day vs +$25.68) and beat
# it on dollars per trade by a wider margin (+$574 vs +$385). 45 was far worse
# (+$5.12/day), so 30 is what the picker aims at.
#
# The WINDOW is wider than the target on purpose. Standard monthly expiries are
# third Fridays, 28-35 days apart, so a tight 21-38 window has calendar holes:
# on 2026-10-12 the monthlies were 5, 40 and 68 days out and a 38-day ceiling
# rejected every monthly-only name for no reason but the calendar. Weekly names
# are unaffected; the names with monthlies only are the thinner ones, which is
# what the OI and slippage gates are for. So the window accommodates the gap and
# OFF-TARGET flags anything more than 10 days from 30 — picked, but labelled.
TARGET_DTE = 30
MIN_DTE = 21
MAX_DTE = 45
DTE_DRIFT_WARN = 10

# Advisory thresholds. These do NOT drop a candidate — every trigger is still
# printed and written. They set the WARN flag, because a filter that silently
# removes rows is how three separate runs produced confident, wrong files.
BE_SIGMA_WARN = 0.80       # breakeven beyond this needs an above-median move
MIN_LEG_OI = 100           # per leg; thinner than this and the fill is a fight
MAX_SPREAD_COST = 25.0     # round-trip bid/ask as a % of the debit
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
    """Liquidity-ranked universe from universe.txt, S&P 500 as a fallback.

    universe.txt is produced by build_universe.py, which ranks every active US
    common stock by dollar volume rather than taking index membership. The
    fallback exists so a missing file degrades to the old behaviour LOUDLY
    rather than silently scanning nothing.
    """
    tickers, source = [], None
    path = HERE / UNIVERSE_FILE
    if path.exists():
        tickers = sorted({s.strip() for s in path.read_text().split() if s.strip()})
        source = UNIVERSE_FILE
    if len(tickers) < 200:
        print(f"!! {UNIVERSE_FILE} missing or too short "
              f"({len(tickers)} names) — falling back to the S&P 500 list.")
        print("!! Run build_universe.py to rebuild the liquidity-ranked universe.")
        tickers, source = [], None

    # GICS sectors where we have them. The wider universe includes non-index
    # names, which get "Other" — the sector cap is a portfolio rule applied by
    # hand, so an unlabelled name is reported, not dropped.
    sectors = {}
    try:
        df = pd.read_csv(SP500_CSV)
        sym = df["Symbol"].str.replace(".", "-", regex=False)
        sectors = dict(zip(sym, df["GICS Sector"]))
        if not tickers:
            tickers = sym.tolist()
            source = "S&P 500 constituents (FALLBACK)"
    except Exception as e:
        if not tickers:
            die(f"no {UNIVERSE_FILE} and could not fetch the S&P 500 list: {e}")
        print(f"   (sector labels unavailable: {e})")

    for t in EXTRA_TICKERS:
        if t not in tickers:
            tickers.append(t)
    if "SPY" not in tickers:
        tickers.append("SPY")
    sectors.setdefault("SPY", "ETF")
    labelled = sum(1 for t in tickers if t in sectors)
    print(f"universe: {len(tickers)} names from {source}  "
          f"({labelled} with a GICS sector, rest 'Other')")
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


def pick_expiry(tk):
    """The tradeable expiry closest to TARGET_DTE, and every expiry on offer.

    Returns (expiry_string, dte, note). A name can trigger on price and still
    be untradeable because the chain has nothing in the window we want — a
    weekly-only name, or a gap across a holiday. That is a REASON, not a blank.
    """
    try:
        exps = list(tk.options or [])
    except Exception as e:
        return None, None, f"chain unavailable ({type(e).__name__})"
    if not exps:
        return None, None, "no expiries listed"
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    dtes = [(e, (pd.Timestamp(e) - today).days) for e in exps]
    usable = [(e, d) for e, d in dtes if MIN_DTE <= d <= MAX_DTE]
    if not usable:
        near = min(dtes, key=lambda x: abs(x[1] - TARGET_DTE))
        return None, None, (f"no expiry in {MIN_DTE}-{MAX_DTE}d "
                            f"(closest {near[0]} at {near[1]}d)")
    best = min(usable, key=lambda x: abs(x[1] - TARGET_DTE))
    return best[0], best[1], ""


def _mid(row):
    """Mid price, and whether it came from a real two-sided quote."""
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2, (ask - bid), True
    last = float(row.get("lastPrice") or 0)
    return (last if last > 0 else np.nan), np.nan, False


def price_the_trade(ticker, spot, sigma, hist):
    """Price the on-spec spread against the REAL chain and judge it.

    Four things the price screen cannot know, all of which have killed trades:

      1. EXPIRY  — does an expiry exist in the window at all?
      2. STRIKES — suggested_width is arithmetic. The chain has actual strikes,
                   and after rounding to them the spread is not the one the
                   spec described. Width is reported as BUILT, not as wanted.
      3. COST    — the debit sets the breakeven, and the breakeven is the only
                   number that says whether the trade can win. Expressed in the
                   stock's own sigma, because 23% of moves reach 1.5 sigma and
                   the median is +0.54 sigma: a breakeven above ~0.8 sigma is
                   asking for an unusual move, whatever the chart looks like.
      4. LIQUIDITY — stock volume says nothing about the option. A $400M/day
                   stock can have a 30%-wide option spread, and that spread is
                   paid twice.
    """
    d = {"opt_note": "", "expiry": "", "dte": np.nan, "k_long": np.nan,
         "k_short": np.nan, "width_built": np.nan, "debit": np.nan,
         "breakeven": np.nan, "be_pct": np.nan, "be_sigma": np.nan,
         "max_profit": np.nan, "rr": np.nan, "leg_vol": np.nan,
         "leg_oi": np.nan, "spread_cost_pct": np.nan,
         "clear_rate": np.nan, "clear_n": np.nan, "two_sided": False}
    try:
        tk = yf.Ticker(ticker, session=SESSION)
        exp, dte, note = pick_expiry(tk)
        if exp is None:
            d["opt_note"] = note
            return d
        d["expiry"], d["dte"] = exp, dte
        calls = tk.option_chain(exp).calls
        if calls is None or not len(calls):
            d["opt_note"] = "empty call chain"
            return d
        ks = np.sort(calls.strike.unique().astype(float))
        if len(ks) < 2:
            d["opt_note"] = f"only {len(ks)} strike(s) listed"
            return d

        # long leg: at the money, per spec
        k1 = float(ks[np.argmin(np.abs(ks - spot))])
        # short leg: 1.5 sigma up, rounded to a strike that actually exists and
        # is strictly above the long leg
        target = k1 + 1.5 * sigma
        above = ks[ks > k1]
        if not len(above):
            d["opt_note"] = "no strike above the money"
            return d
        k2 = float(above[np.argmin(np.abs(above - target))])
        d["k_long"], d["k_short"] = k1, k2
        d["width_built"] = round(k2 - k1, 2)

        rl = calls[calls.strike == k1]
        rs = calls[calls.strike == k2]
        if not len(rl) or not len(rs):
            d["opt_note"] = "strike rows missing from chain"
            return d
        rl, rs = rl.iloc[0], rs.iloc[0]
        ml, sl, okl = _mid(rl)
        ms, ss, oks = _mid(rs)
        if not np.isfinite(ml) or not np.isfinite(ms):
            d["opt_note"] = "no usable prices on one or both legs"
            return d
        debit = ml - ms
        d["two_sided"] = bool(okl and oks)
        if debit <= 0:
            d["opt_note"] = f"non-positive debit ({debit:.2f}) — stale quotes"
            return d

        d["debit"] = round(debit, 2)
        d["breakeven"] = round(k1 + debit, 2)
        d["be_pct"] = round(100 * ((k1 + debit) / spot - 1), 2)
        d["be_sigma"] = round(((k1 + debit) - spot) / sigma, 2) if sigma > 0 else np.nan
        d["max_profit"] = round((k2 - k1) - debit, 2)
        d["rr"] = round(((k2 - k1) - debit) / debit, 2)
        d["leg_vol"] = float(min(rl.get("volume") or 0, rs.get("volume") or 0))
        d["leg_oi"] = float(min(rl.get("openInterest") or 0,
                                rs.get("openInterest") or 0))
        if np.isfinite(sl) and np.isfinite(ss) and ml > 0 and ms > 0:
            # what the round trip costs, as a share of the debit put up
            d["spread_cost_pct"] = round(100 * (sl + ss) / debit, 1)

        # ---- how often has THIS stock cleared THIS breakeven, historically? --
        # Overlapping windows, so n_indep (not the window count) is the honest
        # sample size. Unconditional on the signal, which makes it a mild
        # UNDERSTATEMENT: the trigger is worth roughly +0.85pp of drift. Treat
        # it as a reality check on the breakeven, not as a forecast.
        hold = max(int(round(dte * 5 / 7)), 2)
        s = hist.dropna()
        if len(s) > hold + 40:
            fwd = s.shift(-hold) / s - 1
            fwd = fwd.dropna()
            need = (k1 + debit) / spot - 1
            d["clear_rate"] = round(100 * float((fwd >= need).mean()), 1)
            d["clear_n"] = int(len(fwd) / hold)
        if not d["two_sided"]:
            d["opt_note"] = "one-sided quote — priced off lastPrice, verify in IBKR"
    except Exception as e:
        d["opt_note"] = f"{type(e).__name__}: {e}"
    return d


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

    # ---- GATE 1: enough names actually came back with enough history ------
    scorable = [t for t in tickers
                if t in close.columns and close[t].notna().sum() >= MIN_HIST]
    coverage = len(scorable) / len(tickers)
    print(f"scorable: {len(scorable)}/{len(tickers)} = {coverage:.0%}")
    if coverage < MIN_COVERAGE:
        missing = [t for t in tickers if t not in scorable][:12]
        die(f"only {coverage:.0%} of the universe has {MIN_HIST}+ days of history.\n"
            f"The 200-day average needs that history, so a thin run silently\n"
            f"drops names rather than reporting them. build_universe.py already\n"
            f"requires a year of bars, so a big shortfall here means Yahoo is\n"
            f"throttling this run, not that the names are too young. Re-run.\n"
            f"missing e.g. {missing}")

    close = close[[c for c in close.columns if c in scorable]]
    vol = vol.reindex(index=close.index, columns=close.columns)

    # ---- GATE 2: the FINAL bar must be broadly reported ------------------
    print("\nreporting names by date (last 6):")
    for d in close.index[-6:]:
        print(f"   {d.date()}  {close.loc[d].notna().sum():>4}/{close.shape[1]}")
    # A date almost nobody reports is not a trading day — it is one straggler's
    # stray bar, and reading the last index row blindly lets it speak for the
    # whole universe. 2026-10-01 arrived with 1 of 501 names and failed three
    # consecutive runs on data that was complete through 2026-09-30. Drop those
    # rows first, loudly, so the gates below judge the last REAL bar.
    phantom = close.notna().sum(axis=1) < PHANTOM_BAR * close.shape[1]
    if phantom.any():
        print(f"   dropping {int(phantom.sum())} phantom date(s) "
              f"(<{PHANTOM_BAR:.0%} of names reporting): "
              f"{[str(d.date()) for d in close.index[phantom]]}")
        close = close[~phantom]
        vol = vol[~phantom]

    asof = close.index[-1]
    last_cov = close.loc[asof].notna().sum() / close.shape[1]
    if last_cov < MIN_LAST_BAR_COVERAGE:
        die(f"the final bar ({asof.date()}) has only {last_cov:.0%} of names\n"
            f"reporting. Earlier versions silently dropped this bar and scored\n"
            f"the day before instead, which is how three runs went stale without\n"
            f"an error. Re-run rather than trust a partial bar.")

    # ---- GATE 3: the data must actually be current -----------------------
    # Anchor to New York, not UTC. Past 8pm ET the UTC calendar has already
    # rolled over — on a Friday night it is Saturday — so measuring staleness
    # against a UTC "today" inflates the count by a day. Run #11 died exactly
    # that way: Yahoo had not published Friday's bar yet, the phantom filter
    # above dropped it, and UTC called the run date Saturday, turning a
    # one-session lag into a two-session failure on good data.
    now_et = pd.Timestamp.now(tz="America/New_York")
    # Yahoo rebuilds the consolidated daily bar after the 4pm close and it is
    # not dependably in place until the evening. Before 18:30 ET the newest
    # bar we can legitimately expect is the previous session's.
    probe = now_et.normalize()
    if now_et.hour * 60 + now_et.minute < 18 * 60 + 30:
        probe -= pd.Timedelta(days=1)
    while not np.is_busday(probe.date()):
        probe -= pd.Timedelta(days=1)
    expected = probe.tz_localize(None).normalize()
    behind = int(np.busday_count(asof.date(), expected.date()))
    print(f"\nlast bar {asof.date()}, newest expected {expected.date()} "
          f"({behind} session(s) behind; now {now_et:%Y-%m-%d %H:%M} ET)")
    if behind > MAX_STALE_BDAYS:
        die(f"the newest bar is {asof.date()} but the market has closed on\n"
            f"{expected.date()} — {behind} sessions behind. The scan is STALE.\n\n"
            f"np.is_busday knows weekends but not market holidays, so a single\n"
            f"session of slack is tolerated by design. Two means Yahoo is\n"
            f"genuinely behind: wait and re-run rather than trade off old data.")

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
        "sector": [sectors.get(t, "Other") for t in px.columns],
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
    # Eligibility uses the VALIDATED cross-sectional rank. See the
    # MAX_VOL_RANK comment: rank measured +0.885 sigma holdout (t 5.00) against
    # +0.308 (t 2.31) for the absolute ceiling it briefly replaced.
    out["eligible"] = ((out.dollar_vol_musd >= MIN_DOLLAR_VOL / 1e6)
                       & (out.price >= MIN_PRICE)
                       & (out.realized_vol >= 100 * MIN_RVOL)
                       & (out.vol_rank <= MAX_VOL_RANK))
    out["call_trigger"] = (out.eligible & (out.price > out.sma200)
                           & (out.pct_vs_sma50 <= CALL_TRIGGER * 100))
    out["put_trigger"] = out.eligible & (out.pct_vs_sma200 >= PUT_TRIGGER * 100)
    out["asof_date"] = last.date()

    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(3)
    out = out.sort_values("pct_vs_sma50")

    # ---- GATE 4: last line of defence before anything is written ---------
    if len(out) < MIN_COVERAGE * len(tickers):
        die(f"only {len(out)} of {len(tickers)} names scored out of "
            f"{len(tickers)}.\n"
            f"A file this thin looks perfectly well-formed and is wrong: the\n"
            f"names that failed to download are exactly the ones that cannot\n"
            f"appear as candidates, so the output understates the opportunity\n"
            f"set without any visible error. Re-run rather than trusting it.")

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

        # ---- price the actual spread against the actual chain ---------------
        print(f"pricing the on-spec spread for {len(cand)} candidates...")
        rows = []
        for t, spot, sig in zip(cand.ticker, cand.price, cand.sigma_21d):
            rows.append(price_the_trade(t, float(spot), float(sig),
                                        px[t] if t in px.columns else pd.Series(dtype=float)))
            time.sleep(0.4)
        gates = pd.DataFrame(rows, index=cand.index)
        cand = pd.concat([cand, gates], axis=1)

        # Earnings before expiry is the one hard disqualifier: it is not a
        # mean-reversion trade any more, it is an event bet. 3 of 3 off-spec
        # trades died; the spec exists to be followed.
        ne = pd.to_datetime(cand.next_earnings, errors="coerce")
        ex = pd.to_datetime(cand.expiry, errors="coerce")
        cand["earnings_before_expiry"] = (ne.notna() & ex.notna() & (ne <= ex))

        warn = []
        for r in cand.itertuples():
            w = []
            if r.opt_note:
                w.append("CHAIN")
            if r.earnings_before_expiry:
                w.append("EARNINGS")
            if np.isfinite(r.dte) and abs(r.dte - TARGET_DTE) > DTE_DRIFT_WARN:
                w.append(f"DTE{int(r.dte)}")
            if np.isfinite(r.be_sigma) and r.be_sigma > BE_SIGMA_WARN:
                w.append(f"BE{r.be_sigma:.2f}sig")
            if np.isfinite(r.leg_oi) and r.leg_oi < MIN_LEG_OI:
                w.append(f"OI{int(r.leg_oi)}")
            if np.isfinite(r.spread_cost_pct) and r.spread_cost_pct > MAX_SPREAD_COST:
                w.append(f"SLIP{r.spread_cost_pct:.0f}%")
            warn.append(",".join(w) if w else "clear")
        cand["warn"] = warn
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
    if len(cand) and "warn" in cand:
        print("\n" + "=" * 70)
        print("THE TRADE, PRICED AGAINST THE REAL CHAIN")
        print("=" * 70)
        show = ["ticker", "price", "expiry", "dte", "k_long", "k_short",
                "width_built", "debit", "be_pct", "be_sigma", "rr",
                "clear_rate", "leg_oi", "spread_cost_pct", "warn"]
        print(cand[[c for c in show if c in cand]].to_string(index=False))
        print(f"\n  be_sigma    breakeven in the stock's own 21d sigma. Median move is")
        print(f"              +0.54 sigma and only 23% reach 1.5, so anything over")
        print(f"              {BE_SIGMA_WARN:.2f} needs a better-than-usual move to break even.")
        print(f"  clear_rate  share of trailing {TARGET_DTE}d windows this stock cleared that")
        print(f"              breakeven, unconditional on the signal (so if anything a")
        print(f"              slight understatement). clear_n is the independent-window")
        print(f"              count — it is small, treat it as a sanity check.")
        print(f"  width_built what the real strikes allow, NOT suggested_width.")
        print(f"  warn        advisory only. Nothing is dropped; read the reason.")
        ok = cand[cand.warn == "clear"]
        print(f"\n  {len(ok)} of {len(cand)} candidate(s) clear every gate"
              f"{': ' + ', '.join(ok.ticker) if len(ok) else ''}")
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
