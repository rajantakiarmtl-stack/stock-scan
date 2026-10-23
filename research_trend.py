#!/usr/bin/env python3
"""
Does downtrend PERSISTENCE predict the forward move, among trades that already
pass the current entry rule?

The gap this fills. The system's entry rule is "7% below the 50-day while above
the 200-day". Depth has been measured (7% beats 5% and 6%; 8% was rejected on a
fit/holdout disagreement). Entry timing has been measured three times and
confirmation always loses. The 200-day gate has been measured (below-200 names
fail holdout at +0.068 sigma, t = -2.38). What has NEVER been measured is the
SHAPE OF THE PATH into the dip: whether a stock that has ground lower for weeks
with a falling 50-day behaves differently from one that gapped down twice and
chopped sideways. Every backtested trade was in a recent downtrend by
construction, so "the chart looks bad" cannot be a filter on its own. The
question is whether persistence, conditional on passing the current rule,
separates winners from losers.

METHOD — unchanged from the studies this extends, so the numbers are comparable.

  Outcome is MODEL-FREE: the forward move over the holding horizon expressed in
  units of the stock's own horizon-matched sigma. No option pricing, no implied
  vol assumption, no strike selection. z = fwd_return / (rv60 * sqrt(H/252)).
  A zero-drift random walk gives z = 0. This is the same measure used in the
  below-200-day and earnings studies.

  Standard errors are CLUSTERED BY CALENDAR DATE. This is the whole ballgame on
  this kind of panel: these triggers fire in clusters (99 names on one day in
  June 2026), and an unclustered SE treats that as 99 independent trades. The
  below-200-day result flipped sign under clustering.

  FIT 2021-01-01 .. 2024-12-31, HOLDOUT 2025-01-01 onward. The holdout is not
  used to choose anything. A feature that disagrees across the two windows on a
  thin sample is noise, and is reported as noise.

  MULTIPLE-COMPARISON BUDGET, STATED UP FRONT: 6 pre-registered features, each
  cut into terciles = 18 cells plus 6 top-vs-bottom contrasts. At 18 looks one
  nominal t = 2.1 is expected by chance. Nothing here is a finding unless it
  holds in BOTH windows with the sign the same and |t| > 2.5 in the holdout.

  No feature was chosen after seeing the result. The six below were written
  down before the first run.
"""

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance pandas numpy curl_cffi")

SESSION = None
try:
    from curl_cffi import requests as _cr
    SESSION = _cr.Session(impersonate="chrome")
    print("curl_cffi session active")
except Exception:
    print("!! curl_cffi NOT installed - expect throttling")

HERE = Path(__file__).parent

# ---------------------------------------------------------------- config

YEARS      = 6                 # need 2021 fit window + 200d warmup
HOLD_DAYS  = 21                # trading days; 30 calendar DTE per the spec
FITEND     = pd.Timestamp("2025-01-01")
FITSTART   = pd.Timestamp("2021-01-01")

MIN_DOLLAR_VOL = 100e6
MIN_PRICE      = 50.0
MAX_VOL_RANK   = 0.33          # the VALIDATED filter (+0.885 sigma holdout,
                               # t=5.00) not the 25% absolute currently in
                               # scan.py (+0.308 sigma, t=2.31)
MIN_RVOL       = 10.0          # data-quality floor, percent annualised
CALL_TRIGGER   = -0.07         # 7% below the 50-day

CHUNK, PAUSE, PASSES = 15, 1.5, 4
MIN_HIST = 320

WATCH = ["STT", "BNY", "NTRS", "JPM", "PFG"]   # print live feature values


def die(msg):
    print("\n!! " + msg)
    sys.exit(1)


# ---------------------------------------------------------------- download

def norm_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download(batch):
    global SESSION
    kw = dict(period=f"{YEARS}y", interval="1d", auto_adjust=False,
              progress=False, group_by="column", threads=False)
    if SESSION is not None:
        try:
            return yf.download(batch, session=SESSION, **kw)
        except TypeError:
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
    got_c, got_v, todo = {}, {}, list(names)
    for p in range(PASSES):
        if not todo:
            break
        if p:
            wait = 20 * p
            print(f"  {len(todo)} missing - waiting {wait}s before pass {p+1}")
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
        die("nothing came back - Yahoo is hard-blocking this runner")
    return pd.DataFrame(got_c).sort_index(), pd.DataFrame(got_v).sort_index()


# ------------------------------------------------- per-symbol metric helper

def per_sym(df, fn):
    """Apply fn to each column's OWN non-missing series and reindex back.

    This is not a stylistic choice. Computing a rolling statistic straight off
    a DataFrame whose columns have different start dates reads across a union
    index, and any NaN inside the window poisons the result. That exact bug
    silently discarded 4,005 of 5,658 symbols in build_universe.py and is
    documented in the spec. Never roll across the panel.
    """
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = (fn(s).reindex(df.index) if len(s) > 2
                  else pd.Series(index=df.index, dtype=float))
    return pd.DataFrame(out, index=df.index)


def run_length_true(s):
    """Consecutive trailing True count, per element."""
    b = s.fillna(False).astype(int)
    grp = (b == 0).cumsum()
    return b.groupby(grp).cumsum()


# ---------------------------------------------------------------- main

def main():
    path = HERE / "universe.txt"
    if not path.exists():
        die("universe.txt missing")
    tickers = sorted({t.strip() for t in path.read_text().split() if t.strip()})
    print(f"universe: {len(tickers)} names, pulling {YEARS}y\n")

    close, vol = download(tickers)
    print(f"\ngot {close.shape[1]} tickers, {close.shape[0]} bars, "
          f"{close.index[0].date()} .. {close.index[-1].date()}")

    logret = per_sym(close, lambda s: np.log(s).diff())

    sma50  = per_sym(close, lambda s: s.rolling(50,  min_periods=50 ).mean())
    sma200 = per_sym(close, lambda s: s.rolling(200, min_periods=200).mean())
    rv60   = per_sym(logret, lambda s: s.rolling(60, min_periods=60).std()
                                        * np.sqrt(252) * 100)
    dvol   = per_sym(close * vol,
                     lambda s: s.rolling(60, min_periods=40).median())

    # ---- the six pre-registered persistence features -------------------
    # NOTE: (close < sma50) returns False, not NaN, wherever sma50 is still in
    # its 50-bar warmup. Left alone, a run-length would begin counting on
    # undefined data. Mask the warmup explicitly.
    below   = (close < sma50).where(sma50.notna())
    f_days_below   = per_sym(below.astype(float),
                             lambda s: run_length_true(s > 0.5).astype(float))
    f_sma50_slope  = sma50  / sma50.shift(20)  - 1.0
    f_sma200_slope = sma200 / sma200.shift(60) - 1.0
    f_ret42        = close  / close.shift(42)  - 1.0

    f_days_since_hi = per_sym(close, lambda s: 62 - s.rolling(63, min_periods=63)
                              .apply(np.argmax, raw=True))

    tot63   = per_sym(close, lambda s: np.log(s / s.shift(63)))
    worst3  = per_sym(logret, lambda s: s.rolling(63, min_periods=63)
                      .apply(lambda w: np.sort(w)[:3].sum(), raw=True))
    f_concentration = (worst3 / tot63).where(tot63 < -0.01)

    FEATURES = {
        "sessions below 50-day":        f_days_below,
        "50-day slope, 20 sessions":    f_sma50_slope,
        "200-day slope, 60 sessions":   f_sma200_slope,
        "63-day return":                f_ret42,
        "sessions since 63-day high":   f_days_since_hi,
        "3 worst days / total fall":    f_concentration,
    }

    # ---- point-in-time universe gate -----------------------------------
    # everything lagged one day where a same-day read would be lookahead
    vol_rank = rv60.rank(axis=1, pct=True)

    elig = (  (dvol.shift(1) >= MIN_DOLLAR_VOL)
            & (close         >= MIN_PRICE)
            & (close         >  sma200)
            & (rv60          >= MIN_RVOL)
            & (vol_rank      <= MAX_VOL_RANK) )

    depth   = close / sma50 - 1.0
    trigger = elig & (depth <= CALL_TRIGGER)

    # ---- outcome: forward move in units of own horizon sigma ------------
    fwd     = per_sym(close, lambda s: s.shift(-HOLD_DAYS) / s - 1.0)
    sigma_H = (rv60 / 100.0) * np.sqrt(HOLD_DAYS / 252.0)
    z       = fwd / sigma_H

    rows = []
    tmask = trigger.values
    idx   = close.index
    cols  = close.columns
    for j, c in enumerate(cols):
        hits = np.where(tmask[:, j])[0]
        for i in hits:
            zz = z.values[i, j]
            if not np.isfinite(zz):
                continue
            r = {"date": idx[i], "ticker": c, "z": zz,
                 "depth": depth.values[i, j], "rv60": rv60.values[i, j],
                 "fwd": fwd.values[i, j]}
            ok = True
            for name, F in FEATURES.items():
                v = F.values[i, j]
                if not np.isfinite(v):
                    ok = False
                    break
                r[name] = v
            if ok:
                rows.append(r)

    t = pd.DataFrame(rows)
    t = t[(t.date >= FITSTART)].reset_index(drop=True)
    if t.empty:
        die("no trades after the gate - check the universe gate")

    t["win"] = (t.date < FITEND).map({True: "FIT", False: "HOLDOUT"})
    print(f"\ntrades: {len(t)}   FIT {int((t.win=='FIT').sum())}   "
          f"HOLDOUT {int((t.win=='HOLDOUT').sum())}   "
          f"distinct dates {t.date.nunique()}   tickers {t.ticker.nunique()}")
    print(f"mean names firing on a day that fires: "
          f"{len(t)/max(t.date.nunique(),1):.1f}")

    # ---------------------------------------------------- clustered stats
    def clustered(y, g):
        y = np.asarray(y, float)
        n = len(y)
        if n < 3:
            return np.nan, np.nan, np.nan, 0
        m = y.mean()
        s = pd.Series(y - m).groupby(np.asarray(g)).sum().values
        se = np.sqrt((s ** 2).sum()) / n
        return m, se, (m / se if se > 0 else np.nan), len(np.unique(g))

    def show(lbl, d):
        m, se, tt, k = clustered(d.z, d.date)
        print(f"    {lbl:<26} n {len(d):>5}  dates {k:>4}  "
              f"z {m:+.3f}  SE {se:.3f}  t {tt:+.2f}")
        return m, tt

    print("\n" + "=" * 78)
    print("BASELINE - the current rule, no persistence conditioning")
    print("=" * 78)
    print("  forward 21-day move in units of the stock's own sigma; 0 = random walk")
    for w in ("FIT", "HOLDOUT"):
        show(w, t[t.win == w])
    show("ALL", t)

    print("\n" + "=" * 78)
    print("THE QUESTION - does persistence of the downtrend matter?")
    print("=" * 78)
    print("  terciles computed on the FIT window only, then applied to both.")
    print("  18 cells; a finding needs the same sign in both windows and")
    print("  |t| > 2.5 in the holdout.\n")

    verdict = []
    for name in FEATURES:
        fit = t[t.win == "FIT"]
        try:
            qs = fit[name].quantile([1/3, 2/3]).values
        except Exception:
            continue
        if not np.all(np.isfinite(qs)) or qs[0] == qs[1]:
            print(f"  {name}: degenerate distribution, skipped\n")
            continue

        def bucket(v):
            return "low" if v <= qs[0] else ("mid" if v <= qs[1] else "high")

        t["_b"] = t[name].map(bucket)
        print(f"  {name}   (tercile cuts {qs[0]:+.3f} / {qs[1]:+.3f})")
        got = {}
        for w in ("FIT", "HOLDOUT"):
            print(f"    -- {w}")
            for b in ("low", "mid", "high"):
                d = t[(t.win == w) & (t._b == b)]
                if len(d) < 10:
                    print(f"    {b:<26} n {len(d):>5}  too thin")
                    continue
                m, tt = show(b, d)
                got[(w, b)] = (m, tt)
        if all(k in got for k in (("FIT","low"),("FIT","high"),
                                  ("HOLDOUT","low"),("HOLDOUT","high"))):
            df_ = got[("FIT","high")][0]   - got[("FIT","low")][0]
            dh  = got[("HOLDOUT","high")][0] - got[("HOLDOUT","low")][0]
            agree = "SAME SIGN" if df_ * dh > 0 else "DISAGREE"
            print(f"    high minus low:  FIT {df_:+.3f}   HOLDOUT {dh:+.3f}   {agree}")
            verdict.append((name, df_, dh, agree))
        print()

    print("=" * 78)
    print("SUMMARY - high tercile minus low tercile, forward z")
    print("=" * 78)
    print(f"  {'feature':<30} {'FIT':>9} {'HOLDOUT':>9}   agreement")
    for name, df_, dh, agree in verdict:
        print(f"  {name:<30} {df_:>+9.3f} {dh:>+9.3f}   {agree}")

    # ---------------------------------------------------- live placement
    print("\n" + "=" * 78)
    print("WHERE THE WATCHLIST SITS TODAY on each feature")
    print("=" * 78)
    last = close.index[-1]
    print(f"  last bar {last.date()}")
    fit = t[t.win == "FIT"]
    cuts = {n: fit[n].quantile([1/3, 2/3]).values for n in FEATURES
            if n in fit and fit[n].notna().any()}
    for tk in WATCH:
        if tk not in close.columns:
            print(f"  {tk}: not in universe")
            continue
        i = close.index.get_loc(last)
        j = close.columns.get_loc(tk)
        print(f"\n  {tk}  close {close.values[i,j]:.2f}  "
              f"depth vs 50d {depth.values[i,j]*100:+.2f}%  "
              f"vs 200d {(close.values[i,j]/sma200.values[i,j]-1)*100:+.2f}%  "
              f"rv60 {rv60.values[i,j]:.1f}%  volrank {vol_rank.values[i,j]:.3f}  "
              f"triggers {bool(trigger.values[i,j])}")
        for n, F in FEATURES.items():
            v = F.values[i, j]
            if n in cuts and np.isfinite(v):
                q = cuts[n]
                b = "low" if v <= q[0] else ("mid" if v <= q[1] else "high")
                print(f"      {n:<30} {v:+9.3f}   -> {b}")
            else:
                print(f"      {n:<30} {v:+9.3f}")

    t.drop(columns=["_b"], errors="ignore").to_csv(
        HERE / "research_trend_trades.csv", index=False)
    print(f"\nwrote research_trend_trades.csv ({len(t)} rows)")


if __name__ == "__main__":
    main()
