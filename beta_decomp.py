#!/usr/bin/env python3
"""
Does it matter whether the dip was MARKET-DRIVEN or STOCK-SPECIFIC?

THE QUESTION. Conditional on a call signal firing (7% below the 50-day, above
the 200-day, liquid, calm), split the decline that produced it into the part
beta explains and the part it does not. Does the forward move differ?

  resid = R_stock(21d) - beta * R_spy(21d)

A resid near zero means the stock fell exactly as its market sensitivity
predicted — nothing happened to the company, the market just fell. A large
negative resid means the stock fell well beyond what the market explains, i.e.
something stock-specific. A positive resid means it actually held up better
than beta predicted while still ending below its 50-day.

TWO COMPETING PREDICTIONS, BOTH WRITTEN DOWN BEFORE THE RUN. They disagree, and
the existing evidence in this project supports the SECOND one, which is why this
is worth running rather than assuming.

  H1 (the intuitive one) IDIOSYNCRATIC DIPS REVERT BETTER.
     A stock-specific overreaction has something to revert. A market-driven
     decline leaves the stock correctly priced relative to its peers, so there
     is no particular reason THIS name bounces. Predicts forward z rises as
     resid becomes more negative.

  H2 (what the project's own measurements imply) MARKET-DRIVEN DIPS DO AS WELL
     OR BETTER.
     The spec already measures SPY 4-8% off its high at a 66% win rate against
     53% when the market is within 4% of its high, and the k-of-n rule counts
     "SPY drawdown <= -5%" as a POSITIVE with "NOT SPY drawdown <= -10%" as the
     only flip. Buying into market weakness has measured better, not worse. If
     that is the real mechanism, resid carries no extra information once market
     state is controlled for, and may point the other way.

  PREDICTION RECORDED: I expect H2. H1 is the idea that motivated the test and I
  expect it to fail. Writing that down so the result cannot be re-narrated
  afterwards. This project has had four predictions of mine come out wrong
  (depth concentration, the price-floor example, ETF clustering, and the claim
  that 5% was the more robust trigger) and recording the call up front is the
  only thing that keeps score honestly.

THE CONFOUND THAT DECIDES WHETHER THIS IS READABLE AT ALL. resid is mechanically
tied to the market's own move: when the market falls hard, a stock that merely
tracks it has resid ~ 0, and "market fell hard" is itself a measured POSITIVE.
So a raw resid tercile sort is partly a market-state sort wearing a different
label. The run therefore reports corr(resid, SPY-off-high) first, and the
primary contrast is re-run WITHIN market-state buckets. If the double sort kills
it, the single sort was the confound and nothing is adopted.

  DECISION RULE. A finding requires: same sign in fit AND holdout, |t| > 2.5 in
  the holdout on cluster-robust SEs, and it must survive the double sort against
  market state. A holdout-only result is noise — that pattern has been wrong
  five times here.

  BUDGET. One feature, cut into fit-window terciles: 3 cells + 1 top-vs-bottom
  contrast, plus the 2x3 confound check and two correlation checks. That is
  essentially one hypothesis. Nothing marginal gets adopted.

METHOD. Mirrors research_trend.py and depth_retest.py so numbers stay
comparable. Same universe file, same point-in-time gate, 21-session horizon
(= 30 calendar DTE), fit 2021-2024 / holdout 2025+, SEs CLUSTERED BY CALENDAR
DATE. Outcomes reported side by side:

  z   model-free. forward move / (rv60 * sqrt(21/252)). No option pricing.
  $   the spec structure, ATM long / short leg 1.5x the 1-sigma horizon move,
      real strike increments, BS entry at sigma_iv = 0.85 * rv60 * 1.10, 2%
      slippage, $0.30/leg both sides, HELD TO EXPIRY.

z is believed first; the dollar figure carries modelling assumptions.

NO LOOKAHEAD. beta and idiosyncratic vol at date t use a 250-day window ending
at t — past data only. Tercile cuts are computed on the FIT window alone and
applied to both. The forward move runs from t's close to t+21's close, the same
convention as every other study here.
"""

import sys, time, math, warnings
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

YEARS     = 6
HOLD_DAYS = 21                      # sessions; = 30 calendar DTE per the spec
LOOKBACK  = 21                      # sessions over which the decline is measured
BETA_WIN  = 250                     # same window scan.py uses
FITSTART  = pd.Timestamp("2021-01-01")
FITEND    = pd.Timestamp("2025-01-01")

MIN_DOLLAR_VOL = 100e6
MIN_PRICE      = 50.0
MAX_VOL_RANK   = 0.33
MIN_RVOL       = 10.0
CALL_TRIGGER   = 0.07               # the LIVE rule: 7% below the 50-day

WIDTH_SIGMA = 1.5
IV_MULT     = 0.85 * 1.10
SLIP        = 0.02
COMM        = 0.30 * 2 * 2

# market-state split. 4% is the spec's own boundary: SPY within 4% of its high
# wins 53%, 4-8% off wins 66%. Using the already-measured cut rather than a new
# one keeps this a confound check instead of another free parameter.
MKT_CUT = -0.04

CHUNK, PAUSE, PASSES = 15, 1.5, 4
MIN_HIST = 320


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


# ------------------------------------------------------------- helpers

def per_sym(df, fn):
    """Apply fn to each column's OWN non-missing series, then reindex back.

    Rolling straight off a panel whose columns start on different dates reads
    across the union index and one NaN inside the window poisons the result.
    That bug silently dropped 4,005 of 5,658 symbols in build_universe.py.
    """
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = (fn(s).reindex(df.index) if len(s) > 2
                  else pd.Series(index=df.index, dtype=float))
    return pd.DataFrame(out, index=df.index)


def per_sym_vs_market(df, mkt, fn):
    """Like per_sym, but hands fn the stock series AND the market series
    reindexed onto the STOCK's own dates.

    Aligning the other way — reindexing the stock onto a shared calendar —
    would reintroduce the union-index trap the moment two names have different
    histories.
    """
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        if len(s) <= 2:
            out[c] = pd.Series(index=df.index, dtype=float)
            continue
        m = mkt.reindex(s.index)
        out[c] = fn(s, m).reindex(df.index)
    return pd.DataFrame(out, index=df.index)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return S * _ncdf(d1) - K * _ncdf(d1 - v)


def strike_inc(S):
    if S < 25:
        return 1.0
    if S < 200:
        return 2.5
    return 5.0


def price_trade(S, S_T, rv60_pct):
    """Spec structure at S, held to expiry, settled at S_T."""
    if not all(np.isfinite([S, S_T, rv60_pct])) or S <= 0 or rv60_pct <= 0:
        return None
    T = HOLD_DAYS / 252.0
    sig_h = S * (rv60_pct / 100.0) * math.sqrt(T)
    inc = strike_inc(S)
    k_long = round(S / inc) * inc
    k_short = k_long + max(inc, round(WIDTH_SIGMA * sig_h / inc) * inc)
    if k_short <= k_long or k_long <= 0:
        return None
    iv = (rv60_pct / 100.0) * IV_MULT
    d_ps = bs_call(S, k_long, T, iv) - bs_call(S, k_short, T, iv)
    if d_ps <= 0.05:
        return None
    debit = d_ps * 100.0 * (1.0 + SLIP)
    payoff = min(max(S_T - k_long, 0.0), k_short - k_long) * 100.0
    return payoff - debit - COMM, debit


def clustered(y, g):
    """Mean and cluster-robust SE, clustering on g (calendar date).

    These triggers fire in clusters and an unclustered SE treats 99 names on
    one day as 99 independent trades. The below-200-day result FLIPPED SIGN
    under clustering.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return np.nan, np.nan, np.nan, 0
    m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(g)).sum().values
    se = math.sqrt((s ** 2).sum()) / n
    return m, se, (m / se if se > 0 else np.nan), len(np.unique(g))


def cl_diff(a, b, col):
    """Cluster-robust SE for the difference of two group means.

    Influence function (a.col - ma)/na and -(b.col - mb)/nb, summed within
    date, se = sqrt(sum(s^2)). Treating the two means as independent would
    understate the SE whenever both groups fire on the same day, which they do
    constantly.
    """
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return np.nan, np.nan, np.nan
    ma, mb = a[col].mean(), b[col].mean()
    s = pd.concat([
        pd.DataFrame({"d": a.date.values, "v": (a[col].values - ma) / na}),
        pd.DataFrame({"d": b.date.values, "v": -(b[col].values - mb) / nb}),
    ])
    tot = s.groupby("d").v.sum().values
    se = math.sqrt((tot ** 2).sum())
    d = ma - mb
    return d, se, (d / se if se > 0 else np.nan)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    x, y = x[ok], y[ok]
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------- main

def main():
    path = HERE / "universe.txt"
    if not path.exists():
        die("universe.txt missing")
    tickers = sorted({t.strip() for t in path.read_text().split() if t.strip()})
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"universe: {len(tickers)} names (SPY included), pulling {YEARS}y\n")

    close, vol = download(tickers)
    if "SPY" not in close.columns:
        die("SPY did not come back - the whole decomposition needs it")
    print(f"\ngot {close.shape[1]} tickers, {close.shape[0]} bars, "
          f"{close.index[0].date()} .. {close.index[-1].date()}")

    spy = close["SPY"].dropna()
    close = close.drop(columns=["SPY"])
    vol = vol.drop(columns=["SPY"], errors="ignore")
    spy_ret = spy.pct_change()

    logret = per_sym(close, lambda s: np.log(s).diff())
    pctret = per_sym(close, lambda s: s.pct_change())
    sma50  = per_sym(close, lambda s: s.rolling(50,  min_periods=50 ).mean())
    sma200 = per_sym(close, lambda s: s.rolling(200, min_periods=200).mean())
    rv60   = per_sym(logret, lambda s: s.rolling(60, min_periods=60).std()
                                        * np.sqrt(252) * 100)
    dvol   = per_sym(close * vol,
                     lambda s: s.rolling(60, min_periods=40).median())

    # ---- beta and idiosyncratic vol, 250d ending at t (past data only) ----
    print("computing rolling beta and idiosyncratic vol ...")

    def _beta(s, m):
        r = s.pct_change()
        mr = m.pct_change()
        cov = r.rolling(BETA_WIN, min_periods=200).cov(mr)
        var = mr.rolling(BETA_WIN, min_periods=200).var()
        return cov / var

    beta = per_sym_vs_market(close, spy, _beta)

    def _idio(s, m):
        r = s.pct_change()
        mr = m.pct_change()
        cov = r.rolling(BETA_WIN, min_periods=200).cov(mr)
        var = mr.rolling(BETA_WIN, min_periods=200).var()
        b = cov / var
        return (r - b * mr).rolling(BETA_WIN, min_periods=200).std()

    idio = per_sym_vs_market(close, spy, _idio)

    # ---- the decomposition of the decline into the window ----------------
    r_stock = close / close.shift(LOOKBACK) - 1.0
    r_spy_s = (spy / spy.shift(LOOKBACK) - 1.0).reindex(close.index)
    r_spy = pd.DataFrame(np.repeat(r_spy_s.values[:, None], close.shape[1], 1),
                         index=close.index, columns=close.columns)
    explained = beta * r_spy
    resid = r_stock - explained
    # normalise so names of different volatility are comparable
    resid_z = resid / (idio * np.sqrt(LOOKBACK))

    # market state, the spec's own variable
    spy_off_s = (spy / spy.rolling(252, min_periods=252).max() - 1.0
                 ).reindex(close.index)
    spy_off = pd.DataFrame(np.repeat(spy_off_s.values[:, None], close.shape[1], 1),
                           index=close.index, columns=close.columns)

    vol_rank = rv60.rank(axis=1, pct=True)
    elig = (  (dvol.shift(1) >= MIN_DOLLAR_VOL)
            & (close         >= MIN_PRICE)
            & (close         >  sma200)
            & (rv60          >= MIN_RVOL)
            & (vol_rank      <= MAX_VOL_RANK) )

    depth = close / sma50 - 1.0
    fwd    = per_sym(close, lambda s: s.shift(-HOLD_DAYS) / s - 1.0)
    fwd_px = per_sym(close, lambda s: s.shift(-HOLD_DAYS))
    z_all  = fwd / ((rv60 / 100.0) * np.sqrt(HOLD_DAYS / 252.0))

    # ---- first touch, with an explicit re-arm ---------------------------
    # NOTE: (close >= sma50) returns False, not NaN, through the 50-bar warmup.
    # Unmasked, every name looks "armed" from bar 0 and the first touch lands on
    # undefined data.
    above = (close >= sma50).where(sma50.notna())
    episode = per_sym(above.astype(float),
                      lambda s: (s > 0.5).cumsum().astype(float))

    b = (depth <= -CALL_TRIGGER) & elig & sma50.notna()
    fire = {}
    for c in close.columns:
        bb = b[c].fillna(False).values
        ep = episode[c].values
        seen, f = set(), np.zeros(len(bb), bool)
        for i in range(len(bb)):
            if not bb[i]:
                continue
            e = ep[i]
            if not np.isfinite(e) or e in seen:
                continue
            seen.add(e)
            f[i] = True
        fire[c] = pd.Series(f, index=close.index)
    fire = pd.DataFrame(fire, index=close.index)

    # ---- collect ---------------------------------------------------------
    idx, cols = close.index, close.columns
    M = {k: v.values for k, v in dict(
        close=close, rv60=rv60, z=z_all, fwdpx=fwd_px, depth=depth,
        resid_z=resid_z, resid=resid, beta=beta, rstock=r_stock,
        rspy=r_spy, expl=explained, spyoff=spy_off).items()}
    fv = fire.values
    rows = []
    for j, c in enumerate(cols):
        for i in np.where(fv[:, j])[0]:
            rec = {k: M[k][i, j] for k in M}
            if not np.isfinite([rec["close"], rec["fwdpx"], rec["rv60"],
                                rec["z"], rec["resid_z"], rec["spyoff"],
                                rec["beta"]]).all():
                continue
            pt = price_trade(rec["close"], rec["fwdpx"], rec["rv60"])
            if pt is None:
                continue
            rec["pnl"], rec["debit"] = pt
            rec["date"] = idx[i]
            rec["ticker"] = c
            rec["win"] = rec["pnl"] > 0
            rows.append(rec)

    t = pd.DataFrame(rows)
    if t.empty:
        die("no trades after the gate - check the universe gate")
    t = t[t.date >= FITSTART].reset_index(drop=True)
    t["w"] = np.where(t.date < FITEND, "FIT", "HOLDOUT")
    span = (idx[-1] - max(idx[0], FITSTART)).days / 365.25

    L = "=" * 78
    print(f"\ntrades {len(t)}   FIT {(t.w=='FIT').sum()}   "
          f"HOLDOUT {(t.w=='HOLDOUT').sum()}   dates {t.date.nunique()}   "
          f"tickers {t.ticker.nunique()}   {len(t)/span:.0f}/yr")

    # ---- descriptives ----------------------------------------------------
    print("\n" + L)
    print("HOW MUCH OF A TYPICAL SIGNAL'S DECLINE IS THE MARKET?")
    print(L)
    print(f"  over the {LOOKBACK} sessions into the signal:")
    print(f"    stock move            median {t.rstock.median():+.2%}")
    print(f"    SPY move              median {t.rspy.median():+.2%}")
    print(f"    beta                  median {t.beta.median():+.3f}")
    print(f"    beta-explained part   median {t.expl.median():+.2%}")
    print(f"    residual (stock-specific) median {t.resid.median():+.2%}")
    print(f"    residual in sigma     median {t.resid_z.median():+.2f}")
    share = (t.expl / t.rstock).replace([np.inf, -np.inf], np.nan)
    share = share[(t.rstock < -0.01)]        # ratio is meaningless near zero
    print(f"\n    market share of the decline, median {share.median():+.1%}"
          f"  (n={share.notna().sum()}, only where the stock fell >1%)")
    print("    A ratio is unstable near a zero denominator, so the sorts below")
    print("    all use the normalised residual, never this share.")

    # ---- THE CONFOUND, checked before any result is read -----------------
    print("\n" + L)
    print("THE CONFOUND CHECK — is resid just market state relabelled?")
    print(L)
    c1 = corr(t.resid_z, t.spyoff)
    c2 = corr(t.resid_z, t.depth)
    c3 = corr(t.resid_z, t.beta)
    print(f"  corr(resid_z, SPY off its high) = {c1:+.3f}")
    print(f"  corr(resid_z, depth below 50d)  = {c2:+.3f}")
    print(f"  corr(resid_z, beta)             = {c3:+.3f}")
    print("\n  resid is mechanically tied to the market's own move: when the")
    print("  market falls hard, a stock that merely tracks it has resid ~ 0,")
    print("  and 'market fell hard' is itself a measured POSITIVE. If |corr|")
    print("  with SPY-off-high is large, the single sort below is partly a")
    print("  market-state sort and only the double sort is interpretable.")

    # ---- the primary sort ------------------------------------------------
    fit = t[t.w == "FIT"]
    qs = fit.resid_z.quantile([1/3, 2/3]).values
    if not np.all(np.isfinite(qs)) or qs[0] == qs[1]:
        die("degenerate resid_z distribution")
    print("\n" + L)
    print("H1 vs H2 — forward outcome by how stock-specific the decline was")
    print(L)
    print(f"  fit-window tercile cuts: {qs[0]:+.3f} / {qs[1]:+.3f} (sigma units)")
    print("  'idiosyncratic' = most negative resid (fell well beyond beta)")
    print("  'market-driven' = resid nearest zero or positive\n")

    def bucket(v):
        return ("idiosyncratic" if v <= qs[0]
                else ("middle" if v <= qs[1] else "market-driven"))
    t["bk"] = t.resid_z.map(bucket)
    ORD = ["idiosyncratic", "middle", "market-driven"]

    got = {}
    for w in ("FIT", "HOLDOUT"):
        print(f"  -- {w}")
        for bkt in ORD:
            d = t[(t.w == w) & (t.bk == bkt)]
            if len(d) < 10:
                print(f"     {bkt:<16} n {len(d):>4}  too thin")
                continue
            zm, zse, zt, k = clustered(d.z, d.date)
            pm, pse, pt_, _ = clustered(d.pnl, d.date)
            print(f"     {bkt:<16} n {len(d):>4} dates {k:>4}  "
                  f"z {zm:+.3f} (SE {zse:.3f}, t {zt:+.2f})   "
                  f"$ {pm:+7.0f} (t {pt_:+.2f})  win {d.win.mean():.0%}")
            got[(w, bkt)] = (zm, pm)
        print()

    print("  CONTRAST  idiosyncratic minus market-driven, cluster-robust:")
    verdict = {}
    for w in ("FIT", "HOLDOUT", "ALL"):
        a = t[(t.bk == "idiosyncratic")] if w == "ALL" else t[(t.w == w) & (t.bk == "idiosyncratic")]
        bq = t[(t.bk == "market-driven")] if w == "ALL" else t[(t.w == w) & (t.bk == "market-driven")]
        if len(a) < 10 or len(bq) < 10:
            print(f"    {w:<9} too thin")
            continue
        dz, sez, tz = cl_diff(a, bq, "z")
        dp, sep, tp = cl_diff(a, bq, "pnl")
        print(f"    {w:<9} z {dz:+.3f} (SE {sez:.3f}, t {tz:+.2f})    "
              f"$ {dp:+7.0f} (SE {sep:5.0f}, t {tp:+.2f})")
        verdict[w] = (dz, tz, dp, tp)

    if "FIT" in verdict and "HOLDOUT" in verdict:
        agree = verdict["FIT"][0] * verdict["HOLDOUT"][0] > 0
        print(f"\n    sign agreement on z: FIT {verdict['FIT'][0]:+.3f} / "
              f"HOLDOUT {verdict['HOLDOUT'][0]:+.3f}  -> "
              f"{'SAME SIGN' if agree else '*** DISAGREE - noise ***'}")

    # ---- the double sort, which decides it ------------------------------
    print("\n" + L)
    print("THE DOUBLE SORT — resid tercile WITHIN market state")
    print(L)
    print(f"  market state split at SPY {MKT_CUT:.0%} off its 52-week high,")
    print("  the spec's own boundary (within 4%: 53% win; 4-8% off: 66%).")
    print("  If the contrast survives inside BOTH buckets it is real. If it")
    print("  only exists across them, it was the market-state confound.\n")
    t["mkt"] = np.where(t.spyoff >= MKT_CUT, "market calm", "market stressed")
    dbl = {}
    for ms in ("market calm", "market stressed"):
        sub = t[t.mkt == ms]
        print(f"  {ms}  (n={len(sub)}, SPY off-high "
              f"{sub.spyoff.min():+.1%} .. {sub.spyoff.max():+.1%})")
        for bkt in ORD:
            d = sub[sub.bk == bkt]
            if len(d) < 10:
                print(f"     {bkt:<16} n {len(d):>4}  too thin")
                continue
            zm, zse, zt, k = clustered(d.z, d.date)
            pm, _, pt_, _ = clustered(d.pnl, d.date)
            print(f"     {bkt:<16} n {len(d):>4} dates {k:>4}  "
                  f"z {zm:+.3f} (t {zt:+.2f})   $ {pm:+7.0f} (t {pt_:+.2f})")
        a, bq = sub[sub.bk == "idiosyncratic"], sub[sub.bk == "market-driven"]
        if len(a) >= 10 and len(bq) >= 10:
            dz, sez, tz = cl_diff(a, bq, "z")
            dp, _, tp = cl_diff(a, bq, "pnl")
            print(f"     contrast          z {dz:+.3f} (t {tz:+.2f})   "
                  f"$ {dp:+7.0f} (t {tp:+.2f})")
            dbl[ms] = (dz, tz)
        else:
            print(f"     contrast          too thin to read "
                  f"(idio n={len(a)}, market n={len(bq)})")
        print()

    # ---- and the market-state effect on its own, for scale --------------
    print(L)
    print("FOR SCALE — the market-state effect on its own")
    print(L)
    print("  This is the already-measured variable. If it dominates resid,")
    print("  the honest conclusion is that market state is the real signal.\n")
    for w in ("FIT", "HOLDOUT"):
        for ms in ("market calm", "market stressed"):
            d = t[(t.w == w) & (t.mkt == ms)]
            if len(d) < 10:
                print(f"  {w:<9} {ms:<16} n {len(d):>4}  too thin")
                continue
            zm, _, zt, k = clustered(d.z, d.date)
            pm, _, pt_, _ = clustered(d.pnl, d.date)
            print(f"  {w:<9} {ms:<16} n {len(d):>4} dates {k:>4}  "
                  f"z {zm:+.3f} (t {zt:+.2f})   $ {pm:+7.0f} (t {pt_:+.2f})  "
                  f"win {d.win.mean():.0%}")
    a = t[t.mkt == "market stressed"]
    bq = t[t.mkt == "market calm"]
    if len(a) >= 10 and len(bq) >= 10:
        dz, sez, tz = cl_diff(a, bq, "z")
        dp, _, tp = cl_diff(a, bq, "pnl")
        print(f"\n  stressed minus calm:  z {dz:+.3f} (t {tz:+.2f})   "
              f"$ {dp:+.0f} (t {tp:+.2f})")

    # ---- verdict ---------------------------------------------------------
    print("\n" + L)
    print("VERDICT, against the rule written down before the run")
    print(L)
    if "FIT" not in verdict or "HOLDOUT" not in verdict:
        print("  *** CANNOT BE READ. One window had under 10 trades in a")
        print("      tercile, so there is no fit/holdout comparison to make.")
        print("      This is a sample-size failure, NOT a null result — do not")
        print("      record it as 'no effect'. Needs a longer window or wider")
        print("      buckets.")
        print(f"      windows present: {sorted(verdict)}")
    else:
        dzf, _, _, _ = verdict["FIT"]
        dzh, tzh, _, _ = verdict["HOLDOUT"]
        same = dzf * dzh > 0
        strong = abs(tzh) > 2.5
        print(f"  same sign in both windows?        {same}"
              f"   (FIT {dzf:+.3f}, HOLDOUT {dzh:+.3f})")
        print(f"  |t| > 2.5 in the holdout?         {strong}  (t {tzh:+.2f})")
        if dbl:
            agree = [ms for ms, (dz, _) in dbl.items() if dz * dzh > 0]
            print(f"  survives the double sort?         "
                  f"{len(agree)} of {len(dbl)} market-state buckets agree in sign")
            for ms, (dz, tz) in dbl.items():
                print(f"      {ms:<16} z {dz:+.3f} (t {tz:+.2f})")
        else:
            print("  survives the double sort?         both buckets too thin")
        print(f"\n  direction: "
              f"{'idiosyncratic dips revert BETTER (H1)' if dzh > 0 else 'market-driven dips do as well or BETTER (H2)'}")
        if not (same and strong):
            print("\n  -> NOT A FINDING. Leave the filters alone. Beta stays a")
            print("     reported column, not a gate.")
        elif dbl and len(agree) < len(dbl):
            print("\n  -> the single sort passed but the double sort did NOT hold")
            print("     in every market-state bucket. That is the confound, not")
            print("     a finding. Leave the filters alone.")
        else:
            print("\n  -> passes every pre-registered test. Worth a second look")
            print("     before anything is changed in scan.py.")
    print("\n  Recorded prediction was H2. Score it honestly either way.")

    t.drop(columns=["bk", "mkt"], errors="ignore").to_csv(
        HERE / "beta_decomp_trades.csv", index=False)
    print(f"\nwrote beta_decomp_trades.csv ({len(t)} rows)")


if __name__ == "__main__":
    main()
