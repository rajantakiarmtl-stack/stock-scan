#!/usr/bin/env python3
"""
Re-rank entry depth under the CURRENT exit rule, and settle the paired question.

WHY THIS RE-RUN EXISTS. Depth was measured on 2026-09-24 and 7% was adopted
(94 signals/yr, +$340 OOS) over 5% (444 signals/yr, +$269 OOS). That study is
not wrong, but two things underneath it have changed since:

  1. EXIT RULE. It priced every trade with an exit at 90% of max profit. On
     2026-10-12 early exits were measured and killed at every level tested, and
     the spec moved to HOLD TO EXPIRY. Depth and exit interact: a shallow entry
     needs more of the move to reach a profit target, so a target-based exit
     penalises shallow entries in a way expiry does not. The ranking can move.
  2. VOL FILTER. It ran under the absolute 25% ceiling. Production reverted to
     the validated vol_rank <= 0.33 on 2026-10-29.

AND ONE THING WAS NEVER MEASURED AT ALL. The old table compares two different
POPULATIONS: all declines reaching 5% against the subset reaching 7%. That
conflates two effects -- (a) declines that stop short of 7% may simply be
weaker setups, and (b) on the declines that DO reach 7%, entering early at 5%
means a worse (higher) entry price. Only (b) is a real cost of moving the
trigger, and only (b) can be measured PAIRED, on the same episode, which is far
more powerful than comparing populations.

PRE-REGISTERED, written before the first run:

  H1  On declines that reach BOTH -5% and -7%, waiting for -7% beats entering
      at -5%. Paired, same episode, clustered by the -5% entry date.
  H2  Declines that stop short of -7% still have a positive edge on their own.
  H3  Depth is monotonic in $/trade under hold-to-expiry.

  DECISION RULE. Moving the live trigger to 5% requires: H2 positive in BOTH
  windows, and the H1 paired penalty smaller than the per-trade dollars the
  extra H2 trades bring in. A fit/holdout sign flip on either kills it,
  however good the holdout looks -- that pattern has been wrong 5 times in
  this project.

  BUDGET. 3 pre-registered hypotheses plus one ordered-set monotonicity check
  (which counts as one, not four). At 5% that is ~0.2 expected false
  positives. Nothing marginal gets adopted.

METHOD. Mirrors research_trend.py exactly so numbers stay comparable: same
universe file, same point-in-time gate, same 21-session horizon (= 30 calendar
DTE per the spec), fit 2021-2024 / holdout 2025+, standard errors CLUSTERED BY
CALENDAR DATE. Two outcomes reported side by side:

  z       model-free. forward move / (rv60 * sqrt(21/252)). No option pricing,
          so no IV assumption can flatter it. Zero-drift random walk = 0.
  $       the actual structure. ATM long, short leg 1.5x the 1-sigma horizon
          move, real strike increments, Black-Scholes entry at
          sigma_iv = 0.85 * rv60 * 1.10, 2% slippage, $0.30/leg both sides,
          HELD TO EXPIRY.

If z and $ disagree, the $ number is the one carrying modelling assumptions and
z is believed first.
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
FITSTART  = pd.Timestamp("2021-01-01")
FITEND    = pd.Timestamp("2025-01-01")

MIN_DOLLAR_VOL = 100e6
MIN_PRICE      = 50.0
MAX_VOL_RANK   = 0.33               # production as of 2026-10-29
MIN_RVOL       = 10.0               # data-quality floor, not a strategy rule

WIDTH_SIGMA = 1.5
IV_MULT     = 0.85 * 1.10           # spec: sigma_iv = 0.85 * rv60 * 1.10
SLIP        = 0.02
COMM        = 0.30 * 2 * 2          # $/leg * 2 legs * both sides

# the depth ladder under test. 0.05 is the shallow end of the live spec's
# history; 0.07 is what scan.py fires on today.
LADDER = [0.05, 0.06, 0.07, 0.08]
SHALLOW, DEEP = 0.05, 0.07          # the paired pair

CHUNK, PAUSE, PASSES = 15, 1.5, 4
MIN_HIST = 320


def die(msg):
    print("\n!! " + msg)
    sys.exit(1)


# ---------------------------------------------------------------- download
# lifted unchanged from research_trend.py -- a proven path, and changing it
# would make these numbers non-comparable with the studies this extends.

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

    Not stylistic. Rolling straight off a panel whose columns start on
    different dates reads across the union index, and one NaN inside the
    window poisons the result. That bug silently dropped 4,005 of 5,658
    symbols in build_universe.py. Never roll across the panel.
    """
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = (fn(s).reindex(df.index) if len(s) > 2
                  else pd.Series(index=df.index, dtype=float))
    return pd.DataFrame(out, index=df.index)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sigma):
    """Black-Scholes call, r = 0, matching the spec's pricing convention."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return S * _ncdf(d1) - K * _ncdf(d1 - v)


def strike_inc(S):
    """Listed increments for liquid US equity options."""
    if S < 25:
        return 1.0
    if S < 200:
        return 2.5
    return 5.0


def price_trade(S, S_T, rv60_pct):
    """Build the spec structure at S, hold to expiry, settle at S_T.

    Returns (pnl_dollars, debit_dollars, k_long, k_short) or None if the
    structure is degenerate (zero or negative debit, sub-nickel legs).
    """
    if not all(np.isfinite([S, S_T, rv60_pct])) or S <= 0 or rv60_pct <= 0:
        return None
    T = HOLD_DAYS / 252.0
    sig_h = S * (rv60_pct / 100.0) * math.sqrt(T)        # 1-sigma, horizon
    inc = strike_inc(S)
    k_long = round(S / inc) * inc                        # ATM = nearest listed
    raw_w = WIDTH_SIGMA * sig_h
    k_short = k_long + max(inc, round(raw_w / inc) * inc)
    if k_short <= k_long or k_long <= 0:
        return None
    iv = (rv60_pct / 100.0) * IV_MULT
    d_ps = bs_call(S, k_long, T, iv) - bs_call(S, k_short, T, iv)
    if d_ps <= 0.05:                                     # degenerate
        return None
    debit = d_ps * 100.0 * (1.0 + SLIP)
    payoff = min(max(S_T - k_long, 0.0), k_short - k_long) * 100.0
    return payoff - debit - COMM, debit, k_long, k_short


def clustered(y, g):
    """Mean and cluster-robust SE, clustering on g (calendar date).

    These triggers fire in clusters -- 99 names on one June day -- and an
    unclustered SE treats that as 99 independent trades. The below-200-day
    result FLIPPED SIGN under clustering. This is the whole ballgame.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return np.nan, np.nan, np.nan, 0
    m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(g)).sum().values
    se = math.sqrt((s ** 2).sum()) / n
    return m, se, (m / se if se > 0 else np.nan), len(np.unique(g))


def cl_paired(d, g):
    """Cluster-robust SE for a paired difference, clustering on g."""
    return clustered(d, g)


def fmt(m, se, t, k, n):
    return (f"n {n:>5}  dates {k:>4}  mean {m:>+9.2f}  "
            f"SE {se:>7.2f}  t {t:>+6.2f}")


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

    vol_rank = rv60.rank(axis=1, pct=True)

    elig = (  (dvol.shift(1) >= MIN_DOLLAR_VOL)
            & (close         >= MIN_PRICE)
            & (close         >  sma200)
            & (rv60          >= MIN_RVOL)
            & (vol_rank      <= MAX_VOL_RANK) )

    depth = close / sma50 - 1.0
    fwd   = per_sym(close, lambda s: s.shift(-HOLD_DAYS) / s - 1.0)
    fwd_px = per_sym(close, lambda s: s.shift(-HOLD_DAYS))
    sigma_H = (rv60 / 100.0) * np.sqrt(HOLD_DAYS / 252.0)
    z_all = fwd / sigma_H

    # ---- first touch, with an explicit re-arm -------------------------
    # NOTE: (close >= sma50) returns False, not NaN, through the 50-bar
    # warmup. Unmasked, every name would look "armed" from bar 0 and the
    # first touch would be recorded on undefined data. Mask it.
    above = (close >= sma50).where(sma50.notna())
    episode = per_sym(above.astype(float),
                      lambda s: (s > 0.5).cumsum().astype(float))

    def first_touch(thresh):
        """First close at/below -thresh since the last close above the 50-day."""
        b = (depth <= -thresh) & elig & sma50.notna()
        out = {}
        for c in close.columns:
            bb = b[c].fillna(False).values
            ep = episode[c].values
            seen, fire = set(), np.zeros(len(bb), bool)
            for i in range(len(bb)):
                if not bb[i]:
                    continue
                e = ep[i]
                if not np.isfinite(e) or e in seen:
                    continue
                seen.add(e)
                fire[i] = True
            out[c] = pd.Series(fire, index=close.index)
        return pd.DataFrame(out, index=close.index)

    print("\nbuilding first-touch signals ...")
    touches = {th: first_touch(th) for th in LADDER}

    # ---- collect trades ----------------------------------------------
    idx, cols = close.index, close.columns
    C, R, Z, FP, EPI, DEP = (close.values, rv60.values, z_all.values,
                             fwd_px.values, episode.values, depth.values)

    def collect(mask):
        rows = []
        mv = mask.values
        for j, c in enumerate(cols):
            for i in np.where(mv[:, j])[0]:
                S, S_T, rvv, zz = C[i, j], FP[i, j], R[i, j], Z[i, j]
                if not np.isfinite([S, S_T, rvv, zz]).all():
                    continue
                pt = price_trade(S, S_T, rvv)
                if pt is None:
                    continue
                pnl, debit, kl, ks = pt
                rows.append({
                    "date": idx[i], "ticker": c, "episode": EPI[i, j],
                    "depth": DEP[i, j], "rv60": rvv, "spot": S,
                    "z": zz, "pnl": pnl, "debit": debit,
                    "ror": pnl / debit, "win": pnl > 0,
                    "k_long": kl, "k_short": ks,
                })
        t = pd.DataFrame(rows)
        if t.empty:
            return t
        t = t[t.date >= FITSTART].reset_index(drop=True)
        t["w"] = np.where(t.date < FITEND, "FIT", "HOLDOUT")
        return t

    trades = {th: collect(touches[th]) for th in LADDER}

    # An empty rung is not fatal unless it is one of the two the decision
    # rests on. The deep end of the ladder is thin by construction (the 8%
    # rung held 208 trades in six years), so drop it loudly and carry on
    # rather than killing a 45-minute run over a descriptive row.
    ladder = []
    for th in LADDER:
        if trades[th].empty:
            print(f"  !! no trades at {th:.0%} - dropping that rung")
            continue
        ladder.append(th)
    for req, lbl in ((SHALLOW, "SHALLOW"), (DEEP, "DEEP")):
        if req not in ladder:
            die(f"no trades at the {lbl} rung ({req:.0%}) - the decision "
                f"rests on it, so there is nothing to report. Check the gate.")

    span_yrs = (idx[-1] - max(idx[0], FITSTART)).days / 365.25

    # ============================================================ report
    L = "=" * 78
    print("\n" + L)
    print("H3  THE DEPTH LADDER, HELD TO EXPIRY  (was measured at a 90%-of-max exit)")
    print(L)
    print("  model-free z first, then the modelled structure in dollars.\n")
    print(f"  {'trigger':<10}{'n':>6}{'sig/yr':>8}{'z':>8}{'t(z)':>7}"
          f"{'$/trade':>10}{'t($)':>7}{'ROR':>7}{'win%':>6}"
          f"{'FIT $':>9}{'OOS $':>9}{'capacity $/yr':>15}")
    ladder_rows = []
    for th in ladder:
        t = trades[th]
        zm, zse, zt, zk = clustered(t.z, t.date)
        pm, pse, pt_, pk = clustered(t.pnl, t.date)
        fitm = t.loc[t.w == "FIT", "pnl"].mean()
        oosm = t.loc[t.w == "HOLDOUT", "pnl"].mean()
        per_yr = len(t) / span_yrs
        cap = per_yr * pm
        print(f"  {th:<10.0%}{len(t):>6}{per_yr:>8.0f}{zm:>+8.3f}{zt:>+7.2f}"
              f"{pm:>+10.0f}{pt_:>+7.2f}{t.ror.mean():>+7.0%}"
              f"{t.win.mean():>6.0%}{fitm:>+9.0f}{oosm:>+9.0f}{cap:>+15,.0f}")
        ladder_rows.append((th, per_yr, pm, pt_, fitm, oosm, cap))

    print("\n  capacity $/yr = signals/yr x $/trade. It is a CAPACITY ceiling,")
    print("  not a forecast: taking every signal at 5% means ~36 concurrent")
    print("  positions, which breaks the 2-name sector cap long before cash.")

    mono_fit = [r[4] for r in ladder_rows]
    mono_oos = [r[5] for r in ladder_rows]
    print(f"\n  monotonic in FIT?     {all(a<=b for a,b in zip(mono_fit, mono_fit[1:]))}"
          f"   {['%+.0f'%v for v in mono_fit]}")
    print(f"  monotonic in HOLDOUT? {all(a<=b for a,b in zip(mono_oos, mono_oos[1:]))}"
          f"   {['%+.0f'%v for v in mono_oos]}")

    # ---- H1: the paired test -----------------------------------------
    print("\n" + L)
    print(f"H1  PAIRED: on declines reaching BOTH -{SHALLOW:.0%} and -{DEEP:.0%},")
    print(f"    does waiting for -{DEEP:.0%} beat entering at -{SHALLOW:.0%}?")
    print(L)
    print("    Same episode, same name, so this isolates the ENTRY-PRICE cost of")
    print("    moving early from the population difference. Far more powerful")
    print("    than comparing the two trigger populations against each other.\n")

    sh = trades[SHALLOW].set_index(["ticker", "episode"])
    dp = trades[DEEP].set_index(["ticker", "episode"])
    # first_touch fires at most once per episode per threshold, so these keys
    # must be unique. If they are not, the .loc below silently expands into a
    # cross product and every number after it is quietly wrong -- the exact
    # failure mode that produced three confident wrong files in this project.
    for nm, fr in (("shallow", sh), ("deep", dp)):
        if fr.index.duplicated().any():
            die(f"{nm} leg has duplicate (ticker, episode) keys - "
                f"first_touch is firing more than once per episode")
    common = sh.index.intersection(dp.index)
    if len(common) < 20:
        print(f"    only {len(common)} paired episodes - too thin to read")
        paired = pd.DataFrame()
    else:
        paired = pd.DataFrame({
            "date_sh": sh.loc[common, "date"].values,
            "pnl_sh":  sh.loc[common, "pnl"].values,
            "pnl_dp":  dp.loc[common, "pnl"].values,
            "z_sh":    sh.loc[common, "z"].values,
            "z_dp":    dp.loc[common, "z"].values,
            "dep_sh":  sh.loc[common, "depth"].values,
            "dep_dp":  dp.loc[common, "depth"].values,
        })
        paired["date_dp"] = dp.loc[common, "date"].values
        paired["d_pnl"] = paired.pnl_dp - paired.pnl_sh
        paired["d_z"]   = paired.z_dp   - paired.z_sh
        paired["w"] = np.where(pd.to_datetime(paired.date_sh) < FITEND,
                               "FIT", "HOLDOUT")

        # A pair can fire on the SAME DAY: the shallow leg only triggers on the
        # first day the name is both past -5% AND eligible, and if eligibility
        # arrives late the name may already be past -7%. Those pairs are the
        # identical trade, contribute a structural zero, and would dilute the
        # paired estimate toward "no difference". Drop them and say how many.
        same = (pd.to_datetime(paired.date_sh) == pd.to_datetime(paired.date_dp))
        n_same = int(same.sum())
        # Keep BOTH populations. They answer different questions, and reporting
        # only the conditional one overstates the effect by ~28%:
        #   paired_all = every episode reaching both thresholds. Same-day pairs
        #                are the identical trade, so their difference is a true
        #                zero and belongs in the average. BROADER estimate.
        #   paired     = excludes same-day; conditional on waiting actually
        #                meaning waiting.
        # External audit 2026-11-12.
        paired_all = paired.reset_index(drop=True)
        paired = paired[~same].reset_index(drop=True)
        print(f"    {len(paired)} usable paired episodes "
              f"(of {len(sh)} shallow, {len(dp)} deep; "
              f"{n_same} dropped as same-day)")
        if n_same:
            print(f"    same-day pairs are the identical trade -- they would")
            print(f"    bias the difference toward zero, so they are excluded.")
        if len(paired) < 20:
            print(f"    only {len(paired)} usable pairs left - too thin to read")
            paired = pd.DataFrame()
    if not paired.empty:
        print(f"    mean entry depth: shallow {paired.dep_sh.mean():+.2%}  "
              f"deep {paired.dep_dp.mean():+.2%}")
        print(f"    mean wait: "
              f"{(pd.to_datetime(paired.date_dp) - pd.to_datetime(paired.date_sh)).dt.days.mean():.1f}"
              f" calendar days\n")
        for w in ("FIT", "HOLDOUT", "ALL"):
            p = paired if w == "ALL" else paired[paired.w == w]
            if len(p) < 20:
                print(f"    {w:<9} n {len(p):>4} - too thin to read")
                continue
            m, se, tt, k = cl_paired(p.d_pnl, p.date_sh)
            mz, sez, tz, _ = cl_paired(p.d_z, p.date_sh)
            print(f"    {w:<9} deep minus shallow   $ {m:>+8.0f} (t {tt:>+5.2f})"
                  f"   z {mz:>+6.3f} (t {tz:>+5.2f})   n {len(p):>4} dates {k:>4}")
        fs = paired.loc[paired.w == "FIT", "d_pnl"].mean()
        hs = paired.loc[paired.w == "HOLDOUT", "d_pnl"].mean()
        print(f"\n    sign agreement: FIT {fs:+.0f} / HOLDOUT {hs:+.0f}  -> "
              f"{'SAME SIGN' if fs*hs > 0 else '*** DISAGREE - treat as noise ***'}")

        print("\n    THE TWO ESTIMATES, because they answer different questions:")
        for lbl, fr, note in (
            ("all episodes reaching both", paired_all,
             "same-day pairs in as true zeros -- BROADER estimate"),
            ("excluding same-day pairs  ", paired,
             "conditional on waiting actually meaning waiting")):
            if len(fr) < 3:
                continue
            m, se, tt, k = cl_paired(fr.d_pnl, fr.date_sh)
            mz, _, tz, _ = cl_paired(fr.d_z, fr.date_sh)
            print(f"      {lbl}  n {len(fr):>4}  $ {m:>+8.0f} (t {tt:>+5.2f})"
                  f"   z {mz:>+6.3f} (t {tz:>+5.2f})")
            print(f"        {note}")
        if len(paired_all) and len(paired) and paired.d_pnl.mean():
            ma = paired_all.d_pnl.mean(); me = paired.d_pnl.mean()
            print(f"      -> quoting only the conditional figure overstates the "
                  f"effect by {100*(1 - ma/me):.0f}%")
        print("\n    NOTE this measures the DECISION (enter now vs wait), not the")
        print("    entry-price cost in isolation: the two legs have different exit")
        print("    dates and so different market windows. Calling it an isolation")
        print("    of entry price was wrong. External audit 2026-11-12.")

    # ---- H2: the declines that stop short ----------------------------
    print("\n" + L)
    print(f"H2  Do declines that STOP SHORT of -{DEEP:.0%} pay on their own?")
    print(L)
    print("    These are the trades a shallower trigger actually ADDS. The old")
    print("    study never isolated them -- it folded them in with the earlier")
    print("    entries on declines that went on to reach -7%.\n")
    shr = trades[SHALLOW].copy()
    # Deep-reach is taken from the PRICE PATH, not from the set of deep trades
    # that survived pricing. Deriving it from `dp` let a pricing failure or a
    # missing 21-session forward return relabel an episode as "stopped short",
    # i.e. let pricing failures redefine the H2 population. The dominant leak
    # was the sample edge: 85 of 902 shallow trades (9.4%) sit in the final 42
    # dates, where a later deep entry has no forward data. Eligibility is NOT
    # required here -- the question is whether the decline reached -7%, which is
    # a fact about the price, not about tradeability. External audit 2026-11-12.
    # NOTE the mask must mirror first_touch()'s SIGNAL condition --
    # (depth <= -DEEP) & elig & sma50.notna() -- and drop only the PRICING step.
    # A first attempt used raw `depth` alone, which counted episodes that dipped
    # past -7% while the name was ineligible (below its 200-day, vol_rank too
    # high), so the deep trigger could never have fired on them. On a synthetic
    # panel that inflated deep-reach from 67 to 1,095 episodes, which is how the
    # error was caught. Eligibility is part of the signal; pricing is not.
    deep_mask = (depth <= -DEEP) & elig & sma50.notna()
    deep_reach = set()
    for c in close.columns:
        dm = deep_mask[c].fillna(False).values
        ep = episode[c].values
        for i in np.where(dm)[0]:
            e = ep[i]
            if np.isfinite(e):
                deep_reach.add((c, e))
    shr["reached_deep"] = [ (a, b) in deep_reach
                            for a, b in zip(shr.ticker, shr.episode) ]
    n_priced_key = len(set(zip(dp.index.get_level_values(0),
                               dp.index.get_level_values(1))))
    print(f"    deep-reach from the price path: {len(deep_reach)} episodes; "
          f"from priced trades only: {n_priced_key}. The gap is what the old "
          f"classification mislabelled.")
    for lbl, sub in (("stopped short", shr[~shr.reached_deep]),
                     ("went on to -7%", shr[shr.reached_deep])):
        print(f"    {lbl}")
        for w in ("FIT", "HOLDOUT", "ALL"):
            d = sub if w == "ALL" else sub[sub.w == w]
            if len(d) < 10:
                print(f"      {w:<9} n {len(d):>5}  too thin")
                continue
            m, se, tt, k = clustered(d.pnl, d.date)
            mz, _, tz, _ = clustered(d.z, d.date)
            print(f"      {w:<9} " + fmt(m, se, tt, k, len(d))
                  + f"   z {mz:+.3f} (t {tz:+.2f})  win {d.win.mean():.0%}")
        fs = sub.loc[sub.w == "FIT", "pnl"].mean()
        hs = sub.loc[sub.w == "HOLDOUT", "pnl"].mean()
        print(f"      sign agreement: FIT {fs:+.0f} / HOLDOUT {hs:+.0f}  -> "
              f"{'SAME SIGN' if fs*hs > 0 else '*** DISAGREE ***'}\n")

    # ---- the decision ------------------------------------------------
    print(L)
    print("DECISION, against the rule written down before the run")
    print(L)
    short = shr[~shr.reached_deep]
    s_fit = short.loc[short.w == "FIT", "pnl"].mean()
    s_oos = short.loc[short.w == "HOLDOUT", "pnl"].mean()
    s_ok = np.isfinite(s_fit) and np.isfinite(s_oos) and s_fit > 0 and s_oos > 0
    print(f"  H2 added trades positive in BOTH windows?  {s_ok}"
          f"   (FIT {s_fit:+.0f}, OOS {s_oos:+.0f})")
    if paired.empty:
        print("  H1 paired test was too thin to read, so the net below cannot")
        print("     be computed. Do not move the trigger on H2 alone.")
    elif short.empty:
        print("  no added trades at all - a shallower trigger would change")
        print("     nothing in this universe. Leave the trigger where it is.")
    else:
        pm, _, pt_, _ = cl_paired(paired.d_pnl, paired.date_sh)
        s_all = short.pnl.mean()
        n_short = len(short) / span_yrs
        n_pair = len(paired) / span_yrs
        print(f"  H1 paired cost of entering early:          ${pm:+.0f}/trade "
              f"(t {pt_:+.2f})")
        # max(pm, 0): if waiting turns out to HURT, moving early is not a cost
        # and should not be credited as a benefit either. Count it as zero.
        net = n_short * s_all - n_pair * max(pm, 0.0)
        print(f"  net $/yr of moving 7% -> 5%:               {net:+,.0f}")
        print(f"     = {n_short:.0f} added trades/yr x ${s_all:+.0f} (all-window)"
              f"  minus  {n_pair:.0f} earlier entries/yr x ${max(pm,0):.0f}")
        print(f"  NOTE this nets DOLLARS PER YEAR at one contract each. It is a")
        print(f"       capacity figure: realising it needs every signal taken.")
    print("\n  Adopt 5% ONLY if H2 is positive in both windows AND the net is")
    print("  clearly positive AND no sign flip appeared above. A holdout-only")
    print("  result is noise here -- that has been wrong 5 times in this project.")

    out = pd.concat([trades[th].assign(trigger=th) for th in ladder],
                    ignore_index=True)
    out.to_csv(HERE / "depth_retest_trades.csv", index=False)
    print(f"\nwrote depth_retest_trades.csv ({len(out)} rows)")
    if not paired.empty:
        paired.to_csv(HERE / "depth_retest_paired.csv", index=False)
        print(f"wrote depth_retest_paired.csv ({len(paired)} rows)")


if __name__ == "__main__":
    main()
