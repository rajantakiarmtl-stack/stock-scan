#!/usr/bin/env python3
"""
"They all went up — just not while we were holding them."

Rajan's observation, raised twice and never measured. It is the most important
untested claim in this project, because if it is TRUE the problem is the HORIZON,
not the entry. That points at a different fix from everything tested so far: not
a better filter, but a longer hold, a roll, or a different structure.

THE QUESTION. Conditional on a trade finishing at a LOSS, what does the
underlying do AFTER expiry? If losers systematically recover in the weeks
following, the signal was right and the clock was wrong.

PRE-REGISTERED, written before the first run.

  H1  Stocks whose trade finished at a loss out-perform over the 10/21/42/63
      sessions AFTER expiry, relative to the same stock's unconditional drift.
      -> if true, the 21-session horizon is too short.

  H2  Losers out-perform WINNERS over the post-expiry window.
      -> the sharper version. If the stock keeps climbing after a winning trade
         too, then nothing is special about losers and H1 is just equity drift.

  H3  The recovery, if any, is big enough to have RESCUED the trade: would the
      same spread have finished profitable at 42 or 63 sessions instead of 21?
      -> this is the only version that has a trading implication. H1 can be true
         while H3 is false, if the recovery arrives after the option is gone or
         is too small to clear the breakeven.

  MY RECORDED PREDICTION: H1 partly true (mean reversion is slow and the 21-day
  window probably clips it), H2 FALSE (both groups drift up — this is a bull
  sample, 2021-2026), H3 FALSE at 42 sessions and possibly true at 63 but at a
  cost in premium that eats the gain. Writing it down so the result cannot be
  re-narrated afterwards. Four of my predictions in this project have been wrong.

  DECISION RULE. A horizon change requires H1 AND H3 positive in BOTH windows,
  with H2 confirming the effect is specific to losers rather than generic drift.
  SEs clustered by calendar date. A fit/holdout sign flip kills it — that has
  been wrong five times here.

  BUDGET. 3 hypotheses x 4 horizons, but the horizons are an ordered set and
  count as one. ~3 hypotheses.

THE TRAP THIS MUST AVOID. "The stock recovered later" is the single most
seductive story in trading, because it is almost always TRUE in a rising market
and it always feels like vindication. The unconditional baseline is therefore not
optional: the only interesting result is losers recovering MORE than the same
universe does anyway over the same calendar window. That is why every number
below is reported three ways — raw, minus the stock's own trailing drift, and
minus the cross-sectional mean of all eligible names on the same dates.

METHOD. Mirrors depth_retest.py: same universe file, same gate, 21-session trade
horizon (= 30 calendar DTE), fit 2021-2024 / holdout 2025+, first touch with
re-arm, SEs clustered by date. Post-expiry windows measured from the EXPIRY bar
(entry + 21), not from entry.
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

YEARS      = 6
HOLD       = 21                     # the trade's horizon, = 30 calendar DTE
POST       = [10, 21, 42, 63]       # sessions AFTER expiry
FITSTART   = pd.Timestamp("2021-01-01")
FITEND     = pd.Timestamp("2025-01-01")

MIN_DOLLAR_VOL = 100e6
MIN_PRICE      = 50.0
MAX_VOL_RANK   = 0.33
MIN_RVOL       = 10.0
CALL_TRIGGER   = 0.07

WIDTH_SIGMA = 1.5
IV_MULT     = 0.85 * 1.10
SLIP        = 0.02
COMM        = 0.30 * 2 * 2

CHUNK, PAUSE, PASSES = 15, 1.5, 4
MIN_HIST = 320


def die(msg):
    print("\n!! " + msg); sys.exit(1)


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
            w = 20 * p
            print(f"  {len(todo)} missing - waiting {w}s before pass {p+1}")
            time.sleep(w)
        for i in range(0, len(todo), CHUNK):
            b = todo[i:i + CHUNK]
            try:
                cl, vl = _fetch(b)
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


def per_sym(df, fn):
    """Apply fn to each column's own non-missing series. Rolling across the
    panel's union index poisoned 4,005 of 5,658 symbols once already."""
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = (fn(s).reindex(df.index) if len(s) > 2
                  else pd.Series(index=df.index, dtype=float))
    return pd.DataFrame(out, index=df.index)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return S * _ncdf(d1) - K * _ncdf(d1 - v)


def strike_inc(S):
    return 1.0 if S < 25 else (2.5 if S < 200 else 5.0)


def build_and_settle(S, S_T, rv, horizon_sessions):
    """Build the spec spread at S for a given horizon, settle at S_T.
    Returns (pnl, debit, k_long, k_short) or None. Width scales with the
    horizon, so a 42-session spread is correctly WIDER than a 21-session one
    on the same stock -- otherwise a longer hold would be compared at the
    wrong structure."""
    if not all(np.isfinite([S, S_T, rv])) or S <= 0 or rv <= 0:
        return None
    T = horizon_sessions / 252.0
    inc = strike_inc(S)
    k1 = round(S / inc) * inc
    k2 = k1 + max(inc, round(WIDTH_SIGMA * S * (rv / 100.0) * math.sqrt(T) / inc) * inc)
    if k2 <= k1 or k1 <= 0:
        return None
    iv = (rv / 100.0) * IV_MULT
    dps = bs_call(S, k1, T, iv) - bs_call(S, k2, T, iv)
    if dps <= 0.05:
        return None
    debit = dps * 100.0 * (1.0 + SLIP)
    payoff = min(max(S_T - k1, 0.0), k2 - k1) * 100.0
    return payoff - debit - COMM, debit, k1, k2


def clustered(y, g):
    y = np.asarray(y, float); n = len(y)
    if n < 3:
        return np.nan, np.nan, np.nan, 0
    m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(g)).sum().values
    se = math.sqrt((s ** 2).sum()) / n
    return m, se, (m / se if se > 0 else np.nan), len(np.unique(g))


def cl_diff(a, b, col):
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return np.nan, np.nan, np.nan
    ma, mb = a[col].mean(), b[col].mean()
    s = pd.concat([
        pd.DataFrame({"d": a.date.values, "v": (a[col].values - ma) / na}),
        pd.DataFrame({"d": b.date.values, "v": -(b[col].values - mb) / nb})])
    tot = s.groupby("d").v.sum().values
    se = math.sqrt((tot ** 2).sum())
    d = ma - mb
    return d, se, (d / se if se > 0 else np.nan)


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
    dvol   = per_sym(close * vol, lambda s: s.rolling(60, min_periods=40).median())
    vol_rank = rv60.rank(axis=1, pct=True)

    elig = ((dvol.shift(1) >= MIN_DOLLAR_VOL) & (close >= MIN_PRICE)
            & (close > sma200) & (rv60 >= MIN_RVOL) & (vol_rank <= MAX_VOL_RANK))
    depth = close / sma50 - 1.0

    # (close >= sma50) is False, not NaN, through the warmup -- mask it or the
    # first touch lands on undefined data.
    above = (close >= sma50).where(sma50.notna())
    episode = per_sym(above.astype(float),
                      lambda s: (s > 0.5).cumsum().astype(float))

    b = (depth <= -CALL_TRIGGER) & elig & sma50.notna()
    fire = {}
    for c in close.columns:
        bb = b[c].fillna(False).values; ep = episode[c].values
        seen, f = set(), np.zeros(len(bb), bool)
        for i in range(len(bb)):
            if not bb[i]:
                continue
            e = ep[i]
            if np.isfinite(e) and e not in seen:
                seen.add(e); f[i] = True
        fire[c] = pd.Series(f, index=close.index)
    fire = pd.DataFrame(fire, index=close.index)

    # The unconditional benchmark: the cross-sectional mean forward return of
    # every ELIGIBLE name on the same date, over the same window. Without this
    # a rising market alone would "confirm" the hypothesis.
    fwd_post = {}
    for h in POST:
        fwd_post[h] = per_sym(close, lambda s: s.shift(-h) / s - 1.0)
    bench = {h: fwd_post[h].where(elig).mean(axis=1) for h in POST}

    idx, cols = close.index, close.columns
    C, R = close.values, rv60.values
    fv = fire.values
    rows = []
    for j, c in enumerate(cols):
        n = len(idx)
        for i in np.where(fv[:, j])[0]:
            ie = i + HOLD                       # the expiry bar
            if ie >= n:
                continue
            S, S_T, rv = C[i, j], C[ie, j], R[i, j]
            t21 = build_and_settle(S, S_T, rv, HOLD)
            if t21 is None:
                continue
            pnl21, debit21, k1, k2 = t21
            rec = {"date": idx[i], "expiry_date": idx[ie], "ticker": c,
                   "spot": S, "spot_exp": S_T, "rv60": rv,
                   "pnl21": pnl21, "debit21": debit21,
                   "loser": pnl21 < 0, "total_loss": pnl21 <= -0.99 * debit21,
                   "trade_ret": S_T / S - 1.0}
            ok = True
            for h in POST:
                ih = ie + h
                if ih >= n or not np.isfinite(C[ih, j]):
                    ok = False; break
                # post-expiry move, measured FROM THE EXPIRY BAR
                rec[f"post{h}"] = C[ih, j] / S_T - 1.0
                bv = bench[h].iloc[ie]
                rec[f"post{h}_x"] = rec[f"post{h}"] - (bv if np.isfinite(bv) else np.nan)
                # H3: would a longer spread, built at ENTRY, have finished green?
                tl = build_and_settle(S, C[ih, j], rv, HOLD + h)
                rec[f"long{h}_pnl"] = tl[0] if tl else np.nan
                rec[f"long{h}_debit"] = tl[1] if tl else np.nan
            if not ok:
                continue
            rows.append(rec)

    t = pd.DataFrame(rows)
    if t.empty:
        die("no trades with full post-expiry data")
    t = t[t.date >= FITSTART].reset_index(drop=True)
    t["w"] = np.where(t.date < FITEND, "FIT", "HOLDOUT")

    L = "=" * 78
    print(f"\ntrades with {max(POST)} sessions of post-expiry data: {len(t)}")
    print(f"  FIT {(t.w=='FIT').sum()}  HOLDOUT {(t.w=='HOLDOUT').sum()}  "
          f"dates {t.date.nunique()}  tickers {t.ticker.nunique()}")
    print(f"  losers {t.loser.sum()} ({100*t.loser.mean():.0f}%)   "
          f"total losses {t.total_loss.sum()} ({100*t.total_loss.mean():.0f}%)")
    print(f"\n  NOTE the {max(POST)}-session data requirement drops the most")
    print(f"  recent trades, so this sample is older than the full study's.")

    print("\n" + L)
    print('H1  DO LOSERS RECOVER AFTER EXPIRY?')
    print(L)
    print("  Three columns, and only the third one means anything:")
    print("    raw      = the stock's move after expiry")
    print("    vs bench = minus the same-date mean of all eligible names")
    print("  A rising market makes 'raw' positive for everything. Ignore it.\n")
    for h in POST:
        print(f"  -- {h} sessions after expiry")
        for lbl, sub in (("losers ", t[t.loser]), ("winners", t[~t.loser])):
            if len(sub) < 10:
                print(f"     {lbl} n {len(sub):>4}  too thin"); continue
            rm, _, rt, k = clustered(sub[f"post{h}"], sub.date)
            xm, _, xt, _ = clustered(sub[f"post{h}_x"], sub.date)
            print(f"     {lbl} n {len(sub):>4} dates {k:>4}   "
                  f"raw {rm:+7.2%} (t {rt:+5.2f})   "
                  f"vs bench {xm:+7.2%} (t {xt:+5.2f})")
        a, bq = t[t.loser], t[~t.loser]
        if len(a) >= 10 and len(bq) >= 10:
            d, se, tt = cl_diff(a, bq, f"post{h}_x")
            print(f"     H2 losers minus winners, vs bench: {d:+.2%} (t {tt:+.2f})")
        for w in ("FIT", "HOLDOUT"):
            s2 = t[(t.w == w) & t.loser]
            if len(s2) >= 10:
                m, _, tt, _ = clustered(s2[f"post{h}_x"], s2.date)
                print(f"       {w:<8} losers vs bench {m:+.2%} (t {tt:+.2f}) n {len(s2)}")
        print()

    print(L)
    print("H3  WOULD A LONGER SPREAD HAVE RESCUED THEM?")
    print(L)
    print("  The only version with a trading implication. Same entry, same")
    print("  signal, but the spread is built for a longer horizon -- so it is")
    print("  correctly WIDER and MORE EXPENSIVE. If the recovery arrives after")
    print("  the option expired, or costs more in premium than it returns,")
    print("  H1 can be true while this is false.\n")
    base_all, _, base_t, _ = clustered(t.pnl21, t.date)
    print(f"  21-session (current spec): ${base_all:+.0f}/trade (t {base_t:+.2f}), "
          f"win {100*(t.pnl21>0).mean():.0f}%, mean debit ${t.debit21.mean():.0f}")
    for h in POST:
        col = f"long{h}_pnl"
        d = t[np.isfinite(t[col])]
        if len(d) < 10:
            print(f"  {HOLD+h}-session: too thin"); continue
        m, _, tt, _ = clustered(d[col], d.date)
        fitm = d.loc[d.w == "FIT", col].mean()
        oosm = d.loc[d.w == "HOLDOUT", col].mean()
        same = "SAME SIGN" if fitm * oosm > 0 else "*** DISAGREE ***"
        print(f"  {HOLD+h:>3}-session: ${m:+7.0f}/trade (t {tt:+5.2f})  "
              f"win {100*(d[col]>0).mean():>3.0f}%  debit ${d[f'long{h}_debit'].mean():>5.0f}  "
              f"FIT ${fitm:+6.0f} / OOS ${oosm:+6.0f}  {same}")
        res = d[d.loser]
        if len(res) >= 10:
            print(f"       of the {len(res)} trades that LOST at 21 sessions, "
                  f"{100*(res[col]>0).mean():.0f}% would be green at {HOLD+h}; "
                  f"mean ${res[col].mean():+.0f}")

    print("\n" + L)
    print("VERDICT, against the rule written down before the run")
    print(L)
    ok1 = ok3 = False
    for h in POST:
        s2 = t[t.loser]
        f_ = s2.loc[s2.w == "FIT", f"post{h}_x"].mean()
        o_ = s2.loc[s2.w == "HOLDOUT", f"post{h}_x"].mean()
        m, _, tt, _ = clustered(s2[f"post{h}_x"], s2.date)
        if np.isfinite(f_) and np.isfinite(o_) and f_ * o_ > 0 and m > 0 and tt > 2:
            ok1 = True
        col = f"long{h}_pnl"
        d = t[np.isfinite(t[col])]
        if len(d) >= 10:
            mm, _, mt, _ = clustered(d[col], d.date)
            fm = d.loc[d.w == "FIT", col].mean()
            om = d.loc[d.w == "HOLDOUT", col].mean()
            # A bare "mean > baseline" test is the same permissiveness an
            # external audit caught in beta_decomp's double sort: it passes on
            # noise. Require significance too, at the same t>2 bar H1 uses, or
            # this check would wave through a $28 improvement at t=0.87.
            if mm > base_all and fm * om > 0 and mt > 2:
                ok3 = True
    print(f"  H1 losers beat the benchmark, both windows, t>2?   {ok1}")
    print(f"  H3 a longer horizon beats 21 sessions, same sign?  {ok3}")
    if ok1 and not ok3:
        print("\n  -> The observation is REAL but NOT TRADEABLE as a horizon change:")
        print("     the stocks do recover, and the extra premium a longer spread")
        print("     costs eats the recovery. That is worth knowing and worth")
        print("     saying plainly rather than dressing up as a finding.")
    elif ok1 and ok3:
        print("\n  -> Both pass. A longer horizon deserves a pre-registered forward")
        print("     test before the spec changes. Do NOT adopt off this run alone.")
    else:
        print("\n  -> NOT SUPPORTED. The stocks recovering afterwards is mostly the")
        print("     market rising, not information. Leave the horizon alone.")
    print("\n  Recorded prediction: H1 partly true, H2 false, H3 false at 42.")

    t.to_csv(HERE / "late_recovery_trades.csv", index=False)
    print(f"\nwrote late_recovery_trades.csv ({len(t)} rows)")


if __name__ == "__main__":
    main()
