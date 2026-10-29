#!/usr/bin/env python3
"""
Does the call rule work on ETFs?

Why. The measured bottleneck is that a single stock which falls 7% below its
50-day is, almost by construction, a volatile stock: median rv60 of the deep
names is 60.2% against 31.7% universe-wide, so the depth filter manufactures
exactly the volatility the vol filter rejects. On 2026-10-28, 146 names were 7%+
below their 50-day and ONE was tradeable.

A diversified basket might break that tension. Constituent moves partly cancel,
so an index can fall 7% below its 50-day while its own realized vol stays in the
calm third. ETFs also have NO EARNINGS DATES, which is the most common
disqualifier on every candidate list this system has produced.

ETFs are currently absent from the universe — only SPY, and only because scan.py
bolts it on by hand. The cause is that build_universe.py reads the Nasdaq STOCK
screener, which does not return funds. That is an accident of the data source,
not a decision, which is why this is worth testing rather than assuming.

THE REASON TO EXPECT A NULL, stated before the run so the result cannot be
rationalised afterwards: the below-200-day study found that signals firing in
clusters carry the LEAST edge — that is beta, not alpha. An ETF signal is beta
by construction. If this strategy's edge comes from idiosyncratic overreaction
in individual names, an index of those names will not have it. A flat result
here is the expected outcome, not a failed experiment.

METHOD — same as research_trend.py and price_floor_test.py so numbers compare.

  Model-free outcome: forward 21-session move in units of the stock's own
  horizon-matched sigma. SEs clustered by calendar date. FIT 2021-01-01..
  2024-12-31, HOLDOUT 2025-01-01 on.

  CRITICAL DESIGN POINT: vol_rank is CROSS-SECTIONAL, so it only means anything
  relative to a population. Ranking ~40 ETFs against each other would make
  "calmest third of ETFs" — a different and much weaker filter, since ETFs are
  calm as a class. So the stocks are downloaded TOO and the rank is computed
  over the COMBINED universe. "Calm third" then means the same thing it means
  in production, and the ETF and stock arms are directly comparable.

  BUDGET: one primary test (equity ETFs vs stocks, holdout), plus three
  secondary cuts reported but not counted as findings — bond/commodity ETFs,
  the no-price-floor variant, and the persistence tercile. A finding needs the
  same sign in both windows and |t| > 2.5 on the holdout contrast.
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

YEARS      = 6
HOLD_DAYS  = 21
FITSTART   = pd.Timestamp("2021-01-01")
FITEND     = pd.Timestamp("2025-01-01")

MIN_DOLLAR_VOL = 100e6
MIN_PRICE      = 50.0
MAX_VOL_RANK   = 0.33
MIN_RVOL       = 10.0
CALL_TRIGGER   = -0.07

CHUNK, PAUSE, PASSES = 15, 1.5, 4
MIN_HIST = 320

# Hand-listed because no screener we can reach returns funds. Chosen on
# liquidity and option depth, not on anything to do with the outcome.
EQUITY_ETFS = [
    # broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "RSP", "MDY", "VOO",
    # sector SPDRs
    "XLF", "XLE", "XLK", "XLV", "XLP", "XLU", "XLI", "XLB", "XLY", "XLRE", "XLC",
    # industry
    "SMH", "SOXX", "KRE", "KBE", "XBI", "IBB", "ITB", "XHB", "XRT", "XME",
    "OIH", "XOP", "IYR", "JETS", "TAN",
    # international equity
    "EEM", "EFA", "FXI", "EWJ", "EWZ", "INDA", "VGK", "EWY", "EWT",
]
# Reported separately: these are not equity baskets and the reversion
# mechanism, if any, is a different one. Never pooled with the primary test.
OTHER_ETFS = ["GLD", "SLV", "TLT", "IEF", "HYG", "LQD", "USO", "UNG", "GDX", "GDXJ"]

ALL_ETFS = EQUITY_ETFS + OTHER_ETFS


def die(msg):
    print("\n!! " + msg)
    sys.exit(1)


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


def per_sym(df, fn):
    """Apply fn to each column's OWN non-missing series, then reindex back.

    Never roll across the panel: a union index lets one column's NaN poison
    another's window. Documented in the spec.
    """
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = (fn(s).reindex(df.index) if len(s) > 2
                  else pd.Series(index=df.index, dtype=float))
    return pd.DataFrame(out, index=df.index)


def run_length_true(s):
    b = s.fillna(False).astype(int)
    return b.groupby((b == 0).cumsum()).cumsum()


def clustered(y, g):
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return np.nan, np.nan, np.nan, 0
    m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(g)).sum().values
    se = np.sqrt((s ** 2).sum()) / n
    return m, se, (m / se if se > 0 else np.nan), len(np.unique(g))


def cl_diff(a, b):
    """Clustered SE on (mean a - mean b), pooling both groups' date clusters."""
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return np.nan, np.nan, np.nan
    ma, mb = a.z.mean(), b.z.mean()
    inf = pd.concat([(a.z - ma) / na, -(b.z - mb) / nb])
    s = inf.groupby(pd.concat([a.date, b.date])).sum().values
    se = np.sqrt((s ** 2).sum())
    return ma - mb, se, ((ma - mb) / se if se > 0 else np.nan)


def main():
    path = HERE / "universe.txt"
    if not path.exists():
        die("universe.txt missing")
    stocks = sorted({t.strip() for t in path.read_text().split() if t.strip()})
    stocks = [s for s in stocks if s not in set(ALL_ETFS)]
    names = stocks + ALL_ETFS
    print(f"universe: {len(stocks)} stocks + {len(ALL_ETFS)} ETFs "
          f"({len(EQUITY_ETFS)} equity, {len(OTHER_ETFS)} other), pulling {YEARS}y")
    print("  stocks are pulled TOO so vol_rank is computed over the combined")
    print("  population — ranking ETFs only against each other would change what")
    print("  'calmest third' means and make the arms incomparable.\n")

    close, vol = download(names)
    got_etf = [t for t in ALL_ETFS if t in close.columns]
    missing = [t for t in ALL_ETFS if t not in close.columns]
    print(f"\ngot {close.shape[1]} series, {close.shape[0]} bars, "
          f"{close.index[0].date()} .. {close.index[-1].date()}")
    print(f"ETFs in: {len(got_etf)}/{len(ALL_ETFS)}"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))

    logret = per_sym(close, lambda s: np.log(s).diff())
    sma50  = per_sym(close, lambda s: s.rolling(50,  min_periods=50 ).mean())
    sma200 = per_sym(close, lambda s: s.rolling(200, min_periods=200).mean())
    rv60   = per_sym(logret, lambda s: s.rolling(60, min_periods=60).std()
                                        * np.sqrt(252) * 100)
    dvol   = per_sym(close * vol,
                     lambda s: s.rolling(60, min_periods=40).median())

    vol_rank = rv60.rank(axis=1, pct=True)          # across stocks AND ETFs

    below = (close < sma50).where(sma50.notna())
    sess_below = per_sym(below.astype(float),
                         lambda s: run_length_true(s > 0.5).astype(float))

    gates_noprice = ( (dvol.shift(1) >= MIN_DOLLAR_VOL)
                    & (close         >  sma200)
                    & (rv60          >= MIN_RVOL)
                    & (vol_rank      <= MAX_VOL_RANK) )
    gates = gates_noprice & (close >= MIN_PRICE)

    depth   = close / sma50 - 1.0
    fwd     = per_sym(close, lambda s: s.shift(-HOLD_DAYS) / s - 1.0)
    z       = fwd / ((rv60 / 100.0) * np.sqrt(HOLD_DAYS / 252.0))

    eqset, othset = set(EQUITY_ETFS), set(OTHER_ETFS)

    def harvest(mask):
        rows, m = [], mask.values
        for j, c in enumerate(close.columns):
            kind = ("equity ETF" if c in eqset else
                    "other ETF"  if c in othset else "stock")
            for i in np.where(m[:, j])[0]:
                zz, px = z.values[i, j], close.values[i, j]
                if not (np.isfinite(zz) and np.isfinite(px)):
                    continue
                rows.append({"date": close.index[i], "ticker": c, "kind": kind,
                             "z": zz, "price": px, "depth": depth.values[i, j],
                             "rv60": rv60.values[i, j],
                             "vol_rank": vol_rank.values[i, j],
                             "dvol": dvol.values[i, j] / 1e6,
                             "sess_below": sess_below.values[i, j],
                             "fwd": fwd.values[i, j]})
        t = pd.DataFrame(rows)
        if t.empty:
            return t
        t = t[t.date >= FITSTART].reset_index(drop=True)
        t["win"] = (t.date < FITEND).map({True: "FIT", False: "HOLDOUT"})
        return t

    trig  = gates & (depth <= CALL_TRIGGER)
    t     = harvest(trig)
    trig_np = gates_noprice & (depth <= CALL_TRIGGER)
    t_np  = harvest(trig_np)
    if t.empty:
        die("no trades at all — check the gates")

    for k in ("stock", "equity ETF", "other ETF"):
        d = t[t.kind == k]
        print(f"  {k:<12} {len(d):>5} signals, {d.ticker.nunique():>3} names, "
              f"{d.date.nunique():>4} dates")

    def show(lbl, d):
        if len(d) < 10:
            print(f"    {lbl:<22} n {len(d):>5}   TOO THIN TO TEST")
            return None
        m, se, tt, k = clustered(d.z, d.date)
        print(f"    {lbl:<22} n {len(d):>5}  dates {k:>4}  z {m:+.3f}  "
              f"SE {se:.3f}  t {tt:+.2f}")
        return m

    print("\n" + "=" * 84)
    print("THE PRIMARY TEST — equity ETFs vs stocks, same gates, same period")
    print("=" * 84)
    print("  forward 21-day move in units of own sigma; 0 = random walk\n")
    for w in ("FIT", "HOLDOUT", "ALL"):
        d = t if w == "ALL" else t[t.win == w]
        print(f"  -- {w}")
        show("stocks", d[d.kind == "stock"])
        show("equity ETFs", d[d.kind == "equity ETF"])
        show("other ETFs", d[d.kind == "other ETF"])
        a, b = d[d.kind == "equity ETF"], d[d.kind == "stock"]
        if len(a) >= 10 and len(b) >= 10:
            dd, ds, dt = cl_diff(a, b)
            print(f"    {'ETF minus stock':<22} diff {dd:+.3f}  t {dt:+.2f}")
        print()

    print("=" * 84)
    print("THE CLUSTERING CONCERN, measured rather than asserted")
    print("=" * 84)
    print("  If ETF signals are pure sector beta they should fire together.\n")
    for k in ("stock", "equity ETF"):
        d = t[t.kind == k]
        if d.empty:
            continue
        per_day = d.groupby("date").size()
        print(f"  {k:<12} {len(d):>5} signals on {d.date.nunique():>4} dates "
              f"= {per_day.mean():.1f} per firing day, max {per_day.max()}")
    print()

    print("=" * 84)
    print("SECONDARY CUTS — reported, not counted as findings")
    print("=" * 84)
    print("\n  (a) without the $50 price floor (many ETFs trade below it):")
    for w in ("FIT", "HOLDOUT"):
        d = t_np[(t_np.win == w) & (t_np.kind == "equity ETF")]
        show(f"{w} equity ETFs", d)
    extra = sorted(set(t_np[t_np.kind == "equity ETF"].ticker) -
                   set(t[t.kind == "equity ETF"].ticker))
    print(f"      ETFs the price floor excludes entirely: "
          f"{', '.join(extra) if extra else 'none'}")

    print("\n  (b) persistence tercile (>20 sessions below the 50-day is the")
    print("      favourable bucket, z +0.440 on stocks):")
    for k in ("stock", "equity ETF"):
        d = t[(t.kind == k) & (t.sess_below > 20)]
        show(f"{k} >20 sessions", d)

    print("\n  (c) non-equity ETFs (bonds, metals, energy) — different mechanism:")
    for w in ("FIT", "HOLDOUT"):
        show(f"{w} other ETFs", t[(t.win == w) & (t.kind == "other ETF")])

    print("\n" + "=" * 84)
    print("WHICH ETFs ACTUALLY FIRE, and how often")
    print("=" * 84)
    e = t[t.kind != "stock"]
    if e.empty:
        print("  none")
    else:
        g = (e.groupby(["ticker", "kind"])
               .agg(n=("z", "size"), z=("z", "mean"), rv=("rv60", "median"),
                    vr=("vol_rank", "median"), dv=("dvol", "median"),
                    px=("price", "median"))
               .reset_index().sort_values("n", ascending=False))
        print(f"  {'ticker':<8}{'kind':<12}{'signals':>8}{'mean z':>9}"
              f"{'rv60':>8}{'volrank':>9}{'$vol':>9}{'med px':>9}")
        for _, r in g.iterrows():
            print(f"  {r.ticker:<8}{r['kind']:<12}{int(r.n):>8}{r.z:>+9.3f}"
                  f"{r.rv:>7.1f}%{r.vr:>9.3f}{r.dv:>8.0f}M{r.px:>9.2f}")
        yrs = (t.date.max() - t.date.min()).days / 365.25
        eq = e[e.kind == "equity ETF"]
        print(f"\n  equity-ETF signals per year: {len(eq)/yrs:.1f}  "
              f"(stocks currently produce {len(t[t.kind=='stock'])/yrs:.0f}/yr)")

    t.to_csv(HERE / "etf_trades.csv", index=False)
    print(f"\nwrote etf_trades.csv ({len(t)} rows)")


if __name__ == "__main__":
    main()
