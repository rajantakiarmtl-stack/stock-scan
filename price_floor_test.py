#!/usr/bin/env python3
"""
Does the call-side edge survive BELOW the $50 price floor?

Why this exists. Rajan noticed that 42 names trade more than $100M a day and are
excluded from every scan purely because they cost less than $50 — VZ at $43, BSX
at $42, MNST at $46, TFC at $48. Those are not illiquid companies.

The floor is the weakest-justified rule in the system. `MIN_PRICE = 50` came from
the original spec, not from a measurement. What HAS been measured is the price
PROFILE above the floor: $50-80 names win less often (47%) but return more,
$250+ win more (65%) and return less. Nothing below $50 was ever tested, because
the floor removed those names from every backtest that could have tested it.

So this is not "relax a filter and see if more trades appear." It is: the filter
was never validated, and the band it removes is large and liquid. Measure it.

METHOD — identical to research_trend.py so the numbers are directly comparable.

  Outcome is MODEL-FREE: forward 21-session move in units of the stock's own
  horizon-matched sigma. z = fwd / (rv60 * sqrt(21/252)). Zero-drift random walk
  gives z = 0. No option pricing, so no IV assumption can flatter a cheap stock.

  THIS MATTERS MORE THAN USUAL HERE. A dollar-P&L measure would be biased by
  price: a $40 stock and a $400 stock moving the same PERCENT produce wildly
  different dollar spreads. Sigma units remove that entirely, which is the only
  way to compare price bands honestly.

  SEs CLUSTERED BY CALENDAR DATE. Cheap names cluster by sector (telecom, banks,
  energy), so an unclustered SE would be badly overconfident here.

  FIT 2021-01-01..2024-12-31, HOLDOUT 2025-01-01 on.

  BUDGET, STATED UP FRONT: 6 pre-registered price bands x 2 windows = 12 cells,
  plus one planned contrast (under-50 vs over-50). At 12 looks a nominal t = 2.0
  is expected by chance. A band is a finding only if it holds in BOTH windows
  with the same sign and |t| > 2.5 on the contrast in the holdout.

  THE CONFOUND TO WATCH: cheap stocks may simply be more volatile and less
  liquid. If the sub-$50 band underperforms, that may be volatility or liquidity
  wearing a price costume. So every band reports its median rv60, median dollar
  volume and vol_rank, and the whole test is ALSO re-run with the vol and
  liquidity gates held fixed. If the price effect survives that, it is a price
  effect. If it does not, the floor is redundant with filters we already have.
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
MIN_PRICE      = 15.0          # <-- the floor under test. Production value is 50.
MAX_VOL_RANK   = 0.33
MIN_RVOL       = 10.0
CALL_TRIGGER   = -0.07

CHUNK, PAUSE, PASSES = 15, 1.5, 4
MIN_HIST = 320

BANDS = [(15, 30), (30, 50), (50, 80), (80, 150), (150, 250), (250, 1e9)]


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
    another's window. That bug silently dropped 4,005 of 5,658 symbols in
    build_universe.py and is documented in the spec.
    """
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = (fn(s).reindex(df.index) if len(s) > 2
                  else pd.Series(index=df.index, dtype=float))
    return pd.DataFrame(out, index=df.index)


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

    base = (  (dvol.shift(1) >= MIN_DOLLAR_VOL)
            & (close         >= MIN_PRICE)
            & (close         >  sma200)
            & (rv60          >= MIN_RVOL) )
    elig    = base & (vol_rank <= MAX_VOL_RANK)
    depth   = close / sma50 - 1.0
    trigger = elig & (depth <= CALL_TRIGGER)

    fwd     = per_sym(close, lambda s: s.shift(-HOLD_DAYS) / s - 1.0)
    z       = fwd / ((rv60 / 100.0) * np.sqrt(HOLD_DAYS / 252.0))

    rows, tm = [], trigger.values
    idx, cols = close.index, close.columns
    for j, c in enumerate(cols):
        for i in np.where(tm[:, j])[0]:
            zz = z.values[i, j]
            px = close.values[i, j]
            if not (np.isfinite(zz) and np.isfinite(px)):
                continue
            rows.append({"date": idx[i], "ticker": c, "z": zz, "price": px,
                         "depth": depth.values[i, j], "rv60": rv60.values[i, j],
                         "vol_rank": vol_rank.values[i, j],
                         "dvol": dvol.values[i, j] / 1e6,
                         "fwd": fwd.values[i, j]})
    t = pd.DataFrame(rows)
    t = t[t.date >= FITSTART].reset_index(drop=True)
    if t.empty:
        die("no trades after the gate")
    t["win"] = (t.date < FITEND).map({True: "FIT", False: "HOLDOUT"})

    def label(lo, hi):
        return f"${lo:,.0f}+" if hi > 1e8 else f"${lo:,.0f}-{hi:,.0f}"

    def band_of(p):
        for lo, hi in BANDS:
            if lo <= p < hi:
                return label(lo, hi)
        return "?"

    t["band"] = t.price.map(band_of)
    order = [label(lo, hi) for lo, hi in BANDS]
    unbanded = int((t.band == "?").sum())
    if unbanded:
        print(f"!! {unbanded} trades fell outside every band — check BANDS")

    print(f"\ntrades {len(t)}  FIT {int((t.win=='FIT').sum())}  "
          f"HOLDOUT {int((t.win=='HOLDOUT').sum())}  dates {t.date.nunique()}  "
          f"tickers {t.ticker.nunique()}")
    sub = t[t.price < 50]
    print(f"NEW trades the $50 floor currently excludes: {len(sub)} "
          f"({100*len(sub)/len(t):.1f}% of all), {sub.ticker.nunique()} tickers")

    print("\n" + "=" * 86)
    print("THE QUESTION — forward 21-day move in sigma units, by price band")
    print("=" * 86)
    print("  0 = random walk. The $50 floor removes the first two rows entirely.\n")
    print(f"  {'band':<12}{'n':>6}{'FIT z':>9}{'t':>7}{'n':>6}{'HOLD z':>9}{'t':>7}"
          f"{'medvol':>9}{'med $vol':>10}{'medrank':>9}")
    for b in order:
        d = t[t.band == b]
        if len(d) < 10:
            print(f"  {b:<12}{len(d):>6}   too thin")
            continue
        f_, h_ = d[d.win == "FIT"], d[d.win == "HOLDOUT"]
        mf, sf, tf, _ = clustered(f_.z, f_.date) if len(f_) >= 10 else (np.nan,)*4
        mh, sh, th, _ = clustered(h_.z, h_.date) if len(h_) >= 10 else (np.nan,)*4
        print(f"  {b:<12}{len(f_):>6}{mf:>+9.3f}{tf:>+7.2f}{len(h_):>6}{mh:>+9.3f}{th:>+7.2f}"
              f"{d.rv60.median():>8.1f}%{d.dvol.median():>9.0f}M{d.vol_rank.median():>9.3f}")

    print("\n" + "=" * 86)
    print("THE PLANNED CONTRAST — under $50 vs $50 and above")
    print("=" * 86)
    for w in ("FIT", "HOLDOUT", "ALL"):
        d = t if w == "ALL" else t[t.win == w]
        lo, hi = d[d.price < 50], d[d.price >= 50]
        if len(lo) < 10 or len(hi) < 10:
            print(f"  {w:<9} too thin (under-50 n={len(lo)})")
            continue
        ml, sl, tl, kl = clustered(lo.z, lo.date)
        mh, sh, th, kh = clustered(hi.z, hi.date)
        dd, ds, dt = cl_diff(lo, hi)
        print(f"  {w:<9} under-50 n {len(lo):>4} z {ml:+.3f} (t {tl:+.2f})   "
              f"50+ n {len(hi):>4} z {mh:+.3f} (t {th:+.2f})   "
              f"diff {dd:+.3f} t {dt:+.2f}")

    print("\n" + "=" * 86)
    print("CONFOUND CHECK — is 'cheap' just 'volatile' or 'less liquid'?")
    print("=" * 86)
    lo, hi = t[t.price < 50], t[t.price >= 50]
    for nm, d in (("under $50", lo), ("$50 and up", hi)):
        print(f"  {nm:<12} median rv60 {d.rv60.median():5.1f}%   "
              f"median vol_rank {d.vol_rank.median():.3f}   "
              f"median $vol {d.dvol.median():>5.0f}M   median depth {100*d.depth.median():+.2f}%")
    print()
    print("  Same contrast, but matched on volatility (vol_rank <= 0.20, the calmest fifth):")
    for w in ("FIT", "HOLDOUT"):
        d = t[(t.win == w) & (t.vol_rank <= 0.20)]
        a, b = d[d.price < 50], d[d.price >= 50]
        if len(a) < 10 or len(b) < 10:
            print(f"    {w:<9} too thin (under-50 n={len(a)})")
            continue
        dd, ds, dt = cl_diff(a, b)
        print(f"    {w:<9} under-50 n {len(a):>4} z {a.z.mean():+.3f}   "
              f"50+ n {len(b):>4} z {b.z.mean():+.3f}   diff {dd:+.3f} t {dt:+.2f}")
    print()
    print("  Same contrast, matched on liquidity (>= $500M/day):")
    for w in ("FIT", "HOLDOUT"):
        d = t[(t.win == w) & (t.dvol >= 500)]
        a, b = d[d.price < 50], d[d.price >= 50]
        if len(a) < 10 or len(b) < 10:
            print(f"    {w:<9} too thin (under-50 n={len(a)})")
            continue
        dd, ds, dt = cl_diff(a, b)
        print(f"    {w:<9} under-50 n {len(a):>4} z {a.z.mean():+.3f}   "
              f"50+ n {len(b):>4} z {b.z.mean():+.3f}   diff {dd:+.3f} t {dt:+.2f}")

    print("\n" + "=" * 86)
    print("WHAT LOWERING THE FLOOR WOULD ADD — the names, and how often")
    print("=" * 86)
    if len(sub):
        g = (sub.groupby("ticker")
                .agg(n=("z", "size"), z=("z", "mean"), px=("price", "median"),
                     rv=("rv60", "median"), dv=("dvol", "median"))
                .sort_values("n", ascending=False).head(20))
        print(f"  {'ticker':<8}{'signals':>8}{'mean z':>9}{'med px':>9}{'rv60':>8}{'$vol':>8}")
        for tk, r in g.iterrows():
            print(f"  {tk:<8}{int(r.n):>8}{r.z:>+9.3f}{r.px:>9.2f}{r.rv:>7.1f}%{r.dv:>7.0f}M")
        print(f"\n  extra signals per year from dropping the floor: "
              f"{len(sub)/((t.date.max()-t.date.min()).days/365.25):.1f}")

    t.to_csv(HERE / "price_floor_trades.csv", index=False)
    print(f"\nwrote price_floor_trades.csv ({len(t)} rows)")


if __name__ == "__main__":
    main()
