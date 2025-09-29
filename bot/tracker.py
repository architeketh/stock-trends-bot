import os
import sys
import shutil
import json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from jinja2 import Template

# ---------- Config ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "docs")
STOCKS_FILE = os.path.join(DATA_DIR, "tickers_stocks.txt")
ETFS_FILE   = os.path.join(DATA_DIR, "tickers_etfs.txt")

HISTORY_PERIOD = "1y"                 # enough for YTD
LOOKBACKS = {"day": 1, "month": 21}   # trading days for "month"
RANK_HORIZON = "month"                # "day" | "month" | "ytd"
TOP_N = 10
TITLE = "Top 10 Stocks & ETFs — Price · Day · Month · YTD"
TZ = "America/Chicago"

# Mike's watchlist (kept as provided; missing tickers are skipped gracefully)
MIKE_TICKERS = [
    "VOO","VOOG","VUG","VDIGX","QQQM","AAPL","NVDA","IVV","IWF","SE",
    "FBTC","VV","FXAIZ","AMZN","CLX","CRM","GBTC","ALRM"
]
# ----------------------------


def clean_output_dir():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)


def read_tickers(path, default=None):
    default = default or []
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    return lines


def _to_scalar(x):
    try:
        return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception:
        return float("nan")


def fetch_histories(tickers):
    """Return dict[ticker] -> Series of Close prices. Skip tickers with no data."""
    if not tickers:
        return {}
    df = yf.download(
        tickers=tickers, period=HISTORY_PERIOD, interval="1d",
        auto_adjust=False, progress=False, group_by="ticker", threads=True
    )
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = df[(t, "Close")].dropna()
                if not s.empty:
                    out[t] = s
            except KeyError:
                pass
    else:
        try:
            s = df["Close"].dropna()
            if not s.empty:
                out[tickers[0]] = s
        except Exception:
            pass
    return out


def price_last(s):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return _to_scalar(s.iloc[-1])


def prev_close(s):
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return _to_scalar(s.iloc[-2])


def month_base(s):
    s = s.dropna()
    if len(s) <= LOOKBACKS["month"]:
        return np.nan
    return _to_scalar(s.iloc[-(LOOKBACKS["month"] + 1)])


def ytd_base(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    yr = s.index[-1].year
    this_year = s[s.index.year == yr]
    if this_year.empty:
        return np.nan
    return _to_scalar(this_year.iloc[0])


def names_for_tickers(tickers):
    names = {}
    try:
        info = yf.Tickers(" ".join(tickers))
    except Exception:
        return {t: t for t in tickers}
    for t in tickers:
        try:
            nm = info.tickers[t].info.get("shortName") or info.tickers[t].info.get("longName")
            names[t] = nm or t
        except Exception:
            names[t] = t
    return names


def fetch_index_day_snapshot(symbol):
    """Return (last_price, day_change_abs, day_change_pct) from last two closes for the index."""
    try:
        hist = yf.download(tickers=[symbol], period="5d", interval="1d", progress=False, auto_adjust=False)
        s = hist["Close"].dropna()
        if len(s) < 2:
            return np.nan, np.nan, np.nan
        last = _to_scalar(s.iloc[-1])
        prev = _to_scalar(s.iloc[-2])
        chg = last - prev
        chg_pct = (chg / prev) if prev != 0 else np.nan
        return last, chg, chg_pct
    except Exception:
        return np.nan, np.nan, np.nan


def fmt_pct(x):   return "—" if pd.isna(x) else f"{float(x):+.2%}"
def fmt_price(x): return "—" if pd.isna(x) else f"{float(x):,.2f}"
def fmt_abs(x):   return "—" if pd.isna(x) else f"{float(x):+,.2f}"


def render_html(ctx):
    template = Template("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg:#111; --bg:#fff; --muted:#666; --gain:#0a7f3f; --loss:#a60023; --accent:#0b57d0; --pulse:#c7f0d8; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#0b0b0b; --muted:#9aa0a6; --gain:#4cd26b; --loss:#ff6b81; --accent:#7aa2ff; --pulse:#123d27; }
  }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: var(--fg); background: var(--bg); }
  h1 { margin: 0 0 8px; font-size: 1.75rem; }
  .muted { color: var(--muted); font-size: 0.9rem; }
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; margin-bottom: 18px; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .badge { padding: 8px 12px; border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; display:flex; gap:10px; align-items:baseline; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .gain { color: var(--gain); }
  .loss { color: var(--loss); }
  .status { font-weight: 700; }
  .pulse { animation: pulse-bg 0.6s ease; }
  @keyframes pulse-bg { 0% { background: var(--pulse); } 100% { background: transparent; } }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); }
  th { font-weight: 700; }
  .daywrap { display:flex; gap:6px; align-items:baseline; }
  .dim { color: var(--muted); font-size: .9em; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>

  <!-- Market status + indices -->
  <div class="card" id="banner">
    <div class="row" style="justify-content: space-between;">
      <div class="row">
        <div class="badge"><span class="status" id="market_status">Market …</span></div>
        <div class="badge">
          <div><strong>Dow Jones (DJIA)</strong></div>
          <div class="num">$<span id="dji_price">{{ dji_price }}</span></div>
          <div class="num"><span id="dji_chg" class="{{ 'gain' if dji_chg_val >=0 else 'loss' }}">{{ dji_chg }}</span></div>
          <div class="num"><span id="dji_chg_pct" class="{{ 'gain' if dji_chg_val >=0 else 'loss' }}">{{ dji_chg_pct }}</span></div>
        </div>
        <div class="badge">
          <div><strong>S&amp;P 500</strong></div>
          <div class="num">$<span id="gspc_price">{{ gspc_price }}</span></div>
          <div class="num"><span id="gspc_chg" class="{{ 'gain' if gspc_chg_val >=0 else 'loss' }}">{{ gspc_chg }}</span></div>
          <div class="num"><span id="gspc_chg_pct" class="{{ 'gain' if gspc_chg_val >=0 else 'loss' }}">{{ gspc_chg_pct }}</span></div>
        </div>
      </div>
      <div class="row">
        <div class="muted">Last updated: <span id="last_updated">—</span> (CT)</div>
      </div>
    </div>
    <div class="muted" style="margin-top:8px;">Auto-refreshing every 60 seconds. Day shows $ and %, Month/YTD show %.</div>
    <div id="market_summary" style="margin-top:10px;"></div>
  </div>

  <!-- Top combined table -->
  <div class="card">
    <h2 style="margin:0 0 8px 0;">Combined Top {{ top_n }} (Stocks + ETFs)</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="num">Price</th><th class="num">Day ($ · %)</th><th class="num">Month</th><th class="num">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in top_rows %}
        <tr data-symbol="{{ r['Ticker'] }}" data-prev="{{ r['PrevCloseRaw'] }}" data-mbase="{{ r['MonthBaseRaw'] }}" data-ybase="{{ r['YtdBaseRaw'] }}">
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="num">$<span id="p_{{ r['Ticker'] }}">{{ r['Price'] }}</span></td>
          <td class="num">
            <span class="daywrap">
              <span id="da_{{ r['Ticker'] }}" class="{{ 'gain' if r['DayAbs_val'] >= 0 else 'loss' }}">{{ r['DayAbs'] }}</span>
              <span id="dp_{{ r['Ticker'] }}" class="dim {{ 'gain' if r['Day_val'] >= 0 else 'loss' }}">({{ r['Day'] }})</span>
            </span>
          </td>
          <td class="num"><span id="m_{{ r['Ticker'] }}" class="{{ 'gain' if r['Month_val'] >= 0 else 'loss' }}">{{ r['Month'] }}</span></td>
          <td class="num"><span id="y_{{ r['Ticker'] }}" class="{{ 'gain' if r['YTD_val'] >= 0 else 'loss' }}">{{ r['YTD'] }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Mike's list -->
  <div class="card">
    <h2 style="margin:0 0 8px 0;">Mike's Stocks & ETFs</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="num">Price</th><th class="num">Day ($ · %)</th><th class="num">Month</th><th class="num">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in mike_rows %}
        <tr data-symbol="{{ r['Ticker'] }}" data-prev="{{ r['PrevCloseRaw'] }}" data-mbase="{{ r['MonthBaseRaw'] }}" data-ybase="{{ r['YtdBaseRaw'] }}">
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="num">$<span id="p_{{ r['Ticker'] }}">{{ r['Price'] }}</span></td>
          <td class="num">
            <span class="daywrap">
              <span id="da_{{ r['Ticker'] }}" class="{{ 'gain' if r['DayAbs_val'] >= 0 else 'loss' }}">{{ r['DayAbs'] }}</span>
              <span id="dp_{{ r['Ticker'] }}" class="dim {{ 'gain' if r['Day_val'] >= 0 else 'loss' }}">({{ r['Day'] }})</span>
            </span>
          </td>
          <td class="num"><span id="m_{{ r['Ticker'] }}" class="{{ 'gain' if r['Month_val'] >= 0 else 'loss' }}">{{ r['Month'] }}</span></td>
          <td class="num"><span id="y_{{ r['Ticker'] }}" class="{{ 'gain' if r['YTD_val'] >= 0 else 'loss' }}">{{ r['YTD'] }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <footer class="muted">
    Built by GitHub Actions with yfinance. Not investment advice. Data may be delayed or adjusted.
  </footer>

<script>
  const REFRESH_SYMBOLS = {{ refresh_symbols_json | safe }};

  function fmtPrice(x){ return (x ?? 0).toFixed(2); }
  function fmtPct(x){ return (x >= 0 ? "+" : "") + (x ?? 0).toFixed(2) + "%"; }
  function fmtAbs(x){ const v = (x ?? 0); return (v >= 0 ? "+" : "") + Math.abs(v).toFixed(2); }
  function pulse(el){ if(!el) return; el.classList.add("pulse"); setTimeout(()=>el.classList.remove("pulse"), 600); }
  function setUpdatedNow(){
    const d = new Date();
    const hh = String(d.getHours()).padStart(2,"0");
    const mm = String(d.getMinutes()).padStart(2,"0");
    const ss = String(d.getSeconds()).padStart(2,"0");
    const el = document.getElementById("last_updated");
    el.textContent = `${hh}:${mm}:${ss}`;
    pulse(document.getElementById("banner"));
  }
  function setMarketStatusFromState(state){
    let label = "Market Closed";
    if (state === "REGULAR") label = "Market Open";
    else if (state === "PRE") label = "Pre-Market";
    else if (state === "POST") label = "After Hours";
    document.getElementById("market_status").textContent = label;
  }
  function setMarketStatusFallback(){
    // Fallback: Mon–Fri, 8:30–15:00 CT = Open; 15:00–19:00 = After Hours
    const d = new Date();
    const day = d.getDay(); // 0 Sun ... 6 Sat
    if (day === 0 || day === 6) { setMarketStatusFromState("CLOSED"); return; }
    const mins = d.getHours()*60 + d.getMinutes();
    if (mins >= 8*60+30 && mins <= 15*60) setMarketStatusFromState("REGULAR");
    else if (mins > 15*60 && mins <= 19*60) setMarketStatusFromState("POST");
    else setMarketStatusFromState("CLOSED");
  }

  function recomputeAndRender(sym, quote){
    const row = document.querySelector(`tr[data-symbol="${sym}"]`);
    if (!row) return;

    // Baselines
    let prev  = parseFloat(row.getAttribute("data-prev"));
    const mbase = parseFloat(row.getAttribute("data-mbase"));
    const ybase = parseFloat(row.getAttribute("data-ybase"));
    const price = quote?.regularMarketPrice ?? quote?.price ?? NaN;

    if (isNaN(prev) && quote?.regularMarketPreviousClose != null) prev = +quote.regularMarketPreviousClose;

    // Price
    if (!isNaN(price)) {
      const pEl = document.getElementById("p_"+sym);
      if (pEl){ pEl.textContent = fmtPrice(price); pulse(pEl); }
    }

    // Day: $ and %
    let dayAbs = null, dayPct = null;
    if (!isNaN(prev) && prev !== 0 && !isNaN(price)) {
      dayAbs = price - prev;
      dayPct = (price/prev - 1) * 100.0;
    } else {
      if (quote?.regularMarketChange != null) dayAbs = +quote.regularMarketChange;
      if (quote?.regularMarketChangePercent != null) dayPct = +quote.regularMarketChangePercent;
    }
    const daEl = document.getElementById("da_"+sym);
    const dpEl = document.getElementById("dp_"+sym);
    if (daEl && dayAbs != null) {
      daEl.textContent = fmtAbs(dayAbs);
      daEl.className = (dayAbs >= 0 ? "gain" : "loss");
      pulse(daEl);
    }
    if (dpEl && dayPct != null) {
      dpEl.textContent = "(" + fmtPct(dayPct) + ")";
      dpEl.className = "dim " + (dayPct >= 0 ? "gain" : "loss");
      pulse(dpEl);
    }

    // Month %
    const mEl = document.getElementById("m_"+sym);
    if (mEl && !isNaN(mbase) && mbase !== 0 && !isNaN(price)) {
      const mPct = (price/mbase - 1) * 100.0;
      mEl.textContent = fmtPct(mPct);
      mEl.className = mPct >= 0 ? "gain" : "loss";
      pulse(mEl);
    }

    // YTD %
    const yEl = document.getElementById("y_"+sym);
    if (yEl && !isNaN(ybase) && ybase !== 0 && !isNaN(price)) {
      const yPct = (price/ybase - 1) * 100.0;
      yEl.textContent = fmtPct(yPct);
      yEl.className = yPct >= 0 ? "gain" : "loss";
      pulse(yEl);
    }
  }

  function composeSummary(idxQuotes, allItemQuotes){
    try{
      const dji = idxQuotes["^DJI"], gspc = idxQuotes["^GSPC"];
      const djiPct = dji?.regularMarketChangePercent ?? 0;
      const gspcPct = gspc?.regularMarketChangePercent ?? 0;

      // Find leaders/laggards among the tracked set that have a change%
      const movers = [];
      for (const [sym, q] of Object.entries(allItemQuotes)){
        const p = q?.regularMarketChangePercent;
        if (p == null) continue;
        movers.push([sym, +p]);
      }
      movers.sort((a,b)=>b[1]-a[1]);
      const up = movers.slice(0,3).map(([s,v])=>`${s} ${fmtPct(v)}`);
      const dn = movers.slice(-3).reverse().map(([s,v])=>`${s} ${fmtPct(v)}`);

      const dir = gspcPct >= 0 ? "higher" : "lower";
      const txt = `Market ${dir} so far: S&P 500 ${fmtPct(gspcPct)}, Dow ${fmtPct(djiPct)}. Leaders: ${up.join(", ")}. Laggards: ${dn.join(", ")}.`;
      const el = document.getElementById("market_summary");
      if (el){ el.textContent = txt; }
    }catch(e){ console.log("Summary compose failed:", e); }
  }

  async function refreshIndicesAndStatus(){
    const idxMap = {};
    const url = "https://query2.finance.yahoo.com/v7/finance/quote?symbols=%5EDJI,%5EGSPC&_=" + Date.now();
    try {
      const r = await fetch(url, {cache:"no-store", mode:"cors"});
      const j = await r.json();
      const res = (j && j.quoteResponse && j.quoteResponse.result) || [];
      for (const it of res) { if (it && it.symbol) idxMap[it.symbol] = it; }

      const dji = idxMap["^DJI"];
      if (dji) {
        const price = dji.regularMarketPrice ?? 0;
        const chg = dji.regularMarketChange ?? 0;
        const pct = dji.regularMarketChangePercent ?? 0;
        const cls = chg >= 0 ? "gain" : "loss";
        const pEl = document.getElementById("dji_price");
        const aEl = document.getElementById("dji_chg");
        const pPct = document.getElementById("dji_chg_pct");
        if (pEl){ pEl.textContent = fmtPrice(price); pulse(pEl); }
        if (aEl){ aEl.textContent = (chg>=0?"+":"") + chg.toFixed(2); aEl.className = cls; pulse(aEl); }
        if (pPct){ pPct.textContent = fmtPct(pct); pPct.className = cls; pulse(pPct); }
      }

      const gspc = idxMap["^GSPC"];
      if (gspc) {
        const price = gspc.regularMarketPrice ?? 0;
        const chg = gspc.regularMarketChange ?? 0;
        const pct = gspc.regularMarketChangePercent ?? 0;
        const cls = chg >= 0 ? "gain" : "loss";
        const pEl = document.getElementById("gspc_price");
        const aEl = document.getElementById("gspc_chg");
        const pPct = document.getElementById("gspc_chg_pct");
        if (pEl){ pEl.textContent = fmtPrice(price); pulse(pEl); }
        if (aEl){ aEl.textContent = (chg>=0?"+":"") + chg.toFixed(2); aEl.className = cls; pulse(aEl); }
        if (pPct){ pPct.textContent = fmtPct(pct); pPct.className = cls; pulse(pPct); }
      }

      const anyState = (Object.values(idxMap).find(x => x && x.marketState)?.marketState) || null;
      if (anyState) setMarketStatusFromState(anyState); else setMarketStatusFallback();

    } catch(e){
      console.log("Index/status refresh failed:", e);
      setMarketStatusFallback();
    }
    return idxMap;
  }

  async function refreshTables(){
    const outMap = {};
    if (!REFRESH_SYMBOLS || !REFRESH_SYMBOLS.length) return outMap;

    const size = 40;
    for (let i=0; i<REFRESH_SYMBOLS.length; i+=size) {
      const group = REFRESH_SYMBOLS.slice(i, i+size);
      const url = "https://query2.finance.yahoo.com/v7/finance/quote?symbols=" +
                  encodeURIComponent(group.join(",")) + "&_=" + Date.now();
      try {
        const r = await fetch(url, {cache:"no-store", mode:"cors"});
        const j = await r.json();
        const res = (j && j.quoteResponse && j.quoteResponse.result) || [];
        for (const it of res) {
          if (!it || !it.symbol) continue;
          outMap[it.symbol] = it;
          recomputeAndRender(it.symbol, it);
        }
      } catch(e){ console.log("Table refresh failed:", e); }
    }
    return outMap;
  }

  async function doRefresh(){
    const [idxMap, itemMap] = await Promise.all([refreshIndicesAndStatus(), refreshTables()]);
    composeSummary(idxMap, itemMap);
    setUpdatedNow();
  }

  // Auto-refresh every 60s + initial load
  setInterval(doRefresh, 60000);
  doRefresh();
</script>
</body>
</html>
    """.strip())
    return template.render(**ctx)


def main():
    clean_output_dir()

    # Universe from files
    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])
    universe = list(dict.fromkeys(stocks + etfs))

    # Fetch histories (universe ∪ Mike's list)
    all_symbols = list(dict.fromkeys(universe + MIKE_TICKERS))
    histories = fetch_histories(all_symbols)

    # Compute metrics & baselines (skip symbols with no history)
    rows = []
    for t in all_symbols:
        s = histories.get(t)
        if s is None or len(s.dropna()) == 0:
            continue
        last_price = price_last(s)
        prev = prev_close(s)
        mbase = month_base(s)
        ybase = ytd_base(s)

        day   = (last_price / prev - 1.0) if (pd.notna(last_price) and pd.notna(prev) and prev != 0) else np.nan
        month = (last_price / mbase - 1.0) if (pd.notna(last_price) and pd.notna(mbase) and mbase != 0) else np.nan
        ytd   = (last_price / ybase - 1.0) if (pd.notna(last_price) and pd.notna(ybase) and ybase != 0) else np.nan
        day_abs = (last_price - prev) if (pd.notna(last_price) and pd.notna(prev)) else np.nan

        rows.append({
            "Ticker": t, "Price": last_price,
            "PrevClose": prev, "MonthBase": mbase, "YtdBase": ybase,
            "Day": day, "DayAbs": day_abs, "Month": month, "YTD": ytd
        })
    df = pd.DataFrame(rows)

    # Names
    names = names_for_tickers(df["Ticker"].tolist()) if not df.empty else {}
    df["Name"] = df["Ticker"].map(names).fillna(df["Ticker"])

    # Rank & Top-10 (combined)
    rank_col = {"day": "Day", "month": "Month", "ytd": "YTD"}.get(RANK_HORIZON.lower(), "Month")
    df_top = df[~df[rank_col].isna()].sort_values(by=rank_col, ascending=False).head(TOP_N).copy()

    # Mike's subset (keep specified order)
    order_map = {t: i for i, t in enumerate(MIKE_TICKERS)}
    df_mike = df[df["Ticker"].isin(order_map)].copy()
    df_mike["__order"] = df_mike["Ticker"].map(order_map)
    df_mike = df_mike.sort_values("__order").drop(columns="__order")

    # Prepare HTML rows including baselines (used by JS for live recompute)
    def rows_for_html(df_):
        out = []
        for _, r in df_.iterrows():
            price = r["Price"]; d = r["Day"]; m = r["Month"]; y = r["YTD"]; da = r["DayAbs"]
            out.append({
                "Ticker": r["Ticker"], "Name": r["Name"],
                "Price": "—" if pd.isna(price) else f"{float(price):,.2f}",
                "Day": "—" if pd.isna(d) else f"{float(d):+.2%}",
                "DayAbs": "—" if pd.isna(da) else f"{float(da):+,.2f}",
                "Month": "—" if pd.isna(m) else f"{float(m):+.2%}",
                "YTD": "—" if pd.isna(y) else f"{float(y):+.2%}",
                "Day_val": 0.0 if pd.isna(d) else float(d),
                "DayAbs_val": 0.0 if pd.isna(da) else float(da),
                "Month_val": 0.0 if pd.isna(m) else float(m),
                "YTD_val": 0.0 if pd.isna(y) else float(y),
                "PrevCloseRaw": "" if pd.isna(r["PrevClose"]) else float(r["PrevClose"]),
                "MonthBaseRaw": "" if pd.isna(r["MonthBase"]) else float(r["MonthBase"]),
                "YtdBaseRaw": "" if pd.isna(r["YtdBase"]) else float(r["YtdBase"]),
            })
        return out

    top_rows  = rows_for_html(df_top)
    mike_rows = rows_for_html(df_mike)

    # Symbols to refresh on page (Top10 ∪ Mike)
    refresh_symbols = sorted(set(df_top["Ticker"].tolist()) | set(df_mike["Ticker"].tolist()))

    # Seed index banner from last two closes
    dji_last, dji_chg_abs, dji_chg_pct = fetch_index_day_snapshot("^DJI")
    gspc_last, gspc_chg_abs, gspc_chg_pct = fetch_index_day_snapshot("^GSPC")

    updated_human = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html({
        "title": TITLE,
        "updated_human": updated_human,
        "tz": TZ,
        "rank_horizon": rank_col,
        "top_n": TOP_N,
        "top_rows": top_rows,
        "mike_rows": mike_rows,
        "refresh_symbols_json": json.dumps(refresh_symbols),
        # indices
        "dji_price": "—" if pd.isna(dji_last) else f"{dji_last:,.2f}",
        "dji_chg": "—" if pd.isna(dji_chg_abs) else f"{dji_chg_abs:+,.2f}",
        "dji_chg_pct": "—" if pd.isna(dji_chg_pct) else f"{dji_chg_pct:+.2%}",
        "dji_chg_val": 0 if pd.isna(dji_chg_abs) else float(dji_chg_abs),
        "gspc_price": "—" if pd.isna(gspc_last) else f"{gspc_last:,.2f}",
        "gspc_chg": "—" if pd.isna(gspc_chg_abs) else f"{gspc_chg_abs:+,.2f}",
        "gspc_chg_pct": "—" if pd.isna(gspc_chg_pct) else f"{gspc_chg_pct:+.2%}",
        "gspc_chg_val": 0 if pd.isna(gspc_chg_abs) else float(gspc_chg_abs),
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Built page with reliable live auto-refresh, market status, and daily summary. Missing tickers skipped.")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
