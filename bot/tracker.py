import os, sys, shutil, json
from datetime import datetime
import pandas as pd, numpy as np
import yfinance as yf
from jinja2 import Template

# ---------- Config ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "docs")
STOCKS_FILE = os.path.join(DATA_DIR, "tickers_stocks.txt")
ETFS_FILE   = os.path.join(DATA_DIR, "tickers_etfs.txt")

HISTORY_PERIOD = "1y"                 # enough for YTD baselines
LOOKBACKS = {"day": 1, "month": 21}
TOP_N = 10
TITLE = "Top 10 Stocks & ETFs — Price · Day · Month · YTD"
TZ = "America/Chicago"

# >>>>>>>>>>>>>>>> SET THIS TO YOUR WORKER URL (no trailing slash) <<<<<<<<<<<<<<
WORKER_BASE = "https://broken-night-0891.architek-eth.workers.dev/"

MIKE_TICKERS = ["VOO","VOOG","VUG","VDIGX","QQQM","AAPL","NVDA","IVV","IWF","SE","FBTC","VV","FXAIZ","AMZN","CLX","CRM","GBTC","ALRM"]
# ---------------------------------------------------------------

def clean_output_dir():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

def read_tickers(path, default=None):
    default = default or []
    if not os.path.exists(path): return default
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip() and not x.startswith("#")]

def _to_scalar(x):
    try: return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception: return float("nan")

def fetch_histories(tickers):
    if not tickers: return {}
    df = yf.download(tickers=tickers, period=HISTORY_PERIOD, interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = df[(t, "Close")].dropna()
                if not s.empty: out[t] = s
            except KeyError:
                pass
    else:
        try:
            s = df["Close"].dropna()
            if not s.empty: out[tickers[0]] = s
        except Exception:
            pass
    return out

def price_last(s):
    s = s.dropna()
    return np.nan if len(s)==0 else _to_scalar(s.iloc[-1])

def prev_close(s):
    s = s.dropna()
    return np.nan if len(s)<2 else _to_scalar(s.iloc[-2])

def month_base(s):
    s = s.dropna()
    return np.nan if len(s)<=LOOKBACKS["month"] else _to_scalar(s.iloc[-(LOOKBACKS["month"]+1)])

def ytd_base(s):
    s = s.dropna()
    if s.empty: return np.nan
    yr = s.index[-1].year
    this_year = s[s.index.year==yr]
    return np.nan if this_year.empty else _to_scalar(this_year.iloc[0])

def names_for_tickers(tickers):
    names = {}
    try:
        info = yf.Tickers(" ".join(tickers))
    except Exception:
        return {t:t for t in tickers}
    for t in tickers:
        try:
            nm = info.tickers[t].info.get("shortName") or info.tickers[t].info.get("longName")
            names[t] = nm or t
        except Exception:
            names[t] = t
    return names

def render_html(ctx):
    template = Template("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg:#111; --bg:#fff; --muted:#666; --gain:#0a7f3f; --loss:#a60023; --accent:#0b57d0; --pulse:#c7f0d8; --err:#ffe0e0; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#0b0b0b; --muted:#9aa0a6; --gain:#4cd26b; --loss:#ff6b81; --accent:#7aa2ff; --pulse:#123d27; --err:#4a1212; }
  }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: var(--fg); background: var(--bg); }
  h1 { margin: 0 0 8px; font-size: 1.75rem; }
  .muted { color: var(--muted); font-size: 0.9rem; }
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; margin-bottom: 18px; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .badge { padding: 8px 12px; border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; display:flex; gap:10px; align-items:baseline; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .gain { color: var(--gain); } .loss { color: var(--loss); }
  .status { font-weight: 700; }
  .pulse { animation: pulse-bg 0.6s ease; }
  @keyframes pulse-bg { 0% { background: var(--pulse); } 100% { background: transparent; } }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); }
  th { font-weight: 700; }
  .daywrap { display:flex; gap:6px; align-items:baseline; }
  .dim { color: var(--muted); font-size: .9em; }
  .error { background: var(--err); padding: 6px 10px; border-radius: 8px; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>

  <div class="card" id="banner">
    <div class="row" style="justify-content: space-between;">
      <div class="row">
        <div class="badge"><span class="status" id="market_status">Market …</span></div>
        <div class="badge">
          <div><strong>Dow Jones (DJIA)</strong></div>
          <div class="num">$<span id="dji_price">—</span></div>
          <div class="num"><span id="dji_chg">—</span></div>
          <div class="num"><span id="dji_chg_pct">—</span></div>
        </div>
        <div class="badge">
          <div><strong>S&amp;P 500</strong></div>
          <div class="num">$<span id="gspc_price">—</span></div>
          <div class="num"><span id="gspc_chg">—</span></div>
          <div class="num"><span id="gspc_chg_pct">—</span></div>
        </div>
      </div>
      <div class="row">
        <div class="muted">Last updated: <span id="last_updated">—</span> (CT)</div>
      </div>
    </div>
    <div class="muted" style="margin-top:8px;">Auto-refreshing every 60 seconds. Day shows $ and %, Month/YTD show %.</div>
    <div id="refresh_error" class="error" style="display:none; margin-top:8px;">Last update failed — retrying…</div>
  </div>

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
          <td class="num"><span class="daywrap"><span id="da_{{ r['Ticker'] }}">{{ r['DayAbs'] }}</span> <span id="dp_{{ r['Ticker'] }}" class="dim">({{ r['Day'] }})</span></span></td>
          <td class="num"><span id="m_{{ r['Ticker'] }}">{{ r['Month'] }}</span></td>
          <td class="num"><span id="y_{{ r['Ticker'] }}">{{ r['YTD'] }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

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
          <td class="num"><span class="daywrap"><span id="da_{{ r['Ticker'] }}">{{ r['DayAbs'] }}</span> <span id="dp_{{ r['Ticker'] }}" class="dim">({{ r['Day'] }})</span></span></td>
          <td class="num"><span id="m_{{ r['Ticker'] }}">{{ r['Month'] }}</span></td>
          <td class="num"><span id="y_{{ r['Ticker'] }}">{{ r['YTD'] }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Daily Market Summary</h2>
    <p id="ai_summary" class="muted">—</p>
  </div>

  <footer class="muted">Built on GitHub Pages. Not investment advice. Data may be delayed.</footer>

<script>
  // ======== CONFIG ========
  const WORKER_BASE = "{{ worker_base }}";
  const REFRESH_SYMBOLS = {{ refresh_symbols_json | safe }};

  // ======== UTIL ========
  const $ = (id) => document.getElementById(id);
  function fmtPrice(x){ return (x ?? 0).toFixed(2); }
  function fmtPct(x){ return (x >= 0 ? "+" : "") + (x ?? 0).toFixed(2) + "%"; }
  function fmtAbs(x){ const v = (x ?? 0); return (v >= 0 ? "+" : "") + Math.abs(v).toFixed(2); }
  function pulse(el){ if(!el) return; el.classList.add("pulse"); setTimeout(()=>el.classList.remove("pulse"), 600); }
  function showError(show){ const n=$("refresh_error"); if(n) n.style.display = show ? "block" : "none"; }
  function setUpdatedNow(ok=true){
    const d = new Date(), hh=String(d.getHours()).padStart(2,"0"), mm=String(d.getMinutes()).padStart(2,"0"), ss=String(d.getSeconds()).padStart(2,"0");
    $("last_updated").textContent = `${hh}:${mm}:${ss}`; pulse($("banner")); showError(!ok);
  }
  function setMarketStatusFromState(state){
    let label = "Market Closed";
    if (state === "REGULAR") label = "Market Open";
    else if (state === "PRE") label = "Pre-Market";
    else if (state === "POST") label = "After Hours";
    $("market_status").textContent = label;
  }
  function setMarketStatusFallback(){
    const d = new Date(), day=d.getDay();
    if (day===0 || day===6){ setMarketStatusFromState("CLOSED"); return; }
    const mins = d.getHours()*60 + d.getMinutes();
    if (mins >= 8*60+30 && mins <= 15*60) setMarketStatusFromState("REGULAR");
    else if (mins > 15*60 && mins <= 19*60) setMarketStatusFromState("POST");
    else setMarketStatusFromState("CLOSED");
  }

  // ======== RENDER ========
  function renderIndex(prefix, it){
    if (!it) return;
    const price = it.regularMarketPrice ?? 0;
    const chg = it.regularMarketChange ?? 0;
    const pct = it.regularMarketChangePercent ?? 0;
    $(prefix+"_price").textContent = fmtPrice(price);
    $(prefix+"_chg").textContent = (chg>=0?"+":"") + chg.toFixed(2);
    $(prefix+"_chg_pct").textContent = fmtPct(pct);
    [$(prefix+"_price"), $(prefix+"_chg"), $(prefix+"_chg_pct")].forEach(pulse);
  }

  function recomputeAndRender(sym, quote){
    const row = document.querySelector(`tr[data-symbol="${sym}"]`); if (!row) return;
    let prev  = parseFloat(row.getAttribute("data-prev"));
    const mbase = parseFloat(row.getAttribute("data-mbase"));
    const ybase = parseFloat(row.getAttribute("data-ybase"));
    const price = quote?.regularMarketPrice ?? NaN;
    if (isNaN(prev) && quote?.regularMarketPreviousClose != null) prev = +quote.regularMarketPreviousClose;

    if (!isNaN(price)) { const pEl = $("p_"+sym); if (pEl){ pEl.textContent = fmtPrice(price); pulse(pEl); } }

    let dayAbs=null, dayPct=null;
    if (!isNaN(prev) && prev!==0 && !isNaN(price)) { dayAbs = price - prev; dayPct = (price/prev - 1)*100.0; }
    else { if (quote?.regularMarketChange!=null) dayAbs=+quote.regularMarketChange; if (quote?.regularMarketChangePercent!=null) dayPct=+quote.regularMarketChangePercent; }

    const daEl=$("da_"+sym), dpEl=$("dp_"+sym);
    if (daEl && dayAbs!=null){ daEl.textContent = fmtAbs(dayAbs); pulse(daEl); }
    if (dpEl && dayPct!=null){ dpEl.textContent = "(" + fmtPct(dayPct) + ")"; pulse(dpEl); }

    const mEl=$("m_"+sym);
    if (mEl && !isNaN(mbase) && mbase!==0 && !isNaN(price)){ const mPct=(price/mbase-1)*100.0; mEl.textContent=fmtPct(mPct); pulse(mEl); }
    const yEl=$("y_"+sym);
    if (yEl && !isNaN(ybase) && ybase!==0 && !isNaN(price)){ const yPct=(price/ybase-1)*100.0; yEl.textContent=fmtPct(yPct); pulse(yEl); }
  }

  // ======== QUOTES ========
  async function quotes(symbols){
    const url = `${WORKER_BASE}/quote?symbols=${encodeURIComponent(symbols.join(","))}`;
    const r = await fetch(url, {cache:"no-store"}); if (!r.ok) throw new Error("quotes "+r.status);
    const j = await r.json();
    const res = (j && j.quoteResponse && j.quoteResponse.result) || [];
    const map = {}; res.forEach(it => { if (it?.symbol) map[it.symbol] = it; });
    return map;
  }

  async function refreshIndices(){
    const m = await quotes(["^DJI","^GSPC"]);
    renderIndex("dji", m["^DJI"]); renderIndex("gspc", m["^GSPC"]);
    const state = (Object.values(m).find(x => x?.marketState)?.marketState) || null;
    if (state) setMarketStatusFromState(state); else setMarketStatusFallback();
    return m;
  }

  async function refreshTables(){
    const map = {};
    if (!REFRESH_SYMBOLS?.length) return map;
    const size = 40;
    for (let i=0; i<REFRESH_SYMBOLS.length; i+=size) {
      const group = REFRESH_SYMBOLS.slice(i, i+size);
      const m = await quotes(group);
      Object.entries(m).forEach(([sym, q]) => { map[sym]=q; recomputeAndRender(sym, q); });
    }
    return map;
  }

  // ======== NEWS & SUMMARY ========
  const NEWS = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.bls.gov/feed/bls_latest.rss"
  ];

  async function fetchRss(url){
    const r = await fetch(`${WORKER_BASE}/rss?url=${encodeURIComponent(url)}`, {cache:"no-store"});
    if (!r.ok) throw new Error("rss "+r.status);
    const txt = await r.text();
    const doc = new DOMParser().parseFromString(txt, "application/xml");
    return Array.from(doc.querySelectorAll("item, entry")).map(it => ({
      title: (it.querySelector("title")?.textContent || "").trim(),
      desc: (it.querySelector("description, summary")?.textContent || "").trim(),
      pub:  (it.querySelector("pubDate, updated, published")?.textContent || "").trim()
    }));
  }

  function oneLineSummary(indexMap, headlines){
    const spx = indexMap["^GSPC"], dji = indexMap["^DJI"];
    const spxPct = spx?.regularMarketChangePercent ?? 0;
    const djiPct = dji?.regularMarketChangePercent ?? 0;
    const dir = spxPct >= 0 ? "higher" : "lower";
    const lead = `As of ${new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})} CT, U.S. stocks trade ${dir} (S&P 500 ${fmtPct(spxPct)}, Dow ${fmtPct(djiPct)}).`;

    // Tag-based extraction from last 24h headlines
    const now = Date.now();
    const fresh = headlines.filter(h => (now - (Date.parse(h.pub || "")||now)) < 24*3600*1000);
    const hit = (re) => fresh.find(h => re.test((h.title+" "+h.desc)));
    const fed    = hit(/\b(Fed|FOMC|Powell|rate|cut|hike|dot plot|QT|QE)\b/i);
    const infl   = hit(/\b(CPI|inflation|PCE|core PCE|PPI)\b/i);
    const jobs   = hit(/\b(payrolls|NFP|jobless|unemployment|JOLTS)\b/i);
    const cons   = hit(/\b(retail sales|consumer confidence|Michigan|Conference Board)\b/i);
    const earn   = hit(/\b(earnings|guidance|revenue|sales|EPS|outlook)\b/i);
    const rates  = hit(/\b(Treasury|yield|10-year|curve|spread)\b/i);
    const geo    = hit(/\b(China|tariff|sanction|Ukraine|Gaza|Israel|Russia|Middle East)\b/i);

    const bits = [];
    if (infl)  bits.push(`Inflation: ${infl.title}`);
    if (jobs)  bits.push(`Labor: ${jobs.title}`);
    if (cons)  bits.push(`Consumer: ${cons.title}`);
    if (earn)  bits.push(`Earnings: ${earn.title}`);
    if (rates) bits.push(`Rates: ${rates.title}`);
    if (fed)   bits.push(`Fed: ${fed.title}`);
    if (geo)   bits.push(`Geopolitics: ${geo.title}`);

    const drivers = bits.length ? ` Drivers: ${bits.join(' ')}`
                                : ` Drivers: no major macro headlines in the past few hours.`;
    return (lead + drivers).replace(/\s+/g, " ").trim();
  }

  let lastNews = 0;
  async function refreshNews(indexMap){
    const now = Date.now();
    if (now - lastNews < 5*60*1000) return; // 5-minute throttle
    lastNews = now;
    try {
      const all = (await Promise.all(NEWS.map(fetchRss))).flat();
      $("ai_summary").textContent = oneLineSummary(indexMap, all);
    } catch (e) {
      // keep old text
    }
  }

  // ======== MAIN LOOP ========
  async function tick(){
    try {
      const idx = await refreshIndices();
      const qmap = await refreshTables();
      await refreshNews(idx);
      setUpdatedNow(true);
    } catch(e){
      setUpdatedNow(false);
    }
  }
  setInterval(tick, 60000);
  tick();
</script>
</body>
</html>
    """.strip())
    return template.render(**ctx)

def main():
    clean_output_dir()

    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])
    universe = list(dict.fromkeys(stocks + etfs))
    all_symbols = list(dict.fromkeys(universe + MIKE_TICKERS))

    histories = fetch_histories(all_symbols)

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

    names = {}
    if not df.empty:
        names = names_for_tickers(df["Ticker"].tolist())
    df["Name"] = df["Ticker"].map(names).fillna(df["Ticker"])

    # Rank by Month for Top N
    df_top = df[~df["Month"].isna()].sort_values(by="Month", ascending=False).head(TOP_N).copy()

    # Mike list in given order
    order_map = {t:i for i,t in enumerate(MIKE_TICKERS)}
    df_mike = df[df["Ticker"].isin(order_map)].copy()
    df_mike["__order"] = df_mike["Ticker"].map(order_map)
    df_mike = df_mike.sort_values("__order").drop(columns="__order")

    def rows_for_html(d):
        out=[]
        for _, r in d.iterrows():
            out.append({
                "Ticker": r["Ticker"], "Name": r["Name"],
                "Price": "—" if pd.isna(r["Price"]) else f"{float(r['Price']):,.2f}",
                "Day": "—"   if pd.isna(r["Day"])   else f"{float(r['Day']):+.2%}",
                "DayAbs": "—"if pd.isna(r["DayAbs"])else f"{float(r['DayAbs']):+,.2f}",
                "Month": "—" if pd.isna(r["Month"]) else f"{float(r['Month']):+.2%}",
                "YTD": "—"   if pd.isna(r["YTD"])   else f"{float(r['YTD']):+.2%}",
                "PrevCloseRaw": "" if pd.isna(r["PrevClose"]) else float(r["PrevClose"]),
                "MonthBaseRaw": "" if pd.isna(r["MonthBase"]) else float(r["MonthBase"]),
                "YtdBaseRaw": "" if pd.isna(r["YtdBase"]) else float(r["YtdBase"]),
            })
        return out

    refresh_symbols = sorted(set(df_top["Ticker"].tolist()) | set(df_mike["Ticker"].tolist()))
    if not refresh_symbols:
        refresh_symbols = MIKE_TICKERS

    html = render_html({
        "title": TITLE,
        "top_n": TOP_N,
        "top_rows": rows_for_html(df_top),
        "mike_rows": rows_for_html(df_mike),
        "refresh_symbols_json": json.dumps(refresh_symbols),
        "worker_base": WORKER_BASE
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Built page using Cloudflare Worker for live quotes/news + clean summary paragraph.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
