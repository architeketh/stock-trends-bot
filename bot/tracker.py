import os
import sys
import shutil
import re
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

HISTORY_PERIOD = "1y"              # enough for YTD
LOOKBACKS = {"day": 1, "month": 21}
RANK_HORIZON = "month"             # "day" | "month" | "ytd"
TOP_N = 10
TITLE = "Top 10 Stocks & ETFs — Price, Day, Month, YTD"
TZ = "America/Chicago"

# Mike's watchlist
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


def fetch_histories(tickers):
    """Return dict[ticker] -> Series of Close prices."""
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
    return float(s.iloc[-1]) if len(s) else np.nan


def ret_over_lookback(s, lb_days):
    s = s.dropna()
    if len(s) <= lb_days:
        return np.nan
    last = float(s.iloc[-1])
    past = float(s.iloc[-(lb_days + 1)])
    return (last / past) - 1.0


def ret_ytd(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    yr = s.index[-1].year
    this_year = s[s.index.year == yr]
    if this_year.empty:
        return np.nan
    start = float(this_year.iloc[0])
    last  = float(s.iloc[-1])
    if start == 0:
        return np.nan
    return (last / start) - 1.0


def names_for_tickers(tickers):
    names = {}
    info = yf.Tickers(" ".join(tickers))
    for t in tickers:
        try:
            nm = info.tickers[t].info.get("shortName") or info.tickers[t].info.get("longName")
            names[t] = nm or t
        except Exception:
            names[t] = t
    return names


def to_csv_and_json(df, name):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    json_path = os.path.join(OUT_DIR, f"{name}.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path


def fetch_index_day_snapshot(symbol):
    """Return (last_price, day_change_abs, day_change_pct) from the last two closes for an index."""
    try:
        hist = yf.download(tickers=[symbol], period="5d", interval="1d", progress=False, auto_adjust=False)
        s = hist["Close"].dropna()
        if len(s) < 2:
            return np.nan, np.nan, np.nan
        last = float(s.iloc[-1]); prev = float(s.iloc[-2])
        chg = last - prev; chg_pct = (chg / prev) if prev != 0 else np.nan
        return last, chg, chg_pct
    except Exception:
        return np.nan, np.nan, np.nan


def safe_id(symbol: str) -> str:
    """Generate a DOM-safe id from a ticker (replace non-alnum with underscore)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", symbol)


def render_html(ctx):
    template = Template("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg:#111; --bg:#fff; --muted:#666; --gain:#0a7f3f; --loss:#a60023; --accent:#0b57d0; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#0b0b0b; --muted:#9aa0a6; --gain:#4cd26b; --loss:#ff6b81; --accent:#7aa2ff; }
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
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  button { padding: 8px 12px; border-radius: 10px; border: 1px solid rgba(127,127,127,0.4); background: transparent; cursor: pointer; }
  button:hover { background: rgba(11,87,208,0.08); }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); }
  th { font-weight: 700; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>

  <!-- Index banner + Refresh -->
  <div class="card">
    <div class="row" style="justify-content: space-between;">
      <div class="row">
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
        <button id="refreshBtn" title="Fetch current quotes">↻ Refresh</button>
        <div class="muted">Updated {{ updated_human }} ({{ tz }})</div>
      </div>
    </div>
    <div class="muted" style="margin-top:8px;">Refresh updates the index banner and table Price/Day in place. Month/YTD update on the next build.</div>
  </div>

  <!-- Top combined table -->
  <div class="card">
    <h2 style="margin:0 0 8px 0;">Combined Top {{ top_n }} (Stocks + ETFs)</h2>
    <div class="muted" style="margin-bottom:8px;"><a href="top_combined.csv">CSV</a> · <a href="top_combined.json">JSON</a></div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="num">Price</th><th class="num">Day</th><th class="num">Month</th><th class="num">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in top_rows %}
        <tr data-symbol="{{ r['Ticker'] }}" data-id="{{ r['dom_id'] }}">
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="num">$<span id="p_{{ r['dom_id'] }}">{{ r['Price'] }}</span></td>
          <td class="num"><span id="d_{{ r['dom_id'] }}" class="{{ 'gain' if r['Day_val'] >= 0 else 'loss' }}">{{ r['Day'] }}</span></td>
          <td class="num">{{ r['Month'] }}</td>
          <td class="num">{{ r['YTD'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Mike's list -->
  <div class="card">
    <h2 style="margin:0 0 8px 0;">Mike's Stocks & ETFs</h2>
    <div class="muted" style="margin-bottom:8px;"><a href="mike_watchlist.csv">CSV</a> · <a href="mike_watchlist.json">JSON</a></div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="num">Price</th><th class="num">Day</th><th class="num">Month</th><th class="num">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in mike_rows %}
        <tr data-symbol="{{ r['Ticker'] }}" data-id="{{ r['dom_id'] }}">
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="num">$<span id="p_{{ r['dom_id'] }}">{{ r['Price'] }}</span></td>
          <td class="num"><span id="d_{{ r['dom_id'] }}" class="{{ 'gain' if r['Day_val'] >= 0 else 'loss' }}">{{ r['Day'] }}</span></td>
          <td class="num">{{ r['Month'] }}</td>
          <td class="num">{{ r['YTD'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <footer class="muted">
    Built by GitHub Actions with yfinance. Not investment advice. Data may be delayed or adjusted.
  </footer>

<script>
  // Data bootstrap for refresh
  const REFRESH_SYMBOLS = {{ refresh_symbols_json | safe }};  // array of symbols to fetch (Top10 ∪ Mike)
  const ID_MAP = {{ id_map_json | safe }};                    // {"TICKER": "DOM_SAFE_ID", ...}

  async function refreshIndices() {
    const url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=%5EDJI,%5EGSPC";
    try {
      const r = await fetch(url, {cache: "no-store"});
      const j = await r.json();
      const res = (j && j.quoteResponse && j.quoteResponse.result) || [];
      const bySym = {};
      for (const it of res) bySym[it.symbol] = it;
      function absStr(x){ return (x >= 0 ? "+" : "") + (x || 0).toFixed(2); }
      function pctStr(x){ return (x >= 0 ? "+" : "") + (x || 0).toFixed(2) + "%"; }

      const dji = bySym["^DJI"];
      if (dji) {
        document.getElementById("dji_price").textContent = (dji.regularMarketPrice ?? 0).toFixed(2);
        const chg = dji.regularMarketChange ?? 0;
        const chgPct = dji.regularMarketChangePercent ?? 0;
        const cls = chg >= 0 ? "gain" : "loss";
        const djiChg = document.getElementById("dji_chg");
        const djiChgPct = document.getElementById("dji_chg_pct");
        djiChg.textContent = absStr(chg);
        djiChgPct.textContent = pctStr(chgPct);
        djiChg.className = cls; djiChgPct.className = cls;
      }

      const gspc = bySym["^GSPC"];
      if (gspc) {
        document.getElementById("gspc_price").textContent = (gspc.regularMarketPrice ?? 0).toFixed(2);
        const chg = gspc.regularMarketChange ?? 0;
        const chgPct = gspc.regularMarketChangePercent ?? 0;
        const cls = chg >= 0 ? "gain" : "loss";
        const el1 = document.getElementById("gspc_chg");
        const el2 = document.getElementById("gspc_chg_pct");
        el1.textContent = absStr(chg);
        el2.textContent = pctStr(chgPct);
        el1.className = cls; el2.className = cls;
      }
    } catch (e) { console.log("Index refresh failed:", e); }
  }

  async function refreshTables() {
    if (!REFRESH_SYMBOLS.length) return;
    // Yahoo supports comma-separated symbols; keep requests comfortably sized
    const chunks = [];
    const size = 40;
    for (let i=0; i<REFRESH_SYMBOLS.length; i+=size) chunks.push(REFRESH_SYMBOLS.slice(i, i+size));

    function pctStr(x){ return (x >= 0 ? "+" : "") + (x || 0).toFixed(2) + "%"; }

    try {
      for (const group of chunks) {
        const url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" +
          encodeURIComponent(group.join(","));
        const r = await fetch(url, {cache: "no-store"});
        const j = await r.json();
        const res = (j && j.quoteResponse && j.quoteResponse.result) || [];
        for (const it of res) {
          const sym = it.symbol;
          const id = ID_MAP[sym];
          if (!id) continue;

          const price = it.regularMarketPrice ?? it.price ?? null;
          const chgPct = it.regularMarketChangePercent ?? null;

          if (price !== null) {
            const pEl = document.getElementById("p_" + id);
            if (pEl) pEl.textContent = (+price).toFixed(2);
          }
          if (chgPct !== null) {
            const dEl = document.getElementById("d_" + id);
            if (dEl) {
              const v = +chgPct;
              dEl.textContent = (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
              dEl.className = v >= 0 ? "gain" : "loss";
            }
          }
        }
      }
    } catch (e) { console.log("Table refresh failed:", e); }
  }

  async function doRefresh() {
    await Promise.all([refreshIndices(), refreshTables()]);
  }
  document.getElementById("refreshBtn").addEventListener("click", doRefresh);
</script>
</body>
</html>
    """.strip())
    return template.render(**ctx)


def fmt_pct(x):   return "—" if pd.isna(x) else f"{float(x):+.2%}"
def fmt_price(x): return "—" if pd.isna(x) else f"{float(x):,.2f}"


def main():
    clean_output_dir()

    # Universe from files
    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])
    universe = list(dict.fromkeys(stocks + etfs))

    # Fetch histories (universe ∪ Mike's list)
    all_symbols = list(dict.fromkeys(universe + MIKE_TICKERS))
    histories = fetch_histories(all_symbols)

    # Compute metrics for all symbols
    rows = []
    for t, s in histories.items():
        price = price_last(s)
        day   = ret_over_lookback(s, LOOKBACKS["day"])
        month = ret_over_lookback(s, LOOKBACKS["month"])
        ytd   = ret_ytd(s)
        rows.append({"Ticker": t, "Price": price, "Day": day, "Month": month, "YTD": ytd})
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

    # Save machine-readable
    to_csv_and_json(df_top[["Ticker","Name","Price","Day","Month","YTD"]], "top_combined")
    to_csv_and_json(df_mike[["Ticker","Name","Price","Day","Month","YTD"]], "mike_watchlist")

    # DOM-safe ids + HTML rows
    def rows_for_html(df_):
        out = []
        for _, r in df_.iterrows():
            price = float(r["Price"]) if pd.notna(r["Price"]) else np.nan
            d = float(r["Day"]) if pd.notna(r["Day"]) else np.nan
            m = float(r["Month"]) if pd.notna(r["Month"]) else np.nan
            y = float(r["YTD"]) if pd.notna(r["YTD"]) else np.nan
            dom_id = safe_id(r["Ticker"])
            out.append({
                "Ticker": r["Ticker"], "Name": r["Name"], "dom_id": dom_id,
                "Price": fmt_price(price),
                "Day": fmt_pct(d), "Month": fmt_pct(m), "YTD": fmt_pct(y),
                "Day_val": 0.0 if pd.isna(d) else d,
            })
        return out

    top_rows  = rows_for_html(df_top)
    mike_rows = rows_for_html(df_mike)

    # Symbols to refresh on page (Top10 ∪ Mike)
    refresh_symbols = sorted(set(df_top["Ticker"].tolist()) | set(df_mike["Ticker"].tolist()))
    id_map = {t: safe_id(t) for t in refresh_symbols}

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
        # refresh data
        "refresh_symbols_json": pd.io.json.dumps(refresh_symbols),
        "id_map_json": pd.io.json.dumps(id_map),
        # indices
        "dji_price": "—" if pd.isna(dji_last) else f"{dji_last:,.2f}",
        "dji_chg": "—" if pd.isna(dji_chg_abs) else f"{dji_chg_abs:+,.2f}",
        "dji_chg_pct": "—" if pd.isna(dji_chg_pct) else f"{dji_chg_pct:+.2%}",
        "dji_chg_val": 0 if pd.isna(dji_chg_abs) else dji_chg_abs,
        "gspc_price": "—" if pd.isna(gspc_last) else f"{gspc_last:,.2f}",
        "gspc_chg": "—" if pd.isna(gspc_chg_abs) else f"{gspc_chg_abs:+,.2f}",
        "gspc_chg_pct": "—" if pd.isna(gspc_chg_pct) else f"{gspc_chg_pct:+.2%}",
        "gspc_chg_val": 0 if pd.isna(gspc_chg_abs) else gspc_chg_abs,
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Debug snapshot
    snap = df.copy()
    for c in ["Day","Month","YTD"]:
        if c in snap:
            snap[c] = (snap[c] * 100.0).round(3)
    snap.to_csv(os.path.join(OUT_DIR, "returns_snapshot.csv"), index=False)

    print("✅ Built page with live Refresh for indices + table Price/Day.")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
