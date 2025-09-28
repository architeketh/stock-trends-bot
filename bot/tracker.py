import os
import sys
import shutil
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

# Universe + Watchlist
HISTORY_PERIOD = "1y"              # enough for YTD
LOOKBACKS = {"day": 1, "month": 21}
RANK_HORIZON = "month"             # "day" | "month" | "ytd"
TOP_N = 10
TITLE = "Top 10 Stocks & ETFs — Price, Day, Month, YTD"
TZ = "America/Chicago"

# Mike's watchlist (use exactly as provided)
MIKE_TICKERS = [
    "VOO","VOOG","VUG","VDIGX","QQQM","AAPL","NVDA","IVV","IWF","SE",
    "FBTC","VV","FXAIZ","AMZN","CLX","CRM","GBTC","ALRM"
]
# ----------------------------


def clean_output_dir():
    # Clear docs/ so only current artifacts remain
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
  .muted { color: var(--muted); font-size: 0.9rem; margin-bottom: 16px; }
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; margin-bottom: 18px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); }
  th { font-weight: 700; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .gain { color: var(--gain); }
  .loss { color: var(--loss); }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="muted">
    Updated {{ updated_human }} ({{ tz }}). Universe scanned: {{ universe_count }} tickers. Ranked by <b>{{ rank_horizon|upper }}</b>.
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Combined Top {{ top_n }} (Stocks + ETFs)</h2>
    <div class="muted"><a href="top_combined.csv">CSV</a> · <a href="top_combined.json">JSON</a></div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="num">Price</th><th class="num">Day</th><th class="num">Month</th><th class="num">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in top_rows %}
        <tr>
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="num">${{ r['Price'] }}</td>
          <td class="num {{ 'gain' if r['Day_val'] >= 0 else 'loss' }}">{{ r['Day'] }}</td>
          <td class="num {{ 'gain' if r['Month_val'] >= 0 else 'loss' }}">{{ r['Month'] }}</td>
          <td class="num {{ 'gain' if r['YTD_val'] >= 0 else 'loss' }}">{{ r['YTD'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Mike's Stocks & ETFs</h2>
    <div class="muted"><a href="mike_watchlist.csv">CSV</a> · <a href="mike_watchlist.json">JSON</a></div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="num">Price</th><th class="num">Day</th><th class="num">Month</th><th class="num">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in mike_rows %}
        <tr>
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="num">${{ r['Price'] }}</td>
          <td class="num {{ 'gain' if r['Day_val'] >= 0 else 'loss' }}">{{ r['Day'] }}</td>
          <td class="num {{ 'gain' if r['Month_val'] >= 0 else 'loss' }}">{{ r['Month'] }}</td>
          <td class="num {{ 'gain' if r['YTD_val'] >= 0 else 'loss' }}">{{ r['YTD'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <footer class="muted">
    Built by GitHub Actions with yfinance. Not investment advice. Data may be delayed or adjusted.
  </footer>
</body>
</html>
    """.strip())
    return template.render(**ctx)


def fmt_pct(x):   return "—" if pd.isna(x) else f"{float(x):+.2%}"
def fmt_price(x): return "—" if pd.isna(x) else f"{float(x):,.2f}"


def main():
    clean_output_dir()

    # Universe from files (stocks + etfs)
    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])
    universe = list(dict.fromkeys(stocks + etfs))  # dedupe keep order

    # Fetch histories (universe ∪ Mike's list)
    all_symbols = list(dict.fromkeys(universe + MIKE_TICKERS))
    histories = fetch_histories(all_symbols)

    # Compute metrics for all fetched symbols
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

    # Mike's subset (preserve MIKE_TICKERS order; show if data available)
    df_mike = df[df["Ticker"].isin(MIKE_TICKERS)].copy()
    df_mike["order"] = df_mike["Ticker"].map({t:i for i,t in enumerate(MIKE_TICKERS)})
    df_mike = df_mike.sort_values("order").drop(columns=["order"])

    # Save machine-readable
    to_csv_and_json(df_top[["Ticker","Name","Price","Day","Month","YTD"]], "top_combined")
    to_csv_and_json(df_mike[["Ticker","Name","Price","Day","Month","YTD"]], "mike_watchlist")

    # Prepare HTML rows (with raw values for coloring)
    def rows_for_html(df_):
        out = []
        for _, r in df_.iterrows():
            price = float(r["Price"]) if pd.notna(r["Price"]) else np.nan
            d = float(r["Day"]) if pd.notna(r["Day"]) else np.nan
            m = float(r["Month"]) if pd.notna(r["Month"]) else np.nan
            y = float(r["YTD"]) if pd.notna(r["YTD"]) else np.nan
            out.append({
                "Ticker": r["Ticker"], "Name": r["Name"],
                "Price": fmt_price(price),
                "Day": fmt_pct(d), "Month": fmt_pct(m), "YTD": fmt_pct(y),
                "Day_val": 0.0 if pd.isna(d) else d,
                "Month_val": 0.0 if pd.isna(m) else m,
                "YTD_val": 0.0 if pd.isna(y) else y,
            })
        return out

    top_rows  = rows_for_html(df_top)
    mike_rows = rows_for_html(df_mike)

    updated_human = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html({
        "title": TITLE,
        "updated_human": updated_human,
        "tz": TZ,
        "universe_count": len(universe),
        "rank_horizon": rank_col,
        "top_n": TOP_N,
        "top_rows": top_rows,
        "mike_rows": mike_rows,
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Debug snapshot (percent as %)
    snap = df.copy()
    for c in ["Day","Month","YTD"]:
        if c in snap:
            snap[c] = (snap[c] * 100.0).round(3)
    snap.to_csv(os.path.join(OUT_DIR, "returns_snapshot.csv"), index=False)

    print(f"✅ Built combined Top {TOP_N} and Mike's list (ranked by {rank_col}).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
