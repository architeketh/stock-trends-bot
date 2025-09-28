import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from jinja2 import Template

# ---------- Config ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs")   # GitHub Pages default folder
os.makedirs(OUT_DIR, exist_ok=True)

STOCKS_FILE = os.path.join(DATA_DIR, "tickers_stocks.txt")
ETFS_FILE = os.path.join(DATA_DIR, "tickers_etfs.txt")

# How far back we download daily candles. Needs to be long enough for monthly returns.
HISTORY_PERIOD = "6mo"

# "Weekly" ≈ 5 trading days; "Monthly" ≈ 21 trading days (trading days, not calendar)
LOOKBACKS = {
    "daily": 1,     # last close vs previous close
    "weekly": 5,    # last close vs 5 trading days ago
    "monthly": 21   # last close vs 21 trading days ago
}

TOP_N = 10
TITLE = "Top 10 Stocks & ETFs by Returns"
TZ = "America/Chicago"  # display only (not used by yfinance)
# ----------------------------


def read_tickers(path, default=None):
    if default is None:
        default = []
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
        return [x for x in lines if x and not x.startswith("#")]


def fetch_histories(tickers):
    """
    Returns a dict[ticker] -> pandas Series of Close prices (Date index).
    Uses yfinance download (batched) for speed.
    """
    if not tickers:
        return {}

    # yfinance.download returns multi-column DF when multiple symbols
    df = yf.download(
        tickers=tickers,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    histories = {}
    if isinstance(df.columns, pd.MultiIndex):
        # Multiple tickers case
        for t in tickers:
            try:
                close = df[(t, "Close")].dropna()
                if not close.empty:
                    histories[t] = close
            except KeyError:
                # ticker missing in the batch result (delisted, etc.)
                continue
    else:
        # Single ticker case
        close = df["Close"].dropna()
        if not close.empty:
            histories[tickers[0]] = close

    return histories


def compute_returns(close_series):
    """
    Given a Series of Close prices with Date index, compute returns for each horizon.
    Returns dict with 'daily', 'weekly', 'monthly' (floats) or np.nan if not enough data.
    """
    res = {}
    closes = close_series.dropna()
    if len(closes) < 2:
        # Not even daily
        return {k: np.nan for k in LOOKBACKS.keys()}

    last = closes.iloc[-1]
    for label, lb in LOOKBACKS.items():
        if len(closes) > lb:
            past = closes.iloc[-(lb+1)]
            ret = (last / past) - 1.0
            res[label] = float(ret)
        else:
            res[label] = np.nan
    return res


def leaderboard(df, horizon, top_n=TOP_N):
    # Filter rows with valid returns; sort descending; take top_n
    d = df[~df[horizon].isna()].sort_values(by=horizon, ascending=False).head(top_n).copy()
    # Add pretty %
    d[horizon + "_pct"] = (d[horizon] * 100.0).round(2)
    return d


def to_csv_and_json(df, name):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    json_path = os.path.join(OUT_DIR, f"{name}.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path


def render_html(context):
    template = Template("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg:#111; --bg:#fff; --muted:#666; --accent:#0b57d0; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#0b0b0b; --muted:#9aa0a6; --accent:#7aa2ff; }
  }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: var(--fg); background: var(--bg); }
  h1 { margin: 0 0 8px; font-size: 1.75rem; }
  .muted { color: var(--muted); font-size: 0.9rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); }
  th { font-weight: 700; }
  .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(11,87,208,0.12); color: var(--accent); font-size: 0.8rem; }
  .pct { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }
  footer{ margin-top: 28px; color: var(--muted); font-size: 0.85rem; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="muted">Updated {{ updated_human }} ({{ tz }}). Universe: {{ stock_count }} stocks, {{ etf_count }} ETFs. Prices use the latest available daily close.</div>

  {% for section in sections %}
  <div class="card">
    <h2 style="margin:0 0 8px 0;">{{ section.heading }}</h2>
    <div class="muted">
      Top {{ section.top_n }} by {{ section.horizon|capitalize }} return. 
      <a href="{{ section.csv_name }}.csv">CSV</a> · <a href="{{ section.csv_name }}.json">JSON</a>
    </div>
    <div class="grid">
      {% for table in section.tables %}
      <div>
        <div class="pill">{{ table.label }}</div>
        <table>
          <thead><tr><th>Ticker</th><th>Name</th><th class="pct">Return</th></tr></thead>
          <tbody>
            {% for row in table.rows %}
            <tr>
              <td><a href="https://finance.yahoo.com/quote/{{ row['Ticker'] }}/" target="_blank" rel="noopener">{{ row['Ticker'] }}</a></td>
              <td>{{ row['Name'] }}</td>
              <td class="pct">{{ row['Return'] }}%</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endfor %}

  <footer>
    Built by GitHub Actions with yfinance. Not investment advice. Data may be delayed or adjusted by the provider.
  </footer>
</body>
</html>
    """.strip())
    return template.render(**context)


def get_names_for_tickers(tickers):
    """Quick metadata fetch. Best-effort: if it fails, fall back to ticker as name."""
    names = {}
    info = yf.Tickers(" ".join(tickers))
    for t in tickers:
        try:
            nm = info.tickers[t].info.get("shortName") or info.tickers[t].info.get("longName")
            names[t] = nm if nm else t
        except Exception:
            names[t] = t
    return names


def main():
    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])

    # Fetch price histories (batched)
    hist_stocks = fetch_histories(stocks)
    hist_etfs   = fetch_histories(etfs)

    # Compute returns
    rows = []
    for universe, store in [("Stock", hist_stocks), ("ETF", hist_etfs)]:
        for t, s in store.items():
            r = compute_returns(s)
            rows.append({
                "Ticker": t,
                "Universe": universe,
                **r
            })
    df = pd.DataFrame(rows)

    # Names
    names = get_names_for_tickers(list(df["Ticker"])) if not df.empty else {}
    df["Name"] = df["Ticker"].map(names).fillna(df["Ticker"])

    # Build leaderboards
    sections = []
    updated_human = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    outputs = []

    for horizon in ["daily", "weekly", "monthly"]:
        # Individual leaderboards
        lb_stocks = leaderboard(df[df["Universe"]=="Stock"], horizon)
        lb_etfs   = leaderboard(df[df["Universe"]=="ETF"], horizon)
        # Combined
        lb_all    = leaderboard(df, horizon)

        # Save machine-readable outputs
        st_csv, st_json = to_csv_and_json(
            lb_stocks[["Ticker","Name",horizon]].rename(columns={horizon:"return"}),
            f"top_{horizon}_stocks"
        )
        et_csv, et_json = to_csv_and_json(
            lb_etfs[["Ticker","Name",horizon]].rename(columns={horizon:"return"}),
            f"top_{horizon}_etfs"
        )
        al_csv, al_json = to_csv_and_json(
            lb_all[["Ticker","Name","Universe",horizon]].rename(columns={horizon:"return"}),
            f"top_{horizon}_overall"
        )

        # Prepare rows for HTML tables (pretty %)
        def rows_for_html(df_):
            return [
                {"Ticker": r["Ticker"], "Name": r["Name"], "Return": round(r[horizon]*100.0, 2)}
                for _, r in df_.iterrows()
            ]

        sections.append({
            "heading": f"Top {TOP_N} — {horizon.capitalize()} Returns",
            "horizon": horizon,
            "top_n": TOP_N,
            "csv_name": f"top_{horizon}_overall",
            "tables": [
                {"label": "Overall (Stocks + ETFs)", "rows": rows_for_html(lb_all)},
                {"label": "Stocks Only", "rows": rows_for_html(lb_stocks)},
                {"label": "ETFs Only", "rows": rows_for_html(lb_etfs)},
            ],
        })

    # Render site
    html = render_html({
        "title": TITLE,
        "updated_human": updated_human,
        "tz": TZ,
        "stock_count": len(hist_stocks),
        "etf_count": len(hist_etfs),
        "sections": sections
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Also output a combined snapshot of raw returns for debugging
    snap = df.copy()
    for h in ["daily","weekly","monthly"]:
        snap[h] = (snap[h] * 100.0).round(3)
    snap.to_csv(os.path.join(OUT_DIR, "returns_snapshot.csv"), index=False)

    print("✅ Site generated in docs/index.html")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
