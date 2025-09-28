import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from jinja2 import Template

# Use non-GUI backend for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Config ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs")   # GitHub Pages default folder
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

STOCKS_FILE = os.path.join(DATA_DIR, "tickers_stocks.txt")
ETFS_FILE = os.path.join(DATA_DIR, "tickers_etfs.txt")

# How far back we download daily candles. Needs to be long enough for monthly returns + charts.
HISTORY_PERIOD = "6mo"

# "Weekly" ≈ 5 trading days; "Monthly" ≈ 21 trading days (trading days, not calendar)
LOOKBACKS = {
    "daily": 1,     # last close vs previous close
    "weekly": 5,    # last close vs 5 trading days ago
    "monthly": 21   # last close vs 21 trading days ago
}

TOP_N = 10
TITLE = "Top 10 Stocks & ETFs by Returns"
TZ = "America/Chicago"  # display only

# Chart settings
CHART_LOOKBACK_DAYS = 30   # trading days for daily % change bars
CHART_WIDTH_PX = 360
CHART_HEIGHT_PX = 140
CHART_DPI = 144            # final image DPI
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
        for t in tickers:
            try:
                close = df[(t, "Close")].dropna()
                if not close.empty:
                    histories[t] = close
            except KeyError:
                continue
    else:
        # single-ticker frame
        try:
            close = df["Close"].dropna()
            if not close.empty:
                histories[tickers[0]] = close
        except Exception:
            pass

    return histories


def compute_point_changes(close_series):
    """
    Compute last price, previous price, abs change and pct change vs yesterday.
    """
    closes = close_series.dropna()
    if len(closes) < 2:
        return np.nan, np.nan, np.nan, np.nan
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    chg = last - prev
    chg_pct = (chg / prev) if prev != 0 else np.nan
    return last, prev, chg, chg_pct


def compute_returns(close_series):
    """
    Given a Series of Close prices, compute returns for each horizon in LOOKBACKS.
    Returns dict with 'daily','weekly','monthly' (floats) or np.nan if not enough data.
    """
    res = {}
    closes = close_series.dropna()
    if len(closes) < 2:
        return {k: np.nan for k in LOOKBACKS.keys()}
    last = closes.iloc[-1]
    for label, lb in LOOKBACKS.items():
        if len(closes) > lb:
            past = closes.iloc[-(lb + 1)]
            ret = (last / past) - 1.0
            res[label] = float(ret)
        else:
            res[label] = np.nan
    return res


def leaderboard(df, horizon, top_n=TOP_N):
    d = df[~df[horizon].isna()].sort_values(by=horizon, ascending=False).head(top_n).copy()
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
  :root { --fg:#111; --bg:#fff; --muted:#666; --accent:#0b57d0; --gain:#0a7f3f; --loss:#a60023; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#0b0b0b; --muted:#9aa0a6; --accent:#7aa2ff; --gain:#4cd26b; --loss:#ff6b81; }
  }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: var(--fg); background: var(--bg); }
  h1 { margin: 0 0 8px; font-size: 1.75rem; }
  .muted { color: var(--muted); font-size: 0.9rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); vertical-align: middle; }
  th { font-weight: 700; }
  .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(11,87,208,0.12); color: var(--accent); font-size: 0.8rem; }
  .pct { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; white-space: nowrap; }
  .price { font-variant-numeric: tabular-nums; }
  .gain { color: var(--gain); }
  .loss { color: var(--loss); }
  img.chart { display:block; width: 180px; height: auto; opacity: 0.9; }
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
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Name</th>
              <th class="price">Price</th>
              <th class="pct">Δ (abs)</th>
              <th class="pct">Δ%</th>
              <th class="pct">Return</th>
              <th>Chart</th>
            </tr>
          </thead>
          <tbody>
            {% for row in table.rows %}
            <tr>
              <td><a href="https://finance.yahoo.com/quote/{{ row['Ticker'] }}/" target="_blank" rel="noopener">{{ row['Ticker'] }}</a></td>
              <td>{{ row['Name'] }}</td>
              <td class="price">${{ row['Price'] }}</td>
              <td class="pct {{ 'gain' if row['Chg'] >=0 else 'loss' }}">{{ row['Chg_str'] }}</td>
              <td class="pct {{ 'gain' if row['ChgPct'] >=0 else 'loss' }}">{{ row['ChgPct_str'] }}</td>
              <td class="pct">{{ row['Return'] }}%</td>
              <td>{% if row['Chart'] %}<img class="chart" src="{{ row['Chart'] }}" alt="{{ row['Ticker'] }} daily % change ({{ chart_days }}d)">{% else %}-{% endif %}</td>
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
    Built by GitHub Actions with yfinance & matplotlib. Not investment advice. Data may be delayed or adjusted.
  </footer>
</body>
</html>
    """.strip())
    return template.render(**context)


def get_names_for_tickers(tickers):
    """Best-effort metadata fetch. Falls back to ticker if not available."""
    names = {}
    info = yf.Tickers(" ".join(tickers))
    for t in tickers:
        try:
            nm = info.tickers[t].info.get("shortName") or info.tickers[t].info.get("longName")
            names[t] = nm if nm else t
        except Exception:
            names[t] = t
    return names


def make_daily_gain_chart(ticker, close_series, out_dir=CHART_DIR, lookback=CHART_LOOKBACK_DAYS):
    """
    Save a small bar chart of daily % changes over last `lookback` trading days.
    Returns relative path to saved image or '' if not enough data.
    """
    s = close_series.dropna().astype(float)
    if len(s) < 3:
        return ""

    # Daily % change (r_t = Close_t / Close_{t-1} - 1)
    r = s.pct_change().dropna()
    r = r.iloc[-lookback:] if len(r) > lookback else r

    # Size in inches = pixels / dpi
    fig_w_in = CHART_WIDTH_PX / CHART_DPI
    fig_h_in = CHART_HEIGHT_PX / CHART_DPI
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=CHART_DPI)
    ax.bar(range(len(r)), r.values)
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(r) - 0.5)
    ax.set_title(f"{ticker} daily % change", fontsize=8, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = os.path.join(out_dir, f"{ticker}.png")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, transparent=False)
    plt.close(fig)

    # Return relative path for HTML
    return f"charts/{ticker}.png"


def main():
    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])

    # Fetch price histories (batched)
    hist_stocks = fetch_histories(stocks)
    hist_etfs   = fetch_histories(etfs)

    # Compute metrics (no charts yet)
    rows = []
    for universe, store in [("Stock", hist_stocks), ("ETF", hist_etfs)]:
        for t, s in store.items():
            last, prev, chg_abs, chg_pct = compute_point_changes(s)
            r = compute_returns(s)
            rows.append({
                "Ticker": t,
                "Universe": universe,
                "Price": last,
                "Prev": prev,
                "Chg": chg_abs,
                "ChgPct": chg_pct,
                **r
            })

    df = pd.DataFrame(rows)

    # Names
    names = get_names_for_tickers(list(df["Ticker"])) if not df.empty else {}
    df["Name"] = df["Ticker"].map(names).fillna(df["Ticker"])

    updated_human = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build *all* leaderboards and collect union of Top-10 tickers (any horizon, any table)
    leaders = {}
    tickers_to_chart = set()

    for horizon in ["daily", "weekly", "monthly"]:
        lb_stocks = leaderboard(df[df["Universe"] == "Stock"], horizon)
        lb_etfs   = leaderboard(df[df["Universe"] == "ETF"], horizon)
        lb_all    = leaderboard(df, horizon)

        leaders[horizon] = (lb_stocks, lb_etfs, lb_all)

        for d in (lb_stocks, lb_etfs, lb_all):
            tickers_to_chart.update(d["Ticker"].tolist())

    # Now create charts only for the union of all Top-10 tickers
    chart_map = {}
    for t in sorted(tickers_to_chart):
        series = hist_stocks.get(t) or hist_etfs.get(t)
        if series is not None:
            chart_map[t] = make_daily_gain_chart(t, series)

    # Build sections from cached leaderboards
    sections = []

    def rows_for_html(df_, horizon):
        out = []
        for _, r in df_.iterrows():
            # Force scalars to avoid "truth value of a Series is ambiguous"
            price = float(r["Price"]) if pd.notna(r["Price"]) else np.nan
            chg = float(r["Chg"]) if pd.notna(r["Chg"]) else np.nan
            chg_pct = float(r["ChgPct"]) if pd.notna(r["ChgPct"]) else np.nan
            ret = float(r[horizon]) if pd.notna(r[horizon]) else np.nan

            out.append({
                "Ticker": r["Ticker"],
                "Name": r["Name"],
                "Price": f"{price:,.2f}" if not np.isnan(price) else "—",
                "Chg": chg if not np.isnan(chg) else 0.0,
                "Chg_str": f"{chg:+,.2f}" if not np.isnan(chg) else "—",
                "ChgPct": chg_pct if not np.isnan(chg_pct) else 0.0,
                "ChgPct_str": f"{chg_pct:+.2%}" if not np.isnan(chg_pct) else "—",
                "Return": round(ret * 100.0, 2) if not np.isnan(ret) else "—",
                "Chart": chart_map.get(r["Ticker"], "")
            })
        return out

    for horizon in ["daily", "weekly", "monthly"]:
        lb_stocks, lb_etfs, lb_all = leaders[horizon]

        # Save machine-readable outputs (with price/change fields)
        cols_common = ["Ticker","Name","Universe","Price","Chg","ChgPct",horizon]
        _ = to_csv_and_json(lb_all[cols_common],   f"top_{horizon}_overall")
        _ = to_csv_and_json(lb_stocks[[c for c in cols_common if c != "Universe"]], f"top_{horizon}_stocks")
        _ = to_csv_and_json(lb_etfs[[c for c in cols_common if c != "Universe"]],   f"top_{horizon}_etfs")

        sections.append({
            "heading": f"Top {TOP_N} — {horizon.capitalize()} Returns",
            "horizon": horizon,
            "top_n": TOP_N,
            "csv_name": f"top_{horizon}_overall",
            "tables": [
                {"label": "Overall (Stocks + ETFs)", "rows": rows_for_html(lb_all, horizon)},
                {"label": "Stocks Only", "rows": rows_for_html(lb_stocks, horizon)},
                {"label": "ETFs Only", "rows": rows_for_html(lb_etfs, horizon)},
            ],
        })

    # Render site
    html = render_html({
        "title": TITLE,
        "updated_human": updated_human,
        "tz": TZ,
        "stock_count": len(hist_stocks),
        "etf_count": len(hist_etfs),
        "sections": sections,
        "chart_days": CHART_LOOKBACK_DAYS
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Snapshot for debugging
    snap = df.copy()
    for h in ["daily","weekly","monthly","ChgPct"]:
        if h in snap:
            snap[h] = (snap[h] * 100.0).round(3)
    snap.to_csv(os.path.join(OUT_DIR, "returns_snapshot.csv"), index=False)

    print("✅ Site generated in docs/index.html with prices, daily changes, and charts (leaders only)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
