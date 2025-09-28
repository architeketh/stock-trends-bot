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
ETFS_FILE   = os.path.join(DATA_DIR, "tickers_etfs.txt")

# Download enough history to compute YTD
HISTORY_PERIOD = "1y"

# Lookbacks in trading days
LOOKBACKS = {
    "day": 1,       # last close vs previous close
    "month": 21     # ~1 month (trading days)
}

# Which horizon to use to rank the combined Top-10 table
RANK_HORIZON = "month"   # options: "day" or "month" or "ytd"

TOP_N = 10
TITLE = "Top 10 (Stocks + ETFs) — Combined Leaderboard"
TZ = "America/Chicago"  # display only

# Chart settings (single grouped-bar chart for Top-10)
TOP10_CHART_FILENAME = "top10_returns.png"
CHART_WIDTH_IN = 12
CHART_HEIGHT_IN = 5
CHART_DPI = 144
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


def compute_return_over_lookback(close_series, lb_days):
    """
    last / price_{lb_days_ago} - 1
    """
    s = close_series.dropna()
    if len(s) <= lb_days:
        return np.nan
    last = float(s.iloc[-1])
    past = float(s.iloc[-(lb_days+1)])
    return (last / past) - 1.0


def compute_ytd_return(close_series):
    """
    last / first_close_of_current_year - 1
    """
    s = close_series.dropna()
    if s.empty:
        return np.nan
    year = s.index[-1].year
    s_this_year = s[s.index.year == year]
    if s_this_year.empty:
        return np.nan
    start = float(s_this_year.iloc[0])
    last = float(s.iloc[-1])
    if start == 0:
        return np.nan
    return (last / start) - 1.0


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
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; margin-bottom: 18px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); vertical-align: middle; }
  th { font-weight: 700; }
  .pct, .price { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; white-space: nowrap; }
  .gain { color: var(--gain); }
  .loss { color: var(--loss); }
  footer{ margin-top: 28px; color: var(--muted); font-size: 0.85rem; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  img.chart { max-width: 100%; height: auto; display: block; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="muted">Updated {{ updated_human }} ({{ tz }}). Universe scanned: {{ universe_count }} tickers. Ranked by <b>{{ rank_horizon|upper }}</b> return.</div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Combined Top {{ top_n }} (Stocks + ETFs)</h2>
    <div class="muted">
      <a href="top_combined.csv">CSV</a> · <a href="top_combined.json">JSON</a>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Ticker</th>
          <th>Name</th>
          <th class="price">Price</th>
          <th class="pct">Δ (abs)</th>
          <th class="pct">Δ%</th>
          <th class="pct">Day</th>
          <th class="pct">Month</th>
          <th class="pct">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for row in rows %}
        <tr>
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ row['Ticker'] }}/" target="_blank" rel="noopener">{{ row['Ticker'] }}</a></td>
          <td>{{ row['Name'] }}</td>
          <td class="price">${{ row['Price'] }}</td>
          <td class="pct {{ 'gain' if row['Chg'] >= 0 else 'loss' }}">{{ row['Chg_str'] }}</td>
          <td class="pct {{ 'gain' if row['ChgPct'] >= 0 else 'loss' }}">{{ row['ChgPct_str'] }}</td>
          <td class="pct {{ 'gain' if row['Day'] >= 0 else 'loss' }}">{{ row['Day_str'] }}</td>
          <td class="pct {{ 'gain' if row['Month'] >= 0 else 'loss' }}">{{ row['Month_str'] }}</td>
          <td class="pct {{ 'gain' if row['YTD'] >= 0 else 'loss' }}">{{ row['YTD_str'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Top {{ top_n }} — Day / Month / YTD Returns</h2>
    <img class="chart" src="charts/{{ top10_chart_filename }}" alt="Grouped bar chart: Day / Month / YTD returns for Top {{ top_n }}">
  </div>

  <footer>
    Built by GitHub Actions with yfinance &amp; matplotlib. Not investment advice. Data may be delayed or adjusted.
  </footer>
</body>
</html>
    """.strip())
    return template.render(**context)


def build_top10_chart(df_top, out_path):
    """
    df_top: DataFrame with columns Ticker, Day, Month, YTD (floats, fraction not %), sorted by rank.
    Produces a grouped bar chart.
    """
    tickers = df_top["Ticker"].tolist()
    day = (df_top["Day"].values * 100.0).astype(float)
    month = (df_top["Month"].values * 100.0).astype(float)
    ytd = (df_top["YTD"].values * 100.0).astype(float)

    x = np.arange(len(tickers))
    width = 0.26

    fig, ax = plt.subplots(figsize=(CHART_WIDTH_IN, CHART_HEIGHT_IN), dpi=CHART_DPI)
    ax.bar(x - width, day, width, label="Day")
    ax.bar(x,         month, width, label="Month")
    ax.bar(x + width, ytd, width, label="YTD")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=0, ha="center")
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("%")
    ax.set_title("Top 10: Day / Month / YTD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, transparent=False)
    plt.close(fig)


def main():
    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])
    universe = list(dict.fromkeys(stocks + etfs))  # dedupe keep order

    # Fetch price histories (batched)
    histories = fetch_histories(universe)

    # Compute metrics per ticker
    rows = []
    for t, s in histories.items():
        price, prev, chg_abs, chg_pct = compute_point_changes(s)
        day_ret = compute_return_over_lookback(s, LOOKBACKS["day"])
        month_ret = compute_return_over_lookback(s, LOOKBACKS["month"])
        ytd_ret = compute_ytd_return(s)

        rows.append({
            "Ticker": t,
            "Price": price,
            "Chg": chg_abs,
            "ChgPct": chg_pct,
            "Day": day_ret,
            "Month": month_ret,
            "YTD": ytd_ret
        })

    df = pd.DataFrame(rows)

    # Names
    names = get_names_for_tickers(df["Ticker"].tolist()) if not df.empty else {}
    df["Name"] = df["Ticker"].map(names).fillna(df["Ticker"])

    # Rank & select Top-10 combined
    if RANK_HORIZON.lower() not in {"day", "month", "ytd"}:
        rank_col = "Month"
    else:
        rank_col = {"day": "Day", "month": "Month", "ytd": "YTD"}[RANK_HORIZON.lower()]

    df_top = df[~df[rank_col].isna()].sort_values(by=rank_col, ascending=False).head(TOP_N).copy()

    # Pretty strings for HTML
    def fmt_pct(x):
        return "—" if pd.isna(x) else f"{float(x):+.2%}"

    def fmt_price(x):
        return "—" if pd.isna(x) else f"{float(x):,.2f}"

    def fmt_abs(x):
        return "—" if pd.isna(x) else f"{float(x):+,.2f}"

    rows_html = []
    for _, r in df_top.iterrows():
        price = float(r["Price"]) if pd.notna(r["Price"]) else np.nan
        chg   = float(r["Chg"]) if pd.notna(r["Chg"]) else np.nan
        chgp  = float(r["ChgPct"]) if pd.notna(r["ChgPct"]) else np.nan
        d     = float(r["Day"]) if pd.notna(r["Day"]) else np.nan
        m     = float(r["Month"]) if pd.notna(r["Month"]) else np.nan
        y     = float(r["YTD"]) if pd.notna(r["YTD"]) else np.nan

        rows_html.append({
            "Ticker": r["Ticker"],
            "Name": r["Name"],
            "Price": fmt_price(price),
            "Chg": chg if not np.isnan(chg) else 0.0,
            "Chg_str": fmt_abs(chg),
            "ChgPct": chgp if not np.isnan(chgp) else 0.0,
            "ChgPct_str": fmt_pct(chgp),
            "Day": d if not np.isnan(d) else 0.0,
            "Day_str": fmt_pct(d),
            "Month": m if not np.isnan(m) else 0.0,
            "Month_str": fmt_pct(m),
            "YTD": y if not np.isnan(y) else 0.0,
            "YTD_str": fmt_pct(y),
        })

    # Save machine-readable combined Top-10
    _ = to_csv_and_json(
        df_top[["Ticker","Name","Price","Chg","ChgPct","Day","Month","YTD"]],
        "top_combined"
    )

    # Build the single grouped-bar chart
    chart_path = os.path.join(CHART_DIR, TOP10_CHART_FILENAME)
    build_top10_chart(df_top[["Ticker","Day","Month","YTD"]], chart_path)

    # Render HTML
    updated_human = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html({
        "title": TITLE,
        "updated_human": updated_human,
        "tz": TZ,
        "universe_count": len(histories),
        "top_n": TOP_N,
        "rows": rows_html,
        "rank_horizon": rank_col,
        "top10_chart_filename": TOP10_CHART_FILENAME
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Debug snapshot (in % for returns)
    snap = df.copy()
    for c in ["ChgPct","Day","Month","YTD"]:
        if c in snap:
            snap[c] = (snap[c] * 100.0).round(3)
    snap.to_csv(os.path.join(OUT_DIR, "returns_snapshot.csv"), index=False)

    print(f"✅ Site generated: docs/index.html with combined Top {TOP_N} and single Day/Month/YTD chart (ranked by {rank_col})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
