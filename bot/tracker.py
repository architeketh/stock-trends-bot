import os
import sys
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from jinja2 import Template

# headless matplotlib for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Config ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs")                # GitHub Pages folder
CHART_DIR = os.path.join(OUT_DIR, "charts")

STOCKS_FILE = os.path.join(DATA_DIR, "tickers_stocks.txt")
ETFS_FILE   = os.path.join(DATA_DIR, "tickers_etfs.txt")

HISTORY_PERIOD = "1y"  # enough for YTD
LOOKBACKS = {"day": 1, "month": 21}  # trading days
RANK_HORIZON = "month"  # "day" | "month" | "ytd"

TOP_N = 10
TITLE = "Top 10 Stocks & ETFs — Combined"
TZ = "America/Chicago"

TOP10_CHART_FILENAME = "top10_returns.png"
CHART_WIDTH_IN, CHART_HEIGHT_IN, CHART_DPI = 12, 5, 144
# ----------------------------


def clean_output_dir():
    # Wipe docs/ so old artifacts (stocks/etfs pages) don't persist
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(CHART_DIR, exist_ok=True)


def read_tickers(path, default=None):
    default = default or []
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    return lines


def fetch_histories(tickers):
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


def compute_point_changes(s):
    s = s.dropna()
    if len(s) < 2:
        return np.nan, np.nan, np.nan, np.nan
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
    chg = last - prev
    chg_pct = (chg / prev) if prev != 0 else np.nan
    return last, prev, chg, chg_pct


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
    last = float(s.iloc[-1])
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


def build_chart(df_top, out_path):
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
    ax.set_xticklabels(tickers)
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("%")
    ax.set_title("Top 10: Day / Month / YTD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, transparent=False)
    plt.close(fig)


def render_html(ctx):
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
  .muted { color: var(--muted); font-size: 0.9rem; margin-bottom: 16px; }
  .card { border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 14px; margin-bottom: 18px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px dashed rgba(127,127,127,0.3); vertical-align: middle; }
  th { font-weight: 700; }
  .pct, .price { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; white-space: nowrap; }
  .gain { color: var(--gain); }
  .loss { color: var(--loss); }
  img.chart { max-width: 100%; height: auto; display:block; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="muted">Updated {{ updated_human }} ({{ tz }}). Universe scanned: {{ universe_count }} tickers. Ranked by <b>{{ rank_horizon|upper }}</b>.</div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Combined Top {{ top_n }} (Stocks + ETFs)</h2>
    <div class="muted"><a href="top_combined.csv">CSV</a> · <a href="top_combined.json">JSON</a></div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Name</th>
          <th class="price">Price</th><th class="pct">Δ (abs)</th><th class="pct">Δ%</th>
          <th class="pct">Day</th><th class="pct">Month</th><th class="pct">YTD</th>
        </tr>
      </thead>
      <tbody>
        {% for r in rows %}
        <tr>
          <td>{{ loop.index }}</td>
          <td><a href="https://finance.yahoo.com/quote/{{ r['Ticker'] }}/" target="_blank" rel="noopener">{{ r['Ticker'] }}</a></td>
          <td>{{ r['Name'] }}</td>
          <td class="price">${{ r['Price'] }}</td>
          <td class="pct {{ 'gain' if r['Chg'] >= 0 else 'loss' }}">{{ r['Chg_str'] }}</td>
          <td class="pct {{ 'gain' if r['ChgPct'] >= 0 else 'loss' }}">{{ r['ChgPct_str'] }}</td>
          <td class="pct {{ 'gain' if r['Day'] >= 0 else 'loss' }}">{{ r['Day_str'] }}</td>
          <td class="pct {{ 'gain' if r['Month'] >= 0 else 'loss' }}">{{ r['Month_str'] }}</td>
          <td class="pct {{ 'gain' if r['YTD'] >= 0 else 'loss' }}">{{ r['YTD_str'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Top {{ top_n }} — Day / Month / YTD</h2>
    <img class="chart" src="charts/{{ chart_filename }}" alt="Grouped bar chart of Day/Month/YTD returns for Top {{ top_n }}">
  </div>

  <footer class="muted">
    Built by GitHub Actions with yfinance & matplotlib. Not investment advice. Data may be delayed or adjusted.
  </footer>
</body>
</html>
    """.strip())
    return template.render(**ctx)


def fmt_pct(x):  return "—" if pd.isna(x) else f"{float(x):+.2%}"
def fmt_price(x): return "—" if pd.isna(x) else f"{float(x):,.2f}"
def fmt_abs(x):  return "—" if pd.isna(x) else f"{float(x):+,.2f}"


def main():
    clean_output_dir()

    stocks = read_tickers(STOCKS_FILE, default=["AAPL","MSFT","NVDA","AMZN","GOOGL","META"])
    etfs   = read_tickers(ETFS_FILE,   default=["SPY","QQQ","DIA","IWM","TLT","SMH","ARKK"])
    universe = list(dict.fromkeys(stocks + etfs))  # dedupe, keep order

    histories = fetch_histories(universe)

    # Compute metrics
    rows = []
    for t, s in histories.items():
        price, prev, chg_abs, chg_pct = compute_point_changes(s)
        day   = ret_over_lookback(s, LOOKBACKS["day"])
        month = ret_over_lookback(s, LOOKBACKS["month"])
        ytd   = ret_ytd(s)
        rows.append({"Ticker": t, "Price": price, "Chg": chg_abs, "ChgPct": chg_pct,
                     "Day": day, "Month": month, "YTD": ytd})

    df = pd.DataFrame(rows)

    # Names
    names = names_for_tickers(df["Ticker"].tolist()) if not df.empty else {}
    df["Name"] = df["Ticker"].map(names).fillna(df["Ticker"])

    # Rank column
    rank_col = {"day": "Day", "month": "Month", "ytd": "YTD"}.get(RANK_HORIZON.lower(), "Month")

    # Top-10 combined
    df_top = df[~df[rank_col].isna()].sort_values(by=rank_col, ascending=False).head(TOP_N).copy()

    # Save machine-readable
    to_csv_and_json(df_top[["Ticker","Name","Price","Chg","ChgPct","Day","Month","YTD"]], "top_combined")

    # Chart
    chart_path = os.path.join(CHART_DIR, TOP10_CHART_FILENAME)
    build_chart(df_top[["Ticker","Day","Month","YTD"]], chart_path)

    # HTML rows (pretty)
    rows_html = []
    for _, r in df_top.iterrows():
        price = float(r["Price"]) if pd.notna(r["Price"]) else np.nan
        chg   = float(r["Chg"]) if pd.notna(r["Chg"]) else np.nan
        chgp  = float(r["ChgPct"]) if pd.notna(r["ChgPct"]) else np.nan
        d     = float(r["Day"]) if pd.notna(r["Day"]) else np.nan
        m     = float(r["Month"]) if pd.notna(r["Month"]) else np.nan
        y     = float(r["YTD"]) if pd.notna(r["YTD"]) else np.nan
        rows_html.append({
            "Ticker": r["Ticker"], "Name": r["Name"],
            "Price": fmt_price(price),
            "Chg": chg if not np.isnan(chg) else 0.0, "Chg_str": fmt_abs(chg),
            "ChgPct": chgp if not np.isnan(chgp) else 0.0, "ChgPct_str": fmt_pct(chgp),
            "Day": d if not np.isnan(d) else 0.0, "Day_str": fmt_pct(d),
            "Month": m if not np.isnan(m) else 0.0, "Month_str": fmt_pct(m),
            "YTD": y if not np.isnan(y) else 0.0, "YTD_str": fmt_pct(y),
        })

    # HTML
    updated_human = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html({
        "title": TITLE,
        "updated_human": updated_human,
        "tz": TZ,
        "universe_count": len(histories),
        "rank_horizon": rank_col,
        "top_n": TOP_N,
        "rows": rows_html,
        "chart_filename": TOP10_CHART_FILENAME
    })
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Debug snapshot
    snap = df.copy()
    for c in ["ChgPct","Day","Month","YTD"]:
        if c in snap:
            snap[c] = (snap[c] * 100.0).round(3)
    snap.to_csv(os.path.join(OUT_DIR, "returns_snapshot.csv"), index=False)

    print(f"✅ Combined-only site generated (ranked by {rank_col}).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
