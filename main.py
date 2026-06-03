import pytz
import argparse
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
import uvicorn
from fastapi import FastAPI, BackgroundTasks
import requests
import threading
import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")


# ─── Constants ───────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
NSE_OPEN_HOUR, NSE_OPEN_MIN = 9, 15
NSE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ─── Symbol Fetchers ───────────────────────────────────────────────

def get_nifty50_symbols() -> list[str]:
    nifty50 = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
        "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "NESTLEIND",
        "ULTRACEMCO", "BAJFINANCE", "WIPRO", "HCLTECH", "ONGC", "NTPC",
        "TATASTEEL", "JSWSTEEL", "POWERGRID", "COALINDIA", "ADANIENT",
        "ADANIPORTS", "HINDALCO", "BAJAJFINSV", "DRREDDY", "CIPLA",
        "EICHERMOT", "TECHM", "HEROMOTOCO", "DIVISLAB", "BRITANNIA",
        "GRASIM", "INDUSINDBK", "BAJAJ-AUTO", "APOLLOHOSP", "TATACONSUM",
        "LTIM", "SBILIFE", "HDFCLIFE", "BPCL", "M&M", "TATAMOTORS", "UPL",
    ]
    return [f"{s}.NS" for s in nifty50]


def get_nifty500_symbols() -> list[str]:
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url, storage_options={"User-Agent": NSE_HEADERS["User-Agent"]})
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        symbols = df[col].str.strip().tolist()
        print(f"[✓] Loaded {len(symbols)} Nifty 500 symbols from NSE archives.")
        return [f"{s}.NS" for s in symbols]
    except Exception as e:
        print(f"[!] Could not fetch Nifty 500: {e}  → Falling back to Nifty 50.")
        return get_nifty50_symbols()


def get_all_nse_symbols() -> list[str]:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url, storage_options={"User-Agent": NSE_HEADERS["User-Agent"]})
        col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        symbols = df[col].str.strip().tolist()
        print(f"[✓] Loaded {len(symbols)} NSE equity symbols.")
        return [f"{s}.NS" for s in symbols]
    except Exception as e:
        print(f"[!] Could not fetch all NSE symbols: {e}  → Falling back to Nifty 500.")
        return get_nifty500_symbols()


# ─── ORB Check (with Fake-Breakout Filters) ───────────────────────────────────────────────

def check_orb(
    ticker: str,
    orb_minutes: int = 15,
    min_breakout_pct: float = 0.3,
    rvol_threshold: float = 1.5,
    min_or_range_pct: float = 0.2,
    require_sustained: bool = True,
    require_close_confirm: bool = True,
    require_volume_spike: bool = True,
) -> dict | None:
    """
    Checks if a stock has a HIGH-QUALITY ORB signal.

    Fake-breakout filters:
    ───────────────────────────────────────────────────────────────────
    Filter 1 – Close Confirmation  : A candle must CLOSE beyond OR level.
                                     Wick-only touches are ignored.
    Filter 2 – Minimum Breakout %  : Breakout must exceed `min_breakout_pct`.
                                     Eliminates noise / micro-moves.
    Filter 3 – Relative Volume     : Today's cumulative volume must be
                                     ≥ rvol_threshold × 20-day avg volume.
                                     Low-volume breakouts are usually fake.
    Filter 4 – Sustained Breakout  : Current price must still be beyond the
                                     OR level. Ensures breakout hasn't reversed.
    Filter 5 – OR Range Width      : OR range must be ≥ min_or_range_pct.
                                     Flat/choppy opens produce bad signals.
    Filter 6 – Volume Spike        : The first breakout candle must have
                                     volume > 1.5× average intraday volume.
    ───────────────────────────────────────────────────────────────────
    """
    try:
        ist_now = datetime.now(IST)

        market_open_dt = ist_now.replace(
            hour=NSE_OPEN_HOUR, minute=NSE_OPEN_MIN, second=0, microsecond=0
        )
        or_end_dt = ist_now.replace(
            hour=NSE_OPEN_HOUR,
            minute=NSE_OPEN_MIN + orb_minutes,
            second=0,
            microsecond=0,
        )

        if ist_now < or_end_dt:
            return None  # Opening range not complete yet

        # ── Use yf.Ticker (thread-safe, isolated session per ticker) ──────────
        # NOTE: yf.download() shares internal state across threads causing
        # data from ticker-A to bleed into ticker-B. Ticker.history() is safe.
        t = yf.Ticker(ticker)

        # ── Fetch 1-min intraday data for today ───────────────────────────────
        raw = t.history(period="1d", interval="1m", auto_adjust=True)
        if raw.empty:
            return None

        # Validate: ensure we actually got data (yfinance can return stale cache)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = raw.index.tz_convert(IST)
        today = raw[raw.index.date == ist_now.date()]

        if today.empty or len(today) < orb_minutes:
            return None

        # Sanity-check: OHLC columns must exist and have valid data
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(today.columns):
            return None
        if today[["Open", "High", "Low", "Close"]].isnull().all().any():
            return None

        # ── Opening Range ─────────────────────────────────────────────────────
        or_candles = today[(today.index >= market_open_dt) & (today.index < or_end_dt)]
        if or_candles.empty:
            return None

        or_high = float(or_candles["High"].max())
        or_low  = float(or_candles["Low"].min())

        # Guard against bad data (e.g. all-zero rows from API errors)
        if or_high <= 0 or or_low <= 0 or or_high <= or_low:
            return None

        # ─ Filter 5: OR Range Width ───────────────────────────────────────────
        or_range_pct = ((or_high - or_low) / or_low) * 100
        if or_range_pct < min_or_range_pct:
            return None  # OR too tight → likely flat/choppy open

        # ── Post-OR candles ───────────────────────────────────────────────────
        post_or = today[today.index >= or_end_dt]
        if post_or.empty:
            return None

        current_price  = float(post_or["Close"].iloc[-1])
        avg_candle_vol = float(today["Volume"].mean())
        if avg_candle_vol <= 0:
            return None

        # ── Average daily volume (20-day) for RVOL ────────────────────────────
        # Re-use the same Ticker object — no second download needed
        hist = t.history(period="25d", interval="1d", auto_adjust=True)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            # Exclude today from the avg (use only completed trading days)
            past = hist[hist.index.date < ist_now.date()]
            avg_daily_vol = float(past["Volume"].tail(20).mean()) if not past.empty else 0.0
        else:
            avg_daily_vol = 0.0

        today_vol = float(today["Volume"].sum())
        rvol      = (today_vol / avg_daily_vol) if avg_daily_vol > 0 else 0.0

        # ─ Filter 3: Relative Volume ──────────────────────────────────────────
        if rvol < rvol_threshold:
            return None

        # ── Signal Detection ──────────────────────────────────────────────────
        signal             = None
        breakout_pct       = 0.0
        breakout_candle_ok = True

        # ─ Bullish ORB ────────────────────────────────────────────────────────
        bullish_closes = post_or[post_or["Close"] > or_high]
        if not bullish_closes.empty:
            if require_close_confirm:
                first_bo = bullish_closes.iloc[0]
            else:
                candidates = post_or[post_or["High"] > or_high]
                if candidates.empty:
                    first_bo = bullish_closes.iloc[0]
                else:
                    first_bo = candidates.iloc[0]

            bp = ((float(first_bo["Close"]) - or_high) / or_high) * 100

            # Filter 2: Minimum breakout %
            if bp >= min_breakout_pct:
                # Filter 4: Sustained breakout (price hasn't reversed back inside OR)
                if not require_sustained or current_price > or_high:
                    # Filter 6: Volume spike on the breakout candle
                    if require_volume_spike:
                        bvol = float(first_bo["Volume"])
                        breakout_candle_ok = bvol > (avg_candle_vol * 1.5)

                    if breakout_candle_ok:
                        signal       = "BULLISH ORB"
                        breakout_pct = round(bp, 2)

        # ─ Bearish ORB (only if no bullish signal found) ─────────────────────
        if signal is None:
            bearish_closes = post_or[post_or["Close"] < or_low]
            if not bearish_closes.empty:
                if require_close_confirm:
                    first_bo = bearish_closes.iloc[0]
                else:
                    candidates = post_or[post_or["Low"] < or_low]
                    if candidates.empty:
                        first_bo = bearish_closes.iloc[0]
                    else:
                        first_bo = candidates.iloc[0]

                bp = ((or_low - float(first_bo["Close"])) / or_low) * 100

                if bp >= min_breakout_pct:
                    if not require_sustained or current_price < or_low:
                        if require_volume_spike:
                            bvol = float(first_bo["Volume"])
                            breakout_candle_ok = bvol > (avg_candle_vol * 1.5)

                        if breakout_candle_ok:
                            signal       = "BEARISH ORB"
                            breakout_pct = round(bp, 2)

        if signal is None:
            return None

        return {
            "Ticker"     : ticker.replace(".NS", ""),
            "Signal"     : signal,
            "OR High"    : round(or_high, 2),
            "OR Low"     : round(or_low, 2),
            "OR Range %" : round(or_range_pct, 2),
            "Cur. Price" : round(current_price, 2),
            "Breakout %" : breakout_pct,
            "RVOL"       : round(rvol, 2),
            "Volume"     : f"{int(today_vol):,}",
        }

    except Exception:
        return None


# ─── Scanner ───────────────────────────────────────────────

def scan_orb(
    symbols: list[str],
    orb_minutes: int = 15,
    max_workers: int = 15,
    min_breakout_pct: float = 0.3,
    rvol_threshold: float = 1.5,
    min_or_range_pct: float = 0.2,
    require_sustained: bool = True,
    require_close_confirm: bool = True,
    require_volume_spike: bool = True,
) -> pd.DataFrame:
    results  = []
    total    = len(symbols)
    completed = 0

    print(f"\n[→] Scanning {total} stocks for {orb_minutes}-min ORB signals...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                check_orb,
                sym,
                orb_minutes,
                min_breakout_pct,
                rvol_threshold,
                min_or_range_pct,
                require_sustained,
                require_close_confirm,
                require_volume_spike,
            ): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                results.append(result)
            pct = int((completed / total) * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {completed}/{total}", end="", flush=True)

    print("\n")
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["_sort"] = df["Signal"].apply(lambda x: 0 if "BULLISH" in x else 1)
    df = (
        df.sort_values(["_sort", "Breakout %", "RVOL"], ascending=[True, False, False])
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )
    return df


# ─── Market Hours Guard ───────────────────────────────────────────────

def is_market_hours() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ─── CLI ───────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NSE ORB Scanner with Fake-Breakout Filters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--orb-minutes",       type=int,   default=15,
                   help="Opening range duration in minutes")
    p.add_argument("--index",             choices=["nifty50", "nifty500", "all"],
                   default="nifty500",    help="Index to scan")
    p.add_argument("--workers",           type=int,   default=15,
                   help="Parallel threads (lower = fewer API rate-limit issues)")

    # ── Fake-breakout filter knobs ───────────────────────────────────
    p.add_argument("--min-breakout",      type=float, default=0.3,
                   help="[Filter 2] Min breakout %% beyond OR level (0.3 = 0.3%%)")
    p.add_argument("--rvol",              type=float, default=1.5,
                   help="[Filter 3] Relative volume threshold vs 20-day avg (1.5 = 1.5×)")
    p.add_argument("--min-or-range",      type=float, default=0.2,
                   help="[Filter 5] Min OR range width %% (0.2 = 0.2%%)")
    p.add_argument("--no-close-confirm",  action="store_true",
                   help="[Filter 1] Disable close-confirmation (allow wick touches)")
    p.add_argument("--no-sustained",      action="store_true",
                   help="[Filter 4] Disable sustained-breakout check")
    p.add_argument("--no-vol-spike",      action="store_true",
                   help="[Filter 6] Disable breakout-candle volume spike check")

    p.add_argument("--no-market-check",   action="store_true",
                   help="Skip market hours check (for testing)")
    p.add_argument("--save-csv",          type=str,   default="",
                   help="Save results to CSV file")
    args, unknown = p.parse_known_args() # Changed this line to use parse_known_args()
    return args


app = FastAPI(title="NSE ORB Scanner", version="1.0")

def run_scanner_in_background(webhook_url: str):
    """Runs completely unthrottled on Render"""
    try:
        # 1. Get your full symbol list
        symbols = get_nifty50_symbols() # Change to your 900+ stock list variable/function
        
        print(f"[Render Engine] Starting heavy scan for {len(symbols)} stocks...")
        
        # 2. SAFETY FIX: Bulk download historical data in ONE shot instead of a loop
        # This bypasses Yahoo Finance rate limits completely and takes seconds!
        print("[Render Engine] Bulk downloading market data...")
        data = yf.download(
            tickers=symbols,
            period="2d",      # We only need today and yesterday for ORB
            interval="15m",   # Your 15-minute ORB timeframe
            group_by='ticker',
            timeout=10,       # Force kill hung connections after 10s
            threads=True      # Let yfinance handle the multi-threading internally
        )
        
        print("[Render Engine] Processing technical indicators...")
        # 3. Pass this pre-downloaded data block straight into your custom ORB logic
        # (Make sure to modify your scan_orb function to read from this data object, 
        # or use your existing loop but wrap yf.Ticker(symbol).history with a timeout)
        df = scan_orb(
            symbols,
            orb_minutes=15,
            max_workers=10,
            min_breakout_pct=0.3,
            rvol_threshold=1.5,
            min_or_range_pct=0.2,
            require_sustained=True,
            require_close_confirm=True,
            require_volume_spike=True
        )
        
        if df is None or df.empty:
            payload = {"status": "success", "count": 0, "alerts": []}
        else:
            payload = {"status": "success", "count": len(df), "alerts": df.to_dict(orient="records")}
            
    except Exception as e:
        print(f"[Render Engine] CRITICAL ERROR: {str(e)}")
        payload = {"status": "failed", "error": str(e)}

    # 4. Fire the results straight back to n8n
    try:
        print(f"[Render Engine] Sending results to n8n webhook...")
        requests.post(webhook_url, json=payload, timeout=30)
        print("[Render Engine] Webhook delivered successfully!")
    except Exception as webhook_err:
        print(f"[Render Engine] Failed to ping n8n back: {webhook_err}")


@app.get("/scan")
def trigger_scan(webhook: str, background_tasks: BackgroundTasks):
    """Instantly releases n8n so it doesn't wait around"""
    background_tasks.add_task(run_scanner_in_background, webhook_url=webhook)
    return {"status": "processing", "message": "Render background engine initialized."}


# ─── Main ───────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  ↗  NSE Opening Range Breakout (ORB) Scanner  v2")
    print("=" * 65)
    print(f"  Index           : {args.index.upper()}")
    print(f"  ORB Window      : {args.orb_minutes} min  (from 09:15 IST)")
    print(f"  Threads         : {args.workers}")
    print(f"  Scan Time       : {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("─" * 65)
    print("  Filters Active:")
    print(f"  ✔ [F1] Close confirmation  : {'OFF' if args.no_close_confirm else 'ON  ← wick-only touches rejected'}")
    print(f"  ✔ [F2] Min breakout        : {args.min_breakout:.1f}%%")
    print(f"  ✔ [F3] Relative volume     : ≥ {args.rvol:.1f}× 20-day avg")
    print(f"  ✔ [F4] Sustained breakout  : {'OFF' if args.no_sustained else 'ON  ← reversal after breakout rejected'}")
    print(f"  ✔ [F5] Min OR range        : ≥ {args.min_or_range:.1f}%%")
    print(f"  ✔ [F6] Volume spike candle : {'OFF' if args.no_vol_spike else 'ON  ← breakout candle must have high vol'}")
    print("=" * 65)

#    if not args.no_market_check and not is_market_hours():
#        print("\n[!] NSE is currently CLOSED.")
#        print("    Re-run with --no-market-check to scan anyway.")
#        return

    # ── Load symbols ───────────────────────────────────
    if args.index == "nifty50":
        symbols = get_nifty50_symbols()
    elif args.index == "nifty500":
        symbols = get_nifty500_symbols()
    else:
        symbols = get_all_nse_symbols()

    # ── Run scan ───────────────────────────────────
    df = scan_orb(
        symbols,
        orb_minutes          = args.orb_minutes,
        max_workers          = args.workers,
        min_breakout_pct     = args.min_breakout,
        rvol_threshold       = args.rvol,
        min_or_range_pct     = args.min_or_range,
        require_sustained    = not args.no_sustained,
        require_close_confirm= not args.no_close_confirm,
        require_volume_spike = not args.no_vol_spike,
    )

    # ── Display results ───────────────────────────────────
    print("=" * 65)
    print(f"  ✅  Results  |  {datetime.now(IST).strftime('%H:%M IST')}")
    print("=" * 65)

    if df.empty:
        print("\n  No high-quality ORB breakouts detected.")
        print("  → Try loosening filters:  --min-breakout 0.1  --rvol 1.0\n")
    else:
        print(f"\n  Found {len(df)} genuine ORB signal(s):\n")
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=True))

        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"\n  [▇] Saved to: {args.save_csv}")

    print("\n" + "=" * 65)

    if not df.empty:
        bullish = df[df["Signal"].str.contains("BULLISH")]
        bearish = df[df["Signal"].str.contains("BEARISH")]
        print(f"  ╠ Bullish ORB : {len(bullish)} stock(s)")
        print(f"  ╠ Bearish ORB : {len(bearish)} stock(s)")
        print("=" * 65 + "\n")


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)