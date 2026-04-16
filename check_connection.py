#!/usr/bin/env python3
"""
Alpaca Connection & Trade Readiness Check
==========================================
Run this before starting live_trader.py to verify:
  1. .env file exists with real API credentials
  2. Alpaca paper trading API is reachable
  3. Account is active with sufficient buying power
  4. Market calendar — is today a trading day, and is it open?
  5. SPY data feed is accessible

Usage:
  python check_connection.py
"""

import json
import os
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import pandas as pd
import pytz
from dotenv import load_dotenv

ET = pytz.timezone('America/New_York')

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(msg):      print(f"  {GREEN}[OK]{RESET}  {msg}")
def warn(msg):    print(f"  {YELLOW}[WARN]{RESET} {msg}")
def fail(msg):    print(f"  {RED}[FAIL]{RESET} {msg}")
def section(t):   print(f"\n{'─'*55}\n  {t}\n{'─'*55}")

errors = 0

# ── 1. Credentials (.env) ─────────────────────────────────────────────────────

section("1. Credentials (.env)")

env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    fail(".env file not found.")
    print("\n  Create it from the example:\n"
          "    cp .env.example .env\n"
          "  Then fill in your Alpaca paper trading keys.\n")
    sys.exit(1)

load_dotenv(env_path)

api_key    = os.getenv('ALPACA_API_KEY',    '')
api_secret = os.getenv('ALPACA_API_SECRET', '')

PLACEHOLDER_KEYS    = {'PKXXXXXXXXXXXXXXXXXX', 'api_key', ''}
PLACEHOLDER_SECRETS = {'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'api_secret', ''}

if api_key in PLACEHOLDER_KEYS:
    fail("ALPACA_API_KEY is missing or still a placeholder in .env")
    sys.exit(1)
if api_secret in PLACEHOLDER_SECRETS:
    fail("ALPACA_API_SECRET is missing or still a placeholder in .env")
    sys.exit(1)

ok(f"ALPACA_API_KEY    loaded  ({api_key[:4]}{'*' * (len(api_key) - 4)})")
ok(f"ALPACA_API_SECRET loaded  ({api_secret[:4]}{'*' * (len(api_secret) - 4)})")


# ── 2. Trading API ────────────────────────────────────────────────────────────

section("2. Alpaca Trading API (paper)")

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError as e:
    fail(f"alpaca-py not installed: {e}")
    print("\n  Run:  pip install -r requirements_live.txt\n")
    sys.exit(1)

try:
    trading = TradingClient(api_key, api_secret, paper=True)
    account = trading.get_account()
except Exception as e:
    fail(f"Could not connect to Alpaca: {e}")
    print("\n  Check your keys are for PAPER trading:\n"
          "  https://app.alpaca.markets/paper-trading\n")
    sys.exit(1)

ok("Connected to Alpaca paper trading API")

status = str(account.status)
if status.upper() == 'ACTIVE':
    ok(f"Account status:  {status}")
else:
    warn(f"Account status:  {status}  (expected ACTIVE)")
    errors += 1

portfolio_value = float(account.portfolio_value)
buying_power    = float(account.buying_power)
ok(f"Portfolio value: ${portfolio_value:,.2f}")

if buying_power < 1000:
    warn(f"Buying power:    ${buying_power:,.2f}  (very low — positions may not size correctly)")
    errors += 1
else:
    ok(f"Buying power:    ${buying_power:,.2f}")

shorting_enabled = getattr(account, 'shorting_enabled', True)
if not shorting_enabled:
    warn("Short selling DISABLED — IMP short signals will be skipped")
else:
    ok("Short selling:   enabled")

pdt = getattr(account, 'pattern_day_trader', False)
if pdt:
    warn("Account flagged as Pattern Day Trader (PDT)")
else:
    ok("PDT flag:        not set")


# ── 3. Market calendar ─────────────────────────────────────────────────────────

section("3. Market Calendar")

now_et = datetime.now(ET)
today  = now_et.date()
print(f"  Current time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")

market_open_today = False
session_state     = "unknown"

try:
    cal_req  = GetCalendarRequest(start=str(today), end=str(today))
    calendar = trading.get_calendar(cal_req)
except Exception as e:
    warn(f"Could not fetch market calendar: {e}")
    calendar = []

if not calendar:
    warn(f"No trading session for today ({today}) — likely a holiday or weekend")
    errors += 1
else:
    day = calendar[0]
    ok(f"Today ({today}) is a trading day  |  session {day.open}–{day.close} ET")
    market_open_today = True

    now_time = now_et.time()
    if now_time < dtime(9, 30):
        session_state = "pre_market"
        ok("Market status:  pre-market — run live_trader.py now to self-schedule")
    elif dtime(9, 30) <= now_time <= dtime(15, 59):
        session_state = "open"
        ok("Market status:  OPEN")
    else:
        session_state = "closed"
        warn("Market status:  already closed for today")


# ── 4. SPY data feed ───────────────────────────────────────────────────────────

section("4. SPY Data Feed (SIP)")

try:
    data_client = StockHistoricalDataClient(api_key, api_secret)
    req = StockBarsRequest(
        symbol_or_symbols='SPY',
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=now_et - timedelta(days=7),
        end=now_et,
        adjustment='raw',
        feed='sip'
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.loc['SPY']
    if df.empty:
        warn("SPY data returned 0 bars — possibly outside data availability hours")
    else:
        last_close = float(df['close'].iloc[-1])
        last_date  = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
        ok(f"SPY data feed OK — {len(df)} bar(s) returned")
        ok(f"Last close:      ${last_close:.2f}  ({last_date})")
except Exception as e:
    fail(f"SPY data fetch failed: {e}")
    errors += 1


# ── 5. Open positions ──────────────────────────────────────────────────────────

section("5. Current Positions")

try:
    positions = trading.get_all_positions()
    if not positions:
        ok("No open positions (flat)")
    else:
        for pos in positions:
            side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
            ok(f"{pos.symbol}: {side} {pos.qty} shares  "
               f"(mkt value ${float(pos.market_value):,.2f}, "
               f"unrealised P&L ${float(pos.unrealized_pl):+,.2f})")
except Exception as e:
    warn(f"Could not fetch positions: {e}")


# ── 6. Adaptive parameters ────────────────────────────────────────────────────

section("6. Adaptive Parameters (params.json)")

params_path = Path(__file__).parent / 'params.json'
if params_path.exists():
    try:
        p = json.load(open(params_path))
        inner = p.get('params', p)
        ok("params.json found — adaptive parameters active:")
        for k, v in inner.items():
            print(f"       {k} = {v}")
    except Exception as e:
        warn(f"params.json exists but could not be parsed: {e}")
else:
    ok("params.json absent — strategy will use hardcoded defaults")


# ── Summary ────────────────────────────────────────────────────────────────────

section("Summary")

if not market_open_today:
    print(f"  {YELLOW}NOT A TRADING DAY{RESET} — no session today. "
          "live_trader.py will idle until the next open.")
elif session_state == "pre_market":
    print(f"  {GREEN}READY{RESET} — pre-market. "
          "Start live_trader.py now; it will self-schedule.")
elif session_state == "open":
    print(f"  {GREEN}READY{RESET} — market is open. "
          "You can start live_trader.py immediately.")
elif session_state == "closed":
    print(f"  {YELLOW}MARKET CLOSED{RESET} — today's session has ended. "
          "live_trader.py will wait until tomorrow.")

if errors:
    print(f"\n  {YELLOW}{errors} warning(s) above need attention before trading.{RESET}")
else:
    print(f"\n  {GREEN}All checks passed.{RESET}")

print()
