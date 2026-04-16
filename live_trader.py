#!/usr/bin/env python3
"""
SPY Intraday Momentum Strategy — Paper Trading Live Execution
=============================================================
Based on: "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
Zarattini, Aziz, Barbon (2024) — Swiss Finance Institute Research Paper No. 24-97
https://ssrn.com/abstract=4824172

TWO SIMULTANEOUS SUB-STRATEGIES:
  IMP (Intraday Momentum Portfolio)  — rides momentum signals every 30 min
  MRP (Mean Reversion Portfolio)     — fades overnight gaps > 2% (enter open, exit 10:00 AM)

HOW TO RUN:
  1. Add your Alpaca Paper Trading keys to .env  (see .env.example)
  2. pip install -r requirements_live.txt
  3. python live_trader.py
  4. Leave it running during market hours. It self-schedules everything.
"""

import os
import math
import time
import logging
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ─── Strategy Parameters (defaults from the Zarattini paper) ─────────────────
# These may be overridden by params.json (written by post_session_analyzer.py)
SYMBOL           = 'SPY'
TRADE_FREQ_MIN   = 30        # Evaluate signal every N minutes
BAND_MULT        = 1.0       # Band multiplier (1× sigma_open)
TARGET_VOL       = 0.02      # Daily vol target for position sizing (2%)
MAX_LEVERAGE     = 4.0       # Hard leverage cap
OVERNIGHT_THRESH = 0.02      # Min overnight gap magnitude for MRP trade (2%)
COMMISSION       = 0.0035    # Per-share commission (IB-style)
MIN_COMMISSION   = 0.35      # Min commission per order
SIGMA_LOOKBACK   = 14        # Rolling window (days) for sigma_open
VOL_LOOKBACK     = 15        # Rolling window (days) for daily vol
HIST_DAYS        = 50        # Calendar days of history to fetch pre-market
DAILY_TP_PCT     = 0.20      # Take profit: exit all if daily P&L hits +20%

# ─── Load Adaptive Parameters (from learning pipeline) ───────────────────────
import json
PARAMS_FILE = Path(__file__).parent / 'params.json'
if PARAMS_FILE.exists():
    try:
        _p = json.load(open(PARAMS_FILE))
        if 'params' in _p:
            _p = _p['params']
        BAND_MULT        = _p.get('BAND_MULT', BAND_MULT)
        TARGET_VOL       = _p.get('TARGET_VOL', TARGET_VOL)
        MAX_LEVERAGE     = _p.get('MAX_LEVERAGE', MAX_LEVERAGE)
        OVERNIGHT_THRESH = _p.get('OVERNIGHT_THRESH', OVERNIGHT_THRESH)
        SIGMA_LOOKBACK   = int(_p.get('SIGMA_LOOKBACK', SIGMA_LOOKBACK))
        VOL_LOOKBACK     = int(_p.get('VOL_LOOKBACK', VOL_LOOKBACK))
    except Exception:
        pass  # Fall back to defaults if params.json is corrupted

ET = pytz.timezone('America/New_York')

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = Path('./logs')
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"trader_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def now_et() -> datetime:
    return datetime.now(ET)

def time_et() -> dtime:
    return now_et().time()

def is_market_open() -> bool:
    t = time_et()
    return dtime(9, 30) <= t <= dtime(15, 59)

def minutes_from_open(dt: datetime = None) -> float:
    """Minutes elapsed since 9:30 AM ET (1-based, matching process.py)."""
    if dt is None:
        dt = now_et()
    return (dt.hour * 60 + dt.minute) - (9 * 60 + 30) + 1


# ─── Main Trader Class ────────────────────────────────────────────────────────

class SPYIntradayTrader:
    """
    Live execution engine for the SPY Intraday Momentum strategy.
    Runs fully autonomously during market hours.
    """

    def __init__(self):
        load_dotenv()
        api_key    = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError(
                "Missing ALPACA_API_KEY or ALPACA_API_SECRET.\n"
                "Copy .env.example → .env and fill in your paper trading keys."
            )

        # paper=True → routes to paper-api.alpaca.markets
        self.trading = TradingClient(api_key, api_secret, paper=True)
        self.data    = StockHistoricalDataClient(api_key, api_secret)

        # ── Daily state (reset each session) ──
        self.open_price     = None   # Today's SPY open
        self.prev_close     = None   # Yesterday's adjusted close
        self.sigma_map      = {}     # {check_minute: sigma_open_value}
        self.spx_vol        = None   # 15-day rolling daily vol
        self.aum            = self._get_portfolio_value()
        self.imp_shares     = 0      # Current IMP position (+long / -short)
        self.mrp_shares     = 0      # Current MRP position
        self.mrp_exited     = False  # Flag: MRP already closed today
        self.signal_log     = []     # Intraday signal history
        self.last_check_min = -1     # Prevent double-firing at same minute
        self.start_aum      = self.aum  # Snapshot AUM at session start for TP calc
        self.tp_triggered   = False  # Flag: daily take-profit hit

        log.info(f"Trader initialized | Paper AUM: ${self.aum:,.2f}")
        log.info(f"Strategy params: BAND_MULT={BAND_MULT}, TARGET_VOL={TARGET_VOL}, "
                 f"MAX_LEV={MAX_LEVERAGE}, OVN_THRESH={OVERNIGHT_THRESH}")

    # ── Account / Position Helpers ────────────────────────────────────────────

    def _get_portfolio_value(self) -> float:
        return float(self.trading.get_account().portfolio_value)

    def _get_position_shares(self) -> int:
        """Returns current net SPY shares (positive=long, negative=short, 0=flat)."""
        try:
            pos = self.trading.get_open_position(SYMBOL)
            qty = int(pos.qty)
            side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
            return qty if side == 'long' else -qty
        except Exception:
            return 0

    def _place_order(self, side: OrderSide, qty: int, reason: str):
        if qty <= 0:
            log.warning(f"Skipping order with qty={qty}")
            return None
        log.info(f"  ▶ ORDER: {side.value.upper()} {qty} {SYMBOL}  [{reason}]")
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            ))
            log.info(f"    Order ID: {order.id} | Status: {order.status}")
            return order
        except Exception as e:
            log.error(f"    Order FAILED: {e}")
            return None

    def _adjust_to_target(self, target_shares: int, book: str):
        """
        Move from current position to target_shares.
        book = 'IMP' or 'MRP' — for logging only.
        Reads actual broker position before trading to stay accurate.
        """
        current = self._get_position_shares()
        delta   = target_shares - current

        if delta == 0:
            log.info(f"  {book}: No change needed (already at {current:+d})")
            return

        log.info(f"  {book}: {current:+d} → {target_shares:+d} (delta {delta:+d})")

        if delta > 0:
            self._place_order(OrderSide.BUY, abs(delta), f"{book} signal")
        else:
            self._place_order(OrderSide.SELL, abs(delta), f"{book} signal")

    # ── Data Fetching ─────────────────────────────────────────────────────────

    def _fetch_bars(self, timeframe: TimeFrame, days_back: int) -> pd.DataFrame:
        """Fetch historical bars, return DataFrame indexed by ET datetime."""
        end   = now_et()
        start = end - timedelta(days=days_back)
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment='raw',
            feed='sip'
        )
        bars = self.data.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[SYMBOL]
        df.index = pd.to_datetime(df.index).tz_convert(ET)
        return df

    def _today_bars(self) -> pd.DataFrame:
        """Fetch today's 1-min bars during market hours."""
        df = self._fetch_bars(TimeFrame(1, TimeFrameUnit.Minute), days_back=2)
        today = now_et().date()
        df = df[df.index.date == today]
        df = df.between_time('09:30', '15:59')
        return df

    # ── Pre-Market Preparation ────────────────────────────────────────────────

    def prepare_market(self):
        """
        Run at 9:00 AM ET. Computes:
          - sigma_open for each 30-min check time  (the band threshold)
          - prev_close  (yesterday's close)
          - spx_vol     (15-day rolling daily vol for position sizing)
        """
        log.info("=" * 60)
        log.info("PRE-MARKET PREPARATION")
        log.info("=" * 60)

        # ── 1. Fetch intraday history ──────────────────────────────────────
        log.info(f"Fetching {HIST_DAYS}d of 1-min {SYMBOL} bars...")
        df_1m = self._fetch_bars(TimeFrame(1, TimeFrameUnit.Minute), HIST_DAYS)
        df_1m = df_1m.between_time('09:30', '15:59')
        df_1m['day'] = df_1m.index.date

        # Day open = first bar's open price
        day_opens      = df_1m.groupby('day')['open'].first()
        df_1m['day_open'] = df_1m['day'].map(day_opens)

        # move_open = |close / day_open - 1|  (absolute intraday move from open)
        df_1m['move_open'] = (df_1m['close'] / df_1m['day_open'] - 1).abs()

        # minute_of_day: 1-based from 9:30 (matches process.py formula)
        df_1m['min_from_open'] = (
            (df_1m.index - df_1m.index.normalize()) / pd.Timedelta(minutes=1)
            - (9 * 60 + 30) + 1
        ).round().astype(int)

        today = now_et().date()

        # ── 2. Compute sigma_open for each 30-min check minute ────────────
        # Check minutes: 30, 60, 90, ..., 360  (= bars at 9:59, 10:29, ..., 15:59)
        check_minutes = list(range(30, 361, 30))
        self.sigma_map = {}

        for m in check_minutes:
            hist = df_1m[(df_1m['min_from_open'] == m) & (df_1m['day'] < today)]
            n = len(hist)
            if n >= SIGMA_LOOKBACK:
                sigma = float(hist['move_open'].tail(SIGMA_LOOKBACK).mean())
            elif n > 0:
                sigma = float(hist['move_open'].mean())
            else:
                sigma = 0.005  # fallback: 0.5%
            self.sigma_map[m] = sigma

        # Log a summary of sigmas for the key trading minutes
        log.info("sigma_open by check minute:")
        for m, s in self.sigma_map.items():
            hhmm = f"{9 + (m + 29) // 60}:{((m + 29) % 60):02d}"
            log.info(f"  min {m:3d} (~{hhmm}) → sigma={s:.4f}  "
                     f"UB/LB ±{s*100:.2f}% from open/prev_close")

        # ── 3. Daily bars for prev_close + rolling vol ─────────────────────
        df_d = self._fetch_bars(TimeFrame(1, TimeFrameUnit.Day), days_back=40)
        df_d = df_d[df_d.index.date < today]
        df_d['ret'] = df_d['close'].pct_change()

        self.prev_close = float(df_d['close'].iloc[-1])
        prev_date       = df_d.index[-1].date()

        if len(df_d) >= VOL_LOOKBACK:
            self.spx_vol = float(df_d['ret'].tail(VOL_LOOKBACK - 1).std())
        else:
            self.spx_vol = None

        log.info(f"Previous close ({prev_date}): ${self.prev_close:.2f}")
        log.info(f"15-day daily vol: {f'{self.spx_vol:.4f}' if self.spx_vol else 'N/A (will use max leverage)'}")
        log.info("Pre-market preparation complete. Ready for 9:30 AM open.")

    # ── Market Open ───────────────────────────────────────────────────────────

    def on_market_open(self):
        """
        9:30 AM: record open price, execute MRP trade if overnight gap > 2%.
        MRP logic: if gap > +2% → short (fade rally); if gap < -2% → long (fade selloff).
        Exit MRP at 10:00 AM regardless.
        """
        log.info("=" * 60)
        log.info("MARKET OPEN (9:30 AM)")
        log.info("=" * 60)

        # Give the first bar 45 seconds to form
        time.sleep(45)

        df = self._today_bars()
        if df.empty:
            log.error("Could not fetch open bar. Skipping market open logic.")
            return

        self.open_price   = float(df['open'].iloc[0])
        self.aum          = self._get_portfolio_value()
        self.start_aum    = self.aum   # Snapshot for take-profit calculation
        self.mrp_exited   = False
        self.tp_triggered = False
        self.last_check_min = -1

        ovn_move = (self.open_price / self.prev_close) - 1
        shares   = self._compute_shares()

        log.info(f"Open: ${self.open_price:.2f} | Prev close: ${self.prev_close:.2f} | "
                 f"Overnight gap: {ovn_move*100:+.2f}%")
        log.info(f"Position sizing → {shares} shares")

        if abs(ovn_move) > OVERNIGHT_THRESH:
            direction = int(-np.sign(ovn_move))  # fade the gap
            log.info(f"MRP: Gap {ovn_move*100:+.2f}% > threshold → "
                     f"{'LONG' if direction > 0 else 'SHORT'} {shares} shares")
            self._adjust_to_target(direction * shares, 'MRP')
            self.mrp_shares = direction * shares
        else:
            log.info(f"MRP: Gap {ovn_move*100:+.2f}% below {OVERNIGHT_THRESH*100:.0f}% "
                     f"threshold — no MRP trade")
            self.mrp_shares = 0

    # ── 10:00 AM: MRP Exit + First IMP Signal ────────────────────────────────

    def on_first_check(self):
        """
        10:00 AM: Close MRP position (it's a 30-min gap fade, always exits here),
        then run the first IMP signal check.
        """
        log.info("=" * 60)
        log.info("10:00 AM — MRP EXIT + FIRST IMP SIGNAL")
        log.info("=" * 60)

        # Close MRP
        if self.mrp_shares != 0 and not self.mrp_exited:
            log.info(f"MRP exit: closing {self.mrp_shares:+d} shares")
            self._adjust_to_target(0, 'MRP-exit')
            self.mrp_shares  = 0
            self.mrp_exited  = True
        else:
            log.info("MRP: nothing to close")

        # Small pause so MRP order clears
        time.sleep(3)

        # First IMP signal check
        self._imp_signal_check(check_min=30)

    # ── IMP Signal Check ──────────────────────────────────────────────────────

    def _compute_shares(self) -> int:
        """
        Position size using vol-targeting:
            shares = AUM / open_price × min(target_vol / 15d_vol, max_leverage)
        """
        if self.open_price is None:
            return 0
        if self.spx_vol and not math.isnan(self.spx_vol) and self.spx_vol > 0:
            leverage = min(TARGET_VOL / self.spx_vol, MAX_LEVERAGE)
        else:
            leverage = MAX_LEVERAGE  # fallback: full 4×
        return max(1, round(self.aum / self.open_price * leverage))

    def _compute_vwap(self, df: pd.DataFrame) -> float:
        hlc = (df['high'] + df['low'] + df['close']) / 3
        return float((hlc * df['volume']).sum() / df['volume'].sum())

    def _check_daily_take_profit(self) -> bool:
        """
        Check if daily P&L has hit +20%. If so, close everything and stop trading.
        Returns True if take-profit triggered (caller should skip further signals).
        """
        if self.tp_triggered:
            return True

        current_aum = self._get_portfolio_value()
        daily_pnl_pct = (current_aum - self.start_aum) / self.start_aum

        if daily_pnl_pct >= DAILY_TP_PCT:
            log.info("!" * 60)
            log.info(f"DAILY TAKE-PROFIT TRIGGERED: +{daily_pnl_pct*100:.1f}% "
                     f"(${current_aum - self.start_aum:+,.2f})")
            log.info(f"Start AUM: ${self.start_aum:,.2f} → Current: ${current_aum:,.2f}")
            log.info("Closing all positions and halting trading for today.")
            log.info("!" * 60)

            if self.imp_shares != 0:
                self._adjust_to_target(0, 'IMP-TP')
                self.imp_shares = 0
            if self.mrp_shares != 0:
                self._adjust_to_target(0, 'MRP-TP')
                self.mrp_shares = 0

            self.tp_triggered = True
            self.aum = self._get_portfolio_value()

            self.signal_log.append({
                'timestamp'  : now_et().isoformat(),
                'check_min'  : -1,
                'close'      : 0,
                'vwap'       : 0,
                'ub'         : 0,
                'lb'         : 0,
                'sigma'      : 0,
                'signal'     : 99,  # Special code for take-profit
                'imp_shares' : 0,
                'mrp_shares' : 0,
                'aum'        : round(self.aum, 2),
            })
            return True

        return False

    def _imp_signal_check(self, check_min: int):
        """
        Core IMP logic. Evaluate signal at check_min and adjust position.

        Signal rules (from the paper):
          Long  if close > UB AND close > VWAP
          Short if close < LB AND close < VWAP
          Flat  otherwise

        UB = max(open, prev_close) × (1 + sigma_open)
        LB = min(open, prev_close) × (1 - sigma_open)
        """
        # Check take-profit before evaluating signals
        if self._check_daily_take_profit():
            log.info("IMP check skipped: daily take-profit already triggered")
            return

        if self.open_price is None or self.prev_close is None:
            log.warning("IMP check skipped: open_price or prev_close not set")
            return

        sigma = self.sigma_map.get(check_min)
        if sigma is None:
            log.warning(f"IMP check skipped: no sigma_open for minute {check_min}")
            return

        df = self._today_bars()
        if df.empty:
            log.warning("IMP check skipped: no intraday bars")
            return

        current_close = float(df['close'].iloc[-1])
        vwap          = self._compute_vwap(df)

        UB = max(self.open_price, self.prev_close) * (1 + BAND_MULT * sigma)
        LB = min(self.open_price, self.prev_close) * (1 - BAND_MULT * sigma)

        if current_close > UB and current_close > vwap:
            new_signal = 1
            label = "LONG  (price above upper band + VWAP)"
        elif current_close < LB and current_close < vwap:
            new_signal = -1
            label = "SHORT (price below lower band + VWAP)"
        else:
            new_signal = 0
            label = "FLAT  (price within bands or against VWAP)"

        shares = self._compute_shares()
        target = new_signal * shares

        log.info(f"IMP Signal @ min {check_min}: {label}")
        log.info(f"  Close=${current_close:.2f} | VWAP=${vwap:.2f} | "
                 f"UB=${UB:.2f} | LB=${LB:.2f} | sigma={sigma:.4f}")
        log.info(f"  Target: {target:+d} shares (was {self.imp_shares:+d})")

        self._adjust_to_target(target, 'IMP')
        self.imp_shares = target

        # Log this check
        self.signal_log.append({
            'timestamp'  : now_et().isoformat(),
            'check_min'  : check_min,
            'close'      : round(current_close, 4),
            'vwap'       : round(vwap, 4),
            'ub'         : round(UB, 4),
            'lb'         : round(LB, 4),
            'sigma'      : round(sigma, 6),
            'signal'     : new_signal,
            'imp_shares' : self.imp_shares,
            'mrp_shares' : self.mrp_shares,
            'aum'        : round(self._get_portfolio_value(), 2),
        })

    # ── Force Close ───────────────────────────────────────────────────────────

    def on_force_close(self):
        """
        3:55 PM: Close all positions. No overnight holds.
        """
        log.info("=" * 60)
        log.info("FORCE CLOSE (3:55 PM)")
        log.info("=" * 60)

        if self.imp_shares != 0:
            log.info(f"Closing IMP position: {self.imp_shares:+d} shares")
            self._adjust_to_target(0, 'IMP-EOD')
            self.imp_shares = 0

        if self.mrp_shares != 0:
            log.info(f"Closing MRP position: {self.mrp_shares:+d} shares")
            self._adjust_to_target(0, 'MRP-EOD')
            self.mrp_shares = 0

        time.sleep(5)
        self.aum = self._get_portfolio_value()
        log.info(f"EOD Portfolio Value: ${self.aum:,.2f}")

        # Save intraday signal log
        if self.signal_log:
            today_str = now_et().strftime('%Y%m%d')
            log_path  = LOG_DIR / f"signals_{today_str}.csv"
            pd.DataFrame(self.signal_log).to_csv(log_path, index=False)
            log.info(f"Signal log saved → {log_path}")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        """
        Main event loop. Polls every 30 seconds and fires events at the right times.
        All times are US/Eastern.

        Schedule:
          09:00  →  prepare_market()       pre-market setup
          09:30  →  on_market_open()       record open, MRP trade
          10:00  →  on_first_check()       MRP exit + IMP signal 1
          10:30–15:30 every 30 min → IMP signal check
          15:55  →  on_force_close()       close everything
        """
        log.info("=" * 60)
        log.info("SPY INTRADAY MOMENTUM TRADER — PAPER MODE")
        log.info("=" * 60)
        log.info("Running. All times US/Eastern. Press Ctrl+C to stop.\n")

        # Track which events have fired today
        fired = set()

        def should_fire(event_key: str, target_time: dtime, window_secs: int = 60) -> bool:
            """Returns True once per day when clock is within `window_secs` of target_time."""
            if event_key in fired:
                return False
            t = time_et()
            t_secs  = t.hour * 3600 + t.minute * 60 + t.second
            tg_secs = target_time.hour * 3600 + target_time.minute * 60
            if 0 <= t_secs - tg_secs < window_secs:
                fired.add(event_key)
                return True
            return False

        today_date = now_et().date()

        while True:
            try:
                now = now_et()

                # Reset fired events on new day
                if now.date() != today_date:
                    today_date = now.date()
                    fired.clear()
                    self.imp_shares     = 0
                    self.mrp_shares     = 0
                    self.mrp_exited     = False
                    self.signal_log     = []
                    self.open_price     = None
                    self.last_check_min = -1
                    log.info("New trading day — state reset.")

                # ── Fire scheduled events ────────────────────────────────────
                if should_fire('prepare',    dtime(9,  0)):
                    self.prepare_market()

                if should_fire('open',       dtime(9, 30)):
                    self.on_market_open()

                if should_fire('check_1000', dtime(10, 0)):
                    self.on_first_check()

                # IMP signal checks: 10:30, 11:00, ..., 15:30
                for h in range(10, 16):
                    for m_offset in [30, 0]:
                        if h == 10 and m_offset == 0:
                            continue   # 10:00 already handled above
                        chk_hour = h
                        chk_min  = m_offset
                        if m_offset == 0 and h > 10:
                            pass       # on-the-hour checks
                        chk_time = dtime(chk_hour, chk_min)
                        key      = f"check_{chk_hour:02d}{chk_min:02d}"
                        # minutes_from_open at that clock time:
                        mob = (chk_hour * 60 + chk_min) - (9 * 60 + 30) + 1

                        if should_fire(key, chk_time):
                            # Find which sigma bucket this belongs to (nearest 30-min multiple)
                            sigma_key = int(round(mob / 30.0) * 30)
                            if sigma_key < 30:
                                sigma_key = 30
                            self._imp_signal_check(check_min=sigma_key)

                if should_fire('close', dtime(15, 55)):
                    self.on_force_close()

                time.sleep(30)

            except KeyboardInterrupt:
                log.info("Shutting down gracefully...")
                if self.imp_shares != 0 or self.mrp_shares != 0:
                    log.warning("OPEN POSITIONS EXIST — run on_force_close() to flatten!")
                break
            except Exception as e:
                log.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait a minute before retrying


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    trader = SPYIntradayTrader()
    trader.run()
