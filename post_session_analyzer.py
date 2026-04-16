#!/usr/bin/env python3
"""
POST-SESSION ANALYZER — Autonomous Learning Pipeline
=====================================================
Runs after market close (4:05 PM ET) each trading day.

1. Reads today's signal log + trading log
2. Computes performance metrics (win rate, Sharpe, slippage, regime)
3. Adapts strategy parameters within guardrails
4. Saves adapted params to params.json for tomorrow's session
5. Appends to historical performance DB (CSV-based, no external DB needed)

Architecture principle: REGIME CLASSIFICATION + BOUNDED PARAMETER ADJUSTMENT.
The core edge (VWAP confirmation, 30-min frequency, overnight fade) is NEVER touched.
Only vol-responsive scaling, time-of-day weighting, and band width adapt.
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta, date, time as dtime
from pathlib import Path

import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

ET = pytz.timezone('America/New_York')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
LOG_DIR     = BASE_DIR / 'logs'
PARAMS_FILE = BASE_DIR / 'params.json'
HISTORY_FILE = BASE_DIR / 'logs' / 'performance_history.csv'

# ─── Baseline Parameters (from the Zarattini paper — DO NOT exceed guardrails) ─
BASELINE = {
    'BAND_MULT':        1.0,
    'TARGET_VOL':       0.02,
    'MAX_LEVERAGE':     4.0,
    'OVERNIGHT_THRESH': 0.02,
    'SIGMA_LOOKBACK':   14,
    'VOL_LOOKBACK':     15,
}

# ─── Guardrails: min/max bounds + max daily change ───────────────────────────
GUARDRAILS = {
    'BAND_MULT': {
        'min': 0.70,
        'max': 1.40,
        'max_daily_change': 0.10,  # Max ±0.10 per day
        'ema_alpha': 0.4,          # Smoothing factor (higher = more responsive)
    },
    'SIGMA_LOOKBACK': {
        'min': 7,
        'max': 21,
        'allowed_values': [7, 10, 14, 21],  # Discrete steps only
        'max_daily_change': 1,  # One step in allowed_values per day
    },
    'OVERNIGHT_THRESH': {
        'min': 0.015,
        'max': 0.035,
        'max_daily_change': 0.005,
        'ema_alpha': 0.4,
    },
}

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"analyzer_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def load_current_params() -> dict:
    """Load current params or return baseline if no file exists."""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            return json.load(f)
    return BASELINE.copy()


def save_params(params: dict, reasoning: dict):
    """Save adapted params with timestamp and reasoning."""
    record = {
        'updated_at': datetime.now(ET).isoformat(),
        'params': params,
        'reasoning': reasoning,
    }
    with open(PARAMS_FILE, 'w') as f:
        json.dump(record, f, indent=2)
    log.info(f"Params saved → {PARAMS_FILE}")


def apply_guardrail(param_name: str, new_value: float, old_value: float) -> float:
    """Apply guardrails: clip to bounds, limit daily change, EMA smooth."""
    g = GUARDRAILS.get(param_name)
    if g is None:
        return new_value

    # Discrete parameters (like SIGMA_LOOKBACK)
    if 'allowed_values' in g:
        allowed = sorted(g['allowed_values'])
        # Find closest allowed value
        closest = min(allowed, key=lambda x: abs(x - new_value))
        # Limit step size
        old_idx = allowed.index(min(allowed, key=lambda x: abs(x - old_value)))
        new_idx = allowed.index(closest)
        max_step = g.get('max_daily_change', 1)
        clamped_idx = max(old_idx - max_step, min(old_idx + max_step, new_idx))
        return allowed[clamped_idx]

    # Continuous parameters
    # 1. Clip to absolute bounds
    clipped = max(g['min'], min(g['max'], new_value))

    # 2. EMA smoothing
    alpha = g.get('ema_alpha', 0.4)
    smoothed = alpha * clipped + (1 - alpha) * old_value

    # 3. Limit daily change
    max_change = g.get('max_daily_change', 999)
    delta = smoothed - old_value
    if abs(delta) > max_change:
        smoothed = old_value + max_change * np.sign(delta)

    return round(smoothed, 4)


def kaufman_efficiency_ratio(returns: pd.Series) -> float:
    """
    Kaufman Efficiency Ratio: |sum(returns)| / sum(|returns|)
    Range [0, 1]. Higher = more trending, lower = more choppy.
    """
    if len(returns) < 2 or returns.abs().sum() == 0:
        return 0.5
    return abs(returns.sum()) / returns.abs().sum()


# ─── Main Analysis Pipeline ──────────────────────────────────────────────────

class PostSessionAnalyzer:
    """
    Runs after market close. Analyzes today's performance and adapts parameters
    for tomorrow within strict guardrails.
    """

    def __init__(self):
        load_dotenv(BASE_DIR / '.env')
        api_key    = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        self.data  = StockHistoricalDataClient(api_key, api_secret)
        self.today = datetime.now(ET).date()
        self.params = load_current_params()
        # If params.json has a nested 'params' key (from save_params format)
        if 'params' in self.params and 'BAND_MULT' in self.params['params']:
            self.params = self.params['params']

    def _load_signal_log(self) -> pd.DataFrame:
        """Load today's signal CSV."""
        path = LOG_DIR / f"signals_{self.today.strftime('%Y%m%d')}.csv"
        if not path.exists():
            log.warning(f"No signal log found for today: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def _fetch_daily_bars(self, days: int = 30) -> pd.DataFrame:
        """Fetch daily bars for regime detection."""
        end   = datetime.now(ET)
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start, end=end,
            adjustment='raw', feed='sip'
        )
        bars = self.data.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc['SPY']
        df.index = pd.to_datetime(df.index).tz_convert(ET)
        df['ret'] = df['close'].pct_change()
        return df

    def _fetch_intraday_bars(self) -> pd.DataFrame:
        """Fetch today's 1-min bars for detailed analysis."""
        end   = datetime.now(ET)
        start = end - timedelta(days=2)
        req = StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start, end=end,
            adjustment='raw', feed='sip'
        )
        bars = self.data.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc['SPY']
        df.index = pd.to_datetime(df.index).tz_convert(ET)
        today = self.today
        df = df[df.index.date == today]
        df = df.between_time('09:30', '15:59')
        return df

    # ── Step 1: Compute Performance Metrics ───────────────────────────────────

    def compute_metrics(self, signals: pd.DataFrame) -> dict:
        """Compute comprehensive daily metrics from signal log."""
        metrics = {}

        if signals.empty:
            log.warning("No signals to analyze.")
            return metrics

        # Filter out special signals (take-profit = 99)
        real_signals = signals[signals['signal'].isin([-1, 0, 1])].copy()

        # Daily P&L
        start_aum = signals['aum'].iloc[0]
        end_aum   = signals['aum'].iloc[-1]
        daily_return = (end_aum - start_aum) / start_aum if start_aum > 0 else 0
        metrics['daily_return'] = round(daily_return, 6)
        metrics['daily_pnl']   = round(end_aum - start_aum, 2)
        metrics['start_aum']   = round(start_aum, 2)
        metrics['end_aum']     = round(end_aum, 2)

        # Take-profit triggered?
        metrics['tp_triggered'] = int(99 in signals['signal'].values)

        # Signal counts
        long_signals  = len(real_signals[real_signals['signal'] == 1])
        short_signals = len(real_signals[real_signals['signal'] == -1])
        flat_signals  = len(real_signals[real_signals['signal'] == 0])
        metrics['long_signals']  = long_signals
        metrics['short_signals'] = short_signals
        metrics['flat_signals']  = flat_signals
        metrics['total_signals'] = long_signals + short_signals

        # Win rate (signal-to-signal P&L)
        if len(real_signals) >= 2:
            pnl_changes = real_signals['aum'].diff().dropna()
            wins = (pnl_changes > 0).sum()
            losses = (pnl_changes < 0).sum()
            total_decisions = wins + losses
            metrics['win_rate'] = round(wins / total_decisions, 4) if total_decisions > 0 else 0.5
        else:
            metrics['win_rate'] = 0.5

        # Signal accuracy by hour
        if not real_signals.empty:
            real_signals['hour'] = real_signals['timestamp'].dt.hour
            for h in range(10, 16):
                hour_signals = real_signals[real_signals['hour'] == h]
                if len(hour_signals) >= 1:
                    hour_pnl = hour_signals['aum'].diff().dropna()
                    wins_h = (hour_pnl > 0).sum()
                    total_h = len(hour_pnl[hour_pnl != 0])
                    metrics[f'win_rate_h{h}'] = round(wins_h / total_h, 4) if total_h > 0 else 0.5
                else:
                    metrics[f'win_rate_h{h}'] = 0.5

        # Average sigma used
        if 'sigma' in real_signals.columns:
            metrics['avg_sigma'] = round(real_signals['sigma'].mean(), 6)

        log.info(f"Daily metrics: return={daily_return*100:+.2f}%, "
                 f"win_rate={metrics.get('win_rate', 'N/A')}, "
                 f"signals={metrics.get('total_signals', 0)}, "
                 f"TP={metrics.get('tp_triggered', 0)}")

        return metrics

    # ── Step 2: Regime Detection ──────────────────────────────────────────────

    def detect_regime(self) -> dict:
        """
        Classify current market regime using two metrics:
        1. Kaufman Efficiency Ratio (KER) → trending vs choppy
        2. Vol ratio → high/normal/low volatility
        """
        df = self._fetch_daily_bars(days=30)
        regime = {}

        # KER on last 10 days of daily returns
        recent_ret = df['ret'].dropna().tail(10)
        ker = kaufman_efficiency_ratio(recent_ret)
        regime['ker'] = round(ker, 4)

        if ker > 0.6:
            regime['trend_regime'] = 'TRENDING'
        elif ker > 0.3:
            regime['trend_regime'] = 'MIXED'
        else:
            regime['trend_regime'] = 'CHOPPY'

        # Realized vol vs target vol
        vol_15d = df['ret'].dropna().tail(15).std()
        vol_ratio = vol_15d / 0.02 if vol_15d > 0 else 1.0
        regime['vol_15d']    = round(vol_15d, 6)
        regime['vol_ratio']  = round(vol_ratio, 4)

        if vol_ratio > 1.3:
            regime['vol_regime'] = 'HIGH_VOL'
        elif vol_ratio < 0.7:
            regime['vol_regime'] = 'LOW_VOL'
        else:
            regime['vol_regime'] = 'NORMAL'

        # Intraday range today
        intraday = self._fetch_intraday_bars()
        if not intraday.empty:
            day_high = intraday['high'].max()
            day_low  = intraday['low'].min()
            day_open = intraday['open'].iloc[0]
            regime['intraday_range_pct'] = round((day_high - day_low) / day_open * 100, 2)
        else:
            regime['intraday_range_pct'] = 0

        log.info(f"Regime: {regime['trend_regime']} (KER={ker:.3f}), "
                 f"{regime['vol_regime']} (vol_ratio={vol_ratio:.2f}), "
                 f"intraday_range={regime.get('intraday_range_pct', 0):.2f}%")

        return regime

    # ── Step 3: Adapt Parameters ──────────────────────────────────────────────

    def adapt_parameters(self, metrics: dict, regime: dict) -> tuple[dict, dict]:
        """
        Adjust parameters based on regime + recent performance.
        Returns (new_params, reasoning_dict).

        Rules:
        - BAND_MULT: tighter in high vol (fewer false breaks), wider in low vol
        - SIGMA_LOOKBACK: longer in choppy markets, shorter in trending
        - OVERNIGHT_THRESH: raise if gap-fades underperforming, lower if strong
        """
        old = self.params.copy()
        new = old.copy()
        reasoning = {}

        # ── BAND_MULT adaptation ──────────────────────────────────────────
        vol_regime = regime.get('vol_regime', 'NORMAL')
        baseline_bm = BASELINE['BAND_MULT']

        if vol_regime == 'HIGH_VOL':
            target_bm = baseline_bm * 0.90  # Tighten bands
            reasoning['BAND_MULT'] = f"HIGH_VOL regime → tightening bands (×0.90)"
        elif vol_regime == 'LOW_VOL':
            target_bm = baseline_bm * 1.10  # Widen bands
            reasoning['BAND_MULT'] = f"LOW_VOL regime → widening bands (×1.10)"
        else:
            target_bm = baseline_bm  # Revert toward baseline
            reasoning['BAND_MULT'] = "NORMAL vol → reverting toward baseline"

        new['BAND_MULT'] = apply_guardrail('BAND_MULT', target_bm, old.get('BAND_MULT', baseline_bm))

        # ── SIGMA_LOOKBACK adaptation ─────────────────────────────────────
        trend_regime = regime.get('trend_regime', 'MIXED')

        if trend_regime == 'TRENDING':
            target_sl = 10  # Shorter lookback captures faster trends
            reasoning['SIGMA_LOOKBACK'] = "TRENDING regime → shorter lookback (10d)"
        elif trend_regime == 'CHOPPY':
            target_sl = 21  # Longer lookback smooths noise
            reasoning['SIGMA_LOOKBACK'] = "CHOPPY regime → longer lookback (21d)"
        else:
            target_sl = 14  # Baseline
            reasoning['SIGMA_LOOKBACK'] = "MIXED regime → baseline lookback (14d)"

        new['SIGMA_LOOKBACK'] = int(apply_guardrail(
            'SIGMA_LOOKBACK', target_sl, old.get('SIGMA_LOOKBACK', 14)
        ))

        # ── OVERNIGHT_THRESH adaptation ───────────────────────────────────
        # Load last 20 days of performance history to assess gap-fade success
        gap_fade_success = self._get_gap_fade_success_rate()

        if gap_fade_success is not None:
            if gap_fade_success < 0.40:
                target_ot = old.get('OVERNIGHT_THRESH', 0.02) + 0.005
                reasoning['OVERNIGHT_THRESH'] = (
                    f"Gap-fade win rate {gap_fade_success:.0%} < 40% → raising threshold"
                )
            elif gap_fade_success > 0.65:
                target_ot = old.get('OVERNIGHT_THRESH', 0.02) - 0.005
                reasoning['OVERNIGHT_THRESH'] = (
                    f"Gap-fade win rate {gap_fade_success:.0%} > 65% → lowering threshold"
                )
            else:
                target_ot = old.get('OVERNIGHT_THRESH', 0.02)
                reasoning['OVERNIGHT_THRESH'] = (
                    f"Gap-fade win rate {gap_fade_success:.0%} — holding threshold"
                )
            new['OVERNIGHT_THRESH'] = apply_guardrail(
                'OVERNIGHT_THRESH', target_ot,
                old.get('OVERNIGHT_THRESH', 0.02)
            )
        else:
            reasoning['OVERNIGHT_THRESH'] = "Insufficient gap-fade history — holding"

        # Carry forward unchanged params
        for k in BASELINE:
            if k not in new:
                new[k] = old.get(k, BASELINE[k])

        log.info("Parameter adaptations:")
        for k in ['BAND_MULT', 'SIGMA_LOOKBACK', 'OVERNIGHT_THRESH']:
            old_v = old.get(k, BASELINE[k])
            new_v = new[k]
            changed = "CHANGED" if old_v != new_v else "unchanged"
            log.info(f"  {k}: {old_v} → {new_v} ({changed}) | {reasoning.get(k, '')}")

        return new, reasoning

    def _get_gap_fade_success_rate(self) -> float | None:
        """
        Look at last 20 days of signal logs and count MRP (gap-fade) trades.
        A gap-fade is successful if AUM increased from signal entry to exit.
        Returns None if insufficient data.
        """
        # This is a simplified version — checks if days with overnight signals were profitable
        if not HISTORY_FILE.exists():
            return None

        history = pd.read_csv(HISTORY_FILE)
        if len(history) < 5:
            return None

        recent = history.tail(20)
        # We can't perfectly isolate MRP P&L without more granular logs,
        # but we CAN check daily P&L on days with signals (proxy metric)
        if 'daily_return' in recent.columns and len(recent) > 0:
            win_days = (recent['daily_return'] > 0).sum()
            total_days = len(recent[recent['daily_return'] != 0])
            return win_days / total_days if total_days > 0 else None

        return None

    # ── Step 4: Monthly Backtest Validation ───────────────────────────────────

    def monthly_validation_check(self, metrics: dict) -> bool:
        """
        On first trading day of each month, check if adapted params
        are outperforming baseline. If not, roll back.
        Returns True if rollback occurred.
        """
        if self.today.day > 5:  # Only run first 5 days of month
            return False

        if not HISTORY_FILE.exists():
            return False

        history = pd.read_csv(HISTORY_FILE)
        if len(history) < 20:
            return False

        # Compare last 20 days Sharpe to threshold
        recent = history.tail(20)
        if 'daily_return' in recent.columns:
            returns = recent['daily_return'].dropna()
            if len(returns) > 5 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)

                if sharpe > 1.8:
                    log.warning(f"OVERFITTING ALERT: Sharpe {sharpe:.2f} > 1.8 — "
                                f"suspiciously high. Rolling back to baseline.")
                    self.params = BASELINE.copy()
                    save_params(self.params, {'reason': 'Overfitting rollback',
                                              'sharpe': sharpe})
                    return True

                if sharpe < 0.3:
                    log.warning(f"UNDERPERFORMANCE ALERT: Sharpe {sharpe:.2f} < 0.3 — "
                                f"Rolling back to baseline.")
                    self.params = BASELINE.copy()
                    save_params(self.params, {'reason': 'Underperformance rollback',
                                              'sharpe': sharpe})
                    return True

                log.info(f"Monthly check: 20-day Sharpe = {sharpe:.2f} — within bounds.")

        return False

    # ── Step 5: Save to History ───────────────────────────────────────────────

    def append_history(self, metrics: dict, regime: dict):
        """Append today's performance to the cumulative history CSV."""
        row = {
            'date': self.today.isoformat(),
            **metrics,
            **{f"regime_{k}": v for k, v in regime.items()},
        }
        df_row = pd.DataFrame([row])

        if HISTORY_FILE.exists():
            existing = pd.read_csv(HISTORY_FILE)
            # Don't duplicate today
            existing = existing[existing['date'] != self.today.isoformat()]
            combined = pd.concat([existing, df_row], ignore_index=True)
        else:
            combined = df_row

        combined.to_csv(HISTORY_FILE, index=False)
        log.info(f"Performance history updated → {HISTORY_FILE} ({len(combined)} days)")

    # ── Main Pipeline ─────────────────────────────────────────────────────────

    def run(self):
        """Execute the full post-session analysis pipeline."""
        log.info("=" * 60)
        log.info(f"POST-SESSION ANALYSIS — {self.today}")
        log.info("=" * 60)

        # 1. Load today's signals
        signals = self._load_signal_log()

        # 2. Compute performance metrics
        metrics = self.compute_metrics(signals)

        # 3. Detect market regime
        regime = self.detect_regime()

        # 4. Monthly validation check (may rollback params)
        rolled_back = self.monthly_validation_check(metrics)

        # 5. Adapt parameters (skip if just rolled back)
        if not rolled_back:
            new_params, reasoning = self.adapt_parameters(metrics, regime)
            save_params(new_params, reasoning)
        else:
            reasoning = {'reason': 'Rolled back to baseline'}

        # 6. Save to history
        self.append_history(metrics, regime)

        log.info("=" * 60)
        log.info("Post-session analysis complete.")
        log.info("=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    analyzer = PostSessionAnalyzer()
    analyzer.run()
