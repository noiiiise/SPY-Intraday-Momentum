# SPY Intraday Momentum — Strategy Deep Dive

**Paper:** "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
**Authors:** Zarattini, Aziz, Barbon (2024) — SFI Research Paper No. 24-97
**Download:** https://ssrn.com/abstract=4824172

---

## Why This Edge Exists (The Real Reason It Works)

This isn't a fluke pattern. Two permanent structural forces create intraday momentum in SPY:

**1. Options Market Maker Gamma Hedging**
When market makers sell options, they carry short-gamma exposure. As SPY moves directionally in the morning, they're *mechanically forced* to buy when prices rise and sell when prices fall to stay delta-neutral. This is non-discretionary. It amplifies whatever move started, creating momentum. This force exists as long as the options market exists.

**2. Leveraged ETF Rebalancing**
Products like TQQQ, SPXU, SOXL must rebalance daily to maintain their leverage ratio. On a 1% SPY day, leveraged ETF rebalancing accounts for ~17% of market-on-close volume. On a 5% day, it's >50% of MOC volume. These rebalances happen in a known direction and a predictable window — pushing prices further in the direction the day already moved.

**The strategy captures both.** Enter when a directional signal appears, hold through the afternoon, exit before close.

---

## Two Sub-Strategies Running Simultaneously

### IMP — Intraday Momentum Portfolio
Rides intraday trend. Core of the strategy.

### MRP — Mean Reversion Portfolio
Fades large overnight gaps (>2%). Enters at open, always exits at 10:00 AM.
Logic: gap-and-go moves often partially revert in the first 30 minutes.

---

## IMP Signal Logic (Exact Code Translation)

**Inputs computed pre-market:**
- `sigma_open` — rolling 14-day average of `|close/open - 1|` at each minute of day
  This is a measure of "how much does SPY typically move from open by this time of day"
- `prev_close` — yesterday's closing price
- `spx_vol` — 15-day rolling daily return standard deviation

**Inputs updated live:**
- `open_price` — today's 9:30 AM open
- `current_close` — latest 1-min close price
- `VWAP` — volume-weighted average price (cumulative from open)

**Bands (recalculated at each 30-min check):**
```
UB = max(open, prev_close) × (1 + sigma_open)
LB = min(open, prev_close) × (1 − sigma_open)
```

Using `max/min` of open vs prev_close means the band widens when there's an overnight gap — requiring a *larger* move beyond the gap to trigger a signal. Sensible filter.

**Signal:**
```
close > UB  AND  close > VWAP  →  LONG
close < LB  AND  close < VWAP  →  SHORT
otherwise                       →  FLAT (exit any position)
```

The VWAP filter is crucial. It prevents false signals where price briefly spikes above the band but overall selling pressure dominates (or vice versa). VWAP acts as a "trend confirmation" filter.

**When signals are evaluated:**
Every 30 minutes: 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 15:30 ET

**Positions are held** until the next signal changes them, or until 3:55 PM forced close.

---

## Position Sizing

```
leverage = min(target_vol / 15d_daily_vol, 4.0)
shares   = round(AUM / open_price × leverage)
```

- `target_vol = 0.02` (2% daily vol target)
- `max_leverage = 4.0×` (hard cap)
- If vol is 1% → leverage = 2%. If vol is 0.5% → leverage = 4× (capped).
- If vol data unavailable → default to 4×.

**Example:** AUM=$100K, SPY open=$580, 15d vol=1.2%
- leverage = min(2%/1.2%, 4.0) = 1.67×
- shares = round(100,000 / 580 × 1.67) = **288 shares**

This is **vol-targeting**, not fixed leverage. The strategy automatically reduces size in high-vol environments (like 2022) and increases in calm ones.

---

## MRP Signal Logic

```
overnight_move = open / prev_close − 1

if |overnight_move| > 2%:
    direction = −sign(overnight_move)   ← FADE the gap
    enter position at open
    exit at 10:00 AM regardless of P&L
```

- Gap up >2%: SHORT → expect partial mean reversion in first 30 min
- Gap down >2%: LONG → same logic
- Gap ≤2%: no MRP trade

Same shares calculation as IMP.

---

## Backtested Performance (from the paper)

| Metric | Strategy | S&P 500 |
|--------|----------|---------|
| Total Return (2007–2024) | 1,985% | ~460% |
| Annualized Return | 19.6% | ~10.5% |
| Sharpe Ratio | 1.33 | ~0.65 |
| Max Drawdown | ~18% | ~55% |
| Hit Rate | ~55% | N/A |

The outperformance is consistent across the full 17-year period including 2008, 2020, and 2022 crises.

---

## What the Original Repo Does vs. What `live_trader.py` Does

| | `process.py` (original) | `live_trader.py` (new) |
|---|---|---|
| **Purpose** | Backtest on historical CSV data | Live paper trading via Alpaca API |
| **Data source** | Pre-downloaded CSV files | Real-time Alpaca API (1-min bars) |
| **Execution** | Simulated (calculates P&L mathematically) | Real orders submitted to Alpaca paper account |
| **Scheduling** | Loops through all days in data | Polls clock every 30s, fires at exact times |
| **sigma_open** | Computed for every bar across full history | Computed pre-market for today's check times only |
| **Logging** | Prints stats to terminal | Writes `logs/trader_YYYYMMDD.log` + `logs/signals_YYYYMMDD.csv` |
| **Risk** | Paper math | Paper money (real execution, fake capital) |

---

## Key Risks & Things to Watch in Paper Trading

**1. Execution slippage**
The backtest assumes fills at the close price of the signal bar. In live trading, market orders may fill slightly worse. Watch fill prices in `logs/signals_*.csv`.

**2. Short selling SPY**
Paper trading supports short SPY. Live trading does too (it's highly liquid). Alpaca requires short-selling to be enabled in account settings.

**3. First 30 minutes**
The strategy does NOT trade IMP in the first 30 minutes — it's data collection time. Only MRP trades at the open.

**4. End-of-day closing**
All positions close at 3:55 PM. The 3:55 time leaves 5 minutes buffer before 4 PM to avoid MOC auction participation. Do not modify this.

**5. Dividends**
SPY pays dividends quarterly. The backtest adjusts prev_close for dividends to avoid false gap signals. The live trader uses raw prices from Alpaca — on ex-dividend days (~Mar/Jun/Sep/Dec), the overnight gap signal may misfire. Low-frequency issue but worth noting.

---

## Setup Checklist

- [ ] Get Alpaca paper trading API keys → https://app.alpaca.markets/paper-trading
- [ ] Copy `.env.example` → `.env`, fill in keys
- [ ] `pip install -r requirements_live.txt`
- [ ] Start at 9:00 AM ET on a market day: `python live_trader.py`
- [ ] Watch logs in `logs/` folder
- [ ] Run for 2–4 weeks before evaluating signal quality
- [ ] Compare daily P&L against SPY buy-and-hold over same period

---

## Paper Trading → Live Trading Checklist (FUTURE)

Only move to live after satisfying ALL of these:

- [ ] ≥30 trading days of paper trading with no bugs
- [ ] Hit rate ≥50% (check `logs/signals_*.csv`)
- [ ] No unexpected position sizing errors
- [ ] Short selling confirmed working on paper account
- [ ] Alpaca live account funded and short-selling enabled
- [ ] Change `paper=True` → `paper=False` in `TradingClient()` constructor
- [ ] Start with small size (e.g., AUM=$10K equivalent) before scaling
