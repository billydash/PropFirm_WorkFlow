# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PropFirm_WorkFlow is a trading automation and backtesting system designed to validate trading strategies against FTMO (Funded Trader Made Options) requirements. The project follows a three-stage workflow: data acquisition → backtesting → Monte Carlo simulation for account verification.

## Architecture

### Core Modules

**Data Layer** (`Current/data/pull_data.py`)
- Fetches 1-minute historical bars for SPY using Alpaca Markets API
- Outputs CSV format: `SPY_1min.csv` (used by backtester and simulations)
- Requires `.env` with Alpaca API credentials: `public_key` and `secret_key`
- Can be extended to different symbols or timeframes via parameters

**Backtester** (`Current/Backtesters/ORB.py`)
- Opening Range Breakout (ORB) strategy backtester for 1-minute SPY bars
- Core logic: identifies opening range (default 9:30-9:45 ET), generates entries on breakout, manages positions with stops/targets
- Key classes:
  - `ORBConfig`: parameterizes the entire backtest (dates, entry/exit rules, position sizing, trading hours)
  - `TradeRecord`: captures detailed trade analytics (entry/exit prices, P&L, MAE/MFE, R-multiples)
- Outputs: `orb_trades.csv` (trade log for Monte Carlo) and `orb_equity.png` (equity curve)
- Handles multiple risk modes: fixed distance stops, opposite-of-OR stops, OR multiple stops
- Tracks max adverse excursion (MAE) and max favorable excursion (MFE) per trade for robustness analysis

**Monte Carlo Simulator** (`Current/Challenge Phase/monte_carlo.py`, `Current/Funded Phase/monte_carlo.py`)
- Bootstraps trades from `orb_trades.csv` to simulate FTMO account phases
- Challenge Phase: 10% profit target, 5% daily loss limit, 10% max drawdown
- Verification Phase: 5% profit target, fresh account reset, same loss limits
- Configurable resample modes: `bootstrap_trades` (reshuffle individual trades) or `bootstrap_days` (reshuffle entire trading days)
- Outputs: `mc_results.csv` (pass/fail stats) and `mc_dashboard.png` (distribution of outcomes)

### Workflow Dependencies

```
pull_data.py (fetch data)
    ↓ outputs SPY_1min.csv
ORB.py (backtest strategy)
    ↓ outputs orb_trades.csv
monte_carlo.py (simulate phases)
    ↓ outputs results
```

## Running Commands

### Pull Historical Data
```bash
python3 Current/data/pull_data.py
```
Downloads all SPY 1-minute bars from 2017-01-01 to present. Requires Alpaca API credentials in `.env`. Output: `Current/data/SPY_1min.csv`.

### Run ORB Backtest
```bash
python3 Current/Backtesters/ORB.py
```
Executes the backtest with parameters defined in the ORB file (edit `start_date`, `end_date`, and other `ORBConfig` fields in the script before running). Outputs trade log and equity curve.

### Run Monte Carlo Simulation
```bash
python3 "Current/Challenge Phase/monte_carlo.py"
```
Simulates FTMO Challenge phase based on `orb_trades.csv` from the backtester. Configure account size and iteration count in the script before running.

## Key Configuration Points

### ORBConfig (ORB.py)
- `start_date`, `end_date`: Backtest period (YYYY-MM-DD format)
- `or_start`, `or_duration_minutes`: Opening range time window (default 9:30-9:45 ET)
- `direction_mode`: "long" | "short" | "both" — which directions to trade
- `entry_mode`: "stop" (breakout entry) | "close" (close-of-period entry)
- `stop_mode`: "opposite_or" (stop at opposite OR extreme), "fixed_distance" (fixed $), or "or_multiple" (multiple of OR range)
- `rr`: Risk-to-reward ratio (default 2.0)
- `risk_per_trade_usd`: Fixed risk per trade (default $200)
- `reentry_enabled`: Allow re-entry after exit (default False)
- `commission_per_fill`, `slippage_per_share`: Cost model (current placeholder for CFD assumption)
- `starting_equity`: Initial account size (default $100k)

### MonteCarloConfig (monte_carlo.py)
- `account_size`: Simulated account size (e.g., $50k FTMO account)
- `profit_target_phase1_pct`, `profit_target_phase2_pct`: Phase-specific profit targets
- `daily_loss_limit_pct`, `max_overall_loss_pct`: Risk limits
- `min_trading_days_per_phase`: Minimum active days before phase can be marked passed
- `resample_mode`: How to shuffle trade sequences for Monte Carlo iterations

## Important Notes

### Data & Credentials
- API credentials live in `.env` (gitignored, never committed)
- Alpaca API requires either free tier (IEX feed) or paid subscription (SIP feed for premium data)
- SPY 1-minute bars are cached locally; re-running `pull_data.py` appends new data

### Trade Analysis
- R-multiple: profit or loss expressed as a multiple of the initial risk_per_trade_usd
- MAE (Max Adverse Excursion): worst unrealized loss during the trade; can trigger account drawdown limits even if trade closes profitably
- MFE (Max Favorable Excursion): best unrealized gain during the trade; useful for detecting strategy edge (higher MFE = better average setups)

### FTMO Rules Assumptions
- 1-minute bar mechanics: when a bar touches both stop and target, stop is conservatively assumed to hit first (no intrabar order info)
- Loss limits are checked continuously (intrabar MAE can breach limits)
- Challenge and Verification phases are independent; Verification starts on a fresh account reset
- 4-day minimum trading requirement ensures statistical significance before phase completion

### Funded Phase
- Currently a placeholder; intended for live account simulation under funded conditions (e.g., trailing drawdown, profit targets, etc.)
