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

**Backtester** (`Current/Backtesters/ORB/ORB.py`)
- Opening Range Breakout (ORB) strategy backtester for 1-minute SPY bars
- Core logic: identifies opening range (default 9:30-9:45 ET), generates entries on breakout, manages positions with stops/targets
- Key classes:
  - `ORBConfig`: parameterizes the entire backtest (dates, entry/exit rules, position sizing, trading hours)
  - `TradeRecord`: captures detailed trade analytics (entry/exit prices, P&L, MAE/MFE, R-multiples)
- Outputs: `orb_trades.csv` (trade log for Monte Carlo), `orb_trading_days.csv` (full calendar of real trading days in range, including days with zero fired trades — lets the Monte Carlo scripts sample idle days instead of only days that had a trade), and `orb_equity.png` (equity curve)
- Handles multiple risk modes: fixed distance stops, opposite-of-OR stops, OR multiple stops
- Tracks max adverse excursion (MAE) and max favorable excursion (MFE) per trade for robustness analysis

**Monte Carlo Simulator, Challenge Phase** (`Current/Challenge Phase/monte_carlo.py`)
- Bootstraps trades from `orb_trades.csv` to simulate the FTMO 2-step evaluation
- Challenge (phase 1): 10% profit target; Verification (phase 2): 5% profit target, fresh account reset; both phases share the same 5% daily loss limit and 10% max overall loss
- Configurable resample modes: `bootstrap_trades` (reshuffle individual trades), `bootstrap_days` (reshuffle whole trading days), or `bootstrap_blocks` (circular block bootstrap — draws contiguous runs of historical days via `block_size_days` or a `block_size_range`, preserving cross-day streaks/clustering that the i.i.d. modes destroy)
- Optional risk-per-trade sweep (`run_risk_sweep_enabled`): re-runs the simulation across a range of `risk_pct` values and plots pass rate vs. risk, since pass rate is highly sensitive to position size and a single headline number is misleading without it
- Outputs: `mc_results.csv`, `mc_dashboard.png`, `mc_risk_sweep.csv`, `mc_risk_curve.png`

**Monte Carlo Simulator, Funded Phase** (`Current/Funded Phase/monte_carlo.py`)
- Models the LIVE funded account received after passing both Challenge and Verification. Unlike those phases, there is no profit target and no minimum trading days — the account simply runs until a loss limit is breached, or indefinitely in reality
- Simulates up to a configurable `max_simulated_days` horizon (default ~750 trading days ≈ 3 years) and reports **survival %** at that horizon (Wilson CI), plus days-survived percentiles and final-equity distributions for trials that bust, split by daily-loss vs. overall-loss
- Same three resample modes as Challenge Phase (`bootstrap_trades` / `bootstrap_days` / `bootstrap_blocks`), but one `DaySampler` runs continuously for the whole trial since there are no phase resets
- Models FTMO's actual daily-loss rule precisely: a FIXED dollar offset (5% of Initial Capital) subtracted from the previous day's closing equity — see "FTMO Rules Assumptions" below (Challenge Phase now models this identically)
- Configurable payouts: `payouts_enabled`, `payout_interval_trading_days`, `payout_withdraw_pct` (share of accrued profit withdrawn each cycle), `profit_split_pct` (trader's take-home share of each withdrawal). A dedicated payout sweep (`run_payout_sweep_enabled`) grids payout interval × withdraw % to show how both survival % and total money made trade off against each other
- Outputs: `mc_funded_results.csv`, `mc_funded_dashboard.png`, `mc_payout_sweep.csv`, `mc_payout_sweep_heatmaps.png`
- Explicitly out of scope: FTMO's scaling plan (+25% balance every ~4 months) — account size is fixed for the life of the simulated account

### Workflow Dependencies

```
pull_data.py (fetch data)
    ↓ outputs SPY_1min.csv
ORB.py (backtest strategy)
    ↓ outputs orb_trades.csv, orb_trading_days.csv
Challenge Phase/monte_carlo.py (simulate evaluation: Challenge + Verification)
    ↓ outputs mc_results.csv, mc_risk_sweep.csv
Funded Phase/monte_carlo.py (simulate live funded account survival + payouts)
    ↓ outputs mc_funded_results.csv, mc_payout_sweep.csv
```

## Running Commands

### Pull Historical Data
```bash
python3 Current/data/pull_data.py
```
Downloads all SPY 1-minute bars from 2017-01-01 to present. Requires Alpaca API credentials in `.env`. Output: `Current/data/SPY_1min.csv`.

### Run ORB Backtest
```bash
python3 "Current/Backtesters/ORB/ORB.py"
```
Executes the backtest with parameters defined in the ORB file (edit `start_date`, `end_date`, and other `ORBConfig` fields in the script before running). Outputs trade log and equity curve.

### Run Monte Carlo Simulation — Challenge Phase
```bash
python3 "Current/Challenge Phase/monte_carlo.py"
```
Simulates the FTMO 2-step evaluation (Challenge + Verification) based on `orb_trades.csv` from the backtester. Configure account size, resample mode, and iteration count in the script before running.

### Run Monte Carlo Simulation — Funded Phase
```bash
python3 "Current/Funded Phase/monte_carlo.py"
```
Simulates the live funded account's survival (and payout economics) based on the same `orb_trades.csv`. Configure the simulated day horizon, payout settings, and resample mode in the script before running.

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

### MonteCarloConfig (Challenge Phase/monte_carlo.py)
- `account_size`: Simulated account size (e.g., $50k FTMO account)
- `profit_target_phase1_pct`, `profit_target_phase2_pct`: Phase-specific profit targets
- `daily_loss_limit_pct`, `max_overall_loss_pct`: Risk limits
- `min_trading_days_per_phase`: Minimum active days before phase can be marked passed
- `resample_mode`: `bootstrap_trades` / `bootstrap_days` / `bootstrap_blocks` — how to shuffle trade sequences. All three sample from the full `orb_trading_days.csv` calendar (including zero-trade days); `bootstrap_trades`'s per-day trade count is drawn from the empirical historical distribution rather than a fixed constant (`trades_per_day` is only a fallback if the calendar file is missing)
- `block_size_days`, `block_size_range`: Block length (fixed or randomized range) when `resample_mode="bootstrap_blocks"`
- `run_risk_sweep_enabled`, `risk_sweep_values`, `risk_sweep_simulations`: Sweep pass rate across a range of `risk_pct` values

### FundedMonteCarloConfig (Funded Phase/monte_carlo.py)
- `account_size`, `daily_loss_limit_pct`, `max_overall_loss_pct`: Same account rules and same daily-loss mechanic as Challenge Phase — a fixed dollar offset off the previous day's close, not a percentage of current equity (see "FTMO Rules Assumptions" below)
- `max_simulated_days`: Survival horizon in trading days (right-censoring cap, not a target — there is no profit target on this account)
- `resample_mode`, `block_size_days`, `block_size_range`: Same three modes as Challenge Phase
- `payouts_enabled`, `payout_interval_trading_days`, `payout_withdraw_pct`, `profit_split_pct`: Payout mechanics for the main run
- `run_payout_sweep_enabled`, `payout_sweep_intervals`, `payout_sweep_withdraw_pcts`, `payout_sweep_simulations`: Grid sweep of payout frequency × size

## Important Notes

### Data & Credentials
- API credentials live in `.env` (gitignored, never committed)
- Alpaca API requires either free tier (IEX feed) or paid subscription (SIP feed for premium data)
- `pull_data.py` re-downloads and overwrites the full `START` (2017-01-01) to `datetime.now()` range on every run — it does not append incrementally; `bars.to_csv(output_csv)` replaces the file each time

### Trade Analysis
- R-multiple: profit or loss expressed as a multiple of the initial risk_per_trade_usd
- MAE (Max Adverse Excursion): worst unrealized loss during the trade; can trigger account drawdown limits even if trade closes profitably
- MFE (Max Favorable Excursion): best unrealized gain during the trade; useful for detecting strategy edge (higher MFE = better average setups)

### FTMO Rules Assumptions
- 1-minute bar mechanics: when a bar touches both stop and target, stop is conservatively assumed to hit first (no intrabar order info)
- Loss limits are checked continuously (intrabar MAE can breach limits)
- Challenge and Verification phases are independent; Verification starts on a fresh account reset
- 4-day minimum trading requirement ensures statistical significance before phase completion
- Both Monte Carlo scripts model the daily loss floor identically, matching FTMO's real rule confirmed directly from ftmo.com: a FIXED dollar offset (`daily_loss_limit_pct`% of Initial Capital, e.g. 5%) subtracted from the *previous day's closing equity*, not a percentage of current equity. This mechanic is the same across Challenge, Verification, and the funded Account — FTMO does not vary it by phase. (Challenge Phase previously approximated this as a floating percentage of day-start equity; that was a bug, since fixed 2026-07-07, not an intentional design choice.)
- The overall-loss floor is STATIC at `max_overall_loss_pct`% of Initial Capital for the 2-Step program modeled here (never moves, regardless of equity or payouts). The 1-Step program instead trails the highest EOD balance ever achieved — not modeled in this codebase.
- Both Monte Carlo scripts bootstrap from `orb_trading_days.csv` (written by `ORB.py` alongside `orb_trades.csv`) as the trading-day calendar, so days where the backtest fired zero trades are correctly sampled as idle days rather than being silently absent from every resample mode.

### Funded Phase
- Implemented in `Current/Funded Phase/monte_carlo.py`. Models the live funded account after both evaluation phases are passed: no profit target, no minimum trading days, just survival against the daily and overall loss floors over a configurable day horizon
- Key mechanical property to know when reading results: because the daily-loss offset is a fixed dollar amount, payouts never shrink the cushion against the daily floor, but they do shrink the cushion against the (static, non-moving) overall floor — so frequent/large payouts trade survival for realized income. Separately, if position sizing is `pct_current_equity` (compounding), unchecked account growth over a long horizon can *itself* erode daily-loss safety, since dollar risk-per-trade grows with equity while the daily cushion does not
