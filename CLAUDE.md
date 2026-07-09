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
- Outputs: `orb_trades.csv` (trade log for Monte Carlo), `orb_trading_days.csv` (full calendar of real trading days in range, including days with zero fired trades — lets the Monte Carlo scripts sample idle days instead of only days that had a trade — plus a `day_worst_case_mae_r` column: the worst-case combined drawdown across that day's trades, accounting for same-day overlap/compounding a per-trade check would miss; NaN on zero-trade days), and `orb_equity.png` (equity curve)
- `max_notional_usd` (default `None`, off): optional cap on notional exposure (shares × entry_price), following the same "unconfirmed placeholder" pattern as `slippage_per_share`/`commission_per_fill` — off until FTMO's real leverage/max-lot-size limits for the target instrument are confirmed. `TradeRecord.capped` flags trades where it actually bound
- Handles multiple risk modes: fixed distance stops, opposite-of-OR stops, OR multiple stops
- Tracks max adverse excursion (MAE) and max favorable excursion (MFE) per trade for robustness analysis

**Monte Carlo Simulator, Challenge Phase** (`Current/Challenge Phase/monte_carlo.py`)
- Bootstraps trades from `orb_trades.csv` to simulate the FTMO 2-step evaluation
- Challenge (phase 1): 10% profit target; Verification (phase 2): 5% profit target, fresh account reset; both phases share the same 5% daily loss limit and 10% max overall loss
- Configurable resample modes: `bootstrap_days` (reshuffle whole trading days) or `bootstrap_blocks` (circular block bootstrap — draws contiguous runs of historical days via `block_size_days` or a `block_size_range`, preserving cross-day streaks/clustering that `bootstrap_days` destroys). A third mode, `bootstrap_trades`, was removed 2026-07-09 (see "Gemini Critique Audit" below)
- Optional risk-per-trade sweep (`run_risk_sweep_enabled`): re-runs the simulation across a range of `risk_pct` values and plots pass rate vs. risk, since pass rate is highly sensitive to position size and a single headline number is misleading without it
- Outputs: `mc_results.csv`, `mc_dashboard.png`, `mc_risk_sweep.csv`, `mc_risk_curve.png`

**Monte Carlo Simulator, Funded Phase** (`Current/Funded Phase/monte_carlo.py`)
- Models the LIVE funded account received after passing both Challenge and Verification. Unlike those phases, there is no profit target and no minimum trading days — the account simply runs until a loss limit is breached, or indefinitely in reality
- Simulates up to a configurable `max_simulated_days` horizon (default ~750 trading days ≈ 3 years) and reports **survival %** at that horizon (Wilson CI), plus days-survived percentiles and final-equity distributions for trials that bust, split by daily-loss vs. overall-loss
- Same resample modes as Challenge Phase (`bootstrap_days` / `bootstrap_blocks`), but one `DaySampler` runs continuously for the whole trial since there are no phase resets
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
- `resample_mode`: `bootstrap_days` / `bootstrap_blocks` — how to shuffle trade sequences, both sampling from the full `orb_trading_days.csv` calendar (including zero-trade days). A third mode, `bootstrap_trades` (shuffle individual trades i.i.d.), was removed 2026-07-09: it has no day identity, so it can't carry `day_worst_case_mae_r` (see "Gemini Critique Audit" below), which the daily-loss check depends on to catch same-day overlapping/compounding trades
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

### EV Audit (2026-07-07, full write-up in `Current/Run Files/ev_audit_report.md`)
- **Do not take `EV.py`'s headline output at face value.** The combination math in `EV.py` is correct (verified), but the trade edge feeding it is not: `orb_trades.csv`'s mean r_multiple (0.0417 over 1,685 trades) has a t-stat of only 1.41 (95% significance needs ~1.96; bootstrap 95% CI is [-0.017, +0.100], straddling zero) and turns negative over the most recent 12 months of data (mean R -0.020, Challenge pass rate collapsing from 36.5% to 16.1%, EV/attempt from $2,324 to $198).
- A "zero-edge null" test — demean `r_multiple` to exactly 0 across the trade pool (leave `mae_r` untouched) and rerun the full Challenge → Funded → EV pipeline — still produces ~$666/attempt EV and a 23.6% Challenge pass rate. That means roughly **$650-800 of the reported EV is mechanical** (FTMO's profit targets get hit by variance alone at that rate, withdrawn payouts are never clawed back on a later bust, and the fee refunds on first payout), not attributable to the strategy's actual edge. The edge-driven increment on top of that floor is exactly the part that fails significance and reverses sign in recent data.
- Before trusting any future `EV.py` run (e.g. after regenerating `orb_trades.csv` with new data or a changed `ORBConfig`), re-run the sub-period and zero-edge-null checks described in `ev_audit_report.md` rather than trusting the single full-sample headline figure — the strategy's edge has historically been dominated by one anomalous year (2022) and is not stable across sub-periods.

### Gemini Critique Audit (2026-07-09)
An external critique (Gemini) raised 6 points about SPY-to-CFD execution assumptions and Monte Carlo mechanics. All 6 were fact-checked against the code; 3 were real, actionable bugs (now fixed), 1 was refuted on first pass but on reviewer pushback turned out to be a real (bigger) bug, and 2 were already correctly handled or are inherent, out-of-scope simplifications.

**Fixed:**
- **Position sizing default vs. fixed daily-loss floor**: `position_sizing_mode` defaulted to `"pct_current_equity"` in both Monte Carlo scripts' shipped configs, which lets dollar risk-per-trade compound with equity while FTMO's daily-loss offset stays fixed off Initial Capital — silently shrinking the safety margin over the Funded Phase's long horizon. Default is now `"pct_initial_capital"` in both files; `"pct_current_equity"` remains available, and Funded Phase now reports `avg_max_bust_excursion_ratio` / `p90_max_bust_excursion_ratio` plus a `WARNING` in the results report when that ratio implies a single trade's worst case could consume the whole daily cushion.
- **No margin/lot-size cap in ORB.py**: `shares = math.floor(risk_per_trade_usd / stop_distance)` had no ceiling. Added optional `max_notional_usd` (default `None`, off) — see Backtester section above. Empirically, historical notional over 1,707 trades is median ~$29k / p95 ~$69k / max ~$209k, well under the $1M hypothetical the critique used, but the cap is available once real FTMO leverage/lot limits are confirmed.
- **Signal-erasure bug in `direction_mode="both"`** (originally raised as a "concurrency blindspot" / stacked-drawdown claim): `run_day()`'s shared `scan_from` cursor only looked for the *other* direction's trigger starting at `trade.exit_time + 1min`, so an independent opposite-direction signal occurring while the first trade was still open was silently dropped from `orb_trades.csv` entirely — contradicting `direction_mode="both"`'s own documented contract ("long and short are independent triggers"). Fixed by scanning each direction independently from `or_end_ts` (`_run_direction_trades` in `ORB.py`); recovered 22 previously-erased trades out of 1,685 on the full historical run.
  - A new `compute_day_worst_case_mae_r(trades)` computes each day's true worst-case combined drawdown (in R, no bar data needed), stored as `day_worst_case_mae_r` in `orb_trading_days.csv` and consumed by both Monte Carlo scripts' daily-loss check. It walks a day's trades in entry-time order tracking, at each trade's own worst point, the deficit already run up by trades that closed strictly before it (their realized, cost-inclusive `r_multiple`) plus the mae_r of any trade(s) still concurrently open at that point (which haven't realized anything yet, so their own mae_r is the conservative stand-in, not their eventual r_multiple). Busts attributable to more than one same-day trade are tagged with a distinct `"concurrent_daily_loss"` outcome (not attributed to either individual trade, since the portfolio state failed, not one position) rather than folded into `"daily_loss"`.
  - **This function went through two wrong implementations before landing on the one above** — both looked reasonable and passed the obvious single-trade-day check, and both were only caught by directly comparing old-vs-new Monte Carlo bust rates on the same data rather than trusting the reduction property alone:
    1. A bar-by-bar walk evaluating all open trades at each bar's shared low/high. Bug: on the bar where one trade's exit price (e.g. a stop at the OR level) coincides with another trade's entry (the opposite-direction OR breakout is the SAME price level, so this is common), the entering trade's favorable move within that shared bar partially cancelled the exiting trade's own already-realized excursion, understating risk (verified: produced a lower day-level value than the exiting trade's own isolated mae_r, which is nonsensical).
    2. A flat sum of every trade's independent mae_r. Simpler and safe for genuinely concurrent trades, but still understated days where an earlier trade's realized loss (cost-inclusive `r_multiple`, which can exceed that same trade's own costless `mae_r`) compounded with a small later trade's `mae_r` — found via a direct re-implementation of the old per-trade-sequential Monte Carlo check, run against 200,000 resampled multi-trade days: the two disagreed on whether a trial should bust.
  - Before trusting a future change to this function, rerun that same check: reimplement the pre-change per-trade-sequential logic directly, resample a large number of multi-trade days at an elevated `risk_pct` (so daily-loss actually binds), and confirm zero cases where the old logic busts a trial that the new `day_worst_case_mae_r` does not (and check the reverse direction too — see next point).
  - **Impact check on the current data (2026-07-09):** across 200,000 resampled multi-trade days, there were zero cases where the new day-level check busted a trial the old per-trade-sequential check would not have (the required direction) -- but ALSO zero cases in the other direction. That's because the current `orb_trades.csv` has zero genuinely time-overlapping trade pairs: of 284 same-day trade adjacencies, all are either exact boundary-touches (one trade's exit price/time equals the next's entry -- expected, since opposite-direction OR levels can coincide) or fully sequential with a gap; none actually overlap in wall-clock time. So on this specific historical run, this fix's entire practical effect is the 22 trades recovered by the signal-erasure repair (previous point) plus correctly reproducing the old sequential check's cost-inclusive math -- the concurrent-mae-stacking logic is verified-correct machinery that isn't yet exercised by real data. It would activate automatically if a future config/dataset ever produces true overlap (e.g. a wider opening range or different stop placement that decouples the two directions' trigger levels).
  - At default `risk_pct` (1.0%), daily-loss essentially never binds at all for either phase (`overall_loss` dominates every bust) -- so Fix 3's corrected headline pass/survival numbers are within noise of the pre-fix figures (Challenge ~35.4% pass, Funded ~1.4% survival, matching the "EV Audit" section above). The fix matters at higher `risk_pct` (see the risk sweeps), not at the shipped default.
  - **Process note**: this point was initially misdiagnosed as "refuted" (trades don't overlap in the output, therefore nothing to fix) before the reviewer correctly pushed back that the *reason* they never overlap was itself the bug. Worth remembering when auditing this kind of "the output looks fine" reasoning in the future — absence of a symptom isn't the same as absence of the bug.
- **`bootstrap_trades` resample mode removed** from both Monte Carlo scripts: it samples individual trades i.i.d. with no day identity, so it can't carry `day_worst_case_mae_r`, which the corrected daily-loss check depends on. `resample_mode` is now `Literal["bootstrap_days", "bootstrap_blocks"]` with a `__post_init__` validator that raises on the old value; default changed to `"bootstrap_blocks"` in both dataclasses to match what both `__main__` configs already ran.

**Not fixed (already correct or out of scope):**
- **Flat slippage assumption** (`slippage_per_share = 0.03`, symmetric, no volatility/momentum scaling): a real simplification, but already flagged in-code (`ORB.py` comment) as an unconfirmed placeholder pending real FTMO spread data. Best addressed with a slippage sensitivity sweep as a future exercise, not a code change.
- **Intrabar/reentry blindspot** (can't resolve two triggers within the same 1-minute bar): real, but it's a general property of 1-minute-bar resolution everywhere in the backtester, not specific to `reentry_enabled`. A real fix needs tick or sub-minute data — out of scope for this codebase.
- **Median used in `typical_lifetime_persist`**: `EV.py` already maintains a separate, correctly mean-based `ev_lifetime_persist` as the actual headline EV, and labels the median-based variant "NOT an expected value" in three places (docstring, section header, footnote). The critique re-derived a caveat the code already states.
