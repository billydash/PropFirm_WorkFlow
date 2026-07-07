"""Monte Carlo simulator for the FTMO 50k 2-step Challenge (Challenge + Verification),
bootstrapped from the ORB backtester's trade log (Current/Backtesters/ORB/orb_trades.csv).

Rules modeled (ftmo.com/en/trading-objectives, confirmed live 2026-07-07):
  - Challenge (phase 1): 10% profit target
  - Verification (phase 2): 5% profit target, on a FRESH account reset to account_size
  - Maximum Daily Loss: a FIXED dollar amount (5% of Initial Capital) subtracted from
    the PREVIOUS day's closing equity, recalculated daily. This offset is a static
    dollar figure -- it does NOT scale with the current day's equity. This mechanic
    is identical across Challenge, Verification, AND the funded Account (see
    ../Funded Phase/monte_carlo.py, which implements the same rule) -- FTMO does not
    vary the daily-loss mechanic by phase, only the percentage differs by program
    (3% for 1-Step, 5% for 2-Step).
  - Maximum Overall Loss: 10% of initial capital, STATIC (never moves). This is the
    2-Step program's overall-loss mechanic; the 1-Step program instead trails the
    highest EOD balance ever achieved (not modeled here).
  - Minimum 4 distinct trading days per phase before it can be marked passed
  - Both loss limits are checked continuously against equity, so a trade's MAE
    (worst intra-trade dip) can breach a limit even if the trade closes as a win.

Assumption: each historical trade's r_multiple and mae_r are unitless ratios
(relative to the ORB backtest's own risk_per_trade_usd), so rescaling them by
a different risk_amount here correctly reproduces the same relative edge,
independent of the original $/share mechanics.

Trading calendar: day_groups/day_trade_counts are built from
Backtesters/ORB/orb_trading_days.csv (the full set of real trading days the ORB
backtest ran over) when that file is present alongside trades_csv_path, so days
that produced zero trades are correctly represented as idle days during resampling
instead of being silently absent from the trade pool.
"""

import math
import os
from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ResampleMode = Literal["bootstrap_trades", "bootstrap_days", "bootstrap_blocks"]
PositionSizingMode = Literal["pct_current_equity", "pct_initial_capital", "fixed_usd"]
Outcome = Literal["passed", "daily_loss", "overall_loss", "timeout"]


@dataclass
class MonteCarloConfig:
    # Path to trade CSV from ORB backtester. Default points to ../Backtesters/ORB/orb_trades.csv
    trades_csv_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "..", "Backtesters/ORB", "orb_trades.csv"
        )
    )

    # === FTMO Account & Phase Rules ===
    # Simulated account starting capital (e.g., $50k for FTMO challenge)
    account_size: float = 50_000.0

    # Phase 1 (Challenge) profit target as percentage of account_size (e.g., 10% = $5k on $50k)
    profit_target_phase1_pct: float = 10.0

    # Phase 2 (Verification) profit target as percentage; resets on fresh account
    profit_target_phase2_pct: float = 5.0

    # Maximum Daily Loss, as % of starting_capital (account_size for this phase).
    # Converted internally to a FIXED dollar offset (starting_capital * pct/100)
    # subtracted from the PREVIOUS day's closing equity -- this offset never scales
    # with current equity, only the anchor (previous close) moves day to day. This
    # matches FTMO's real rule (confirmed live from ftmo.com) and is the same
    # mechanic ../Funded Phase/monte_carlo.py already implements.
    # If a trade's MAE breaches this, phase fails immediately
    daily_loss_limit_pct: float = 5.0

    # Maximum overall (cumulative) loss allowed (% of initial account_size)
    # This is a static floor; never moves. If equity drops below it, phase fails
    max_overall_loss_pct: float = 10.0

    # Minimum number of distinct trading days required before phase can pass
    # Ensures statistical significance (typical FTMO requirement: 4 days)
    min_trading_days_per_phase: int = 4

    # === Trade Resampling Mode ===
    # "bootstrap_trades": shuffle individual trades, drawing a per-day trade count
    #     from the empirical historical distribution each simulated day (see trades_per_day)
    # "bootstrap_days": shuffle entire historical trading days (preserves correlation within days)
    # "bootstrap_blocks": draw contiguous RUNS of historical days (see block settings below).
    #     This is the least optimistic mode: it preserves losing/winning streaks and
    #     volatility clustering ACROSS days, which the two i.i.d. modes destroy.
    resample_mode: ResampleMode = "bootstrap_trades"

    # FALLBACK ONLY for "bootstrap_trades" mode: fixed number of trades to draw per
    # simulated day, used only if orb_trading_days.csv (the trading-day calendar)
    # isn't found alongside trades_csv_path. When the calendar IS found (the normal
    # case), the per-day trade count is instead drawn from the empirical historical
    # distribution (including zero-trade days), which is more accurate than any
    # single fixed constant -- see load_trade_pool / DaySampler.
    trades_per_day: int = 2

    # === Block Resampling (only used when resample_mode == "bootstrap_blocks") ===
    # Circular block bootstrap: each block starts at a random historical day and takes
    # `block_size_days` consecutive days, wrapping past the end of the history back to the
    # start so every day is equally likely and blocks keep their full length. Preserving
    # runs of consecutive real days keeps cross-day streaks intact and gives more realistic
    # (higher) failure rates than shuffling days independently.
    #
    # Fixed block length in trading days. Used when block_size_range is None.
    block_size_days: int = 5
    #
    # Optional (min, max) INCLUSIVE range of block lengths in trading days. If set, each
    # block's length is drawn uniformly from [min, max] (so blocks vary in size); this
    # overrides block_size_days. Set to None to use the fixed block_size_days instead.
    block_size_range: Optional[Tuple[int, int]] = None

    # === Position Sizing ===
    # Method for determining risk per trade:
    #   "pct_current_equity": risk_pct % of current equity (compounding, grows with profits)
    #   "pct_initial_capital": risk_pct % of starting account_size (fixed, no compounding)
    #   "fixed_usd": flat dollar amount (risk_usd field) per trade, regardless of equity
    position_sizing_mode: PositionSizingMode = "pct_current_equity"

    # Risk percentage (used for pct_current_equity or pct_initial_capital modes)
    # E.g., 1.0 means risk 1% per trade. Typical range: 0.5% to 2%
    risk_pct: float = 1.0

    # Fixed risk amount in dollars (used for fixed_usd mode)
    # E.g., 500.0 means risk $500 per trade regardless of current equity
    risk_usd: float = 500.0

    # === Simulation Control ===
    # Maximum trading days allowed per phase before timeout (prevents infinite loops)
    # If profit target not met after this many days, trial marked as "timeout" failure
    max_trading_days_per_phase: int = 100

    # Number of Monte Carlo trials to run (higher = more accurate stats, slower runtime)
    # Typical: 10k-50k simulations. 10k gives ~1-2% margin of error on pass rate
    num_simulations: int = 10_000

    # Random seed for reproducibility. Set to None for random behavior each run
    random_seed: Optional[int] = 42

    # Number of equity curves to save for visualization (plots first N trial curves)
    # Higher = more curves in the dashboard plot, but larger output. Typical: 50-200
    sample_curves_to_keep: int = 50

    # === Risk-per-trade Sweep (pass-rate curve) ===
    # When enabled, after the main run the simulator re-runs the Monte Carlo across a range
    # of risk_pct values and plots overall pass rate (with 95% CI) vs risk. Pass rate is
    # extremely sensitive to position size, so a single headline number is misleading
    # without the sizing it assumes. Only meaningful for the pct_* sizing modes.
    run_risk_sweep_enabled: bool = True

    # Risk-per-trade values (% of equity/capital) to evaluate in the sweep.
    risk_sweep_values: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)

    # Simulations per risk level in the sweep. Lower than num_simulations to keep the
    # multi-point sweep fast; raise for tighter confidence bands on the curve.
    risk_sweep_simulations: int = 3_000


@dataclass
class PhaseResult:
    outcome: Outcome
    days_taken: int
    trades_taken: int
    final_equity: float
    max_drawdown_pct: float  # peak-to-trough dip during this phase, % of starting_capital
    equity_curve: list


@dataclass
class TrialResult:
    phase1_outcome: Outcome
    phase2_outcome: Optional[Outcome]
    overall_pass: bool
    overall_fail_reason: Optional[str]
    total_days: int
    total_trades: int
    final_equity: float
    max_drawdown_pct: float
    equity_curve: list


def _max_drawdown_pct(equity_curve: list, starting_capital: float) -> float:
    curve = np.array(equity_curve)
    running_peak = np.maximum.accumulate(curve)
    drawdown = (running_peak - curve) / starting_capital * 100
    return drawdown.max()


def load_trade_pool(config: MonteCarloConfig):
    """Returns (flat_trades, day_groups, day_trade_counts, trade_pool_df).

    day_groups/day_trade_counts are built over the FULL historical trading
    calendar (orb_trading_days.csv, written by the ORB backtester alongside
    trades_csv_path), not just the days that happen to appear in trades_csv_path.
    Days with zero fired trades have no row in the trades CSV at all, so without
    the calendar they'd be silently invisible to every resample mode -- both
    "bootstrap_days"/"bootstrap_blocks" (which sample day_groups directly) and
    "bootstrap_trades" (which uses day_trade_counts to draw a realistic per-day
    trade count, including zero, instead of a fixed constant).
    """
    df = pd.read_csv(config.trades_csv_path, parse_dates=["date"])
    df = df.sort_values(["date", "entry_time"]).reset_index(drop=True)

    flat_trades = df[["r_multiple", "mae_r"]].to_numpy()

    trades_by_date = {
        date: day_df[["r_multiple", "mae_r"]].to_numpy()
        for date, day_df in df.groupby("date")
    }

    calendar_path = os.path.join(os.path.dirname(config.trades_csv_path), "orb_trading_days.csv")
    if os.path.exists(calendar_path):
        calendar_dates = pd.read_csv(calendar_path, parse_dates=["date"])["date"]
        day_groups = [trades_by_date.get(d, np.empty((0, 2))) for d in calendar_dates]
    else:
        print(
            f"WARNING: trading-day calendar not found at {calendar_path} -- falling back "
            "to a trades-only calendar (days with zero fired trades are silently excluded "
            "from resampling, which overstates trading frequency). Re-run ORB.py to "
            "regenerate orb_trading_days.csv."
        )
        day_groups = list(trades_by_date.values())

    day_trade_counts = np.array([len(day) for day in day_groups])

    return flat_trades, day_groups, day_trade_counts, df


class DaySampler:
    """Yields one simulated trading day of (r_multiple, mae_r) trades at a time.

    - bootstrap_trades: draw a per-day trade COUNT from the empirical historical
      distribution in day_trade_counts (including zero-trade days), then draw that
      many individual trades i.i.d. Falls back to the fixed `trades_per_day`
      constant only if day_trade_counts wasn't built from a real calendar (see
      load_trade_pool).
    - bootstrap_days:   draw one whole historical day i.i.d. (may be empty).
    - bootstrap_blocks: draw contiguous RUNS of historical days (circular block
      bootstrap), handing them back one day at a time so cross-day streaks are
      preserved. The remaining days of the current block are held as instance
      state, so a FRESH sampler must be created per phase (done in simulate_phase).
    """

    def __init__(self, config: MonteCarloConfig, flat_trades, day_groups, day_trade_counts, rng):
        self.config = config
        self.flat_trades = flat_trades
        self.day_groups = day_groups
        self.day_trade_counts = day_trade_counts
        self.rng = rng
        self._block_buffer: list = []

    def next_day(self):
        mode = self.config.resample_mode
        if mode == "bootstrap_trades":
            n = int(self.rng.choice(self.day_trade_counts))
            if n == 0:
                return self.flat_trades[:0]
            idx = self.rng.integers(0, len(self.flat_trades), size=n)
            return self.flat_trades[idx]
        if mode == "bootstrap_days":
            idx = self.rng.integers(0, len(self.day_groups))
            return self.day_groups[idx]
        # bootstrap_blocks: refill the buffer with a fresh random block when empty
        if not self._block_buffer:
            self._block_buffer = self._draw_block()
        return self._block_buffer.pop(0)

    def _draw_block(self) -> list:
        n = len(self.day_groups)
        if self.config.block_size_range is not None:
            lo, hi = self.config.block_size_range
            length = int(self.rng.integers(lo, hi + 1))
        else:
            length = self.config.block_size_days
        length = max(1, length)
        start = int(self.rng.integers(0, n))
        # Circular block: wrap past the end back to the start so every start
        # index is valid and every historical day is equally likely to appear.
        return [self.day_groups[(start + i) % n] for i in range(length)]


def simulate_phase(
    config: MonteCarloConfig,
    starting_capital: float,
    profit_target_pct: float,
    flat_trades,
    day_groups,
    day_trade_counts,
    rng,
) -> PhaseResult:
    equity = starting_capital
    overall_floor = starting_capital * (1 - config.max_overall_loss_pct / 100)
    target_equity = starting_capital * (1 + profit_target_pct / 100)
    # Fixed $ offset, computed once from starting_capital (FTMO's real rule: a
    # static dollar amount off the PREVIOUS day's closing equity, not a % of
    # current equity). Matches ../Funded Phase/monte_carlo.py's mechanic. Since
    # Verification resets to a fresh starting_capital, this offset is correctly
    # recomputed per-phase rather than carried over from phase 1.
    daily_loss_offset = starting_capital * config.daily_loss_limit_pct / 100

    distinct_trading_days = 0
    trades_taken = 0
    equity_curve = [equity]

    # Fresh sampler per phase so any in-progress block does not leak across the
    # phase 1 -> phase 2 account reset.
    sampler = DaySampler(config, flat_trades, day_groups, day_trade_counts, rng)

    for day in range(1, config.max_trading_days_per_phase + 1):
        day_start_equity = equity
        # day_start_equity is already the previous day's closing equity (day 1
        # anchors to starting_capital), so subtracting the fixed offset here
        # reproduces FTMO's real "previous close minus fixed $" daily floor.
        daily_floor = day_start_equity - daily_loss_offset

        day_trades = sampler.next_day()
        if len(day_trades) > 0:
            distinct_trading_days += 1

        for r_multiple, mae_r in day_trades:
            if config.position_sizing_mode == "pct_current_equity":
                risk_amount = equity * config.risk_pct / 100
            elif config.position_sizing_mode == "pct_initial_capital":
                risk_amount = starting_capital * config.risk_pct / 100
            else:  # fixed_usd
                risk_amount = config.risk_usd

            worst_case_equity = equity - mae_r * risk_amount

            if worst_case_equity <= overall_floor:
                equity_curve.append(worst_case_equity)
                return PhaseResult(
                    "overall_loss", day, trades_taken + 1, worst_case_equity,
                    _max_drawdown_pct(equity_curve, starting_capital), equity_curve,
                )
            if worst_case_equity <= daily_floor:
                equity_curve.append(worst_case_equity)
                return PhaseResult(
                    "daily_loss", day, trades_taken + 1, worst_case_equity,
                    _max_drawdown_pct(equity_curve, starting_capital), equity_curve,
                )

            equity += r_multiple * risk_amount
            trades_taken += 1
            equity_curve.append(equity)

        if equity >= target_equity and distinct_trading_days >= config.min_trading_days_per_phase:
            return PhaseResult(
                "passed", day, trades_taken, equity,
                _max_drawdown_pct(equity_curve, starting_capital), equity_curve,
            )

    return PhaseResult(
        "timeout", config.max_trading_days_per_phase, trades_taken, equity,
        _max_drawdown_pct(equity_curve, starting_capital), equity_curve,
    )


def simulate_trial(config: MonteCarloConfig, flat_trades, day_groups, day_trade_counts, rng) -> TrialResult:
    phase1 = simulate_phase(
        config, config.account_size, config.profit_target_phase1_pct,
        flat_trades, day_groups, day_trade_counts, rng
    )

    if phase1.outcome != "passed":
        return TrialResult(
            phase1_outcome=phase1.outcome,
            phase2_outcome=None,
            overall_pass=False,
            overall_fail_reason=phase1.outcome,
            total_days=phase1.days_taken,
            total_trades=phase1.trades_taken,
            final_equity=phase1.final_equity,
            max_drawdown_pct=phase1.max_drawdown_pct,
            equity_curve=phase1.equity_curve,
        )

    phase2 = simulate_phase(
        config, config.account_size, config.profit_target_phase2_pct,
        flat_trades, day_groups, day_trade_counts, rng
    )

    overall_pass = phase2.outcome == "passed"
    return TrialResult(
        phase1_outcome=phase1.outcome,
        phase2_outcome=phase2.outcome,
        overall_pass=overall_pass,
        overall_fail_reason=None if overall_pass else phase2.outcome,
        total_days=phase1.days_taken + phase2.days_taken,
        total_trades=phase1.trades_taken + phase2.trades_taken,
        final_equity=phase2.final_equity,
        max_drawdown_pct=max(phase1.max_drawdown_pct, phase2.max_drawdown_pct),
        equity_curve=phase1.equity_curve + phase2.equity_curve,
    )


def run_monte_carlo(config: MonteCarloConfig):
    flat_trades, day_groups, day_trade_counts, trade_pool_df = load_trade_pool(config)
    rng = np.random.default_rng(config.random_seed)

    rows = []
    sample_curves = []
    for i in range(config.num_simulations):
        trial = simulate_trial(config, flat_trades, day_groups, day_trade_counts, rng)
        rows.append(
            {
                "phase1_outcome": trial.phase1_outcome,
                "phase2_outcome": trial.phase2_outcome,
                "overall_pass": trial.overall_pass,
                "overall_fail_reason": trial.overall_fail_reason,
                "total_days": trial.total_days,
                "total_trades": trial.total_trades,
                "final_equity": trial.final_equity,
                "max_drawdown_pct": trial.max_drawdown_pct,
            }
        )
        if i < config.sample_curves_to_keep:
            sample_curves.append(trial.equity_curve)

    return pd.DataFrame(rows), sample_curves, trade_pool_df


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score confidence interval for a binomial proportion, as percentages."""
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (center - margin) / denom * 100, (center + margin) / denom * 100


def compute_summary_stats(results_df: pd.DataFrame, config: MonteCarloConfig) -> dict:
    n = len(results_df)
    phase1_pass = results_df["phase1_outcome"] == "passed"
    overall_pass = results_df["overall_pass"]
    passed_df = results_df[overall_pass]
    failed_df = results_df[~overall_pass]

    fail_reason_counts = results_df.loc[~overall_pass, "overall_fail_reason"].value_counts()
    ci_low, ci_high = _wilson_ci(int(overall_pass.sum()), n)

    stats = {
        "num_simulations": n,
        "overall_pass_pct": 100 * overall_pass.sum() / n,
        "overall_pass_ci_low": ci_low,
        "overall_pass_ci_high": ci_high,
        "phase1_pass_pct": 100 * phase1_pass.sum() / n,
        "phase2_pass_pct": 100 * overall_pass.sum() / phase1_pass.sum() if phase1_pass.sum() > 0 else 0.0,
        "fail_daily_loss_pct": 100 * fail_reason_counts.get("daily_loss", 0) / n,
        "fail_overall_loss_pct": 100 * fail_reason_counts.get("overall_loss", 0) / n,
        "fail_timeout_pct": 100 * fail_reason_counts.get("timeout", 0) / n,
        "avg_days_passed": passed_df["total_days"].mean() if not passed_df.empty else 0.0,
        "median_days_passed": passed_df["total_days"].median() if not passed_df.empty else 0.0,
        "p25_days_passed": passed_df["total_days"].quantile(0.25) if not passed_df.empty else 0.0,
        "p75_days_passed": passed_df["total_days"].quantile(0.75) if not passed_df.empty else 0.0,
        "p10_days_passed": passed_df["total_days"].quantile(0.10) if not passed_df.empty else 0.0,
        "p90_days_passed": passed_df["total_days"].quantile(0.90) if not passed_df.empty else 0.0,
        "avg_trades_passed": passed_df["total_trades"].mean() if not passed_df.empty else 0.0,
        "median_trades_passed": passed_df["total_trades"].median() if not passed_df.empty else 0.0,
        "avg_days_failed": failed_df["total_days"].mean() if not failed_df.empty else 0.0,
        "avg_final_equity_passed": passed_df["final_equity"].mean() if not passed_df.empty else 0.0,
        "median_final_equity_passed": passed_df["final_equity"].median() if not passed_df.empty else 0.0,
        "avg_final_equity_failed": failed_df["final_equity"].mean() if not failed_df.empty else 0.0,
        "avg_max_drawdown_pct": results_df["max_drawdown_pct"].mean(),
        "median_max_drawdown_pct": results_df["max_drawdown_pct"].median(),
        "p90_max_drawdown_pct": results_df["max_drawdown_pct"].quantile(0.90),
        "avg_max_drawdown_pct_passed": passed_df["max_drawdown_pct"].mean() if not passed_df.empty else 0.0,
        "avg_max_drawdown_pct_failed": failed_df["max_drawdown_pct"].mean() if not failed_df.empty else 0.0,
    }
    return stats


def compute_input_data_stats(trade_pool_df: pd.DataFrame) -> dict:
    return {
        "num_trades": len(trade_pool_df),
        "date_start": trade_pool_df["date"].min().date(),
        "date_end": trade_pool_df["date"].max().date(),
        "win_rate_pct": 100 * (trade_pool_df["net_pnl"] > 0).mean(),
        "avg_r_multiple": trade_pool_df["r_multiple"].mean(),
        "avg_mae_r": trade_pool_df["mae_r"].mean(),
        "avg_mfe_r": trade_pool_df["mfe_r"].mean(),
        "worst_r_multiple": trade_pool_df["r_multiple"].min(),
        "best_r_multiple": trade_pool_df["r_multiple"].max(),
    }


def run_risk_sweep(base_config: MonteCarloConfig, risk_values=None) -> pd.DataFrame:
    """Re-run the Monte Carlo across a range of risk_pct values and collect the
    overall pass rate (with CI) and failure breakdown at each level.

    Uses the same random_seed for every point (common random numbers), so the
    curve reflects the effect of risk sizing rather than sampling noise between
    points. Runs risk_sweep_simulations trials per point to keep it fast.
    """
    risk_values = risk_values if risk_values is not None else base_config.risk_sweep_values
    rows = []
    for rp in risk_values:
        cfg = replace(
            base_config,
            risk_pct=rp,
            num_simulations=base_config.risk_sweep_simulations,
        )
        results_df, _, _ = run_monte_carlo(cfg)
        s = compute_summary_stats(results_df, cfg)
        rows.append(
            {
                "risk_pct": rp,
                "overall_pass_pct": s["overall_pass_pct"],
                "ci_low": s["overall_pass_ci_low"],
                "ci_high": s["overall_pass_ci_high"],
                "phase1_pass_pct": s["phase1_pass_pct"],
                "fail_daily_loss_pct": s["fail_daily_loss_pct"],
                "fail_overall_loss_pct": s["fail_overall_loss_pct"],
                "fail_timeout_pct": s["fail_timeout_pct"],
            }
        )
    return pd.DataFrame(rows)


def format_risk_sweep_report(sweep_df: pd.DataFrame) -> str:
    width = 60
    lines = [
        "RISK-PER-TRADE SWEEP".center(width),
        "=" * width,
        f"{'Risk %':>7}{'Pass %':>9}{'95% CI':>17}{'Daily':>8}{'Overall':>9}{'T-out':>7}",
        "-" * width,
    ]
    for _, r in sweep_df.iterrows():
        ci = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        lines.append(
            f"{r['risk_pct']:>7.2f}{r['overall_pass_pct']:>9.2f}{ci:>17}"
            f"{r['fail_daily_loss_pct']:>8.1f}{r['fail_overall_loss_pct']:>9.1f}{r['fail_timeout_pct']:>7.1f}"
        )
    best = sweep_df.loc[sweep_df["overall_pass_pct"].idxmax()]
    lines += [
        "-" * width,
        f"Peak pass rate {best['overall_pass_pct']:.2f}% at risk {best['risk_pct']:.2f}%",
        "=" * width,
    ]
    return "\n".join(lines)


def plot_risk_curve(sweep_df: pd.DataFrame, config: MonteCarloConfig, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    rp = sweep_df["risk_pct"]

    # Left: overall pass rate vs risk, with 95% Wilson CI band
    ax = axes[0]
    ax.plot(rp, sweep_df["overall_pass_pct"], marker="o", color="#2a78d6", linewidth=1.8, label="Overall pass rate")
    ax.fill_between(rp, sweep_df["ci_low"], sweep_df["ci_high"], color="#2a78d6", alpha=0.18, label="95% CI")
    best = sweep_df.loc[sweep_df["overall_pass_pct"].idxmax()]
    ax.axvline(best["risk_pct"], color="#2ca02c", linestyle="--", linewidth=1,
               label=f"Peak {best['overall_pass_pct']:.1f}% @ {best['risk_pct']:.2f}%")
    ax.set_title("Overall Pass Rate vs Risk per Trade")
    ax.set_xlabel("Risk per trade (% of equity)")
    ax.set_ylabel("Overall pass rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Right: failure-mode breakdown vs risk
    ax = axes[1]
    ax.plot(rp, sweep_df["fail_daily_loss_pct"], marker="o", color=_OUTCOME_COLORS["daily_loss"], label="Daily loss breach")
    ax.plot(rp, sweep_df["fail_overall_loss_pct"], marker="o", color=_OUTCOME_COLORS["overall_loss"], label="Overall loss breach")
    ax.plot(rp, sweep_df["fail_timeout_pct"], marker="o", color=_OUTCOME_COLORS["timeout"], label="Timed out")
    ax.set_title("Failure Modes vs Risk per Trade")
    ax.set_xlabel("Risk per trade (% of equity)")
    ax.set_ylabel("% of trials")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Risk Sensitivity  ({config.resample_mode}, {config.risk_sweep_simulations:,} sims/point)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=120)
    plt.close()


def format_config_report(config: MonteCarloConfig) -> str:
    width = 60
    sizing_line = {
        "pct_current_equity": f"{config.risk_pct}% of current equity (compounding)",
        "pct_initial_capital": f"{config.risk_pct}% of initial capital (fixed)",
        "fixed_usd": f"${config.risk_usd:,.2f} flat",
    }[config.position_sizing_mode]

    lines = [
        "=" * width,
        "MONTE CARLO CONFIG".center(width),
        "=" * width,
        f"{'Account size':28}${config.account_size:,.2f}",
        f"{'Phase 1 target':28}{config.profit_target_phase1_pct}%",
        f"{'Phase 2 target':28}{config.profit_target_phase2_pct}%",
        f"{'Daily loss limit':28}{config.daily_loss_limit_pct}% of IC (static $, off prev close)",
        f"{'Max overall loss':28}{config.max_overall_loss_pct}% (static, of initial capital)",
        f"{'Min trading days/phase':28}{config.min_trading_days_per_phase}",
        f"{'Resample mode':28}{config.resample_mode}",
    ]
    if config.resample_mode == "bootstrap_trades":
        lines.append(f"{'Trades per day':28}empirical (calendar) / {config.trades_per_day} fallback")
    if config.resample_mode == "bootstrap_blocks":
        if config.block_size_range is not None:
            block_line = f"{config.block_size_range[0]}-{config.block_size_range[1]} days (random per block)"
        else:
            block_line = f"{config.block_size_days} days (fixed)"
        lines.append(f"{'Block size':28}{block_line}")
    lines += [
        f"{'Position sizing':28}{sizing_line}",
        f"{'Max trading days/phase':28}{config.max_trading_days_per_phase}",
        f"{'Simulations':28}{config.num_simulations:,}",
        f"{'Random seed':28}{config.random_seed}",
        "=" * width,
    ]
    return "\n".join(lines)


def format_input_data_report(stats: dict) -> str:
    width = 60
    lines = [
        "INPUT TRADE POOL (from orb_trades.csv)".center(width),
        "=" * width,
        f"{'Trades':28}{stats['num_trades']}",
        f"{'Date range':28}{stats['date_start']}  to  {stats['date_end']}",
        f"{'Win rate':28}{stats['win_rate_pct']:.2f}%",
        f"{'Avg R-multiple':28}{stats['avg_r_multiple']:.2f}R",
        f"{'Avg MAE':28}{stats['avg_mae_r']:.2f}R",
        f"{'Avg MFE':28}{stats['avg_mfe_r']:.2f}R",
        f"{'Best / Worst trade':28}{stats['best_r_multiple']:.2f}R / {stats['worst_r_multiple']:.2f}R",
        "=" * width,
    ]
    return "\n".join(lines)


def format_results_report(stats: dict) -> str:
    width = 60
    lines = [
        "MONTE CARLO RESULTS".center(width),
        "=" * width,
        "-- Pass probability " + "-" * (width - 20),
        f"{'Simulations run':28}{stats['num_simulations']:,}",
        f"{'Overall pass rate':28}{stats['overall_pass_pct']:.2f}%",
        f"{'  95% confidence interval':28}[{stats['overall_pass_ci_low']:.2f}%, {stats['overall_pass_ci_high']:.2f}%]",
        f"{'Phase 1 pass rate':28}{stats['phase1_pass_pct']:.2f}%",
        f"{'Phase 2 pass rate*':28}{stats['phase2_pass_pct']:.2f}%",
        "  * conditional on passing phase 1",
        "",
        "-- Failure breakdown " + "-" * (width - 21),
        f"{'Daily loss breach':28}{stats['fail_daily_loss_pct']:.2f}%",
        f"{'Overall loss breach':28}{stats['fail_overall_loss_pct']:.2f}%",
        f"{'Timed out':28}{stats['fail_timeout_pct']:.2f}%",
        "",
        "-- Days to pass (passing trials) " + "-" * (width - 33),
        f"{'Average':28}{stats['avg_days_passed']:.1f} days",
        f"{'Median':28}{stats['median_days_passed']:.1f} days",
        f"{'25th / 75th percentile':28}{stats['p25_days_passed']:.1f} / {stats['p75_days_passed']:.1f} days",
        f"{'10th / 90th percentile':28}{stats['p10_days_passed']:.1f} / {stats['p90_days_passed']:.1f} days",
        f"{'Avg / median trades taken':28}{stats['avg_trades_passed']:.1f} / {stats['median_trades_passed']:.1f}",
        "",
        "-- Drawdown (peak-to-trough, % of starting capital) " + "-" * (width - 53),
        f"{'Average (all trials)':28}{stats['avg_max_drawdown_pct']:.2f}%",
        f"{'Median (all trials)':28}{stats['median_max_drawdown_pct']:.2f}%",
        f"{'90th percentile':28}{stats['p90_max_drawdown_pct']:.2f}%",
        f"{'Average, passing trials':28}{stats['avg_max_drawdown_pct_passed']:.2f}%",
        f"{'Average, failing trials':28}{stats['avg_max_drawdown_pct_failed']:.2f}%",
        "",
        "-- Final equity " + "-" * (width - 16),
        f"{'Avg, passing trials':28}${stats['avg_final_equity_passed']:,.2f}",
        f"{'Median, passing trials':28}${stats['median_final_equity_passed']:,.2f}",
        f"{'Avg, failing trials':28}${stats['avg_final_equity_failed']:,.2f}",
        "",
        "-- Failing trials " + "-" * (width - 18),
        f"{'Avg days before failing':28}{stats['avg_days_failed']:.1f} days",
        "=" * width,
    ]
    return "\n".join(lines)


_OUTCOME_COLORS = {
    "passed": "#2ca02c",
    "daily_loss": "#ff7f0e",
    "overall_loss": "#d62728",
    "timeout": "#7f7f7f",
}


def _classify_outcome(row) -> str:
    return "passed" if row["overall_pass"] else row["overall_fail_reason"]


def plot_dashboard(
    results_df: pd.DataFrame,
    sample_curves: list,
    trade_pool_df: pd.DataFrame,
    config: MonteCarloConfig,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    outcomes = results_df.apply(_classify_outcome, axis=1)
    passed_df = results_df[results_df["overall_pass"]]
    failed_df = results_df[~results_df["overall_pass"]]

    # 1. Sample equity paths
    ax = axes[0, 0]
    for curve in sample_curves:
        ax.plot(curve, alpha=0.5, linewidth=0.7)
    ax.axhline(config.account_size * (1 + config.profit_target_phase1_pct / 100), color="green", linestyle="--", label="Phase 1 target", linewidth=1)
    ax.axhline(config.account_size * (1 - config.daily_loss_limit_pct / 100), color="orange", linestyle=":", label="Daily floor (day 1)", linewidth=1)
    ax.axhline(config.account_size * (1 - config.max_overall_loss_pct / 100), color="red", linestyle="--", label="Overall floor", linewidth=1)
    ax.set_title("Sample Equity Paths")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=7)

    # 2. Outcome breakdown pie
    ax = axes[0, 1]
    counts = outcomes.value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=[_OUTCOME_COLORS.get(k, "#999999") for k in counts.index])
    ax.set_title("Outcome Breakdown")

    # 3. Days-to-pass histogram
    ax = axes[0, 2]
    if not passed_df.empty:
        ax.hist(passed_df["total_days"], bins=20, color="#2a78d6")
    ax.set_title("Days to Pass (passing trials)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Count")

    # 4. Trades-taken histogram
    ax = axes[1, 0]
    if not passed_df.empty:
        ax.hist(passed_df["total_trades"], bins=20, color="#1baf7a")
    ax.set_title("Trades Taken (passing trials)")
    ax.set_xlabel("Trades")
    ax.set_ylabel("Count")

    # 5. Final equity distribution, passed vs failed
    ax = axes[1, 1]
    if not passed_df.empty:
        ax.hist(passed_df["final_equity"], bins=20, alpha=0.6, label="passed", color=_OUTCOME_COLORS["passed"])
    if not failed_df.empty:
        ax.hist(failed_df["final_equity"], bins=20, alpha=0.6, label="failed", color=_OUTCOME_COLORS["overall_loss"])
    ax.set_title("Final Equity Distribution")
    ax.set_xlabel("Equity ($)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # 6. Failure reason bar chart
    ax = axes[1, 2]
    fail_counts = failed_df["overall_fail_reason"].value_counts()
    ax.bar(fail_counts.index, fail_counts.values, color=[_OUTCOME_COLORS.get(k, "#999999") for k in fail_counts.index])
    ax.set_title("Failure Reasons")
    ax.set_ylabel("Count")

    # 7. Pass rate by phase
    ax = axes[2, 0]
    phase1_rate = 100 * (results_df["phase1_outcome"] == "passed").mean()
    phase2_rate = 100 * results_df["overall_pass"].mean() / (phase1_rate / 100) if phase1_rate > 0 else 0.0
    ax.bar(["Phase 1", "Phase 2*"], [phase1_rate, phase2_rate], color=["#2a78d6", "#1baf7a"])
    ax.set_title("Pass Rate by Phase (*cond. on phase 1)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    # 8. Convergence of the overall pass-rate estimate
    ax = axes[2, 1]
    running_pass_rate = 100 * results_df["overall_pass"].expanding().mean()
    ax.plot(running_pass_rate.values, color="#4a3aa7", linewidth=1)
    ax.set_title("Pass Rate Convergence")
    ax.set_xlabel("Simulation #")
    ax.set_ylabel("Cumulative pass rate (%)")

    # 9. Input trade pool R-multiple distribution
    ax = axes[2, 2]
    ax.hist(trade_pool_df["r_multiple"], bins=20, color="#eda100")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Input Trade R-Multiple Distribution")
    ax.set_xlabel("R multiple")
    ax.set_ylabel("Count")

    fig.suptitle("FTMO Monte Carlo Dashboard", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    # === CUSTOMIZE THESE SETTINGS BEFORE RUNNING ===
    config = MonteCarloConfig(
        # Reshuffle mode: "bootstrap_trades", "bootstrap_days", or "bootstrap_blocks".
        # "bootstrap_blocks" preserves cross-day streaks and is the least optimistic.
        resample_mode="bootstrap_blocks",

        # If resample_mode="bootstrap_trades", how many trades per simulated day
        trades_per_day=2,

        # Block resampling (only used when resample_mode="bootstrap_blocks"):
        #   block_size_days   -> fixed run length in trading days
        #   block_size_range  -> (min, max) to draw a random length per block; overrides
        #                        block_size_days. Set to None to use the fixed size.
        block_size_days=5,
        block_size_range=None,  # e.g. (3, 15) for variable-length blocks

        # Position sizing: "pct_current_equity", "pct_initial_capital", or "fixed_usd"
        position_sizing_mode="pct_current_equity",

        # Risk per trade as % of equity (only used for pct_* modes)
        risk_pct=1.0,

        # Number of simulations to run. Higher = more accurate but slower (10k to 50k typical)
        num_simulations=10_000,

        # Risk-sweep pass-rate curve: sweep risk_pct and plot pass rate vs risk.
        run_risk_sweep_enabled=True,
        risk_sweep_values=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
        risk_sweep_simulations=3_000,

        # Optional: change account_size, profit targets, loss limits, etc. here
        # account_size=100_000.0,
        # profit_target_phase1_pct=10.0,
        # daily_loss_limit_pct=5.0,
    )

    results_df, sample_curves, trade_pool_df = run_monte_carlo(config)

    out_dir = os.path.dirname(__file__)
    print(format_config_report(config))
    print()
    print(format_input_data_report(compute_input_data_stats(trade_pool_df)))
    print()
    print(format_results_report(compute_summary_stats(results_df, config)))

    results_csv = os.path.join(out_dir, "mc_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved per-trial results to {results_csv}")

    dashboard_png = os.path.join(out_dir, "mc_dashboard.png")
    plot_dashboard(results_df, sample_curves, trade_pool_df, config, dashboard_png)
    print(f"Saved dashboard to {dashboard_png}")

    if config.run_risk_sweep_enabled:
        print("\nRunning risk-per-trade sweep...")
        sweep_df = run_risk_sweep(config)
        print()
        print(format_risk_sweep_report(sweep_df))

        sweep_csv = os.path.join(out_dir, "mc_risk_sweep.csv")
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"\nSaved risk sweep to {sweep_csv}")

        risk_curve_png = os.path.join(out_dir, "mc_risk_curve.png")
        plot_risk_curve(sweep_df, config, risk_curve_png)
        print(f"Saved risk curve to {risk_curve_png}")
