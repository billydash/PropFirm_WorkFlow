"""Monte Carlo simulator for the FTMO 50k 2-Step FUNDED (live) account,
bootstrapped from the ORB backtester's trade log (Current/Backtesters/ORB/orb_trades.csv).

This models the account a trader receives AFTER passing both the Challenge and
Verification phases (see ../Challenge Phase/monte_carlo.py for that evaluation).
The funded account has no profit target and no minimum trading days: it simply
runs until either loss limit is breached, or indefinitely in reality. This
simulator answers: given the strategy's historical trade distribution, how long
does the funded account survive, and how much does the trader get paid before it
fails or an assumed time horizon is reached?

Rules modeled (ftmo.com/en/trading-objectives, 2026):
  - Maximum Daily Loss: account equity cannot drop below the PREVIOUS day's
    closing equity minus a FIXED dollar amount equal to daily_loss_limit_pct%
    of the Initial Capital. This offset is a static dollar figure -- it does
    NOT scale with the current day's equity.
  - Maximum Overall Loss: static floor equal to max_overall_loss_pct% down from
    the Initial Capital. This floor is anchored to the ORIGINAL initial capital
    forever and never moves, regardless of current equity or payouts.
  - No profit target, no minimum trading days (both only apply during the
    Challenge/Verification phases).

Both this file and ../Challenge Phase/monte_carlo.py compute the daily floor
identically: a static dollar offset (daily_loss_limit_pct% of Initial Capital)
subtracted from the previous day's closing equity. Challenge Phase previously
approximated this as a floating percentage of day-start equity; that was a bug,
fixed 2026-07-07, not an intentional design choice -- see CLAUDE.md.

Payouts: FTMO pays out on a real-money cadence of roughly every 14 calendar
days. This simulator only advances in TRADING days (no calendar/weekend
modeling), so `payout_interval_trading_days` (default 10) is a documented
approximation of that cadence, not an exact calendar reproduction.

Key mechanical property worth remembering when reading the results: because the
daily-loss offset is a FIXED dollar amount, payouts never shrink the cushion
against the daily floor (today's start equity is already post-withdrawal, and
the gap to the daily floor is always exactly that fixed offset). Payouts DO
shrink the cushion against the overall floor, since that floor never moves while
every withdrawal pulls equity back down toward the Initial Capital. So frequent
or large payouts trade survival (mostly via more overall-loss busts) for
realized income -- see the payout sweep at the bottom of this run.

Trading calendar: day_groups/day_mae_rs are built from
Backtesters/ORB/orb_trading_days.csv (the full set of real trading days the ORB
backtest ran over) when that file is present alongside trades_csv_path, so days
that produced zero trades are correctly represented as idle days during resampling
instead of being silently absent from the trade pool.
"""

import math
import os
from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Tuple, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ResampleMode = Literal["bootstrap_days", "bootstrap_blocks"]
PositionSizingMode = Literal["pct_current_equity", "pct_initial_capital", "fixed_usd"]
Outcome = Literal["survived", "daily_loss", "overall_loss", "concurrent_daily_loss"]


@dataclass
class FundedMonteCarloConfig:
    # Path to trade CSV from ORB backtester. Default points to ../Backtesters/ORB/orb_trades.csv
    trades_csv_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "..", "Backtesters/ORB", "orb_trades.csv"
        )
    )

    # === FTMO Funded Account Rules ===
    # Initial Capital (IC) of the funded account (e.g., $50k)
    account_size: float = 50_000.0

    # Maximum Daily Loss, as % of IC. Converted internally to a FIXED dollar
    # offset (IC * pct/100) subtracted from the PREVIOUS day's closing equity.
    # This offset never scales with current equity -- only the anchor (previous
    # close) moves day to day.
    daily_loss_limit_pct: float = 5.0

    # Maximum Overall Loss, as % of IC. Floor = IC * (1 - pct/100), computed
    # ONCE from the original account_size and never recomputed against current
    # or peak equity. Static for the life of the account.
    max_overall_loss_pct: float = 10.0

    # No profit_target field: the funded account has none (unlike Challenge/
    # Verification). No min_trading_days field: none required either.

    # === Trade Resampling Mode ===
    # "bootstrap_days": shuffle entire historical trading days (preserves correlation within days)
    # "bootstrap_blocks": draw contiguous RUNS of historical days (circular block
    #     bootstrap), preserving cross-day streaks/clustering that the i.i.d.
    #     modes destroy. See block settings below.
    # A third mode, "bootstrap_trades" (shuffle individual trades i.i.d.), was
    # removed: it has no day identity, so it can't carry day_worst_case_mae_r
    # (see orb_trading_days.csv / ORB.py's compute_day_worst_case_mae_r), which
    # the daily-loss check below depends on to catch same-day overlapping/
    # compounding trades. Use "bootstrap_days" or "bootstrap_blocks" instead.
    resample_mode: ResampleMode = "bootstrap_blocks"

    # === Block Resampling (only used when resample_mode == "bootstrap_blocks") ===
    # Fixed block length in trading days. Used when block_size_range is None.
    block_size_days: int = 5
    # Optional (min, max) INCLUSIVE range of block lengths in trading days; if set,
    # each block's length is drawn uniformly from this range, overriding block_size_days.
    block_size_range: Optional[Tuple[int, int]] = None

    # === Position Sizing ===
    #   "pct_current_equity": risk_pct % of current equity (compounding)
    #   "pct_initial_capital": risk_pct % of starting account_size (fixed, no compounding)
    #   "fixed_usd": flat dollar amount (risk_usd field) per trade
    # Default is "pct_initial_capital": the daily-loss offset is a FIXED dollar
    # amount off Initial Capital (see module docstring) and never rescales, so
    # "pct_current_equity" lets dollar risk-per-trade compound upward over this
    # account's long, uncapped horizon while the daily cushion stays flat --
    # silently shrinking the safety margin as equity grows. "pct_current_equity"
    # is still fully supported for anyone who deliberately wants that tradeoff;
    # see avg_max_bust_excursion_ratio / the WARNING in the report below for a
    # diagnostic of how close it's cutting things.
    position_sizing_mode: PositionSizingMode = "pct_initial_capital"
    risk_pct: float = 1.0
    risk_usd: float = 500.0

    # === Simulation Control ===
    # Right-censoring horizon in trading days (~3 years default). The funded
    # account has no profit target, so it would otherwise run forever -- this
    # caps compute and lets us report "% still alive at N days" as the headline.
    max_simulated_days: int = 750

    num_simulations: int = 10_000
    random_seed: Optional[int] = 42
    sample_curves_to_keep: int = 50

    # === Payouts ===
    # Master toggle for the main single run (the payout sweep below always
    # enables payouts and varies interval/withdraw_pct regardless of this flag).
    payouts_enabled: bool = True

    # Cadence of payout events, in TRADING days. Documented approximation of
    # FTMO's real ~14 calendar-day payout cycle (see module docstring).
    payout_interval_trading_days: int = 10

    # % of profit accrued since the last payout event (equity - watermark, floored
    # at 0) that gets withdrawn from tracked equity at each payout event.
    payout_withdraw_pct: float = 100.0

    # % of each withdrawn dollar that is the trader's actual take-home pay (the
    # rest is FTMO's cut). Affects only the "money made" reporting metric, not
    # the equity/survival mechanics (the FULL withdrawal amount always leaves
    # tracked equity regardless of this split).
    profit_split_pct: float = 80.0

    # === Payout Sweep (frequency x size sensitivity analysis) ===
    run_payout_sweep_enabled: bool = True
    payout_sweep_intervals: Tuple[int, ...] = (5, 10, 15, 20, 30, 45)
    payout_sweep_withdraw_pcts: Tuple[float, ...] = (25.0, 50.0, 75.0, 100.0)
    payout_sweep_simulations: int = 2_000

    # === Risk-per-trade Sweep (survival % and trader take-home vs risk_pct) ===
    # Holds payout settings and everything else fixed at this config's values and
    # only varies risk_pct, mirroring how the payout sweep above holds risk_pct
    # fixed and varies payout settings instead.
    run_risk_sweep_enabled: bool = True
    risk_sweep_values: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    risk_sweep_simulations: int = 2_000

    def __post_init__(self):
        if self.resample_mode not in get_args(ResampleMode):
            raise ValueError(
                f"resample_mode={self.resample_mode!r} is not valid -- 'bootstrap_trades' "
                "was removed (it has no day identity to carry day_worst_case_mae_r). "
                f"Use one of {get_args(ResampleMode)}."
            )


@dataclass
class TrialResult:
    outcome: Outcome
    days_survived: int
    trades_taken: int
    final_equity: float
    max_drawdown_pct: float
    total_payouts_usd: float
    trader_take_home_usd: float
    num_payout_events: int
    equity_curve: list
    max_bust_excursion_ratio: float


def _max_drawdown_pct(equity_curve: list, starting_capital: float) -> float:
    curve = np.array(equity_curve)
    running_peak = np.maximum.accumulate(curve)
    drawdown = (running_peak - curve) / starting_capital * 100
    return drawdown.max()


def load_trade_pool(config: FundedMonteCarloConfig):
    """Returns (day_groups, day_mae_rs, trade_pool_df).

    day_groups/day_mae_rs are built over the FULL historical trading calendar
    (orb_trading_days.csv, written by the ORB backtester alongside
    trades_csv_path), not just the days that happen to appear in trades_csv_path.
    Days with zero fired trades have no row in the trades CSV at all, so without
    the calendar they'd be silently invisible to resampling.

    day_mae_rs holds each calendar day's day_worst_case_mae_r (from ORB.py's
    compute_day_worst_case_mae_r -- the worst-case combined drawdown across that
    day's trades, accounting for same-day overlap/compounding that per-trade
    checks miss), aligned index-for-index with day_groups. NaN if the calendar
    file predates that column (re-run ORB.py to regenerate it).
    """
    df = pd.read_csv(config.trades_csv_path, parse_dates=["date"])
    df = df.sort_values(["date", "entry_time"]).reset_index(drop=True)

    trades_by_date = {
        date: day_df[["r_multiple", "mae_r"]].to_numpy()
        for date, day_df in df.groupby("date")
    }

    calendar_path = os.path.join(os.path.dirname(config.trades_csv_path), "orb_trading_days.csv")
    if os.path.exists(calendar_path):
        calendar_df = pd.read_csv(calendar_path, parse_dates=["date"])
        calendar_dates = calendar_df["date"]
        day_groups = [trades_by_date.get(d, np.empty((0, 2))) for d in calendar_dates]
        if "day_worst_case_mae_r" in calendar_df.columns:
            day_mae_rs = calendar_df["day_worst_case_mae_r"].to_numpy()
        else:
            print(
                "WARNING: orb_trading_days.csv predates day_worst_case_mae_r -- the "
                "daily-loss check will fall back to each day's single largest per-trade "
                "mae_r, which understates same-day overlap/compounding. Re-run ORB.py."
            )
            day_mae_rs = np.array(
                [max((t[1] for t in day), default=np.nan) for day in day_groups]
            )
    else:
        print(
            f"WARNING: trading-day calendar not found at {calendar_path} -- falling back "
            "to a trades-only calendar (days with zero fired trades are silently excluded "
            "from resampling, which overstates trading frequency). Re-run ORB.py to "
            "regenerate orb_trading_days.csv."
        )
        day_groups = list(trades_by_date.values())
        day_mae_rs = np.array([max((t[1] for t in day), default=np.nan) for day in day_groups])

    return day_groups, day_mae_rs, df


class DaySampler:
    """Yields one simulated trading day at a time, as (trades, day_mae_r):
    trades is an array of (r_multiple, mae_r) rows, day_mae_r is that day's
    day_worst_case_mae_r (worst-case combined drawdown across the day's trades).

    - bootstrap_days:   draw one whole historical day i.i.d. (may be empty).
    - bootstrap_blocks: draw contiguous RUNS of historical days (circular block
      bootstrap), handing them back one day at a time so cross-day streaks are
      preserved.

    Unlike the Challenge Phase version, the funded account has no phase resets,
    so exactly ONE sampler is constructed per trial and used continuously for
    the account's entire simulated life (see simulate_trial).
    """

    def __init__(self, config: FundedMonteCarloConfig, day_groups, day_mae_rs, rng):
        self.config = config
        self.day_groups = day_groups
        self.day_mae_rs = day_mae_rs
        self.rng = rng
        self._block_buffer: list = []

    def next_day(self):
        mode = self.config.resample_mode
        if mode == "bootstrap_days":
            idx = int(self.rng.integers(0, len(self.day_groups)))
            return self.day_groups[idx], self.day_mae_rs[idx]
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
        return [
            (self.day_groups[(start + i) % n], self.day_mae_rs[(start + i) % n])
            for i in range(length)
        ]


def simulate_trial(
    config: FundedMonteCarloConfig, day_groups, day_mae_rs, rng
) -> TrialResult:
    ic = config.account_size
    daily_loss_offset = ic * config.daily_loss_limit_pct / 100  # fixed $, computed once
    overall_floor = ic * (1 - config.max_overall_loss_pct / 100)  # fixed $, anchored to original IC forever

    equity = ic
    equity_at_last_payout = ic  # withdrawal watermark
    total_payouts = 0.0
    num_payout_events = 0
    trades_taken = 0
    equity_curve = [equity]
    max_bust_excursion_ratio = 0.0  # worst-case single-day drawdown $ (day_mae_r) / fixed daily cushion $, across the trial

    sampler = DaySampler(config, day_groups, day_mae_rs, rng)  # one sampler, whole trial

    for day in range(1, config.max_simulated_days + 1):
        # 1. Payout event at the START of the day, before that day's trades, using
        #    equity carried over from the previous day's close. This ordering is
        #    what makes the daily-loss cushion invariant to withdrawals (see below).
        if config.payouts_enabled and day % config.payout_interval_trading_days == 0:
            accrued = max(0.0, equity - equity_at_last_payout)
            withdrawal = accrued * config.payout_withdraw_pct / 100
            if withdrawal > 0:
                equity -= withdrawal
                total_payouts += withdrawal
                num_payout_events += 1
            equity_at_last_payout = equity  # reset watermark to POST-withdrawal equity
            equity_curve.append(equity)

        # 2. Recompute today's daily floor from POST-withdrawal equity. Because
        #    day_start_equity already reflects any withdrawal above, the gap to
        #    daily_floor is always exactly daily_loss_offset -- a fixed dollar
        #    amount -- regardless of withdrawal history. overall_floor never
        #    moves. Whichever floor is numerically HIGHER is the one equity
        #    actually crosses first as it declines intraday, so that must be
        #    the attributed bust reason (do not unconditionally prefer one).
        day_start_equity = equity
        daily_floor = day_start_equity - daily_loss_offset
        if daily_floor >= overall_floor:
            effective_floor = daily_floor
            bust_reason: Outcome = "daily_loss"
        else:
            effective_floor = overall_floor
            bust_reason = "overall_loss"

        # 3. Day-level worst-case check. risk_amount is anchored EXCLUSIVELY to
        #    day-start equity (never intraday floating equity) -- this matches
        #    FTMO's actual daily-loss mechanic (a fixed offset off the prior
        #    close) and keeps it consistent with day_mae_r, which is itself a
        #    multiple of a single fixed risk unit for the whole day (see ORB.py's
        #    compute_day_worst_case_mae_r). day_mae_r reduces exactly to a single
        #    trade's own mae_r on 1-trade days, and captures same-day overlap or
        #    sequential compounding (whichever produced it) on multi-trade days,
        #    which the old per-trade-isolated check couldn't see either way.
        day_trades, day_mae_r = sampler.next_day()
        num_day_trades = len(day_trades)

        if config.position_sizing_mode == "pct_current_equity":
            risk_amount = day_start_equity * config.risk_pct / 100
        elif config.position_sizing_mode == "pct_initial_capital":
            risk_amount = ic * config.risk_pct / 100
        else:  # fixed_usd
            risk_amount = config.risk_usd

        if num_day_trades > 0 and not math.isnan(day_mae_r):
            max_bust_excursion_ratio = max(
                max_bust_excursion_ratio, (day_mae_r * risk_amount) / daily_loss_offset
            )
            worst_case_equity = day_start_equity - day_mae_r * risk_amount

            if worst_case_equity <= effective_floor:
                equity_curve.append(worst_case_equity)
                # Can't attribute a joint day-level breach to one trade -- the
                # portfolio state failed, not an individual position.
                outcome = (
                    "concurrent_daily_loss"
                    if bust_reason == "daily_loss" and num_day_trades > 1
                    else bust_reason
                )
                return TrialResult(
                    outcome=outcome,
                    days_survived=day,
                    trades_taken=trades_taken + num_day_trades,
                    final_equity=worst_case_equity,
                    max_drawdown_pct=_max_drawdown_pct(equity_curve, ic),
                    total_payouts_usd=total_payouts,
                    trader_take_home_usd=total_payouts * config.profit_split_pct / 100,
                    num_payout_events=num_payout_events,
                    equity_curve=equity_curve,
                    max_bust_excursion_ratio=max_bust_excursion_ratio,
                )

        for r_multiple, _ in day_trades:
            equity += r_multiple * risk_amount
            trades_taken += 1
            equity_curve.append(equity)

    # Reached the horizon without busting: right-censored "survived"
    return TrialResult(
        outcome="survived",
        days_survived=config.max_simulated_days,
        trades_taken=trades_taken,
        final_equity=equity,
        max_drawdown_pct=_max_drawdown_pct(equity_curve, ic),
        total_payouts_usd=total_payouts,
        trader_take_home_usd=total_payouts * config.profit_split_pct / 100,
        num_payout_events=num_payout_events,
        equity_curve=equity_curve,
        max_bust_excursion_ratio=max_bust_excursion_ratio,
    )


def run_monte_carlo(config: FundedMonteCarloConfig):
    day_groups, day_mae_rs, trade_pool_df = load_trade_pool(config)
    rng = np.random.default_rng(config.random_seed)

    rows = []
    sample_curves = []
    for i in range(config.num_simulations):
        trial = simulate_trial(config, day_groups, day_mae_rs, rng)
        rows.append(
            {
                "outcome": trial.outcome,
                "days_survived": trial.days_survived,
                "trades_taken": trial.trades_taken,
                "final_equity": trial.final_equity,
                "max_drawdown_pct": trial.max_drawdown_pct,
                "total_payouts_usd": trial.total_payouts_usd,
                "trader_take_home_usd": trial.trader_take_home_usd,
                "num_payout_events": trial.num_payout_events,
                "max_bust_excursion_ratio": trial.max_bust_excursion_ratio,
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


def _percentile_block(df: pd.DataFrame, col: str) -> dict:
    if df.empty:
        return {"avg": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "avg": df[col].mean(),
        "median": df[col].median(),
        "p10": df[col].quantile(0.10),
        "p25": df[col].quantile(0.25),
        "p75": df[col].quantile(0.75),
        "p90": df[col].quantile(0.90),
    }


def compute_summary_stats(results_df: pd.DataFrame, config: FundedMonteCarloConfig) -> dict:
    n = len(results_df)
    survived = results_df["outcome"] == "survived"
    daily_bust = results_df["outcome"] == "daily_loss"
    overall_bust = results_df["outcome"] == "overall_loss"
    concurrent_bust = results_df["outcome"] == "concurrent_daily_loss"
    busted = ~survived

    survived_df = results_df[survived]
    busted_df = results_df[busted]
    daily_df = results_df[daily_bust]
    overall_df = results_df[overall_bust]
    concurrent_df = results_df[concurrent_bust]

    ci_low, ci_high = _wilson_ci(int(survived.sum()), n)

    stats = {
        "num_simulations": n,
        "survival_pct": 100 * survived.sum() / n,
        "survival_ci_low": ci_low,
        "survival_ci_high": ci_high,
        "bust_daily_loss_pct": 100 * daily_bust.sum() / n,
        "bust_overall_loss_pct": 100 * overall_bust.sum() / n,
        # Same daily-loss limit as bust_daily_loss_pct, but the breach couldn't be
        # attributed to one trade -- multiple same-day trades jointly tripped it.
        "bust_concurrent_daily_loss_pct": 100 * concurrent_bust.sum() / n,
        "days_survived_busted": _percentile_block(busted_df, "days_survived"),
        "days_survived_daily": _percentile_block(daily_df, "days_survived"),
        "days_survived_overall": _percentile_block(overall_df, "days_survived"),
        "avg_final_equity_survived": survived_df["final_equity"].mean() if not survived_df.empty else 0.0,
        "median_final_equity_survived": survived_df["final_equity"].median() if not survived_df.empty else 0.0,
        "avg_final_equity_daily": daily_df["final_equity"].mean() if not daily_df.empty else 0.0,
        "avg_final_equity_overall": overall_df["final_equity"].mean() if not overall_df.empty else 0.0,
        "avg_final_equity_concurrent": concurrent_df["final_equity"].mean() if not concurrent_df.empty else 0.0,
        "avg_max_drawdown_pct": results_df["max_drawdown_pct"].mean(),
        "median_max_drawdown_pct": results_df["max_drawdown_pct"].median(),
        "p90_max_drawdown_pct": results_df["max_drawdown_pct"].quantile(0.90),
        "avg_max_drawdown_pct_survived": survived_df["max_drawdown_pct"].mean() if not survived_df.empty else 0.0,
        "avg_max_drawdown_pct_busted": busted_df["max_drawdown_pct"].mean() if not busted_df.empty else 0.0,
        "avg_total_payouts_usd": results_df["total_payouts_usd"].mean(),
        "median_total_payouts_usd": results_df["total_payouts_usd"].median(),
        "avg_trader_take_home_usd": results_df["trader_take_home_usd"].mean(),
        "median_trader_take_home_usd": results_df["trader_take_home_usd"].median(),
        "avg_num_payout_events": results_df["num_payout_events"].mean(),
        # Worst-case single-trade MAE $ / fixed daily-loss cushion $, across each trial's
        # lifetime. Only meaningful diagnostic under "pct_current_equity" sizing, where
        # risk_amount compounds with equity while the cushion (daily_loss_offset) stays
        # fixed -- a ratio approaching/exceeding 1.0 means a single trade's worst case
        # could plausibly consume the entire daily cushion by itself. See position_sizing_mode.
        "avg_max_bust_excursion_ratio": results_df["max_bust_excursion_ratio"].mean(),
        "p90_max_bust_excursion_ratio": results_df["max_bust_excursion_ratio"].quantile(0.90),
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


def run_payout_sweep(
    base_config: FundedMonteCarloConfig, intervals=None, withdraw_pcts=None
) -> pd.DataFrame:
    """2D grid sweep over (payout_interval_trading_days, payout_withdraw_pct),
    holding risk_pct and everything else fixed at base_config's values. Mirrors
    the Challenge Phase risk-sweep pattern: dataclasses.replace(), a fixed shared
    random_seed, and a lower simulation count per grid point.

    Because payout logic consumes no RNG draws, the DaySampler's trade-draw
    sequence is identical across every cell for a given trial index (common
    random numbers) -- the grid isolates the effect of payout parameters from
    sampling noise.
    """
    intervals = intervals if intervals is not None else base_config.payout_sweep_intervals
    withdraw_pcts = withdraw_pcts if withdraw_pcts is not None else base_config.payout_sweep_withdraw_pcts

    rows = []
    for interval in intervals:
        for pct in withdraw_pcts:
            cfg = replace(
                base_config,
                payouts_enabled=True,
                payout_interval_trading_days=interval,
                payout_withdraw_pct=pct,
                num_simulations=base_config.payout_sweep_simulations,
            )
            results_df, _, _ = run_monte_carlo(cfg)
            s = compute_summary_stats(results_df, cfg)
            rows.append(
                {
                    "payout_interval_trading_days": interval,
                    "payout_withdraw_pct": pct,
                    "survival_pct": s["survival_pct"],
                    "survival_ci_low": s["survival_ci_low"],
                    "survival_ci_high": s["survival_ci_high"],
                    "avg_total_money_made": results_df["trader_take_home_usd"].mean(),
                    "median_total_money_made": results_df["trader_take_home_usd"].median(),
                    "avg_num_payout_events": results_df["num_payout_events"].mean(),
                    # Reported alongside the money-made metric so a low-frequency/
                    # low-withdraw-pct cell (near-$0 realized income) isn't misread
                    # as strictly worse -- its survivors may hold large unrealized equity.
                    "avg_final_equity_survived": s["avg_final_equity_survived"],
                }
            )
    return pd.DataFrame(rows)


def run_risk_sweep(base_config: FundedMonteCarloConfig, risk_values=None) -> pd.DataFrame:
    """1D sweep over risk_pct, holding payout settings and everything else fixed at
    base_config's values. Mirrors run_payout_sweep's pattern (dataclasses.replace,
    fixed shared random_seed, lower simulation count per point) but is the inverse
    axis: this holds payout params fixed and varies risk_pct instead.
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
                "survival_pct": s["survival_pct"],
                "survival_ci_low": s["survival_ci_low"],
                "survival_ci_high": s["survival_ci_high"],
                "bust_daily_loss_pct": s["bust_daily_loss_pct"],
                "bust_overall_loss_pct": s["bust_overall_loss_pct"],
                "avg_total_money_made": results_df["trader_take_home_usd"].mean(),
                "median_total_money_made": results_df["trader_take_home_usd"].median(),
                "avg_final_equity_survived": s["avg_final_equity_survived"],
            }
        )
    return pd.DataFrame(rows)


def format_config_report(config: FundedMonteCarloConfig) -> str:
    width = 60
    sizing_line = {
        "pct_current_equity": f"{config.risk_pct}% of current equity (compounding)",
        "pct_initial_capital": f"{config.risk_pct}% of initial capital (fixed)",
        "fixed_usd": f"${config.risk_usd:,.2f} flat",
    }[config.position_sizing_mode]

    lines = [
        "=" * width,
        "FUNDED PHASE MONTE CARLO CONFIG".center(width),
        "=" * width,
        f"{'Account size (Initial Capital)':32}${config.account_size:,.2f}",
        f"{'Daily loss limit':32}{config.daily_loss_limit_pct}% of IC (static $, off prev close)",
        f"{'Max overall loss':32}{config.max_overall_loss_pct}% of IC (static, never moves)",
        f"{'Profit target':32}none (funded account)",
        f"{'Min trading days':32}none (funded account)",
        f"{'Resample mode':32}{config.resample_mode}",
    ]
    if config.resample_mode == "bootstrap_blocks":
        if config.block_size_range is not None:
            block_line = f"{config.block_size_range[0]}-{config.block_size_range[1]} days (random per block)"
        else:
            block_line = f"{config.block_size_days} days (fixed)"
        lines.append(f"{'Block size':32}{block_line}")
    lines += [
        f"{'Position sizing':32}{sizing_line}",
        f"{'Max simulated days (horizon)':32}{config.max_simulated_days}",
        f"{'Simulations':32}{config.num_simulations:,}",
        f"{'Random seed':32}{config.random_seed}",
        "-" * width,
        f"{'Payouts enabled':32}{config.payouts_enabled}",
    ]
    if config.payouts_enabled:
        lines += [
            f"{'Payout interval':32}{config.payout_interval_trading_days} trading days (~2wk approx)",
            f"{'Payout withdraw %':32}{config.payout_withdraw_pct}% of accrued profit",
            f"{'Profit split (trader take)':32}{config.profit_split_pct}%",
        ]
    lines.append("=" * width)
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


def format_results_report(stats: dict, config: FundedMonteCarloConfig) -> str:
    width = 60
    db = stats["days_survived_busted"]
    dd = stats["days_survived_daily"]
    do = stats["days_survived_overall"]
    lines = [
        "FUNDED PHASE MONTE CARLO RESULTS".center(width),
        "=" * width,
        "-- Survival probability " + "-" * (width - 25),
        f"{'Simulations run':28}{stats['num_simulations']:,}",
        f"{'Survival rate (to horizon)':28}{stats['survival_pct']:.2f}%",
        f"{'  95% confidence interval':28}[{stats['survival_ci_low']:.2f}%, {stats['survival_ci_high']:.2f}%]",
        "",
        "-- Bust breakdown " + "-" * (width - 18),
        f"{'Daily loss breach (1 trade)':28}{stats['bust_daily_loss_pct']:.2f}%",
        f"{'Daily loss breach (concurrent)':28}{stats['bust_concurrent_daily_loss_pct']:.2f}%",
        f"{'Overall loss breach':28}{stats['bust_overall_loss_pct']:.2f}%",
        "",
        "-- Days survived, busted trials " + "-" * (width - 32),
        f"{'All busted: avg / median':28}{db['avg']:.1f} / {db['median']:.1f} days",
        f"{'  10th / 90th percentile':28}{db['p10']:.1f} / {db['p90']:.1f} days",
        f"{'Daily-loss busts: avg / median':28}{dd['avg']:.1f} / {dd['median']:.1f} days",
        f"{'Overall-loss busts: avg / median':28}{do['avg']:.1f} / {do['median']:.1f} days",
        "",
        "-- Final equity " + "-" * (width - 16),
        f"{'Avg, survived to horizon':28}${stats['avg_final_equity_survived']:,.2f}",
        f"{'Median, survived to horizon':28}${stats['median_final_equity_survived']:,.2f}",
        f"{'Avg, daily-loss busts':28}${stats['avg_final_equity_daily']:,.2f}",
        f"{'Avg, overall-loss busts':28}${stats['avg_final_equity_overall']:,.2f}",
        "",
        "-- Drawdown (peak-to-trough, % of initial capital) " + "-" * (width - 52),
        f"{'Average (all trials)':28}{stats['avg_max_drawdown_pct']:.2f}%",
        f"{'Median (all trials)':28}{stats['median_max_drawdown_pct']:.2f}%",
        f"{'90th percentile':28}{stats['p90_max_drawdown_pct']:.2f}%",
        f"{'Average, survived':28}{stats['avg_max_drawdown_pct_survived']:.2f}%",
        f"{'Average, busted':28}{stats['avg_max_drawdown_pct_busted']:.2f}%",
        "",
        "-- Payouts & take-home (all trials) " + "-" * (width - 36),
        f"{'Avg / median total withdrawn':28}${stats['avg_total_payouts_usd']:,.2f} / ${stats['median_total_payouts_usd']:,.2f}",
        f"{'Avg / median trader take-home':28}${stats['avg_trader_take_home_usd']:,.2f} / ${stats['median_trader_take_home_usd']:,.2f}",
        f"{'Avg payout events per trial':28}{stats['avg_num_payout_events']:.1f}",
        "=" * width,
    ]
    if config.position_sizing_mode == "pct_current_equity" and stats["p90_max_bust_excursion_ratio"] > 1.0:
        lines += [
            "",
            "WARNING: position_sizing_mode='pct_current_equity' -- dollar risk-per-trade",
            "compounds with equity while the daily-loss cushion stays FIXED (see module",
            f"docstring). p90 max_bust_excursion_ratio = {stats['p90_max_bust_excursion_ratio']:.2f}"
            f" (avg {stats['avg_max_bust_excursion_ratio']:.2f}): in the worst 10% of trials,",
            "a single day's worst-case excursion (day_worst_case_mae_r) could by itself",
            "consume the entire fixed daily cushion. Consider 'pct_initial_capital' instead.",
            "=" * width,
        ]
    return "\n".join(lines)


def format_payout_sweep_report(sweep_df: pd.DataFrame) -> str:
    width = 60
    lines = [
        "PAYOUT FREQUENCY x SIZE SWEEP".center(width),
        "=" * width,
    ]
    for pct in sorted(sweep_df["payout_withdraw_pct"].unique()):
        block = sweep_df[sweep_df["payout_withdraw_pct"] == pct].sort_values(
            "payout_interval_trading_days"
        )
        lines.append(f"-- Withdraw {pct:.0f}% of accrued profit each payout " + "-" * 10)
        lines.append(
            f"{'Interval(days)':>15}{'Survival%':>11}{'95% CI':>17}{'Avg $ made':>13}{'Avg equity*':>13}"
        )
        for _, r in block.iterrows():
            ci = f"[{r['survival_ci_low']:.1f},{r['survival_ci_high']:.1f}]"
            lines.append(
                f"{r['payout_interval_trading_days']:>15.0f}{r['survival_pct']:>11.2f}{ci:>17}"
                f"{r['avg_total_money_made']:>13,.0f}{r['avg_final_equity_survived']:>13,.0f}"
            )
        lines.append("")
    lines.append("* avg final equity among trials that survived to the horizon (unrealized, not yet paid out)")

    best_survival = sweep_df.loc[sweep_df["survival_pct"].idxmax()]
    best_money = sweep_df.loc[sweep_df["avg_total_money_made"].idxmax()]
    lines += [
        "-" * width,
        f"Best survival: {best_survival['survival_pct']:.1f}% at interval="
        f"{best_survival['payout_interval_trading_days']:.0f}d, withdraw={best_survival['payout_withdraw_pct']:.0f}%",
        f"Best avg $ made: ${best_money['avg_total_money_made']:,.0f} at interval="
        f"{best_money['payout_interval_trading_days']:.0f}d, withdraw={best_money['payout_withdraw_pct']:.0f}%",
        "(these two typically trade off against each other)",
        "=" * width,
    ]
    return "\n".join(lines)


def format_risk_sweep_report(sweep_df: pd.DataFrame) -> str:
    width = 60
    lines = [
        "RISK-PER-TRADE SWEEP".center(width),
        "=" * width,
        f"{'Risk %':>7}{'Survival%':>11}{'95% CI':>17}{'Daily':>8}{'Overall':>9}{'Avg $ made':>13}",
        "-" * width,
    ]
    for _, r in sweep_df.iterrows():
        ci = f"[{r['survival_ci_low']:.1f},{r['survival_ci_high']:.1f}]"
        lines.append(
            f"{r['risk_pct']:>7.2f}{r['survival_pct']:>11.2f}{ci:>17}"
            f"{r['bust_daily_loss_pct']:>8.1f}{r['bust_overall_loss_pct']:>9.1f}{r['avg_total_money_made']:>13,.0f}"
        )

    best_survival = sweep_df.loc[sweep_df["survival_pct"].idxmax()]
    best_money = sweep_df.loc[sweep_df["avg_total_money_made"].idxmax()]
    lines += [
        "-" * width,
        f"Best survival: {best_survival['survival_pct']:.1f}% at risk={best_survival['risk_pct']:.2f}%",
        f"Best avg $ made: ${best_money['avg_total_money_made']:,.0f} at risk={best_money['risk_pct']:.2f}%",
        "(these two typically trade off against each other)",
        "=" * width,
    ]
    return "\n".join(lines)


# Status colors (reserved semantics: survived=good, daily_loss=warning, overall_loss=critical)
_OUTCOME_COLORS = {
    "survived": "#0ca30c",
    "daily_loss": "#fab219",
    "concurrent_daily_loss": "#d17a17",  # same family as daily_loss, darker: same limit, joint cause
    "overall_loss": "#d03b3b",
}

# Sequential single-hue ramps (light -> dark), one hue per magnitude chart, no rainbow.
_BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
_AQUA_SEQ = LinearSegmentedColormap.from_list(
    "aqua_seq", ["#d3f3e7", "#9de0c4", "#5cc7a0", "#1baf7a", "#0f7a54"]
)


def plot_dashboard(
    results_df: pd.DataFrame,
    sample_curves: list,
    trade_pool_df: pd.DataFrame,
    config: FundedMonteCarloConfig,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    survived_df = results_df[results_df["outcome"] == "survived"]
    daily_df = results_df[results_df["outcome"] == "daily_loss"]
    concurrent_df = results_df[results_df["outcome"] == "concurrent_daily_loss"]
    overall_df = results_df[results_df["outcome"] == "overall_loss"]
    busted_df = results_df[results_df["outcome"] != "survived"]

    ic = config.account_size
    daily_loss_offset = ic * config.daily_loss_limit_pct / 100
    overall_floor = ic * (1 - config.max_overall_loss_pct / 100)

    # 1. Sample equity paths
    ax = axes[0, 0]
    for curve in sample_curves:
        ax.plot(curve, alpha=0.5, linewidth=0.7)
    ax.axhline(ic - daily_loss_offset, color=_OUTCOME_COLORS["daily_loss"], linestyle=":", label="Daily floor (day 1)", linewidth=1)
    ax.axhline(overall_floor, color=_OUTCOME_COLORS["overall_loss"], linestyle="--", label="Overall floor (static)", linewidth=1)
    ax.set_title("Sample Equity Paths")
    ax.set_xlabel("Step (trades + payout events)")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=7)

    # 2. Outcome breakdown pie
    ax = axes[0, 1]
    counts = results_df["outcome"].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=[_OUTCOME_COLORS.get(k, "#999999") for k in counts.index])
    ax.set_title("Outcome Breakdown")

    # 3. Days-survived histogram, busted trials, by reason
    ax = axes[0, 2]
    if not daily_df.empty:
        ax.hist(daily_df["days_survived"], bins=20, alpha=0.6, label="daily_loss", color=_OUTCOME_COLORS["daily_loss"])
    if not concurrent_df.empty:
        ax.hist(concurrent_df["days_survived"], bins=20, alpha=0.6, label="concurrent_daily_loss", color=_OUTCOME_COLORS["concurrent_daily_loss"])
    if not overall_df.empty:
        ax.hist(overall_df["days_survived"], bins=20, alpha=0.6, label="overall_loss", color=_OUTCOME_COLORS["overall_loss"])
    ax.set_title("Days Survived (busted trials)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # 4. Final equity distribution: survivors vs each bust reason
    ax = axes[1, 0]
    if not survived_df.empty:
        ax.hist(survived_df["final_equity"], bins=20, alpha=0.6, label="survived", color=_OUTCOME_COLORS["survived"])
    if not daily_df.empty:
        ax.hist(daily_df["final_equity"], bins=20, alpha=0.6, label="daily_loss", color=_OUTCOME_COLORS["daily_loss"])
    if not concurrent_df.empty:
        ax.hist(concurrent_df["final_equity"], bins=20, alpha=0.6, label="concurrent_daily_loss", color=_OUTCOME_COLORS["concurrent_daily_loss"])
    if not overall_df.empty:
        ax.hist(overall_df["final_equity"], bins=20, alpha=0.6, label="overall_loss", color=_OUTCOME_COLORS["overall_loss"])
    ax.set_title("Final Equity Distribution")
    ax.set_xlabel("Equity ($)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # 5. Drawdown histogram (all trials)
    ax = axes[1, 1]
    ax.hist(results_df["max_drawdown_pct"], bins=20, color="#2a78d6")
    ax.set_title("Max Drawdown Distribution (all trials)")
    ax.set_xlabel("Drawdown (% of IC)")
    ax.set_ylabel("Count")

    # 6. Bust-reason bar chart
    ax = axes[1, 2]
    fail_counts = busted_df["outcome"].value_counts()
    ax.bar(fail_counts.index, fail_counts.values, color=[_OUTCOME_COLORS.get(k, "#999999") for k in fail_counts.index])
    ax.set_title("Bust Reasons")
    ax.set_ylabel("Count")

    # 7. Survival-rate convergence
    ax = axes[2, 0]
    running_survival = 100 * (results_df["outcome"] == "survived").expanding().mean()
    ax.plot(running_survival.values, color="#4a3aa7", linewidth=1)
    ax.set_title("Survival Rate Convergence")
    ax.set_xlabel("Simulation #")
    ax.set_ylabel("Cumulative survival rate (%)")

    # 8. Payout activity
    ax = axes[2, 1]
    if config.payouts_enabled:
        ax.hist(results_df["total_payouts_usd"], bins=20, color="#1baf7a")
        ax.set_xlabel("Total withdrawn ($)")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "Payouts disabled", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Total Payouts per Trial ($)")

    # 9. Input trade pool R-multiple distribution
    ax = axes[2, 2]
    ax.hist(trade_pool_df["r_multiple"], bins=20, color="#eda100")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Input Trade R-Multiple Distribution")
    ax.set_xlabel("R multiple")
    ax.set_ylabel("Count")

    fig.suptitle("FTMO Funded Phase Monte Carlo Dashboard", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_payout_sweep_heatmaps(
    sweep_df: pd.DataFrame, config: FundedMonteCarloConfig, output_path: str
) -> None:
    survival_grid = sweep_df.pivot(
        index="payout_withdraw_pct", columns="payout_interval_trading_days", values="survival_pct"
    )
    money_grid = sweep_df.pivot(
        index="payout_withdraw_pct", columns="payout_interval_trading_days", values="avg_total_money_made"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    def _annotated_heatmap(ax, grid, cmap, title, fmt, cbar_label):
        im = ax.imshow(grid.values, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{r:g}%" for r in grid.index])
        ax.set_xlabel("Payout interval (trading days)")
        ax.set_ylabel("Payout withdraw %")
        ax.set_title(title)
        vmin, vmax = np.nanmin(grid.values), np.nanmax(grid.values)
        mid = (vmin + vmax) / 2
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid.values[i, j]
                text_color = "white" if val > mid else "black"
                ax.text(j, i, fmt(val), ha="center", va="center", color=text_color, fontsize=8)
        fig.colorbar(im, ax=ax, label=cbar_label)

    _annotated_heatmap(axes[0], survival_grid, _BLUE_SEQ, "Survival % to Horizon", lambda v: f"{v:.0f}%", "Survival %")
    _annotated_heatmap(axes[1], money_grid, _AQUA_SEQ, "Avg Trader Take-Home ($)", lambda v: f"${v:,.0f}", "Avg $ made")

    fig.suptitle(
        f"Payout Frequency x Size Sensitivity  ({config.resample_mode}, {config.payout_sweep_simulations:,} sims/cell)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_risk_sweep_curve(
    sweep_df: pd.DataFrame, config: FundedMonteCarloConfig, output_path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    rp = sweep_df["risk_pct"]

    # Left: survival % vs risk, with 95% Wilson CI band
    ax = axes[0]
    ax.plot(rp, sweep_df["survival_pct"], marker="o", color="#2a78d6", linewidth=1.8, label="Survival %")
    ax.fill_between(rp, sweep_df["survival_ci_low"], sweep_df["survival_ci_high"], color="#2a78d6", alpha=0.18, label="95% CI")
    best_survival = sweep_df.loc[sweep_df["survival_pct"].idxmax()]
    ax.axvline(best_survival["risk_pct"], color=_OUTCOME_COLORS["survived"], linestyle="--", linewidth=1,
               label=f"Peak {best_survival['survival_pct']:.1f}% @ {best_survival['risk_pct']:.2f}%")
    ax.set_title("Survival % vs Risk per Trade")
    ax.set_xlabel("Risk per trade (% of equity)")
    ax.set_ylabel("Survival rate to horizon (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Right: trader take-home $ vs risk
    ax = axes[1]
    ax.plot(rp, sweep_df["avg_total_money_made"], marker="o", color="#1baf7a", linewidth=1.8, label="Avg trader take-home")
    ax.plot(rp, sweep_df["median_total_money_made"], marker="o", color="#1baf7a", linewidth=1, linestyle=":", alpha=0.7, label="Median trader take-home")
    best_money = sweep_df.loc[sweep_df["avg_total_money_made"].idxmax()]
    ax.axvline(best_money["risk_pct"], color="#4a3aa7", linestyle="--", linewidth=1,
               label=f"Peak ${best_money['avg_total_money_made']:,.0f} @ {best_money['risk_pct']:.2f}%")
    ax.set_title("Trader Take-Home vs Risk per Trade")
    ax.set_xlabel("Risk per trade (% of equity)")
    ax.set_ylabel("Trader take-home ($)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Risk Sensitivity  ({config.resample_mode}, {config.risk_sweep_simulations:,} sims/point)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    # === CUSTOMIZE THESE SETTINGS BEFORE RUNNING ===
    # Every field below is a FundedMonteCarloConfig setting -- see the dataclass
    # definition near the top of this file for the full field list and defaults.
    # Fields not set explicitly here just use that dataclass's default value.
    config = FundedMonteCarloConfig(
        # --- How historical trades are resampled into simulated days ---
        # "bootstrap_days":   draw one whole historical trading day i.i.d.
        # "bootstrap_blocks": draw contiguous RUNS of historical days (circular block
        #     bootstrap) -- preserves cross-day streaks/clustering "bootstrap_days"
        #     destroys. Length is `block_size_days`, or randomized via `block_size_range`.
        resample_mode="bootstrap_blocks",
        block_size_days=5,          # only used when resample_mode == "bootstrap_blocks" and block_size_range is None
        block_size_range=None,      # e.g. (3, 15) to randomize block length instead of using block_size_days

        # --- Position sizing: how much of the account is risked per trade ---
        # "pct_current_equity":  risk_pct% of CURRENT equity (compounds as the account grows/shrinks)
        # "pct_initial_capital": risk_pct% of the ORIGINAL account_size (fixed dollar risk, no compounding)
        # "fixed_usd":           a flat dollar amount every trade (risk_usd field, unused here)
        # NOTE: "pct_current_equity" compounds risk-per-trade with equity while the
        # daily-loss cushion stays fixed -- see position_sizing_mode's docstring above.
        position_sizing_mode="pct_initial_capital",
        risk_pct=1.0,                # % risked per trade under the *_pct sizing modes above

        # --- Simulation horizon and sample count ---
        # The funded account has no profit target, so it would run forever without a
        # cap. max_simulated_days is a right-censoring horizon, not a target: trials
        # still alive at this many trading days are reported as "survived".
        max_simulated_days=750,      # 750 days ~3 trading years
        num_simulations=10_000,      # number of independent account lifetimes simulated

        # --- Payouts for the single main run (see also the sweep below) ---
        payouts_enabled=True,                  # master on/off switch for withdrawals in this run
        payout_interval_trading_days=20,       # how often (in trading days) a payout event occurs; ~10 trading days approximates FTMO's real ~14 calendar-day cadence
        payout_withdraw_pct=75.0,             # % of profit accrued SINCE THE LAST PAYOUT that gets withdrawn each cycle (0 = never withdraw, 100 = withdraw all accrued profit)
        profit_split_pct=80.0,                 # % of each withdrawn dollar that is the TRADER'S take-home pay (the rest is FTMO's cut); affects only the money-made reporting, not equity/survival mechanics

        # --- Payout frequency x size sensitivity sweep ---
        # Runs a whole extra grid of simulations (independent of the single run above)
        # to show how survival % and total money made trade off across different
        # payout cadences and withdrawal sizes. Disable to skip this and save time.
        run_payout_sweep_enabled=True,
        payout_sweep_intervals=(5, 10, 15, 20, 30, 45),        # payout_interval_trading_days values to test
        payout_sweep_withdraw_pcts=(25.0, 50.0, 75.0, 100.0),  # payout_withdraw_pct values to test
        payout_sweep_simulations=2_000,                        # simulations per grid cell (lower than num_simulations to keep the sweep fast)

        # --- Risk-per-trade sensitivity sweep ---
        # Runs another extra grid of simulations (independent of the single run and
        # the payout sweep above) varying only risk_pct, holding payout settings
        # fixed at whatever's set above. Produces survival % and trader take-home
        # vs risk_pct. Disable to skip this and save time.
        run_risk_sweep_enabled=True,
        risk_sweep_values=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),  # risk_pct values to test
        risk_sweep_simulations=2_000,                                            # simulations per point (lower than num_simulations to keep the sweep fast)

        # --- Optional: account size and loss-limit rules (defaults shown; uncomment to override) ---
        # account_size=100_000.0,       # Initial Capital of the funded account, in dollars
        # daily_loss_limit_pct=5.0,     # Max Daily Loss, as % of Initial Capital (fixed $ offset off the previous day's close)
        # max_overall_loss_pct=10.0,    # Max Overall Loss, as % of Initial Capital (static floor, never moves)
    )

    results_df, sample_curves, trade_pool_df = run_monte_carlo(config)

    out_dir = os.path.dirname(__file__)
    print(format_config_report(config))
    print()
    print(format_input_data_report(compute_input_data_stats(trade_pool_df)))
    print()
    print(format_results_report(compute_summary_stats(results_df, config), config))

    results_csv = os.path.join(out_dir, "mc_funded_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved per-trial results to {results_csv}")

    dashboard_png = os.path.join(out_dir, "mc_funded_dashboard.png")
    plot_dashboard(results_df, sample_curves, trade_pool_df, config, dashboard_png)
    print(f"Saved dashboard to {dashboard_png}")

    if config.run_payout_sweep_enabled:
        print("\nRunning payout frequency x size sweep...")
        sweep_df = run_payout_sweep(config)
        print()
        print(format_payout_sweep_report(sweep_df))

        sweep_csv = os.path.join(out_dir, "mc_payout_sweep.csv")
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"\nSaved payout sweep to {sweep_csv}")

        heatmaps_png = os.path.join(out_dir, "mc_payout_sweep_heatmaps.png")
        plot_payout_sweep_heatmaps(sweep_df, config, heatmaps_png)
        print(f"Saved payout sweep heatmaps to {heatmaps_png}")

    if config.run_risk_sweep_enabled:
        print("\nRunning risk-per-trade sweep...")
        risk_sweep_df = run_risk_sweep(config)
        print()
        print(format_risk_sweep_report(risk_sweep_df))

        risk_sweep_csv = os.path.join(out_dir, "mc_risk_sweep.csv")
        risk_sweep_df.to_csv(risk_sweep_csv, index=False)
        print(f"\nSaved risk sweep to {risk_sweep_csv}")

        risk_curve_png = os.path.join(out_dir, "mc_risk_curve.png")
        plot_risk_sweep_curve(risk_sweep_df, config, risk_curve_png)
        print(f"Saved risk curve to {risk_curve_png}")
