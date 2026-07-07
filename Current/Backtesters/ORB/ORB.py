"""Configurable Opening Range Breakout (ORB) backtester for 1-minute SPY bars.

Assumption: when a single bar's high/low touches both the stop and the target,
we assume the stop is hit first (conservative tie-break — 1-min OHLC bars
don't reveal intrabar order).
"""

import math
import os
from dataclasses import dataclass, field
from datetime import time
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd

DirectionMode = Literal["long", "short", "both"]
EntryMode = Literal["stop", "close"]
StopMode = Literal["opposite_or", "fixed_distance", "or_multiple"]
Direction = Literal["long", "short"]


@dataclass
class ORBConfig:
    start_date: str  # "YYYY-MM-DD", required
    end_date: str  # "YYYY-MM-DD", required

    csv_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "SPY_1min.csv"
        )
    )
    timezone: str = "America/New_York"

    or_start: time = time(9, 30)
    or_duration_minutes: int = 15

    direction_mode: DirectionMode = "both"
    entry_mode: EntryMode = "stop"

    stop_mode: StopMode = "opposite_or"
    fixed_stop_distance: float = 1.0  # used when stop_mode == "fixed_distance"
    or_stop_multiple: float = 1.0  # used when stop_mode == "or_multiple"

    rr: float = 2.0
    risk_per_trade_usd: float = 200.0

    last_entry_time: time = time(11, 30)
    session_close: time = time(16, 0)

    reentry_enabled: bool = False
    max_reentries_per_direction: int = 0

    # ROUGH PLACEHOLDER for eventual FTMO execution: this strategy would likely run as
    # FTMO's S&P 500 index CFD (confirmed zero-commission), not real SPY shares, so
    # commission stays $0 but slippage is widened vs. a real-shares assumption to
    # approximate a CFD spread (~0.4-0.5 index points ~= $0.03-0.05/SPY-equivalent
    # share). Revisit both once FTMO's actual live spread/commission is confirmed.
    commission_per_fill: float = 0.0
    slippage_per_share: float = 0.03

    starting_equity: float = 100_000.0


@dataclass
class TradeRecord:
    date: object
    direction: Direction
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: object
    exit_price: float
    exit_reason: str  # "stop" | "target" | "eod"
    shares: int
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    mae_usd: float  # max adverse excursion (worst unrealized drawdown), in $
    mfe_usd: float  # max favorable excursion (best unrealized gain), in $
    mae_r: float  # mae_usd expressed in R (multiples of risk_per_trade_usd)
    mfe_r: float  # mfe_usd expressed in R (multiples of risk_per_trade_usd)


def load_data(config: ORBConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert(config.timezone)
    df = df.sort_values("timestamp")
    df = df[
        (df["timestamp"] >= pd.Timestamp(config.start_date, tz=config.timezone))
        & (
            df["timestamp"]
            < pd.Timestamp(config.end_date, tz=config.timezone) + pd.Timedelta(days=1)
        )
    ]
    return df.reset_index(drop=True)


def iter_sessions(df: pd.DataFrame):
    for trading_date, day_df in df.groupby(df["timestamp"].dt.date):
        yield trading_date, day_df.reset_index(drop=True)


def compute_opening_range(day_df: pd.DataFrame, config: ORBConfig) -> Optional[tuple]:
    or_start_ts = day_df["timestamp"].iloc[0].normalize() + pd.Timedelta(
        hours=config.or_start.hour, minutes=config.or_start.minute
    )
    or_end_ts = or_start_ts + pd.Timedelta(minutes=config.or_duration_minutes)

    window = day_df[
        (day_df["timestamp"] >= or_start_ts) & (day_df["timestamp"] < or_end_ts)
    ]
    if window.empty:
        return None
    return window["high"].max(), window["low"].min(), or_end_ts


def compute_stop_target(
    entry_price: float, direction: Direction, or_high: float, or_low: float, config: ORBConfig
) -> tuple:
    or_width = or_high - or_low

    if config.stop_mode == "opposite_or":
        stop_price = or_low if direction == "long" else or_high
    elif config.stop_mode == "fixed_distance":
        stop_price = (
            entry_price - config.fixed_stop_distance
            if direction == "long"
            else entry_price + config.fixed_stop_distance
        )
    else:  # or_multiple
        offset = or_width * config.or_stop_multiple
        stop_price = entry_price - offset if direction == "long" else entry_price + offset

    stop_distance = abs(entry_price - stop_price)
    target_price = (
        entry_price + stop_distance * config.rr
        if direction == "long"
        else entry_price - stop_distance * config.rr
    )
    return stop_price, target_price, stop_distance


def find_next_breakout(
    day_df: pd.DataFrame,
    scan_from_ts,
    or_high: float,
    or_low: float,
    config: ORBConfig,
    directions_allowed: set,
) -> Optional[tuple]:
    """Return (direction, trigger_time, trigger_price, bar_index) for the first
    valid breakout at/after scan_from_ts and at/before last_entry_time, or None."""
    last_entry_ts = day_df["timestamp"].iloc[0].normalize() + pd.Timedelta(
        hours=config.last_entry_time.hour, minutes=config.last_entry_time.minute
    )

    candidates = day_df[
        (day_df["timestamp"] >= scan_from_ts) & (day_df["timestamp"] <= last_entry_ts)
    ]

    for idx, bar in candidates.iterrows():
        if config.entry_mode == "stop":
            if "long" in directions_allowed and bar["high"] >= or_high:
                return "long", bar["timestamp"], or_high, idx
            if "short" in directions_allowed and bar["low"] <= or_low:
                return "short", bar["timestamp"], or_low, idx
        else:  # close
            if "long" in directions_allowed and bar["close"] >= or_high:
                next_bar = day_df[day_df.index > idx].head(1)
                if next_bar.empty:
                    continue
                return "long", next_bar["timestamp"].iloc[0], next_bar["open"].iloc[0], next_bar.index[0]
            if "short" in directions_allowed and bar["close"] <= or_low:
                next_bar = day_df[day_df.index > idx].head(1)
                if next_bar.empty:
                    continue
                return "short", next_bar["timestamp"].iloc[0], next_bar["open"].iloc[0], next_bar.index[0]
    return None


def simulate_trade(
    day_df: pd.DataFrame,
    entry_idx: int,
    entry_time,
    entry_price: float,
    direction: Direction,
    or_high: float,
    or_low: float,
    config: ORBConfig,
) -> TradeRecord:
    stop_price, target_price, stop_distance = compute_stop_target(
        entry_price, direction, or_high, or_low, config
    )
    shares = math.floor(config.risk_per_trade_usd / stop_distance) if stop_distance > 0 else 0

    close_ts = day_df["timestamp"].iloc[0].normalize() + pd.Timedelta(
        hours=config.session_close.hour, minutes=config.session_close.minute
    )

    forward = day_df[day_df.index >= entry_idx]
    forward = forward[forward["timestamp"] <= close_ts]

    exit_time = forward["timestamp"].iloc[-1]
    exit_price = forward["close"].iloc[-1]
    exit_reason = "eod"

    worst_price = entry_price
    best_price = entry_price

    for _, bar in forward.iterrows():
        if direction == "long":
            worst_price = min(worst_price, bar["low"])
            best_price = max(best_price, bar["high"])
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
        else:
            worst_price = max(worst_price, bar["high"])
            best_price = min(best_price, bar["low"])
            stop_hit = bar["high"] >= stop_price
            target_hit = bar["low"] <= target_price

        if stop_hit:
            exit_time, exit_price, exit_reason = bar["timestamp"], stop_price, "stop"
            break
        if target_hit:
            exit_time, exit_price, exit_reason = bar["timestamp"], target_price, "target"
            break

    if direction == "long":
        gross_pnl = (exit_price - entry_price) * shares
        mae_usd = (entry_price - worst_price) * shares
        mfe_usd = (best_price - entry_price) * shares
    else:
        gross_pnl = (entry_price - exit_price) * shares
        mae_usd = (worst_price - entry_price) * shares
        mfe_usd = (entry_price - best_price) * shares

    costs = config.commission_per_fill * 2 + config.slippage_per_share * shares * 2
    net_pnl = gross_pnl - costs
    r_multiple = net_pnl / (config.risk_per_trade_usd) if config.risk_per_trade_usd else 0.0
    mae_r = mae_usd / config.risk_per_trade_usd if config.risk_per_trade_usd else 0.0
    mfe_r = mfe_usd / config.risk_per_trade_usd if config.risk_per_trade_usd else 0.0

    return TradeRecord(
        date=day_df["timestamp"].iloc[0].date(),
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        shares=shares,
        gross_pnl=gross_pnl,
        costs=costs,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        mae_usd=mae_usd,
        mfe_usd=mfe_usd,
        mae_r=mae_r,
        mfe_r=mfe_r,
    )


def run_day(day_df: pd.DataFrame, config: ORBConfig) -> list:
    or_result = compute_opening_range(day_df, config)
    if or_result is None:
        return []
    or_high, or_low, or_end_ts = or_result

    if config.direction_mode == "both":
        all_directions = {"long", "short"}
    else:
        all_directions = {config.direction_mode}

    trades = []
    reentries_used = {"long": 0, "short": 0}
    active_directions = set(all_directions)
    scan_from = or_end_ts

    while active_directions:
        breakout = find_next_breakout(day_df, scan_from, or_high, or_low, config, active_directions)
        if breakout is None:
            break
        direction, trigger_time, trigger_price, entry_idx = breakout

        trade = simulate_trade(
            day_df, entry_idx, trigger_time, trigger_price, direction, or_high, or_low, config
        )
        trades.append(trade)

        scan_from = trade.exit_time + pd.Timedelta(minutes=1)

        can_reenter = (
            config.reentry_enabled
            and trade.exit_reason == "stop"
            and reentries_used[direction] < config.max_reentries_per_direction
        )
        if can_reenter:
            reentries_used[direction] += 1
        else:
            active_directions.discard(direction)

    return trades


def run_backtest(config: ORBConfig) -> pd.DataFrame:
    df = load_data(config)
    all_trades = []
    for _, day_df in iter_sessions(df):
        all_trades.extend(run_day(day_df, config))

    if not all_trades:
        return pd.DataFrame()

    return pd.DataFrame([vars(t) for t in all_trades])


def compute_stats(trades_df: pd.DataFrame, config: ORBConfig) -> dict:
    if trades_df.empty:
        return {"total_trades": 0}

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    equity = config.starting_equity + trades_df["net_pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (drawdown / running_max).min() * 100

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()

    # Sharpe/Sortino from daily returns (net P&L per trading day / starting equity),
    # annualized with sqrt(252). Risk-free rate assumed 0.
    daily_pnl = trades_df.groupby("date")["net_pnl"].sum()
    daily_returns = daily_pnl / config.starting_equity
    ann_factor = math.sqrt(252)

    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std(ddof=1)
    downside = daily_returns[daily_returns < 0]
    downside_std = math.sqrt((downside**2).mean()) if not downside.empty else 0.0

    sharpe = ann_factor * mean_daily / std_daily if std_daily > 0 else float("nan")
    sortino = ann_factor * mean_daily / downside_std if downside_std > 0 else float("nan")

    stats = {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": 100 * len(wins) / len(trades_df),
        "total_net_pnl": trades_df["net_pnl"].sum(),
        "avg_net_pnl": trades_df["net_pnl"].mean(),
        "avg_win": wins["net_pnl"].mean() if not wins.empty else 0.0,
        "avg_loss": losses["net_pnl"].mean() if not losses.empty else 0.0,
        "best_trade": trades_df["net_pnl"].max(),
        "worst_trade": trades_df["net_pnl"].min(),
        "avg_r_multiple": trades_df["r_multiple"].mean(),
        "avg_mae_r": trades_df["mae_r"].mean(),
        "avg_mfe_r": trades_df["mfe_r"].mean(),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "sharpe": sharpe,
        "sortino": sortino,
        "total_costs": trades_df["costs"].sum(),
        "max_drawdown_usd": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "starting_equity": config.starting_equity,
        "final_equity": equity.iloc[-1],
    }
    return stats, equity


def format_config_report(config: ORBConfig) -> str:
    width = 60
    lines = [
        "=" * width,
        "ORB BACKTEST CONFIG".center(width),
        "=" * width,
        f"{'Date range':24}{config.start_date}  to  {config.end_date}",
        f"{'Symbol data':24}{os.path.basename(config.csv_path)}",
        f"{'Direction mode':24}{config.direction_mode}",
        f"{'Entry mode':24}{config.entry_mode}",
        f"{'Opening range':24}{config.or_start.strftime('%H:%M')} ET, {config.or_duration_minutes} min",
        f"{'Stop mode':24}{config.stop_mode}",
    ]
    if config.stop_mode == "fixed_distance":
        lines.append(f"{'Fixed stop distance':24}${config.fixed_stop_distance:,.2f}")
    if config.stop_mode == "or_multiple":
        lines.append(f"{'OR stop multiple':24}{config.or_stop_multiple}x")
    lines += [
        f"{'Reward:risk (RR)':24}{config.rr}",
        f"{'Risk per trade':24}${config.risk_per_trade_usd:,.2f}",
        f"{'Last entry time':24}{config.last_entry_time.strftime('%H:%M')} ET",
        f"{'Session close':24}{config.session_close.strftime('%H:%M')} ET",
        f"{'Re-entry':24}{'enabled, max ' + str(config.max_reentries_per_direction) + '/direction' if config.reentry_enabled else 'disabled'}",
        f"{'Commission per fill':24}${config.commission_per_fill:,.2f}",
        f"{'Slippage per share':24}${config.slippage_per_share:,.4f}",
        f"{'Starting equity':24}${config.starting_equity:,.2f}",
        "=" * width,
    ]
    return "\n".join(lines)


def format_stats_report(stats: dict) -> str:
    width = 60

    def usd(key):
        return f"${stats[key]:,.2f}"

    def pct(key):
        return f"{stats[key]:.2f}%"

    lines = [
        "ORB BACKTEST RESULTS".center(width),
        "=" * width,
        "-- Trade counts " + "-" * (width - 16),
        f"{'Total trades':24}{stats['total_trades']}",
        f"{'Wins / Losses':24}{stats['wins']} / {stats['losses']}",
        f"{'Win rate':24}{pct('win_rate_pct')}",
        "",
        "-- P&L " + "-" * (width - 7),
        f"{'Total net P&L':24}{usd('total_net_pnl')}",
        f"{'Avg net P&L / trade':24}{usd('avg_net_pnl')}",
        f"{'Avg win':24}{usd('avg_win')}",
        f"{'Avg loss':24}{usd('avg_loss')}",
        f"{'Best trade':24}{usd('best_trade')}",
        f"{'Worst trade':24}{usd('worst_trade')}",
        f"{'Total costs':24}{usd('total_costs')}",
        "",
        "-- Trade quality " + "-" * (width - 17),
        f"{'Avg R-multiple':24}{stats['avg_r_multiple']:.2f}R",
        f"{'Avg MAE':24}{stats['avg_mae_r']:.2f}R",
        f"{'Avg MFE':24}{stats['avg_mfe_r']:.2f}R",
        f"{'Profit factor':24}{stats['profit_factor']:.2f}",
        f"{'Sharpe (annualized)':24}{stats['sharpe']:.2f}",
        f"{'Sortino (annualized)':24}{stats['sortino']:.2f}",
        "",
        "-- Equity " + "-" * (width - 10),
        f"{'Starting equity':24}{usd('starting_equity')}",
        f"{'Final equity':24}{usd('final_equity')}",
        f"{'Max drawdown':24}{usd('max_drawdown_usd')} ({pct('max_drawdown_pct')})",
        "=" * width,
    ]
    return "\n".join(lines)


def plot_equity_curve(equity: pd.Series, output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(equity.values)
    plt.title("ORB Strategy Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    config = ORBConfig(
        start_date="2020-08-01",  # "YYYY-MM-DD", first date included in the backtest
        end_date="2026-06-30",  # "YYYY-MM-DD", last date included in the backtest

        # Which breakout direction(s) to trade:
        #   "long"  - only take OR-high breakouts
        #   "short" - only take OR-low breakouts
        #   "both"  - trade both sides; long and short are independent triggers
        #             (a short signal is still taken even if a long already fired)
        direction_mode="both",

        # How a breakout must occur to trigger an entry:
        #   "stop"  - intrabar: a bar's high/low touching the OR level fills immediately
        #             at that level (classic breakout-stop behavior)
        #   "close" - a bar must CLOSE beyond the OR level; entry then fills at the
        #             next bar's open (slower, fewer false breakouts)
        entry_mode="stop",

        # How the stop-loss price is placed:
        #   "opposite_or"    - stop at the far side of the opening range
        #                       (long stop = or_low, short stop = or_high)
        #   "fixed_distance" - stop a fixed $ distance from entry (set fixed_stop_distance)
        #   "or_multiple"    - stop = entry +/- (OR width * or_stop_multiple)
        stop_mode="opposite_or",

        rr=2.0,  # reward:risk multiple — target = entry +/- (stop_distance * rr)
        risk_per_trade_usd=100.0,  # fixed $ risked per trade; shares = floor(risk_per_trade_usd / stop_distance)

        # Whether a direction (long/short) can re-trigger after being stopped out
        # (never after a target hit) later the same day.
        reentry_enabled=True,
        # Max re-entries allowed per direction per day; only used when reentry_enabled is True.
        max_reentries_per_direction=1,

        or_duration_minutes=30,  # opening-range window length in minutes

        last_entry_time=time(11, 00),  # last time a NEW breakout entry may be triggered

        # Other available options (all keep their ORBConfig defaults unless set here):
        #   csv_path                - path to the 1-min OHLCV CSV to backtest
        #   timezone                - IANA timezone the UTC timestamps are converted to
        #   or_start, or_duration_minutes - opening-range window: [or_start, or_start + duration)
        #   fixed_stop_distance     - $ distance from entry; only used when stop_mode == "fixed_distance"
        #   or_stop_multiple        - multiple of OR width; only used when stop_mode == "or_multiple"
        #   last_entry_time         - last time a NEW breakout entry may be triggered
        #   session_close           - time any still-open position is force-closed
        #   commission_per_fill     - flat $ commission charged per fill (entry + exit each incur this)
        #   slippage_per_share      - per-share slippage applied unfavorably to entry/exit fills
        #   starting_equity         - starting balance, used only for the equity curve/drawdown %
    )

    trades_df = run_backtest(config)

    out_dir = os.path.dirname(__file__)
    print(format_config_report(config))
    if trades_df.empty:
        print("\nNo trades generated for the given config/date range.")
    else:
        stats, equity = compute_stats(trades_df, config)
        print()
        print(format_stats_report(stats))

        trades_csv = os.path.join(out_dir, "orb_trades.csv")
        trades_df.to_csv(trades_csv, index=False)
        print(f"\nSaved trade log to {trades_csv}")

        equity_png = os.path.join(out_dir, "orb_equity.png")
        plot_equity_curve(equity, equity_png)
        print(f"Saved equity curve to {equity_png}")
