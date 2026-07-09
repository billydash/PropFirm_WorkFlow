"""Configurable Opening Range Breakout (ORB) backtester for 1-minute SPY bars.

Assumption: when a single bar's high/low touches both the stop and the target,
we assume the stop is hit first (conservative tie-break — 1-min OHLC bars
don't reveal intrabar order). hi
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

    # ROUGH PLACEHOLDER, off by default: caps notional exposure (shares * entry_price)
    # to approximate a live broker's leverage/max-lot-size limits, which this backtester
    # otherwise ignores entirely (shares are sized purely off risk_per_trade_usd /
    # stop_distance with no ceiling). Leave None until FTMO's actual leverage and max
    # lot size for the target instrument are confirmed -- do not guess a number.
    max_notional_usd: Optional[float] = None

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
    capped: bool  # True if max_notional_usd reduced shares below the uncapped value


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
            # Resting-stop fill: normally fills exactly at the OR level, but if the
            # bar's OPEN already gapped through the level (price moved past it before
            # this bar even started), a real resting order fills at that worse open
            # price instead -- the level was never available to trade at. Empirically
            # ~23% of breakout bars in this data gap through (mean overshoot $0.07/share).
            if "long" in directions_allowed and bar["high"] >= or_high:
                fill_price = max(bar["open"], or_high)
                return "long", bar["timestamp"], fill_price, idx
            if "short" in directions_allowed and bar["low"] <= or_low:
                fill_price = min(bar["open"], or_low)
                return "short", bar["timestamp"], fill_price, idx
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

    capped = False
    if config.max_notional_usd is not None:
        capped_shares = math.floor(config.max_notional_usd / entry_price)
        if capped_shares < shares:
            shares = max(0, capped_shares)
            capped = True

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
        capped=capped,
    )


def _run_direction_trades(
    day_df: pd.DataFrame,
    direction: Direction,
    or_high: float,
    or_low: float,
    or_end_ts,
    config: ORBConfig,
) -> list:
    """Runs one direction's entry + re-entry sequence in isolation, always scanning
    from or_end_ts. Called once per direction so that under direction_mode="both",
    long and short are genuinely independent triggers (per that mode's docstring)
    instead of one direction's scan being gated behind the other's exit time."""
    trades = []
    reentries_used = 0
    scan_from = or_end_ts
    active = True

    while active:
        breakout = find_next_breakout(day_df, scan_from, or_high, or_low, config, {direction})
        if breakout is None:
            break
        _, trigger_time, trigger_price, entry_idx = breakout

        trade = simulate_trade(
            day_df, entry_idx, trigger_time, trigger_price, direction, or_high, or_low, config
        )
        trades.append(trade)

        scan_from = trade.exit_time + pd.Timedelta(minutes=1)

        can_reenter = (
            config.reentry_enabled
            and trade.exit_reason == "stop"
            and reentries_used < config.max_reentries_per_direction
        )
        if can_reenter:
            reentries_used += 1
        else:
            active = False

    return trades


def run_day(day_df: pd.DataFrame, config: ORBConfig) -> list:
    or_result = compute_opening_range(day_df, config)
    if or_result is None:
        return []
    or_high, or_low, or_end_ts = or_result

    if config.direction_mode == "both":
        directions = ["long", "short"]
    else:
        directions = [config.direction_mode]

    trades = []
    for direction in directions:
        trades.extend(_run_direction_trades(day_df, direction, or_high, or_low, or_end_ts, config))

    # Each direction was scanned independently (see _run_direction_trades), so under
    # "both" mode a long and short may genuinely interleave in wall-clock time --
    # sort by entry so orb_trades.csv reads chronologically, as before.
    trades.sort(key=lambda t: t.entry_time)
    return trades


def compute_day_worst_case_mae_r(trades: list) -> float:
    """Worst-case combined drawdown (in R, relative to day-start equity) across ALL
    of a day's trades, walking them in entry-time order and tracking, at each
    trade's own worst point, the deficit already run up by trades that closed
    strictly before it (their REALIZED r_multiple, which includes costs) plus the
    OWN worst-case mae_r of any trade(s) still concurrently open at that point
    (which haven't realized anything yet, so their eventual r_multiple can't be
    used -- their own mae_r is the conservative stand-in).

    Two earlier versions of this function were tried and both had real bugs,
    verified directly against specific historical days rather than assumed:
    1. A bar-by-bar walk evaluating all open trades at each bar's shared low/high.
       On the bar where one trade's exit price (e.g. a stop at the OR level)
       coincides with another trade's entry (the opposite-direction OR breakout is
       the SAME price level, so this is common, not rare), the entering trade's
       favorable move within that shared bar partially cancelled the exiting
       trade's own already-realized mae_usd, UNDERSTATING the day's true worst
       case (2023-12-29-style day: computed less than the first trade's own mae_r
       alone, which is nonsensical -- a second trade can't make a day safer).
    2. A flat sum of every trade's independent mae_r. Simpler, and correctly
       conservative for genuinely concurrent trades, but still understated
       days with a large realized loss (cost-inclusive r_multiple, which can
       exceed that trade's own costless mae_r) on an EARLIER trade compounding
       with a small later trade's mae_r -- e.g. the same 2023-12-29 day: this
       trade's mae_r (1.0388) is smaller than the magnitude of its own realized
       r_multiple (-1.1172, worse due to costs), so a naive sum understated the
       true carry-over into the next trade's check by ~0.08R.

    This version reduces EXACTLY to a single trade's own mae_r on 0/1-trade days
    (verified: max diff 0.0 across 1,179 single-trade historical days), and was
    verified against a direct re-implementation of the Monte Carlo scripts'
    original per-trade-sequential check on 20,000 resampled two-trade days: zero
    cases where the old sequential check would have busted a trial that this
    formula's resulting day_worst_case_mae_r does not.

    Returns NaN for zero-trade days.
    """
    if not trades:
        return float("nan")

    ordered = sorted(trades, key=lambda t: t.entry_time)
    open_trades: list = []
    realized_r = 0.0
    worst_deficit = 0.0

    for t in ordered:
        still_open = []
        for o in open_trades:
            if o.exit_time <= t.entry_time:
                realized_r += o.r_multiple
            else:
                still_open.append(o)
        open_trades = still_open

        concurrent_open_mae = sum(o.mae_r for o in open_trades)
        worst_deficit = max(worst_deficit, t.mae_r + concurrent_open_mae - realized_r)
        open_trades.append(t)

    return max(0.0, worst_deficit)


def run_backtest(config: ORBConfig) -> tuple:
    """Returns (trades_df, trading_days_df). trading_days_df has one row per REAL
    trading day found in the price data within [start_date, end_date], regardless
    of whether that day produced any trades -- this is the authoritative trading
    calendar the Monte Carlo simulators bootstrap from, so days with zero fired
    trades aren't silently invisible to resampling (see orb_trading_days.csv)."""
    df = load_data(config)
    all_trades = []
    trading_days = []
    day_worst_case_mae_rs = []
    for trading_date, day_df in iter_sessions(df):
        trading_days.append(trading_date)
        day_trades = run_day(day_df, config)
        all_trades.extend(day_trades)
        day_worst_case_mae_rs.append(compute_day_worst_case_mae_r(day_trades))

    trading_days_df = pd.DataFrame(
        {"date": trading_days, "day_worst_case_mae_r": day_worst_case_mae_rs}
    )

    if not all_trades:
        return pd.DataFrame(), trading_days_df

    return pd.DataFrame([vars(t) for t in all_trades]), trading_days_df


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
        "pct_trades_capped": 100 * trades_df["capped"].mean(),
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
        f"{'Trades notional-capped':24}{pct('pct_trades_capped')}",
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


def build_equity_curves(
    trades_df: pd.DataFrame, trading_days_df: pd.DataFrame, config: ORBConfig
) -> tuple:
    """Build daily strategy-equity and buy-and-hold-equity series, both indexed by
    every real trading day in [start_date, end_date] (from trading_days_df), so
    no-trade days carry equity forward flat instead of being skipped."""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days_df["date"]).unique()))

    daily_pnl = trades_df.groupby("date")["net_pnl"].sum() if not trades_df.empty else pd.Series(dtype=float)
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    daily_pnl = daily_pnl.reindex(days, fill_value=0.0)
    strategy_equity = pd.Series(config.starting_equity + daily_pnl.cumsum().values, index=days)

    price_df = load_data(config)
    daily_close = price_df.groupby(price_df["timestamp"].dt.date)["close"].last()
    daily_close.index = pd.to_datetime(daily_close.index)
    daily_close = daily_close.reindex(days).ffill().bfill()
    buy_hold_shares = config.starting_equity / daily_close.iloc[0]
    buy_hold_equity = buy_hold_shares * daily_close

    return strategy_equity, buy_hold_equity


def plot_equity_curve(
    strategy_equity: pd.Series, buy_hold_equity: pd.Series, output_path: str
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(strategy_equity.index, strategy_equity.values, label="ORB Strategy")
    plt.plot(
        buy_hold_equity.index,
        buy_hold_equity.values,
        label="Buy & Hold (SPY)",
        alpha=0.7,
    )
    plt.title("ORB Strategy Equity Curve vs. Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.gcf().autofmt_xdate()
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

    trades_df, trading_days_df = run_backtest(config)

    out_dir = os.path.dirname(__file__)
    print(format_config_report(config))

    trading_days_csv = os.path.join(out_dir, "orb_trading_days.csv")
    trading_days_df.to_csv(trading_days_csv, index=False)
    print(f"Saved trading-day calendar ({len(trading_days_df)} days) to {trading_days_csv}")

    if trades_df.empty:
        print("\nNo trades generated for the given config/date range.")
    else:
        stats, equity = compute_stats(trades_df, config)
        print()
        print(format_stats_report(stats))

        trades_csv = os.path.join(out_dir, "orb_trades.csv")
        trades_df.to_csv(trades_csv, index=False)
        print(f"\nSaved trade log to {trades_csv}")

        strategy_equity, buy_hold_equity = build_equity_curves(trades_df, trading_days_df, config)
        equity_png = os.path.join(out_dir, "orb_equity.png")
        plot_equity_curve(strategy_equity, buy_hold_equity, equity_png)
        print(f"Saved equity curve to {equity_png}")
