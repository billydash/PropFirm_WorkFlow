"""Opening Range Breakout (ORB) backtester for 1-minute SPY bars.

A fresh, self-contained implementation. Reads the 1-minute OHLCV CSV, builds a
daily opening range, trades breakouts of that range with configurable entry /
stop / target logic, then prints a detailed statistics report and renders a
9-panel matplotlib dashboard.

Intrabar tie-break: when a single 1-minute bar's high/low touches BOTH the stop
and the target, the stop is assumed to fill first (conservative — 1-min OHLC
bars do not reveal the true intrabar path).
"""

import os
from dataclasses import dataclass, field
from datetime import time
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DirectionMode = Literal["long", "short", "both"]
EntryMode = Literal["stop", "close"]
StopMode = Literal["opposite_or", "fixed_distance", "or_multiple"]
Direction = Literal["long", "short"]

# ---------------------------------------------------------------------------
# Palette (validated categorical + status hues, light surface).
# ---------------------------------------------------------------------------
C_BLUE = "#2a78d6"
C_AQUA = "#1baf7a"
C_VIOLET = "#4a3aa7"
C_ORANGE = "#eb6834"
C_WIN = "#0ca30c"      # status: good
C_LOSS = "#d03b3b"     # status: critical
C_GRID = "#d9d8d4"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_SURFACE = "#fcfcfb"


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
class Trade:
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
    hold_minutes: float
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    mae_usd: float  # max adverse excursion (worst unrealized loss), $
    mfe_usd: float  # max favorable excursion (best unrealized gain), $
    mae_r: float
    mfe_r: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(config: ORBConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert(config.timezone)
    df = df.sort_values("timestamp").reset_index(drop=True)

    start = pd.Timestamp(config.start_date, tz=config.timezone)
    end = pd.Timestamp(config.end_date, tz=config.timezone) + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    return df.reset_index(drop=True)


def session_ts(day_df: pd.DataFrame, t: time) -> pd.Timestamp:
    """Build a timezone-aware timestamp on this session's date at time ``t``."""
    return day_df["timestamp"].iloc[0].normalize() + pd.Timedelta(
        hours=t.hour, minutes=t.minute
    )


# ---------------------------------------------------------------------------
# Strategy pieces
# ---------------------------------------------------------------------------
def compute_opening_range(day_df: pd.DataFrame, config: ORBConfig) -> Optional[tuple]:
    or_start_ts = session_ts(day_df, config.or_start)
    or_end_ts = or_start_ts + pd.Timedelta(minutes=config.or_duration_minutes)
    window = day_df[
        (day_df["timestamp"] >= or_start_ts) & (day_df["timestamp"] < or_end_ts)
    ]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min()), or_end_ts


def compute_stop_target(
    entry_price: float,
    direction: Direction,
    or_high: float,
    or_low: float,
    config: ORBConfig,
) -> tuple:
    or_width = or_high - or_low
    if config.stop_mode == "opposite_or":
        stop_price = or_low if direction == "long" else or_high
    elif config.stop_mode == "fixed_distance":
        d = config.fixed_stop_distance
        stop_price = entry_price - d if direction == "long" else entry_price + d
    else:  # or_multiple
        offset = or_width * config.or_stop_multiple
        stop_price = entry_price - offset if direction == "long" else entry_price + offset

    stop_distance = abs(entry_price - stop_price)
    if direction == "long":
        target_price = entry_price + stop_distance * config.rr
    else:
        target_price = entry_price - stop_distance * config.rr
    return stop_price, target_price, stop_distance


def find_next_entry(
    day_df: pd.DataFrame,
    scan_from_ts,
    or_high: float,
    or_low: float,
    config: ORBConfig,
    directions_allowed: set,
) -> Optional[tuple]:
    """First valid breakout at/after ``scan_from_ts`` up to ``last_entry_time``.

    Returns (direction, entry_time, entry_price, entry_idx) or None.
    """
    last_entry_ts = session_ts(day_df, config.last_entry_time)
    window = day_df[
        (day_df["timestamp"] >= scan_from_ts) & (day_df["timestamp"] <= last_entry_ts)
    ]

    for idx, bar in window.iterrows():
        if config.entry_mode == "stop":
            # Intrabar touch fills at the OR level, or at the open if the bar
            # gapped clean through it.
            if "long" in directions_allowed and bar["high"] >= or_high:
                fill = max(or_high, float(bar["open"]))
                return "long", bar["timestamp"], fill, idx
            if "short" in directions_allowed and bar["low"] <= or_low:
                fill = min(or_low, float(bar["open"]))
                return "short", bar["timestamp"], fill, idx
        else:  # close: bar must CLOSE beyond level, fill next bar's open
            broke_long = "long" in directions_allowed and bar["close"] >= or_high
            broke_short = "short" in directions_allowed and bar["close"] <= or_low
            if broke_long or broke_short:
                nxt = day_df[day_df.index > idx].head(1)
                if nxt.empty:
                    continue
                direction = "long" if broke_long else "short"
                return (
                    direction,
                    nxt["timestamp"].iloc[0],
                    float(nxt["open"].iloc[0]),
                    nxt.index[0],
                )
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
) -> Trade:
    stop_price, target_price, stop_distance = compute_stop_target(
        entry_price, direction, or_high, or_low, config
    )
    shares = int(config.risk_per_trade_usd // stop_distance) if stop_distance > 0 else 0

    close_ts = session_ts(day_df, config.session_close)
    fwd = day_df[(day_df.index >= entry_idx) & (day_df["timestamp"] <= close_ts)]

    exit_time = fwd["timestamp"].iloc[-1]
    exit_price = float(fwd["close"].iloc[-1])
    exit_reason = "eod"
    worst = best = entry_price

    for _, bar in fwd.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if direction == "long":
            worst, best = min(worst, lo), max(best, hi)
            stop_hit, target_hit = lo <= stop_price, hi >= target_price
        else:
            worst, best = max(worst, hi), min(best, lo)
            stop_hit, target_hit = hi >= stop_price, lo <= target_price

        if stop_hit:  # conservative tie-break: stop before target
            exit_time, exit_price, exit_reason = bar["timestamp"], stop_price, "stop"
            break
        if target_hit:
            exit_time, exit_price, exit_reason = bar["timestamp"], target_price, "target"
            break

    if direction == "long":
        gross = (exit_price - entry_price) * shares
        mae = (entry_price - worst) * shares
        mfe = (best - entry_price) * shares
    else:
        gross = (entry_price - exit_price) * shares
        mae = (worst - entry_price) * shares
        mfe = (entry_price - best) * shares

    costs = config.commission_per_fill * 2 + config.slippage_per_share * shares * 2
    net = gross - costs
    risk = config.risk_per_trade_usd
    hold = (exit_time - entry_time).total_seconds() / 60.0

    return Trade(
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
        hold_minutes=hold,
        gross_pnl=gross,
        costs=costs,
        net_pnl=net,
        r_multiple=net / risk if risk else 0.0,
        mae_usd=mae,
        mfe_usd=mfe,
        mae_r=mae / risk if risk else 0.0,
        mfe_r=mfe / risk if risk else 0.0,
    )


def run_day(day_df: pd.DataFrame, config: ORBConfig) -> list:
    or_result = compute_opening_range(day_df, config)
    if or_result is None:
        return []
    or_high, or_low, or_end_ts = or_result

    all_dirs = {"long", "short"} if config.direction_mode == "both" else {config.direction_mode}
    active = set(all_dirs)
    reentries = {"long": 0, "short": 0}
    scan_from = or_end_ts
    trades = []

    while active:
        entry = find_next_entry(day_df, scan_from, or_high, or_low, config, active)
        if entry is None:
            break
        direction, entry_time, entry_price, entry_idx = entry
        trade = simulate_trade(
            day_df, entry_idx, entry_time, entry_price, direction, or_high, or_low, config
        )
        trades.append(trade)
        scan_from = trade.exit_time + pd.Timedelta(minutes=1)

        can_reenter = (
            config.reentry_enabled
            and trade.exit_reason == "stop"
            and reentries[direction] < config.max_reentries_per_direction
        )
        if can_reenter:
            reentries[direction] += 1
        else:
            active.discard(direction)

    return trades


def run_backtest(config: ORBConfig) -> pd.DataFrame:
    df = load_data(config)
    all_trades = []
    for _, day_df in df.groupby(df["timestamp"].dt.date):
        all_trades.extend(run_day(day_df.reset_index(drop=True), config))
    if not all_trades:
        return pd.DataFrame()
    tdf = pd.DataFrame([vars(t) for t in all_trades])
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["exit_time"] = pd.to_datetime(tdf["exit_time"])
    tdf["date"] = pd.to_datetime(tdf["date"])
    return tdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def max_consecutive(mask: pd.Series) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def compute_stats(trades_df: pd.DataFrame, config: ORBConfig) -> tuple:
    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    equity = config.starting_equity + trades_df["net_pnl"].cumsum()
    running_max = equity.cummax()
    dd = equity - running_max
    max_dd_usd = float(dd.min())
    max_dd_pct = float((dd / running_max).min() * 100)

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    r = trades_df["r_multiple"]

    win_rate = len(wins) / len(trades_df)
    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0
    expectancy = trades_df["net_pnl"].mean()
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")

    daily = trades_df.groupby("date")["net_pnl"].sum()

    stats = {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": 100 * win_rate,
        "trading_days": trades_df["date"].nunique(),
        "trades_per_day": len(trades_df) / max(trades_df["date"].nunique(), 1),
        "total_net_pnl": float(trades_df["net_pnl"].sum()),
        "gross_pnl": float(trades_df["gross_pnl"].sum()),
        "total_costs": float(trades_df["costs"].sum()),
        "expectancy": float(expectancy),
        "expectancy_r": float(r.mean()),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(payoff),
        "best_trade": float(trades_df["net_pnl"].max()),
        "worst_trade": float(trades_df["net_pnl"].min()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_r_multiple": float(r.mean()),
        "r_std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "sharpe_per_trade": float(r.mean() / r.std(ddof=1)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0,
        "avg_mae_r": float(trades_df["mae_r"].mean()),
        "avg_mfe_r": float(trades_df["mfe_r"].mean()),
        "avg_hold_min": float(trades_df["hold_minutes"].mean()),
        "max_consec_wins": max_consecutive(trades_df["net_pnl"] > 0),
        "max_consec_losses": max_consecutive(trades_df["net_pnl"] <= 0),
        "best_day": float(daily.max()),
        "worst_day": float(daily.min()),
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "starting_equity": config.starting_equity,
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": 100 * (float(equity.iloc[-1]) - config.starting_equity) / config.starting_equity,
    }
    return stats, equity


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------
def format_config_report(config: ORBConfig) -> str:
    w = 62
    lines = [
        "=" * w,
        "ORB BACKTEST — CONFIG".center(w),
        "=" * w,
        f"{'Date range':26}{config.start_date}  ->  {config.end_date}",
        f"{'Data file':26}{os.path.basename(config.csv_path)}",
        f"{'Opening range':26}{config.or_start.strftime('%H:%M')} ET + {config.or_duration_minutes} min",
        f"{'Direction mode':26}{config.direction_mode}",
        f"{'Entry mode':26}{config.entry_mode}",
        f"{'Stop mode':26}{config.stop_mode}",
    ]
    if config.stop_mode == "fixed_distance":
        lines.append(f"{'Fixed stop distance':26}${config.fixed_stop_distance:,.2f}")
    if config.stop_mode == "or_multiple":
        lines.append(f"{'OR stop multiple':26}{config.or_stop_multiple}x")
    lines += [
        f"{'Reward : risk':26}{config.rr} : 1",
        f"{'Risk per trade':26}${config.risk_per_trade_usd:,.2f}",
        f"{'Last entry time':26}{config.last_entry_time.strftime('%H:%M')} ET",
        f"{'Session close':26}{config.session_close.strftime('%H:%M')} ET",
        f"{'Re-entry':26}"
        + ("enabled, max %d/direction" % config.max_reentries_per_direction if config.reentry_enabled else "disabled"),
        f"{'Commission per fill':26}${config.commission_per_fill:,.2f}",
        f"{'Slippage per share':26}${config.slippage_per_share:,.4f}",
        f"{'Starting equity':26}${config.starting_equity:,.2f}",
        "=" * w,
    ]
    return "\n".join(lines)


def format_stats_report(s: dict) -> str:
    w = 62

    def usd(k):
        return f"${s[k]:,.2f}"

    lines = [
        "ORB BACKTEST — RESULTS".center(w),
        "=" * w,
        "-- Activity " + "-" * (w - 12),
        f"{'Total trades':26}{s['total_trades']}",
        f"{'Trading days':26}{s['trading_days']}  ({s['trades_per_day']:.2f} trades/day)",
        f"{'Wins / Losses':26}{s['wins']} / {s['losses']}",
        f"{'Win rate':26}{s['win_rate_pct']:.2f}%",
        f"{'Avg hold':26}{s['avg_hold_min']:.1f} min",
        "",
        "-- P&L " + "-" * (w - 7),
        f"{'Total net P&L':26}{usd('total_net_pnl')}",
        f"{'Gross P&L':26}{usd('gross_pnl')}",
        f"{'Total costs':26}{usd('total_costs')}",
        f"{'Expectancy / trade':26}{usd('expectancy')}  ({s['expectancy_r']:+.3f}R)",
        f"{'Avg win / Avg loss':26}{usd('avg_win')} / {usd('avg_loss')}",
        f"{'Payoff ratio':26}{s['payoff_ratio']:.2f}",
        f"{'Best / Worst trade':26}{usd('best_trade')} / {usd('worst_trade')}",
        f"{'Best / Worst day':26}{usd('best_day')} / {usd('worst_day')}",
        "",
        "-- Edge & risk " + "-" * (w - 15),
        f"{'Profit factor':26}{s['profit_factor']:.2f}",
        f"{'Avg R-multiple':26}{s['avg_r_multiple']:+.3f}R  (std {s['r_std']:.2f})",
        f"{'Sharpe (per trade)':26}{s['sharpe_per_trade']:.3f}",
        f"{'Avg MAE / MFE':26}{s['avg_mae_r']:.2f}R / {s['avg_mfe_r']:.2f}R",
        f"{'Max consec W / L':26}{s['max_consec_wins']} / {s['max_consec_losses']}",
        "",
        "-- Equity " + "-" * (w - 10),
        f"{'Starting equity':26}{usd('starting_equity')}",
        f"{'Final equity':26}{usd('final_equity')}",
        f"{'Total return':26}{s['total_return_pct']:+.2f}%",
        f"{'Max drawdown':26}{usd('max_drawdown_usd')}  ({s['max_drawdown_pct']:.2f}%)",
        "=" * w,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9-panel dashboard
# ---------------------------------------------------------------------------
def plot_dashboard(
    trades_df: pd.DataFrame, equity: pd.Series, stats: dict, config: ORBConfig, out_path: str
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": C_SURFACE,
            "axes.facecolor": C_SURFACE,
            "axes.edgecolor": C_GRID,
            "axes.labelcolor": C_INK2,
            "axes.titlecolor": C_INK,
            "text.color": C_INK,
            "xtick.color": C_INK2,
            "ytick.color": C_INK2,
            "font.size": 9,
            "axes.grid": True,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(
        f"ORB Backtest Dashboard — SPY  |  {config.start_date} to {config.end_date}  |  "
        f"{stats['total_trades']} trades",
        fontsize=15,
        fontweight="bold",
        color=C_INK,
    )

    def style(ax, title):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

    eq = equity.reset_index(drop=True)
    x = np.arange(len(eq))

    # 1 — Equity curve
    ax = axes[0, 0]
    ax.plot(x, eq.values, color=C_BLUE, linewidth=2)
    ax.axhline(config.starting_equity, color=C_INK2, linewidth=1, linestyle="--", alpha=0.6)
    ax.fill_between(x, config.starting_equity, eq.values, where=eq.values >= config.starting_equity,
                    color=C_BLUE, alpha=0.10)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity ($)")
    style(ax, "Equity Curve")

    # 2 — Underwater drawdown
    ax = axes[0, 1]
    running_max = eq.cummax()
    dd_pct = (eq - running_max) / running_max * 100
    ax.fill_between(x, dd_pct.values, 0, color=C_LOSS, alpha=0.35)
    ax.plot(x, dd_pct.values, color=C_LOSS, linewidth=1.2)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (%)")
    style(ax, f"Underwater Curve  (max {stats['max_drawdown_pct']:.1f}%)")

    # 3 — Monthly P&L (diverging)
    ax = axes[0, 2]
    monthly = trades_df.set_index("entry_time")["net_pnl"].resample("ME").sum()
    labels = [d.strftime("%b %y") for d in monthly.index]
    colors = [C_WIN if v >= 0 else C_LOSS for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors, width=0.8)
    ax.axhline(0, color=C_INK2, linewidth=0.8)
    step = max(1, len(labels) // 12)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Net P&L ($)")
    style(ax, "Monthly Net P&L")

    # 4 — Net P&L per trade distribution
    ax = axes[1, 0]
    ax.hist(trades_df["net_pnl"], bins=40, color=C_BLUE, alpha=0.85, edgecolor=C_SURFACE, linewidth=0.5)
    ax.axvline(0, color=C_INK2, linewidth=1, linestyle="--")
    ax.axvline(stats["expectancy"], color=C_ORANGE, linewidth=1.5,
               label=f"mean ${stats['expectancy']:.0f}")
    ax.set_xlabel("Net P&L per trade ($)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Trade P&L Distribution")

    # 5 — R-multiple distribution
    ax = axes[1, 1]
    r = trades_df["r_multiple"]
    bins = np.linspace(r.min(), r.max(), 40)
    ax.hist(r[r > 0], bins=bins, color=C_WIN, alpha=0.85, edgecolor=C_SURFACE, linewidth=0.5, label="win")
    ax.hist(r[r <= 0], bins=bins, color=C_LOSS, alpha=0.85, edgecolor=C_SURFACE, linewidth=0.5, label="loss")
    ax.axvline(0, color=C_INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("R-multiple")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)
    style(ax, f"R-Multiple Distribution  (avg {stats['avg_r_multiple']:+.2f}R)")

    # 6 — MAE vs MFE scatter
    ax = axes[1, 2]
    win_mask = trades_df["net_pnl"] > 0
    ax.scatter(trades_df.loc[win_mask, "mae_r"], trades_df.loc[win_mask, "mfe_r"],
               s=18, color=C_WIN, alpha=0.6, label="win", edgecolors="none")
    ax.scatter(trades_df.loc[~win_mask, "mae_r"], trades_df.loc[~win_mask, "mfe_r"],
               s=18, color=C_LOSS, alpha=0.6, label="loss", edgecolors="none")
    ax.set_xlabel("MAE (R)")
    ax.set_ylabel("MFE (R)")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Excursion: MAE vs MFE")

    # 7 — Long vs Short net P&L
    ax = axes[2, 0]
    by_dir = trades_df.groupby("direction")
    dirs = ["long", "short"]
    pnls = [by_dir.get_group(d)["net_pnl"].sum() if d in by_dir.groups else 0.0 for d in dirs]
    wr = [100 * (by_dir.get_group(d)["net_pnl"] > 0).mean() if d in by_dir.groups else 0.0 for d in dirs]
    cnt = [len(by_dir.get_group(d)) if d in by_dir.groups else 0 for d in dirs]
    bar_colors = [C_WIN if v >= 0 else C_LOSS for v in pnls]
    bars = ax.bar(dirs, pnls, color=bar_colors, width=0.6)
    ax.axhline(0, color=C_INK2, linewidth=0.8)
    for b, w_, c in zip(bars, wr, cnt):
        ax.annotate(f"n={c}\n{w_:.0f}% win",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom" if b.get_height() >= 0 else "top",
                    fontsize=8, color=C_INK2)
    ax.set_ylabel("Net P&L ($)")
    style(ax, "Long vs Short")

    # 8 — Exit reason breakdown
    ax = axes[2, 1]
    order = ["target", "stop", "eod"]
    reason_color = {"target": C_WIN, "stop": C_LOSS, "eod": C_AQUA}
    counts = trades_df["exit_reason"].value_counts()
    vals = [int(counts.get(k, 0)) for k in order]
    bars = ax.bar(order, vals, color=[reason_color[k] for k in order], width=0.6)
    for b, v in zip(bars, vals):
        pct = 100 * v / len(trades_df)
        ax.annotate(f"{v}\n({pct:.0f}%)", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color=C_INK2)
    ax.set_ylabel("Trades")
    style(ax, "Exit Reason")

    # 9 — Summary stat panel (text)
    ax = axes[2, 2]
    ax.axis("off")
    ax.set_title("Summary", fontsize=11, fontweight="bold", pad=8)
    rows = [
        ("Net P&L", f"${stats['total_net_pnl']:,.0f}"),
        ("Total return", f"{stats['total_return_pct']:+.1f}%"),
        ("Win rate", f"{stats['win_rate_pct']:.1f}%"),
        ("Profit factor", f"{stats['profit_factor']:.2f}"),
        ("Expectancy", f"${stats['expectancy']:.0f}  ({stats['expectancy_r']:+.2f}R)"),
        ("Payoff ratio", f"{stats['payoff_ratio']:.2f}"),
        ("Sharpe / trade", f"{stats['sharpe_per_trade']:.2f}"),
        ("Max drawdown", f"{stats['max_drawdown_pct']:.1f}%"),
        ("Max consec loss", f"{stats['max_consec_losses']}"),
        ("Trades / day", f"{stats['trades_per_day']:.2f}"),
    ]
    y = 0.95
    for label, val in rows:
        color = C_INK
        if label in ("Net P&L", "Total return"):
            num = stats["total_net_pnl"] if label == "Net P&L" else stats["total_return_pct"]
            color = C_WIN if num >= 0 else C_LOSS
        ax.text(0.02, y, label, fontsize=10, color=C_INK2, va="top")
        ax.text(0.98, y, val, fontsize=10, fontweight="bold", color=color, va="top", ha="right")
        y -= 0.095

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, facecolor=C_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = ORBConfig(
        start_date="2022-01-01",
        end_date="2022-12-31",
        direction_mode="both",
        entry_mode="stop",
        stop_mode="opposite_or",
        rr=2.0,
        risk_per_trade_usd=100.0,
        or_duration_minutes=30,
        last_entry_time=time(11, 00),
        reentry_enabled=True,
        max_reentries_per_direction=1,
    )

    out_dir = os.path.dirname(__file__)
    print(format_config_report(config))

    trades_df = run_backtest(config)
    if trades_df.empty:
        print("\nNo trades generated for the given config / date range.")
    else:
        stats, equity = compute_stats(trades_df, config)
        print()
        print(format_stats_report(stats))

        trades_csv = os.path.join(out_dir, "orb2_trades.csv")
        trades_df.to_csv(trades_csv, index=False)
        print(f"\nSaved trade log      -> {trades_csv}")

        dash_png = os.path.join(out_dir, "orb2_dashboard.png")
        plot_dashboard(trades_df, equity, stats, config, dash_png)
        print(f"Saved 9-panel dash   -> {dash_png}")
