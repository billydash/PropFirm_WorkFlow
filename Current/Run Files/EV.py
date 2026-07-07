"""Expected-value calculator for the full prop firm journey: pay the Challenge fee,
attempt the Challenge + Verification (Current/Challenge Phase/monte_carlo.py), and --
if you pass -- run the funded account until it busts or survives to the simulated
horizon (Current/Funded Phase/monte_carlo.py).

This script does NOT run its own simulation. It reads the per-trial results already
written by those two scripts (mc_results.csv, mc_funded_results.csv) and combines:
  - Challenge pass rate (probability of ever reaching a funded account)
  - Funded phase's average trader take-home (the "avg payout" side of the EV)
  - The one-time Challenge fee, which FTMO refunds 100% "with your first reward
    withdrawal" (confirmed directly on ftmo.com/en/how-it-works/, 2026-07-07) -- i.e.
    refunded only if the trader both passes AND lives long enough as a funded trader
    to receive at least one nonzero payout, not merely on passing Verification.

Run Current/Challenge Phase/monte_carlo.py and Current/Funded Phase/monte_carlo.py
first (in that order) so their output CSVs exist and reflect the same account_size /
strategy assumptions before running this script.
"""

import math
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class EVConfig:
    # Per-trial results from the two Monte Carlo scripts (must be run first)
    challenge_results_path: str = os.path.join(
        os.path.dirname(__file__), "..", "Challenge Phase", "mc_results.csv"
    )
    funded_results_path: str = os.path.join(
        os.path.dirname(__file__), "..", "Funded Phase", "mc_funded_results.csv"
    )

    # One-time Challenge (Step 1 + Step 2) evaluation fee in USD for the account
    # size the two Monte Carlo runs used (both scripts default to account_size =
    # $50,000). ftmo.com's pricing table is rendered client-side and couldn't be
    # scraped directly; this is a best estimate from third-party FTMO fee trackers
    # as of 2026-07-07 (~$319-345 for the $50k 2-Step Challenge). VERIFY against
    # https://ftmo.com/en/#pricing before trusting this for a real decision, and
    # update it here -- it is the single most important number in this file.
    challenge_fee_usd: float = 399.0

    # FTMO's stated policy (ftmo.com/en/how-it-works/): "Receive a 100% refund of
    # your initial fee with your first reward withdrawal." The refund is tied to
    # actually RECEIVING a nonzero payout as a funded trader, not just to passing
    # Verification -- a trader who passes but busts the funded account before the
    # first payout cycle pays the fee and never gets it back.
    fee_refunded_on_first_payout: bool = True


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score confidence interval for a binomial proportion, as percentages."""
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (center - margin) / denom * 100, (center + margin) / denom * 100


def load_challenge_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run 'Current/Challenge Phase/monte_carlo.py' first."
        )
    return pd.read_csv(path)


def load_funded_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run 'Current/Funded Phase/monte_carlo.py' first."
        )
    return pd.read_csv(path)


def compute_challenge_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    passed = df["overall_pass"].astype(bool)
    passed_df = df[passed]
    failed_df = df[~passed]

    ci_low, ci_high = _wilson_ci(int(passed.sum()), n)
    fail_reason_counts = failed_df["overall_fail_reason"].value_counts()
    phase1_pass_pct = 100 * (df["phase1_outcome"] == "passed").sum() / n

    return {
        "num_simulations": n,
        "pass_rate": passed.sum() / n,
        "pass_pct": 100 * passed.sum() / n,
        "pass_ci_low": ci_low,
        "pass_ci_high": ci_high,
        "phase1_pass_pct": phase1_pass_pct,
        "fail_overall_loss_pct": 100 * fail_reason_counts.get("overall_loss", 0) / n,
        "fail_daily_loss_pct": 100 * fail_reason_counts.get("daily_loss", 0) / n,
        "fail_timeout_pct": 100 * fail_reason_counts.get("timeout", 0) / n,
        "avg_days_passed": passed_df["total_days"].mean() if not passed_df.empty else 0.0,
        "median_days_passed": passed_df["total_days"].median() if not passed_df.empty else 0.0,
        "p10_days_passed": passed_df["total_days"].quantile(0.10) if not passed_df.empty else 0.0,
        "p25_days_passed": passed_df["total_days"].quantile(0.25) if not passed_df.empty else 0.0,
        "p75_days_passed": passed_df["total_days"].quantile(0.75) if not passed_df.empty else 0.0,
        "p90_days_passed": passed_df["total_days"].quantile(0.90) if not passed_df.empty else 0.0,
        "avg_trades_passed": passed_df["total_trades"].mean() if not passed_df.empty else 0.0,
        "avg_days_failed": failed_df["total_days"].mean() if not failed_df.empty else 0.0,
    }


def compute_funded_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    survived = df["outcome"] == "survived"
    busted_df = df[~survived]
    got_payout = df["num_payout_events"] >= 1

    ci_low, ci_high = _wilson_ci(int(survived.sum()), n)

    return {
        "num_simulations": n,
        "survival_pct": 100 * survived.sum() / n,
        "survival_ci_low": ci_low,
        "survival_ci_high": ci_high,
        "bust_daily_loss_pct": 100 * (df["outcome"] == "daily_loss").sum() / n,
        "bust_overall_loss_pct": 100 * (df["outcome"] == "overall_loss").sum() / n,
        "avg_days_survived": df["days_survived"].mean(),
        "median_days_survived": df["days_survived"].median(),
        "avg_days_survived_busted": busted_df["days_survived"].mean() if not busted_df.empty else 0.0,
        "refund_rate": got_payout.sum() / n,
        "refund_rate_pct": 100 * got_payout.sum() / n,
        "avg_num_payout_events": df["num_payout_events"].mean(),
        "avg_total_payouts_usd": df["total_payouts_usd"].mean(),
        "avg_take_home_usd": df["trader_take_home_usd"].mean(),
        "median_take_home_usd": df["trader_take_home_usd"].median(),
        "p10_take_home_usd": df["trader_take_home_usd"].quantile(0.10),
        "p25_take_home_usd": df["trader_take_home_usd"].quantile(0.25),
        "p75_take_home_usd": df["trader_take_home_usd"].quantile(0.75),
        "p90_take_home_usd": df["trader_take_home_usd"].quantile(0.90),
    }


def compute_ev_stats(challenge_stats: dict, funded_stats: dict, config: EVConfig) -> dict:
    """Combines Challenge pass probability with the funded account's payout
    distribution into an expected value per Challenge attempt.

    p_pass          = probability a single Challenge purchase reaches a funded account
    p_refund        = P(at least one nonzero payout | funded), i.e. P(fee gets refunded | funded)
    e_take_home     = E[trader take-home | funded], averaged over ALL funded trials
                       (already 0 for trials that busted before any payout)

    Net cash flow of ONE attempt:
      fail Challenge:                          -fee
      pass, never get a payout as funded:       -fee              (take-home is 0, no refund)
      pass, get >=1 payout as funded:            take_home         (fee paid then refunded)

    Per-attempt EV = p_pass * e_take_home - fee * (1 - p_pass * p_refund)

    Lifetime EV (retry with a fresh Challenge purchase after every failed attempt,
    until the Challenge is eventually passed) works out to:
      e_take_home - fee * (1 / p_pass - p_refund)
    """
    fee = config.challenge_fee_usd
    p_pass = challenge_stats["pass_rate"]
    p_refund = funded_stats["refund_rate"] if config.fee_refunded_on_first_payout else 0.0
    e_take_home = funded_stats["avg_take_home_usd"]

    ev_single_attempt = p_pass * e_take_home - fee * (1 - p_pass * p_refund)
    expected_attempts_to_pass = 1 / p_pass if p_pass > 0 else float("inf")
    expected_fees_gross = fee * expected_attempts_to_pass
    ev_lifetime_persist = (
        e_take_home - fee * (expected_attempts_to_pass - p_refund) if p_pass > 0 else -fee
    )

    return {
        "challenge_fee_usd": fee,
        "p_pass": p_pass,
        "p_refund_given_funded": p_refund,
        "avg_take_home_given_funded": e_take_home,
        "ev_single_attempt": ev_single_attempt,
        "roi_single_attempt_pct": 100 * ev_single_attempt / fee if fee else 0.0,
        "expected_attempts_to_pass": expected_attempts_to_pass,
        "expected_fees_paid_gross": expected_fees_gross,
        "ev_lifetime_persist": ev_lifetime_persist,
    }


def format_report(challenge_stats: dict, funded_stats: dict, ev_stats: dict) -> str:
    width = 64
    lines = [
        "=" * width,
        "PROP FIRM WORKFLOW -- EXPECTED VALUE".center(width),
        "=" * width,
        "",
        "-- Challenge Phase (Current/Challenge Phase/mc_results.csv) " + "-" * (width - 62),
        f"{'Simulations':30}{challenge_stats['num_simulations']:,}",
        f"{'Overall pass rate':30}{challenge_stats['pass_pct']:.2f}%",
        f"{'  95% confidence interval':30}[{challenge_stats['pass_ci_low']:.2f}%, {challenge_stats['pass_ci_high']:.2f}%]",
        f"{'Phase 1 (Challenge) pass rate':30}{challenge_stats['phase1_pass_pct']:.2f}%",
        f"{'Fail: overall loss':30}{challenge_stats['fail_overall_loss_pct']:.2f}%",
        f"{'Fail: daily loss':30}{challenge_stats['fail_daily_loss_pct']:.2f}%",
        f"{'Fail: timeout':30}{challenge_stats['fail_timeout_pct']:.2f}%",
        "",
        f"{'Days to pass -- average':30}{challenge_stats['avg_days_passed']:.1f} days",
        f"{'Days to pass -- median':30}{challenge_stats['median_days_passed']:.1f} days",
        f"{'Days to pass -- 10th/90th pct':30}{challenge_stats['p10_days_passed']:.1f} / {challenge_stats['p90_days_passed']:.1f} days",
        f"{'Days to pass -- 25th/75th pct':30}{challenge_stats['p25_days_passed']:.1f} / {challenge_stats['p75_days_passed']:.1f} days",
        f"{'Trades to pass -- average':30}{challenge_stats['avg_trades_passed']:.1f}",
        f"{'Days before failing -- avg':30}{challenge_stats['avg_days_failed']:.1f} days",
        "",
        "-- Funded Phase (Current/Funded Phase/mc_funded_results.csv) " + "-" * (width - 63),
        f"{'Simulations':30}{funded_stats['num_simulations']:,}",
        f"{'Survival rate (to horizon)':30}{funded_stats['survival_pct']:.2f}%",
        f"{'  95% confidence interval':30}[{funded_stats['survival_ci_low']:.2f}%, {funded_stats['survival_ci_high']:.2f}%]",
        f"{'Bust: daily loss':30}{funded_stats['bust_daily_loss_pct']:.2f}%",
        f"{'Bust: overall loss':30}{funded_stats['bust_overall_loss_pct']:.2f}%",
        f"{'Days survived -- average':30}{funded_stats['avg_days_survived']:.1f} days",
        f"{'Days survived -- median':30}{funded_stats['median_days_survived']:.1f} days",
        f"{'Days survived -- avg (busted)':30}{funded_stats['avg_days_survived_busted']:.1f} days",
        "",
        f"{'Got >=1 payout (fee refund)':30}{funded_stats['refund_rate_pct']:.2f}%",
        f"{'Avg payout events / trial':30}{funded_stats['avg_num_payout_events']:.2f}",
        f"{'Avg total payouts (gross)':30}${funded_stats['avg_total_payouts_usd']:,.2f}",
        f"{'Avg trader take-home':30}${funded_stats['avg_take_home_usd']:,.2f}",
        f"{'Median trader take-home':30}${funded_stats['median_take_home_usd']:,.2f}",
        f"{'Take-home 10th/90th pct':30}${funded_stats['p10_take_home_usd']:,.0f} / ${funded_stats['p90_take_home_usd']:,.0f}",
        f"{'Take-home 25th/75th pct':30}${funded_stats['p25_take_home_usd']:,.0f} / ${funded_stats['p75_take_home_usd']:,.0f}",
        "",
        "-- Expected Value " + "-" * (width - 18),
        f"{'Challenge fee':30}${ev_stats['challenge_fee_usd']:,.2f}",
        f"{'P(pass Challenge)':30}{ev_stats['p_pass']:.4f}",
        f"{'P(fee refunded | funded)':30}{ev_stats['p_refund_given_funded']:.4f}",
        f"{'E[take-home | funded]':30}${ev_stats['avg_take_home_given_funded']:,.2f}",
        "",
        f"{'EV, single attempt':30}${ev_stats['ev_single_attempt']:,.2f}",
        f"{'ROI, single attempt':30}{ev_stats['roi_single_attempt_pct']:.1f}%",
        "  one Challenge purchase, no retry if you fail",
        "",
        f"{'Expected attempts to pass':30}{ev_stats['expected_attempts_to_pass']:.2f}",
        f"{'Expected gross fees paid':30}${ev_stats['expected_fees_paid_gross']:,.2f}",
        f"{'EV, persist until passed':30}${ev_stats['ev_lifetime_persist']:,.2f}",
        "  buy a fresh Challenge after every failed attempt until you pass",
        "=" * width,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    config = EVConfig(
        # Verify against https://ftmo.com/en/#pricing before trusting this figure.
        challenge_fee_usd=345.0,
    )

    challenge_df = load_challenge_results(config.challenge_results_path)
    funded_df = load_funded_results(config.funded_results_path)

    challenge_stats = compute_challenge_stats(challenge_df)
    funded_stats = compute_funded_stats(funded_df)
    ev_stats = compute_ev_stats(challenge_stats, funded_stats, config)

    print(format_report(challenge_stats, funded_stats, ev_stats))

    out_path = os.path.join(os.path.dirname(__file__), "ev_summary.csv")
    challenge_row = {f"challenge_{k}": v for k, v in challenge_stats.items()}
    funded_row = {f"funded_{k}": v for k, v in funded_stats.items()}
    pd.DataFrame([{**challenge_row, **funded_row, **ev_stats}]).to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
