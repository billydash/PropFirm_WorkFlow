# EV Audit Report — 2026-07-08

**Question:** Is the expected value reported by `EV.py` a realistic estimate of the payoff from undertaking the FTMO 50k 2-step Challenge with this ORB strategy?

**Verdict:** No — not as a reliable point estimate. The combination math in `EV.py` is correct, but the trade edge feeding it is too weak to support the confidence the single headline number ($2,368/attempt, 686% ROI) implies. A large share of that number is mechanical (challenge structure), not attributable to trading skill.

Method: audited every script in the pipeline (`ORB.py`, `Challenge Phase/monte_carlo.py`, `Funded Phase/monte_carlo.py`, `EV.py`), then re-ran the actual `simulate_trial()` functions from both Monte Carlo scripts against alternative slices of the historical trade data to stress-test the inputs (not just re-derive the same headline number).

---

## 1. What's solid in the code

- `EV.py`'s formulas (`ev_single_attempt = p_pass·e_take_home − fee·(1 − p_pass·p_refund)`, the persist-until-passed variant, refund conditioning) are internally consistent — no bugs.
- Both Monte Carlo scripts run in `bootstrap_blocks` mode (most conservative of the three resample options — preserves cross-day streaks), with matching `account_size`, `risk_pct`, and block settings, so the Challenge and Funded stats combined in `EV.py` are apples-to-apples.
- The daily-loss mechanic (fixed $ offset off previous close, not a floating %) is correctly and identically implemented in both phase scripts, matching FTMO's documented rule.
- My reimplementation reproduced the production run almost exactly (36.45% pass / $2,324 EV vs. production's 36.85% / $2,368 at lower sim count) — confirms the sub-period tests below are faithful to the real pipeline, not an artifact of a divergent reimplementation.

## 2. The core problem: the edge isn't statistically distinguishable from zero

From `orb_trades.csv` (1,685 trades, 2020-08-03 to 2026-06-30):

| Metric | Value |
|---|---|
| mean r_multiple | 0.0417 |
| std r_multiple | 1.216 |
| t-statistic vs. 0 | **1.41** (need ~1.96 for 95% significance) |
| bootstrap 95% CI of mean | **[-0.017, +0.100]** — straddles zero |
| bootstrap draws with mean ≤ 0 | ~8% |

Year-by-year mean R-multiple: 2020 −0.046, 2021 +0.027, **2022 +0.105 (outlier)**, 2023 +0.073, 2024 +0.047, 2025 +0.016, 2026 partial −0.014. The edge peaks in 2022 and decays toward/through zero afterward.

## 3. Sub-period robustness ("many trials, different data")

Reran the actual `simulate_trial` functions from both MC scripts (not a reimplementation) on different historical windows:

| Scenario | n trades | mean R | t-stat | P(pass) | EV/attempt | EV persist |
|---|---:|---:|---:|---:|---:|---:|
| Full sample (production) | 1685 | 0.042 | 1.41 | 36.5% | $2,324 | $6,375 |
| First half (2020–23) | 818 | 0.046 | 1.09 | 36.6% | $2,583 | $7,056 |
| Second half (2023–26) | 867 | 0.038 | 0.90 | 35.8% | $2,100 | $5,861 |
| **Last 12 months** | 287 | **−0.020** | **−0.30** | **16.1%** | **$198** | **$1,234** |
| Excl. 2022 (best year) | 1400 | 0.029 | 0.88 | 27.8% | $1,307 | $4,696 |

The most recent year of data — the window most relevant to starting the challenge today — collapses the pass rate from 36% to 16% and the per-attempt EV from $2,324 to $198. Dropping just the single best year (2022) nearly halves the persist-EV.

## 4. The decisive test: zero-edge null

Demeaned `r_multiple` to exactly 0 (leaving `mae_r`/variance untouched) and reran the full pipeline:

| | Actual edge | **Zero edge** |
|---|---:|---:|
| P(pass Challenge) | 36.5% | **23.6%** |
| E[take-home\|funded] | $7,038 | **$4,020** |
| EV, single attempt | $2,324 | **$666** |
| EV, persist | $6,375 | **$2,820** |

Even with **no edge whatsoever**, the pipeline reports a strongly positive EV. That's mechanical, not skill: FTMO's 10%/5% targets are hit by variance alone ~24% of the time; payouts withdrawn along the way are never clawed back on a later bust; and the fee is refunded on first payout. Roughly **$650–800 of "EV" is structural**. Everything above that mechanical floor is what the measured edge is supposedly buying — and that increment is exactly the part with a t-stat of 1.41 that evaporates in the last 12 months of data.

## 5. Un-quantifiable but directional additional risks

- **Data feed**: `pull_data.py` uses Alpaca's free-tier `feed="iex"` — IEX carries a small minority of consolidated volume. Every stop/target/breakout trigger in `ORB.py` depends on precise intrabar high/low touches, which an IEX-only tape can misrepresent vs. the SIP-consolidated tape a real broker fills against.
- **Instrument mismatch**: backtested on SPY ETF shares; intended live execution is FTMO's S&P 500 CFD (acknowledged as a "rough placeholder" in `ORB.py`'s own comments) — different spread, hours, and slippage.
- **Parameter history**: git log shows `or_duration_minutes` (15→30), `last_entry_time` (11:30→11:00), and `reentry_enabled` (False→True) were hand-adjusted against this same full-history sample with no held-out test set — in-sample selection risk on top of the statistical noise above.
- Real-trader behavioral factors (deviating from the mechanical backtest under evaluation pressure) aren't modeled at all.

## Bottom line

The code is logically sound; the number it produces isn't reliable. Treat the ~$650–800/attempt mechanical floor as the only defensible part of this EV. The additional ~$1,700 (single-attempt) or ~$3,500 (persist) the strategy claims to add on top rests on an edge estimate that fails a basic significance test and reverses sign in the most recent year of data. The honest statement is "expected value is plausibly positive but not distinguishable from the mechanical floor given available data," not "$2,368 per attempt, 686% ROI."

---

*Methodology note for reproducing/updating this audit: re-run `simulate_trial()` from both `monte_carlo.py` scripts against date-filtered slices of `orb_trades.csv`/`orb_trading_days.csv`, and against a demeaned (`r_multiple - mean`) "zero-edge null" version of the trade pool, to separate the mechanical EV floor from the edge-driven component. Re-run this whenever `orb_trades.csv` is regenerated with new data or a changed strategy config.*
