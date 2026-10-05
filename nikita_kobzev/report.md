# European Power Fair Value — Germany/Luxembourg
**Nikita Kobzev · nikitakob2004@gmail.com**

Option A: next-day hourly Day-Ahead price, EUR/MWh.

## Data and information timing
The supplied SMARD extract contains 17,520 hourly records, 27 September 2024–26 September 2026 (Europe/Berlin), and three physical drivers: load, wind and solar. The user supplied the acquisition parameters; the original builder was not supplied. `data/PROVENANCE.md` records modules, transformations, source checks and the preserved input hash. A fresh-source check for 2026-09-26 found 36 load/solar actual-value differences; sampled prices match. The input is preserved; those specific differences leave OOS features unchanged. Source revisions versus compilation differences cannot be resolved without the original snapshot.

Forecasts are issued at **11:00 Europe/Berlin on D−1**. SMARD renewable day-ahead forecasts may arrive at 18:00, so these and residual-load forecasts are excluded. Load forecasts have an earlier reporting deadline but this extract lacks issue/revision timestamps; they are also excluded from the primary model. [SMARD forecast data][1]; [SMARD guide, pp. 8, 45][2].

Same-hour actuals and errors are prohibited. Physical lag-24 values are also prohibited: evening D−1 generation is unknown at the morning origin. Instead, load/wind/solar use 48- and 168-hour elapsed lags, assuming availability three hours after interval end. This prevents future-delivery information under that assumption. Revised historical actuals remain a limitation: this is a timing-aware retrospective study, not a vintage-certified backtest.

## QA and feature design
UTC timestamps are unique, sorted and uninterrupted. Derived arithmetic, calendars and stored lags pass checks. The only nulls are expected lag warm-up rows. Four DST days contain 23 or 25 hours; daily values use their actual hour counts. No prices are removed or clipped. Training negative-price share is 5.1%; training range is -250.32 to 936.28 EUR/MWh.

Fourteen features combine previous-day/week same-local-hour prices, prior-day price mean/std, six lagged physical values and hour/weekday/month/weekend. Calendar price lags average a repeated source hour and interpolate a missing spring source hour within its already published day. Model physical lags use UTC. The hourly target remains consistent across the 1 October 2025 introduction of quarter-hourly market intervals. [ENTSO-E SDAC implementation][3].

## Validation and results
Training: 2024-10-04–2026-02-28. Validation: 2026-03-01–2026-08-29. Final OOS: 2026-08-30–2026-09-26 (672 hours). Exact employer OOS dates were absent; these are configurable researcher-selected historical periods.

Daily/weekly persistence, scaled one-hot linear regression and CatBoost share the same observations. Two preset CatBoost configurations are compared; validation MAE chooses **catboost**. Selection is saved before OOS scoring, followed by refit on train plus validation. Within OOS, model parameters stay fixed and lagged information advances daily. Thus predictions represent sequential next-day forecasts, not a multiweek forecast made at one origin. All metrics below are EUR/MWh.

| Model | Validation MAE | Validation RMSE | OOS MAE | OOS RMSE |
|---|---:|---:|---:|---:|
| catboost | 27.88 | 40.95 | 45.22 | 61.83 |
| linear_regression | 30.14 | 42.68 | 42.83 | 58.33 |
| daily_persistence | 31.54 | 49.83 | 47.51 | 70.17 |
| weekly_persistence | 39.14 | 59.99 | 57.33 | 80.75 |

The best OOS MAE belongs to linear_regression, not the validation winner; the original selection is retained to avoid test-set reselection. Validation negative-price MAE for the selected model is 42.35, highlighting a weak regime.

CatBoost parameters: `{"depth": 6, "iterations": 500, "l2_leaf_reg": 5, "learning_rate": 0.05}`, RMSE training loss, seed 42, one CPU thread. Its leading training-fit importances are price_prev_day_same_hour (26.3%), price_prev_day_std (12.3%), price_prev_day_mean (11.7%); importance is not causality. Removing all physical drivers from CatBoost gives validation MAE 28.57. A separate fixed CatBoost fit gives September/October 2025 MAE 28.75/26.22; seasonality prevents causal attribution to the interval change. Negative/high-price, peak/off-peak and weekday/weekend validation metrics are saved without favourable-regime selection.

## Fair value and prompt curve
Daily baseload fair value is the mean of hourly predictions. The last OOS day, 2026-09-26, has fair value **95.91 EUR/MWh**. Validation's 80th percentile daily absolute error is 34.16 EUR/MWh, a descriptive uncertainty reference rather than calibrated coverage.

No verified historical prompt quote was acquired. The timestamped interface accepts public German base week/month quotes and rejects incompatible contracts. It anchors last-week weekday/hour shape to a quote, then replaces only tomorrow's hours: **ΔF = H_day/H_contract × (FV_day − curve-implied day)**. Other contract hours remain at the anchored shape. Positive/negative ΔF implies conditional upward/downward pressure, not measured trading alpha. Actual execution would require bid/ask, risk-premium and transaction-cost checks. Sequential daily forecasts cannot establish a same-origin monthly valuation. [EEX data hub][4].

Invalidate the view on material demand/renewable revisions, outages, fuel/carbon moves, out-of-support inputs, worsening errors or stale/mismatched quotes. No strategy P&L is claimed.

## Programmatic LLM and reproducibility
Python supplies validation evidence to an LLM that chooses at most three whitelisted diagnostics. Python validates and executes them; a second call synthesizes the results. This automates review triage across metric tables while keeping numerical calculations deterministic. The LLM cannot alter features, labels, selection or OOS predictions. LLM integration is implemented; live execution is incomplete. Diagnostic selection: skipped; review: skipped. Prompts, exact inputs and response/status artifacts are logged. The current review is an explicitly labelled deterministic fallback. Supply OPENAI_API_KEY and run `python main.py --llm live --llm-only`.

After installing pinned dependencies, `python main.py` recreates outputs offline with fixed seeds. Key limits are missing vintage archives, absent executable curve quotes, the historical OOS-window ambiguity, and incomplete live LLM execution where indicated above. The next research priority is timestamped pre-auction renewable forecasts, followed by broader rolling-origin validation.

[1]: https://www.smard.de/page/en/wiki-article/5884/206318
[2]: https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf
[3]: https://www.entsoe.eu/network_codes/cacm/implementation/sdac/
[4]: https://www.eex.com/en/market-data/market-data-hub
