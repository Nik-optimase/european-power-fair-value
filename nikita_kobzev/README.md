# European Power Fair Value

**Nikita Kobzev — nikitakob2004@gmail.com**

Germany/Luxembourg · Option A · next-day hourly Day-Ahead prices in EUR/MWh.

This prototype produces sequential next-day forecasts, daily baseload fair values,
and a conditional translation into German prompt-contract views. It is a
historical research exercise, not a live trading system.

**Completion status:** the numerical pipeline is runnable without credentials.
The required genuine programmatic LLM calls remain outstanding until an API key
is supplied. `outputs/completion_status.json` is the authoritative status after
each run. No API response is simulated. Historical curve quotes have not been
acquired; the delivery-matched interface and methodology are provided explicitly.

## Run

Python 3.10 or 3.11 is recommended; verified environment details are in
`outputs/verification.json`. Create an environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows activate with `.venv\Scripts\activate`.
After dependency installation, **`python main.py`** is the single command for
the full pipeline. Paths resolve relative to this repository. No Spark or
notebook execution is required. The default run needs no network and uses the
committed input CSV. CPU threads and random seeds are fixed; exact floating-point
identity across different CPU architectures is not promised.

Useful commands:

```bash
python -m unittest discover -s tests -v
python main.py --config config.json --llm off
python main.py --llm live --llm-only
```

The last command requires `OPENAI_API_KEY` to be set in the environment. Set it
through your shell or secrets manager; never commit it. `OPENAI_MODEL` optionally
overrides the default `gpt-6-luna`. A ChatGPT login is not an API key.
The code does not automatically load `.env`. `--llm live` explicitly permits two
API requests; the default `replay` mode never sends network requests.

## Dataset and QA

The input has 17,520 observations and 28 columns, from **2024-09-27 00:00 to
2026-09-26 23:00 Europe/Berlin**. Price covers the DE/LU bidding zone; fundamentals
cover country DE. This geographic mismatch is documented rather than silently
relabelled. The file was supplied by the project owner and is copied unchanged.
See `data/PROVENANCE.md` for the endpoint, module IDs, transformations, licensing,
SHA256, and independently re-fetched source samples.

The second source check found differences in actual load/solar for the final
delivery day; all sampled prices still match. The supplied file stays unchanged.
Those specific last-day differences do not enter any OOS model feature, but they
reinforce the need for archived source vintages. Counts and evidence are in
`outputs/source_recheck.json` and `data/PROVENANCE.md`.

Critical QA fails on duplicate/out-of-order UTC timestamps, missing hourly
intervals, mismatched local timestamps, missing/invalid physical values,
incorrect component sums, inconsistent derived columns or invalid stored lags.
The initial 24/168 missing lag rows are expected. Model warm-up removes the first
seven local days. Negative prices and genuine price spikes are retained. EDA
uses training data only; full-file QA checks integrity without choosing models.

Local 23/25-hour days are handled explicitly. UTC is the unique delivery key.
The supplied hourly series is used consistently before and after the October
2025 market-interval change; it does not reconstruct quarter-hourly execution.

Optional source reconstruction (not needed for ordinary reproduction):

```bash
python data/fetch_smard.py --start 2024-09-27 --end 2026-09-27
```

The end date is exclusive. This creates `data/de_power_refetched.csv`, never
overwrites the primary input, and may differ if SMARD revises its history.
The original `build_de_power_dataset_v2.py` was named but not supplied;
`fetch_smard.py` is a new implementation of the provided acquisition specification.

## Forecast origin and leakage policy

Each day D is forecast at **11:00 Europe/Berlin on D−1**, before the day-ahead
auction. The core model uses 14 features:

| Group | Features | Timing treatment |
|---|---|---|
| Price history | `price_prev_day_same_hour`, `price_prev_week_same_hour`, `price_prev_day_mean`, `price_prev_day_std` | Conservatively assumed published by 14:00 on the day preceding their delivery |
| Physical history | `load_actual_lag_48`, `load_actual_lag_168`, `wind_actual_lag_48`, `wind_actual_lag_168`, `solar_actual_lag_48`, `solar_actual_lag_168` | Elapsed UTC-hour lags; assumed published three hours after interval end |
| Calendar | `hour`, `day_of_week`, `month`, `is_weekend` | Known calendar, computed in Europe/Berlin |

Same-local-hour price baselines average repeated source hours and interpolate a
missing spring hour from adjacent hours of the already published historical day.
The provided row-shift price lags are checked for integrity but rebuilt using
this calendar policy. Physical lags intentionally represent elapsed UTC hours.

Excluded: same-hour realized load/wind/solar; all same-hour forecast errors;
physical lag-24 values; SMARD same-hour wind/solar forecasts; residual-load
forecasts; and archived same-hour load forecasts. **Lag 24 alone is not proof of
availability:** D−1 evening actuals are still future information at 11:00 D−1.

Renewable forecasts may be published at 18:00 D−1. Initial load forecasts have an
earlier reporting deadline, but the supplied history has no version timestamps
and may contain revisions. Excluding them avoids treating the word “forecast”
as proof of historical availability. Historical actuals can also be revised;
the three-hour delay is an explicit research assumption, not a recovered
publication record. This is **not a vintage-certified backtest**.

`outputs/feature_timing_audit.csv` checks every non-calendar feature against
every applicable forecast origin under these assumptions. The tests also
perturb unavailable future data to check that a forecast's features do not change.

## Experiment

| Split | Local delivery dates, inclusive | Purpose |
|---|---|---|
| Training | 2024-10-04–2026-02-28 | Fit models and preprocessing |
| Validation | 2026-03-01–2026-08-29 | Select model and CatBoost configuration |
| Final OOS | 2026-08-30–2026-09-26 | Final historical evaluation, 672 hours |

The employer email did not specify exact OOS dates. These researcher-selected
historical dates are editable in `config.json`; no official dates are invented.
No current-day live forecast is claimed from an extract ending in September.

Daily and weekly persistence, standardized one-hot linear regression, and
CatBoost share the same rows. CatBoost uses a fixed two-configuration comparison,
RMSE loss, no clipping, one thread and seed 42. The lowest validation MAE wins
(RMSE breaks ties). `selection.json` is written before final OOS scoring.
Each model is refit on train plus validation for final comparison; the selected
model is never changed based on OOS scores. Its parameters stay fixed throughout
the OOS window while observed history advances each day.

`outputs/model_comparison.csv`, `oos_metrics.csv`, `catboost_candidates.csv` and
`report.md` contain computed results. CatBoost feature importance describes the
training fit, regardless of which model wins; it is not causal attribution.
The committed default run selects **CatBoost** (500 trees, depth 6, learning
rate 0.05, L2 regularization 5). Validation MAE/RMSE are **27.88/40.95 EUR/MWh**;
final OOS MAE/RMSE are **45.22/61.83**. Linear regression is better on final OOS
(MAE 42.83), but the model is not reselected using that result.
Validation diagnostics cover negative/high-price hours (high >200 EUR/MWh),
weekday/weekend, and peak (Mon–Fri 08:00–20:00 local)/off-peak.
The default structural diagnostic uses a separate model trained before
September 2025 and evaluates September/October with identical parameters.
Differences may reflect seasonality; they do not identify a causal market-change
effect. Feature ablation is explanatory only and cannot change selection.

## Prompt curve translation

`predictions.csv` has exactly `datetime,y_pred`, with ISO 8601 UTC timestamps.
`outputs/daily_fair_value.csv` averages the actual number of delivery hours.
The validation 80th percentile daily absolute error is a descriptive reference
band, not a calibrated prediction interval.

`data/prompt_curve.csv` is deliberately header-only until genuine public quotes
are available. Its columns are:

```text
observed_at,delivery_start,delivery_end,product,price_eur_mwh,source_url
```

Timestamps must include offsets. Delivery end is exclusive. Supported products
are `DE_BASE_WEEK` (Monday to Monday) and `DE_BASE_MONTH` (first to first), with
local-midnight boundaries. Each quote must have a public HTTPS source and be
observed before the forecast origin, at most 120 hours old. Delivery must not
have started and must contain the whole forecast day. Unsupported or malformed
rows fail; future/stale/nonmatching quotes are not used.

For each eligible quote, build a weekday/hour shape from the prior seven full
local delivery days whose DA prices are already published. Shift that shape to
average to the contract quote. Replace only tomorrow's shaped price by its model
fair value, leaving other contract hours unchanged:

```text
day_spread = model_day_fair_value - curve_implied_day
conditional_curve_revision = forecast_day_hours / contract_hours * day_spread
conditional_curve_mark = observed_contract_quote + conditional_curve_revision
```

Positive revisions suggest upward pressure on the conditional contract value;
negative revisions suggest downward pressure. This is a view to investigate
against executable quotes, risk premia, uncertainty, liquidity and costs.
It is not a full-contract forecast or a tested long/short strategy. Pooling the
sequential next-day forecasts into a same-origin monthly forecast would leak
information. Invalidate on forecast revisions, outages, fuel/carbon shocks,
out-of-support inputs, deteriorating errors or invalid quote timing.

EEX's public hub was researched; a timestamped historical quote was not acquired.
No artificial price or P&L is included. `outputs/trading_view.json` records the
status and `curve_revisions.json` is empty until quotes match.

## Programmatic LLM workflow

1. Python computes validation metrics, regimes, drift, feature importance and
   largest errors. Final OOS labels are never supplied to the LLM.
2. A Responses API call selects 1–3 operations from a fixed whitelist via JSON
   Schema. Python rejects invalid/duplicate/unlisted operations.
3. Python executes only those functions. No generated code is executed.
4. A second call writes a short review using only the supplied evidence.

`ai_logs/` contains exact prompts, evidence, request bodies and response/status
artifacts. Successful real responses are cached under request SHA256 hashes;
default runs replay them only when the complete request matches. Missing keys,
invalid selections or API failures use an explicitly labelled deterministic
fallback. They **do not** satisfy the real-call requirement. Live reviews should
be inspected for factual fidelity before submission. Numerical results remain
deterministic; new live LLM prose can vary.
Incomplete/refused/error API responses are retained in full. Invalid cache files
fall back without network access, and reports distinguish partial success from
two successful stages.

## Contents

```text
nikita_kobzev/
  README.md, requirements.txt, config.json, main.py, report.md, predictions.csv
  src/          data, QA, features, models, validation, diagnostics, LLM, trading, plots, reporting
  data/         preserved CSV, provenance, optional fetcher, curve input contract
  figures/      six compact EDA/evaluation figures
  outputs/      QA, metrics, predictions, timing audit, selection, trading and runtime evidence
  ai_logs/      prompts, structured inputs, real responses or explicit skips
  tests/        information timing, DST, OOS isolation, quote and LLM contracts
```

Limitations are explicit in the report. Priorities for extension are archived
pre-auction weather/renewable forecasts, verified release vintages, public
timestamped curve quotes, and broader rolling-origin validation. Deployment and
automatic order placement are outside this prototype.
