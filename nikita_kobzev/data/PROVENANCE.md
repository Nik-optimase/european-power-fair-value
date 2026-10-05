# Data provenance

Source attribution: **Bundesnetzagentur | SMARD.de**. Data licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as described in the
[SMARD user guide, pp. 9–10](https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf).
The extract is a transformed compilation, not an endorsement by the source.

The project owner supplied `de_power_2y_clean.csv` and the acquisition details
below. The CSV was preserved byte for byte. SHA256:

```text
d762d2e8068e2ee2ad8f3b324087187c12cfa46107c051af6e924a02320f2bc6
```

Public POST endpoint:
`https://www.smard.de/nip-download-manager/nip/download/market-data`.
Request settings: region `DE`, resolution `hour`, format `CSV`, type `discrete`,
language `de`; local interval 2024-09-27 00:00 through 2026-09-26 23:59:59 in
Europe/Berlin. Request timestamps are Unix milliseconds. Country-DE physical
fundamentals are explanatory proxies for DE/LU bidding-zone prices.

| Module | Input column |
|---:|---|
| 8004169 | price_da_eur_mwh |
| 5000410 | load_actual_mwh |
| 6000411 | load_forecast_mwh |
| 1004067 | wind_onshore_actual_mwh |
| 1001225 | wind_offshore_actual_mwh |
| 1004068 | solar_actual_mwh |
| 2000123 | wind_onshore_forecast_mwh |
| 2003791 | wind_offshore_forecast_mwh |
| 2000125 | solar_forecast_mwh |

According to the owner, raw series were joined by delivery interval. Repeated
autumn 02:00 intervals were distinguished by occurrence index before converting
Europe/Berlin timestamps into unique UTC keys. Total wind equals onshore plus
offshore; errors equal actual minus forecast; residual-load forecast equals load
forecast minus wind and solar forecasts. Calendar fields and elapsed-hour lags
were added. The pipeline independently checks these equalities and rebuilds its
own forecasting features. No source values were clipped or imputed here.

The original `build_de_power_dataset_v2.py` was mentioned but not supplied.
`fetch_smard.py` reconstructs the supplied acquisition procedure. Its DST parser
compares returned local delivery labels to a complete expected UTC timeline.
It does not claim to reconstruct missing historical issue/revision vintages.

Independent source spot checks during implementation:

- The price endpoint returned the same starting values for 27 September 2024.
- All nine modules were fetched for the 25-hour day 26 October 2025. All 225
  source values match the supplied file exactly. Numerical evidence is in
  `outputs/source_spot_checks.csv`.
- These checks validate a sample and the source mapping, not all 17,520 rows
  against a fresh download. They are retained research evidence; normal offline
  runs do not re-fetch the network or overwrite this table.

A second check on 5 October 2026 fetched all nine modules for 26 September 2026
(216 source values). All 24 prices and all values in the other six unchanged
series match. Actual load differs in 23 hours (maximum absolute difference
2.50 MWh); actual solar differs in 13 hours (maximum 320.82 MWh). A second fetch
of these two series confirmed the differences. See `outputs/source_recheck.json`
and `outputs/source_recheck_differences.csv` for evidence.

The preserved input was not overwritten. Without the original extraction
snapshot/builder, later source revisions cannot be distinguished from compilation
differences. Replacing only these last-day actual values in an isolated feature
calculation leaves every OOS feature unchanged: they are too recent to enter any
48/168-hour lag in this holdout. This result applies to this sample only and does
not establish vintage correctness elsewhere in the dataset.

## Availability and market references

- [SMARD forecast data](https://www.smard.de/page/en/wiki-article/5884/206318): renewable day-ahead forecasts may be reported by 18:00 D−1. They are QA/EDA-only here.
- [SMARD guide, pp. 8 and 45](https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf): history can be revised; the load-forecast reporting deadline precedes day-ahead close by two hours. A current archive is not a record of original releases.
- [ENTSO-E SDAC implementation](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/): successful transition to 15-minute market time units in September 2025; delivery starts 1 October. This study uses the hourly SMARD representation required by the case.
- [EPEX interval and hourly-index description](https://www.epexspot.com/en/new-15-minute-products-market-coupling): market-coupling timing and hourly aggregation context.
- [EEX public market-data overview](https://www.eex.com/en/market-data): public hub displays recent history; longer histories are offered via DataSource. A reliably timestamped historical prompt-contract quote was not acquired for this deliverable. Final settlement averages are not used as ex-ante futures quotes.
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): Responses API JSON Schema contract used for bounded diagnostic selection.

The public EEX widget's documented-in-page filter request was attempted at
`https://api.eex-group.com/pub/customise-widget/filter-data-with-scope` and
returned HTTP 403 (`fault filter abort`). No quote was recovered and no access
restriction was bypassed. This is an acquisition limitation of this run, not a
claim that EEX has no public data.

Source documentation was reviewed on 4 October 2026. The historical CSV contains
no vintage timestamps, so publication-delay assumptions are documented explicitly
in README and the report rather than claimed as observed facts.
