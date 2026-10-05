# European Power Fair Value

**Nikita Kobzev — nikitakob2004@gmail.com**

Cobblestone Energy case study · Germany/Luxembourg · Option A: next-day hourly Day-Ahead electricity prices in EUR/MWh.

## Review the submission

- [Report](nikita_kobzev/report.md)
- [Detailed methodology and run instructions](nikita_kobzev/README.md)
- [Out-of-sample predictions](nikita_kobzev/predictions.csv)
- [Validation metrics](nikita_kobzev/outputs/model_comparison.csv) and [final OOS metrics](nikita_kobzev/outputs/oos_metrics.csv)
- [Data provenance](nikita_kobzev/data/PROVENANCE.md)
- [LLM prompts and execution logs](nikita_kobzev/ai_logs/)

## Run

Use Python 3.10 or 3.11. From the cloned repository:

```bash
cd nikita_kobzev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate with `.venv\Scripts\activate`.
The default pipeline runs offline using the included dataset and fixed seeds.

## Results and scope

Validation: 1 March–29 August 2026. Final historical OOS: 30 August–26 September 2026, with 672 sequential next-day hourly predictions.

CatBoost was selected on validation MAE (27.88 EUR/MWh). Its OOS MAE is 45.22 EUR/MWh; linear regression performs better on OOS (42.83), and that result does not change the validation-based selection.

The forecasting pipeline is completed and validated. LLM integration is implemented; live execution was not performed because API credentials were unavailable. Logs contain exact prompts and explicit skip statuses, not generated LLM responses. This leaves the assignment's real-call requirement unmet. With credentials, `python main.py --llm live --llm-only` runs the implemented integration.

Historical prompt-curve quotes were not acquired. The conditional translation method and quote interface are implemented; no executable trading performance is claimed. Publication-vintage assumptions and fresh-source discrepancies are disclosed in the report.
