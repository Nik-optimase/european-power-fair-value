# Deterministic model review — no LLM response

Selected by validation MAE: catboost. MAE 27.88 EUR/MWh; RMSE 40.95 EUR/MWh.

Python executed the recorded diagnostics. Inspect selected_diagnostics.json and regime_metrics.csv for numerical findings. Historical source revisions remain unverified; unversioned delivery-hour forecasts were excluded.

The real-call requirement is incomplete. Set OPENAI_API_KEY and run `python main.py --llm live --llm-only` to obtain genuine selection and review responses.
