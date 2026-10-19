# ml/

The machine-learning package. Services import its serving parts (features, models, NLP, summarisation, the registry loader); everything else runs in the `ml-trainer` container (`docker compose run --rm ml-trainer python -m ...`) or as the `ml-monitor` service. Runs, metrics and models are tracked in MLflow; datasets and reference snapshots are written under `data/` (git-ignored).

| Module | Purpose |
| --- | --- |
| `features.py`, `labels.py`, `hints.py`, `alarm.py` | Feature engineering shared by training and serving, weak congestion labels, anomaly hints, the alarm rules (shared by anomaly_service and the evaluation) |
| `data/` | Versioned datasets: real rows from Postgres (`real.py`), synthetic series with ground-truth episodes (`synthetic.py`), `build.py` |
| `models/` | The LSTM autoencoder and the congestion classifiers as MLflow pyfuncs, and the rule-based fallbacks |
| `nlp/` | The incident classifier (TF-IDF baseline, two-headed DistilBERT), NER and rule entities, camera matching, report stage |
| `summarize/` | Fact context, the template, generation with local models, the faithfulness guard, prompts, reference summaries |
| `training/` | `train_congestion`, `train_anomaly`, `train_nlp` |
| `evaluation/` | Offline and online evaluation of the anomaly detector, fusion, the text models and the summarizer |
| `monitoring/` | Drift checks with Evidently, reference snapshots, the `ml-monitor` scheduler and CLI |
| `lifecycle/` | The retrain CLI, the promotion gate and its records |
| `registry.py` | Loads `@champion` models from the MLflow registry, with hot reload and rule-based fallbacks |

How each model is trained, evaluated, monitored and promoted is described in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#congestion-and-anomaly-models).
