# ml/

Offline ML work, run on the host or in a dedicated container (not part of `docker compose up`).

| Folder | Purpose |
| --- | --- |
| `training/` | Training scripts and configs for CV, NLP and anomaly models |
| `evaluation/` | Offline evaluation, benchmarks and model comparison |
| `monitoring/` | Drift detection and production model-quality checks |

Trained artifacts go to `models/`; experiment runs are tracked in `mlruns/` (both git-ignored).
