# Real-Time Multimodal Traffic Intelligence Platform

Ingests live video, synthetic/real traffic events and text reports, runs CV, NLP and
anomaly-detection models over them, fuses the signals into a single traffic picture,
and serves it to a React dashboard.

> **Status:** the infrastructure (Kafka, PostgreSQL, MLflow), `video_ingest`, `event_simulator`, `cv_inference`, `anomaly_service`, `nlp_service`, `fusion_service`, `summarizer` and `api` are implemented, with trained congestion, anomaly and incident text models. The other app services are placeholders that start, log, and idle.

## Screenshots

The dashboard while a `multi_vehicle_accident` scenario runs on Hyde Park Corner at ×6 (see [Dashboard](#dashboard)). All times are in `DISPLAY_TIMEZONE` (Europe/London).

| | |
| --- | --- |
| ![Overview during an incident](docs/screenshots/overview-during-incident.png) **Overview** during the demo incident: the run's banner, KPIs, the map with an open-incident badge, the live feed with the produced incident highlighted, and the four annotated streams. | ![Incident detail](docs/screenshots/incident-detail.png) **Incident detail:** the latest summary and its generation path, the status timeline across versions, the confidence breakdown, and the linked reports (tagged with the scenario that sent them), anomalies, metric windows and summaries. |
| ![Camera detail](docs/screenshots/camera-detail.png) **Camera detail** on the 15-minute range: the live stream, the congestion timeline in 1-minute buckets, the metric charts and recent anomalies. | ![Demo control](docs/screenshots/demo-control.png) **Demo control:** the trigger form, the run triggered from this browser linked to the incident it produced, and recent scenario runs. |
| ![Reports](docs/screenshots/reports.png) **Reports:** the live report feed with NLP type, severity and camera match, each with its confidence. | ![Models](docs/screenshots/models.png) **Models:** the version in service, the champion and its key MLflow metrics for each model, and a placeholder for drift status. |
| ![Overview in light mode](docs/screenshots/overview-light.png) **Light mode.** | |

## Architecture

```
video_ingest ──► cv_inference ──┐
event_simulator ───────────────┼──► fusion_service ──► summarizer
nlp_service ───────────────────┤            │
anomaly_service ───────────────┘            ▼
                                            api ──► dashboard
```

| Service | Responsibility |
| --- | --- |
| `video_ingest` | Read camera / RTSP / file streams and publish frames |
| `event_simulator` | Generate synthetic traffic events for dev and testing |
| `cv_inference` | Vehicle/pedestrian detection, tracking, counting |
| `nlp_service` | Incident type and severity, entities and camera matching for text reports |
| `anomaly_service` | Anomaly detection over traffic metrics and events |
| `fusion_service` | Combine vision, text and sensor signals into a unified state |
| `summarizer` | Natural-language summaries of current conditions |
| `api` | HTTP/WebSocket API for the dashboard (host port `API_PORT`, default 8001) |
| `dashboard` | React + Vite + TypeScript UI (port 5173) |

### Infrastructure

| Component | Image | Purpose |
| --- | --- | --- |
| `kafka` | `apache/kafka:3.8.0` | Event bus. Single node in KRaft mode (no ZooKeeper). |
| `kafka-ui` | `provectuslabs/kafka-ui:v0.7.2` | Browse topics, messages and consumer groups |
| `postgres` | `postgres:16-alpine` | Application database (`traffic`) and MLflow backend store (`mlflow`) |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.17.2` + psycopg2 | Experiment tracking and model registry. Artifacts are kept in the `mlflow_artifacts` volume. |
| `adminer` | `adminer:4.8.1` | Web SQL client for Postgres |

All five have healthchecks. The 8 app services wait for `kafka` and `postgres` to report healthy before they start.

Kafka has two listeners:

| Listener | Address | Used by |
| --- | --- | --- |
| Internal | `kafka:29092` | Containers on the compose network |
| External | `localhost:9092` (`KAFKA_PORT`) | Code running on your machine |

### Kafka topics

Created by `scripts/create_topics.ps1`, which `up.ps1` runs once Kafka is healthy. Each data topic gets 3 partitions and replication factor 1. Automatic topic creation is disabled, so a misspelled topic name fails instead of silently creating a new topic. Each topic has a Pydantic schema in `traffic_common.schemas` (see [shared/README.md](shared/README.md)).

| Topic | Carries |
| --- | --- |
| `video.frames` | Frames or frame references from `video_ingest` |
| `events.raw` | Raw traffic events from `event_simulator` and other feeds |
| `reports.text` | Free-text incident reports |
| `detections` | CV detections and tracks |
| `traffic.metrics` | Aggregated flow, speed and occupancy metrics |
| `anomalies` | Output of `anomaly_service` |
| `nlp.results` | Incident type, severity, entities and matched camera per report (`nlp_service`) |
| `incidents` | Fused incidents from `fusion_service` |
| `summaries` | Text from `summarizer` |
| `drift.alerts` | Model drift alerts from `ml/monitoring` |
| `simulator.control` | Commands for `event_simulator` (start/stop scenarios) |

There is also one dead-letter topic per consumer group, `dlq.<group>`:

- **Topics:** `dlq.cv-inference`, `dlq.nlp-service`, `dlq.anomaly-service`, `dlq.fusion-service`, `dlq.summarizer`, `dlq.api`, `dlq.ml-monitoring`, `dlq.event-simulator`. Each has 1 partition and keeps messages for 14 days.
- **What lands there:** messages that fail validation, or that still fail after retries.
- **Inspecting them:** browse them in Kafka UI. Each record carries `dlq.*` headers explaining why it failed.

`create_topics.ps1` prints a Kafka warning for every topic with a `.` in its name. It's harmless: the warning is about topics that differ only by `.` versus `_`, and none of ours do.

## Video ingest

`video_ingest` runs one worker thread per row in the `cameras` table, and re-reads the table every `CAMERA_REFRESH_SECONDS`. New cameras start, removed ones stop, and a camera whose `stream_url` changes is restarted.

For each camera it:

1. **Reads** the source. `stream_url` can be a file under `/data/videos` (which maps to `./data/videos`), an `rtsp://` URL, or an `http(s)://` URL. Files play in real time and loop forever, simulating a live feed.
2. **Samples** frames at `FRAME_SAMPLE_FPS` (default 5). Below that, vehicles move further than their own length between frames and cv_inference loses their tracks.
3. **Resizes** frames to at most `FRAME_MAX_WIDTH`, keeping the aspect ratio, and JPEG-encodes them at `FRAME_JPEG_QUALITY`.
4. **Writes** each frame to the shared `frames` volume as `/data/frames/<camera_id>/<timestamp>.jpg`. The volume is read-write for `video_ingest` and read-only for `cv_inference` and `api`.
5. **Publishes** a `FrameMessage` to `video.frames`, keyed by camera.

When a source fails, the worker logs it and retries with exponential backoff (`SOURCE_RETRY_*`); the other cameras keep running. A cleanup thread deletes frames older than `FRAME_RETENTION_MINUTES`, so consumers must handle a frame file that has already gone. Every `METRICS_LOG_INTERVAL_SECONDS`, each camera logs a health line:

```
health camera=cam-old-street state=streaming frames_read=897 frames_published=60 fps=2.00 errors=0
```

All settings are in `.env` (see `.env.example`).

### Sample videos

```powershell
python scripts\download_sample_videos.py
```

The script downloads one openly licensed traffic video per demo camera from Wikimedia Commons into `data/videos/`. It uses the 480p versions, a few MB each, and points each camera's `stream_url` at its file. Licence and author are read from the Commons API, printed, and saved to `data/videos/ATTRIBUTION.md`. If a download fails, the script generates a synthetic video instead, so the pipeline still runs.

Licences and authors are listed under [Video credits](#video-credits).

Options: `--force` (re-download), `--synthetic-only`, `--skip-db`.

### Watching frames

- **Kafka UI:** <http://localhost:8080/ui/clusters/traffic-intel/all-topics/video.frames/messages>
- **Console consumer:**

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:29092 --topic video.frames --max-messages 5 --property print.key=true --property print.headers=true
```

## CV inference

`cv_inference` consumes `video.frames` (consumer group `cv-inference`). It detects cars, trucks, buses, motorcycles, bicycles and people with YOLO26n, follows them with ByteTrack (one tracker per camera), and publishes:

| Output | Where | When |
| --- | --- | --- |
| `Detection` | `detections` | One per processed frame: class, confidence, normalised box, track id, `speed_kmh`, `stopped`, `in_roadway` |
| `TrafficMetrics` | `traffic.metrics` and the `traffic_metrics` table | Every `CV_METRICS_WINDOW_SECONDS` (5 s) per camera |
| Annotated frame | `/data/annotated/<camera_id>/latest.jpg` on the `annotated` volume (read-only in `api`) | Every processed frame, replaced atomically |

`TrafficMetrics` contains:

- **Counts:** unique tracks per class and unique vehicles.
- **Speeds:** average and 85th-percentile speed, one value per vehicle track.
- **Density:** mean number of vehicles inside the roadway per frame.
- **Incidents:** stopped vehicles and pedestrians inside the roadway. A person whose box mostly overlaps a detected bicycle or motorcycle counts as a rider, not a pedestrian.
- **Quality:** mean detection confidence and frames analysed.

**Keeping up in real time on CPU:**

- Frames older than `CV_MAX_FRAME_AGE_SECONDS` (5 s) are skipped as stale.
- Each camera keeps only its newest waiting frame; older ones are dropped and counted as `superseded`.
- A frame whose file retention has already deleted is skipped with a debug log.
- In all three cases the offset is still committed.

Every 30 s each camera logs `processed`, `fps`, inference latency (mean and p95), capture-to-publish lag, and the stale, superseded, missing and error counts.

### Model loading and the CPU backend

At startup the service:

1. Loads `vehicle-detector@champion` from the MLflow registry. On first run, it registers the pretrained `yolo26n.pt` as version 1 instead, logging the model name, input size and class list as run parameters.
2. Caches the weights in the `weights` volume, so restarts don't download them again.
3. Exports the model to OpenVINO once, also cached in the volume.
4. Benchmarks PyTorch and OpenVINO on a real frame and logs both latencies. `CV_BACKEND` picks the backend: `openvino` (default), `pytorch`, or `auto` for whichever is faster. If the export fails, it falls back to PyTorch.
5. Records the version in use, with the latency figures, in `model_versions`.

Inference uses `CV_THREADS` threads, which defaults to half the cores, at most 6. Inside the Docker VM all vCPUs look the same, so using every core puts work on efficiency cores and next to video decoding. On a Core Ultra 5 225H, 14 threads were about 80× slower than 6. OpenVINO is recompiled without CPU pinning for the same reason, which took it from 31 ms to 19 ms per frame. PyTorch is CPU-only (the `whl/cpu` wheels).

### Zones and speed

Each camera's `zone_config` holds a `roadway` polygon (normalised coordinates) and a `pixels_per_metre` scale. It can also hold `pixels_per_metre_y` for traffic moving towards the camera, plus the `reference_width` the scales were measured at. Without a config, the whole frame is the roadway and the frame is taken to be `CV_DEFAULT_FRAME_WIDTH_METRES` (30 m) wide.

```powershell
python scripts\write_zone_configs.py                       # calibrated defaults for the 4 demo cameras
python scripts\write_zone_configs.py --preview data\zone_previews   # draw them on video frames to check
```

cv_inference re-reads zone configs every `CAMERA_REFRESH_SECONDS`, so no restart is needed. How each value is computed:

- **Speed:** ground-point displacement (bottom centre of the box) over the last 2 s, smoothed.
- **Stopped:** speed below `CV_STOPPED_SPEED_KMH` (3) for `CV_STOPPED_AFTER_SECONDS` (5).
- **Accuracy:** a single scale ignores perspective, so speeds are approximate. They are most accurate where vehicles cross the area the scale was measured in.

### Viewing annotated frames

```powershell
docker compose cp cv_inference:/data/annotated/cam-hyde-park-corner/latest.jpg $env:TEMP\annotated.jpg; Invoke-Item $env:TEMP\annotated.jpg
```

## Congestion and anomaly models

All training runs in Docker, in the `ml-trainer` service. It uses the same CPU-only PyTorch wheels as cv_inference, only starts on demand, and mounts `ml/` and `data/`:

```powershell
docker compose run --rm ml-trainer python -m ml.training.train_congestion
docker compose run --rm ml-trainer python -m ml.training.train_anomaly
docker compose run --rm ml-trainer python -m ml.data.build --no-real      # just build a dataset
```

Both training commands accept these options:

| Option | Effect |
| --- | --- |
| `--real-hours N` | Use only the last N hours of real data |
| `--no-real` | Train on synthetic data only |
| `--synthetic-days N` | How much synthetic data to generate |
| `--seed N` | Random seed |
| `--no-promote` | Train and log, but don't register the model or move `@champion` |

`train_anomaly` also takes `--epochs`, `--threshold-percentile` and `--score-method`.

### Data

Datasets are written as versioned parquet under `data/ml/<name>/<UTC time>-<hash>/`, with a `stats.json`. The version, SHA-256 and stats are logged to each MLflow run. Every row records where it came from:

- **Real (`origin=real`):** `traffic_metrics` rows, each joined with the latest loop-detector reading for the same camera at or before the window end (within 30 s). anomaly_service stores those readings from `events.raw` in `sensor_readings`.
  - **Congestion labels** are *weak*: thresholds on speed relative to the camera's free-flow speed, density relative to its median, and stopped vehicles (`ml/labels.py`).
  - **Scenario windows** from `scenario_runs` are anomalies during the active phases. The final, resolution phase is excluded from training.
- **Synthetic (`origin=synthetic`):** per-camera series with daily patterns and injected episodes, each with ground truth (`ml/data/synthetic.py`). The episodes are:
  - an incident: sudden speed drop, stopped vehicles and a density spike;
  - a gradual congestion build-up;
  - a sensor dropout;
  - a camera failure, where detection confidence collapses;
  - pedestrians in the road.

### Congestion classifier

The classifier predicts `free_flow`, `moderate`, `heavy` or `standstill` from engineered window features: current values plus rolling statistics over the last minute (`ml/features.py`). Two models are compared on a time-based split per camera:

- **Gradient boosting baseline** (scikit-learn `HistGradientBoostingClassifier`);
- **Small PyTorch MLP.**

Each is a child run in the MLflow experiment `congestion-classifier`, which records parameters, per-class precision, recall and F1, a confusion matrix, and permutation feature importance. The model with the higher test macro-F1 is registered as `congestion-classifier` with alias `champion`.

### Anomaly detector

The detector is an LSTM autoencoder over 12 windows of 5 s per camera, trained on normal periods only. Its inputs are measured against the camera's own recent past:

- **Relative inputs:** speeds, counts, density, occupancy and frames are each divided by the camera's rolling 10-minute median, which excludes the most recent minute.
- **Absolute inputs:** stopped vehicles, pedestrians, detection confidence and the presence flags.
- **Normalisation:** per camera.

Absolute levels and time of day would make the daily pattern and slow drift look abnormal, especially with less than a day of real history.

The score is the larger of two parts. Contributions in the `Anomaly` message are keyed `reconstruction:<feature>` or `level:<feature>`, and `details.fired_by` says which part produced the score.

- **Reconstruction:** the largest per-feature reconstruction error over the last 3 windows. Each feature's error is divided by the 99th percentile of that feature's error on normal data for the camera, so a break in one smooth channel such as loop speed isn't drowned out by noisy ones. This part catches sudden changes.
- **Level (distance from normal):** how far each relative feature is from 1, and how far stopped vehicles and pedestrians exceed their baseline. Each distance is divided by its own normal 99th percentile and multiplied by `--level-weight` (default 1; 0 turns it off). This part keeps scoring a *sustained* deviation, which the autoencoder alone reconstructs too well and stops flagging. `frames` (frames processed per window) is a pipeline-health signal, not traffic, so it has no level term and its baseline is never frozen; otherwise a lasting dip in processing throughput would raise, and hold open, an alert on every camera at once.

Thresholds are per camera: the 99.5th percentile of the combined score on normal validation data, with a global fallback. The model also accepts an optional input column `baseline_<column>` for every baselined column. Where set, it replaces the rolling baseline, which is how anomaly_service freezes baselines during an alert.

The run is evaluated two ways, using the service's own alarm rules (`ml/alarm.py`):

- **Synthetic incident episodes:** recall, precision, detection delay and false alarms per hour, overall and per episode type.
- **Real scenario runs:** whether an alarm fired during each run's active phases, and how fast.

Everything is logged to the MLflow experiment `traffic-anomaly-detector`, and the model is registered as `traffic-anomaly-detector` with alias `champion`.

Both models are MLflow **pyfunc** models. They take raw rows (one per camera per 5 s window) and compute features themselves. The per-camera references, normalisation, error scales and thresholds are stored inside the model, and the `ml` package is bundled through `code_paths`, so serving can't drift from training.

## Text models

Three parts, all in `ml/nlp/`:

- **Incident classifier:** incident type (8 classes) and severity (4 classes). It is trained in the `ml-trainer` like the other models and registered in MLflow.
- **Entity extraction:** a pretrained NER model plus rules. It isn't trained here.
- **Camera matching:** fuzzy matching against the simulator's gazetteers. It isn't trained here either.

```powershell
docker compose run --rm ml-trainer python -m ml.training.train_nlp            # ~25 min on CPU
docker compose run --rm ml-trainer python -m ml.training.train_nlp --skip-transformer   # baseline only, seconds
```

The training command takes these options:

| Option | Effect |
| --- | --- |
| `--epochs N` | DistilBERT epochs (default 3) |
| `--max-length N` | Token limit per report (default 64) |
| `--batch-size N` | Training batch size |
| `--learning-rate X` | DistilBERT learning rate |
| `--val-fraction X` | Share of the train split held out for validation |
| `--threads N` | PyTorch threads |
| `--skip-entities` | Skip the NER and camera-matching evaluation |
| `--limit N` | Subsample, for a quick smoke run |
| `--no-promote` | Train and log, but don't register the model or move `@champion` |

### Incident classifier

Two candidates run as nested runs in the MLflow experiment `incident-text-classifier`:

- **`tfidf-baseline`:** TF-IDF on word 1–2-grams and character 2–5-grams, with one logistic regression per task.
- **`distilbert`:** `distilbert-base-uncased` with one shared encoder and two heads, so each report needs one forward pass. Fine-tuned on CPU with a 64-token limit, 3 epochs, per-batch padding, and the best epoch chosen by validation macro-F1.

**Data split:** 15% of `reports_train.csv`, stratified, is held out for validation. `reports_test.csv` uses held-out phrasing and location forms, and is used for the final comparison.

**What each run logs:**
- params;
- accuracy, and macro and weighted F1;
- precision, recall and F1 for every class of both tasks;
- confusion matrices (PNG and CSV);
- accuracy by report source;
- `worst_misclassified_test.csv`: wrong predictions, most confident first;
- training time and inference time per report.

**Selection:** the candidate with the higher mean test macro-F1 is registered as `incident-text-classifier` `@champion`. It is an MLflow pyfunc that bundles the tokenizer, encoder, heads and label maps. Input is a `text` column. Output is `incident_type`, `severity`, their confidences, and JSON `probabilities` for both tasks.

Held-out test set, current champion (version 1):

| Model | Task | Accuracy | Macro-F1 | Train time | Inference |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | incident type | 0.900 | 0.903 | 16 s | 0.15 ms/report |
| TF-IDF + logistic regression | severity | 0.792 | 0.782 | | |
| DistilBERT (2 heads) | incident type | **0.955** | **0.956** | 22 min (3 epochs, 10 threads) | 20 ms/report |
| DistilBERT (2 heads) | severity | **0.890** | **0.888** | | |

Both candidates score about 0.999 on validation, because the validation split shares the training templates. The gap shows only on the test split's unseen phrasing. There, DistilBERT is clearly better, especially on severity.

### Entities and camera matching

- **Entities:**
  - Locations come from `dslim/bert-base-NER`. LOC spans are kept; ORG and MISC spans are kept only if the gazetteer confirms them.
  - Gazetteer matches are added where NER misses. The cased NER model misses most lowercase driver and social posts.
  - Vehicle types, lane information and times come from rules.
  - Entity labels are `LOCATION`, `VEHICLE`, `LANE` and `TIME`.
- **Camera matching:**
  - It reuses `traffic_common.gazetteer`, the data the event simulator writes reports from: official names, streets, landmarks, abbreviations, misspellings and hashtags. Gazetteers for other cameras are derived from their name.
  - Text and aliases are normalised the same way: lowercase, "&" becomes "and", standard abbreviations are expanded (St, Rd, Sq, Rbt, Pk, Cnr, X), and hashtags are split.
  - Each alias is compared with word windows of the text by fuzzy ratio (at least 0.85). Short aliases such as HPC and E&C must match exactly.
  - Confidence drops when a second camera scores almost as well. Below 0.7 the result is "no camera".
  - The test split's held-out location forms are never read, and a test checks that.

Results on the held-out test set:

| Measure | NER only | Gazetteer only | Combined (used) |
| --- | --- | --- | --- |
| Location found, overlapping the labelled mention | 0.65 | 0.81 | **0.93** |
| Location exact match | 0.36 | 0.66 | 0.75 |
| Precision of extracted location spans | 0.95 | 0.93 | 0.91 |

| Camera matching | Value |
| --- | --- |
| Accuracy (no camera counts as wrong) | **0.871** |
| Coverage (a camera was returned) | 0.872 |
| Precision when a camera was returned | 0.999 |
| By source: driver / police / operator / social media | 0.83 / 0.84 / 0.93 / 0.90 |

Almost every miss is a place name the gazetteer doesn't contain, such as "Brook Drive" or "Duke of Wellington Place". These come from the test split's held-out location forms, so the matcher returns no camera rather than guessing.

### Online evaluation

```powershell
docker compose run --rm ml-trainer python -m ml.evaluation.nlp_online --hours 24
```

The command joins `nlp_results` with the simulator's ground truth in `simulated_reports` for the last N hours.

- **Measures:** incident and severity accuracy and macro-F1, plus camera accuracy, coverage and precision. Each is reported overall and per source style: driver, police, operator, social media.
- **Lost reports:** it also counts reports that have no result.
- **Logging:** results go to the `incident-text-classifier` experiment, tagged `eval=online`.
- **Options:** `--model-version` limits it to one version; `--no-mlflow` only prints.

## Model limitations

- **The congestion classifier learns rules, not ground truth.** Its labels are derived by rule. Real data gets weak labels from thresholds on speed ratio, density ratio and stopped vehicles (`ml/labels.py`), and synthetic data is labelled by the same rule applied to the generator's noise-free state. Its near-perfect test scores (macro-F1 about 0.99) mostly show that it reproduces that rule. They are not evidence of accuracy against human-labelled traffic states, and there is no such evaluation yet.
- **Real data is short and partly synthetic.** It covers about a day. The camera metrics come from looped sample videos, and the loop-detector readings come from the event simulator. Simulated incidents therefore show up only in the sensor channel, never in the video.
- **The anomaly baseline is short-term.** The detector compares each camera with its own last ~10 minutes, frozen while an alert is active. It doesn't know what is normal for the time of day. A slowdown that builds over more than about 10 minutes, or an incident during an already-congested peak, is detected late or not at all; the evening-peak scenario runs were mostly missed.
- **A time-of-day baseline is planned** once more real data exists, ideally several weeks per camera, so each camera can be compared with its usual traffic at that hour and weekday.
- **The text models have only seen simulated reports.** Reports come from templates with injected noise. The test split's unseen phrasing is a fairer check than validation, but real reports will be more varied, and accuracy on them is unknown until real labelled reports exist.
- **Camera matching only knows the gazetteer.** Places outside it (side streets, new landmarks) get no camera, and nothing is geocoded. The camera a report is tagged with by its source (the simulator sets one) is recorded in `details.reported_camera_id` for evaluation, but never used for matching.
- **Summaries are short and template-like.** The local models are small, and the guard rejects anything it can't verify. ROUGE-L is measured against references we wrote ourselves from the same facts, so it rewards fact coverage, not prose quality.
- **Camera artefacts cause false alarms.** Elephant & Castle's parked vans flicker in and out of "stopped" and cause most real false alarms (about 2 per camera-hour overall). Per-camera zone configs that exclude parking areas would reduce that.

## Anomaly service

`anomaly_service` consumes `traffic.metrics` and `events.raw` (consumer group `anomaly-service`):

- **Sensor readings:** loop-detector readings are stored in `sensor_readings`. A new consumer group starts from the earliest offset, so it backfills the history.
- **Rolling window:** each metrics window is joined with the latest reading and added to the camera's rolling 12-minute history. Replayed history older than 60 s isn't scored.
- **Scoring:** every 5 s each camera's history goes through both champion models. The congestion level, score and threshold are written to `camera_congestion`.
- **Alerts:** an alert *starts* after `ANOMALY_MIN_CONSECUTIVE` (2) checks at or above the camera's threshold.
  - **Frozen baseline:** when an alert starts, the camera's baselines (10-minute medians ending a minute earlier) are frozen. Until the alert ends, the model receives them as `baseline_<column>` inputs, so a long incident doesn't become the new normal.
  - **Hold rule:** the alert stays open while loop or camera speed (mean of the last 3 windows) is below `ANOMALY_HOLD_SPEED_RATIO` (0.6) × the frozen baseline, or stopped vehicles exceed it by more than `ANOMALY_HOLD_STOPPED_MARGIN` (1). The hold lasts at most `ANOMALY_MAX_HOLD_SECONDS` (1800). After that the frozen baseline is released too: a change that lasts that long becomes the new normal, so the score can fall and the alert end (without this, a lasting change in a camera's view kept one alert open for hours).
  - **End:** once the hold is off, the alert ends after `ANOMALY_CLEAR_AFTER` (3) checks below `ANOMALY_CLEAR_RATIO` (0.8) × threshold. A new alert can't start until `ANOMALY_COOLDOWN_SECONDS` (120) after the last one ended.
  - **Severity:** comes from the episode's peak score ÷ threshold, using `ANOMALY_SEVERITY_BOUNDS` (`2,4,8`): below 2× is low, then medium, high, critical.
  - **Updates:** an `update` event is published when severity rises or a new hint appears, at most every `ANOMALY_UPDATE_MIN_SECONDS` (15), so downstream services see escalation.
- **System-wide degradation:** each check, every camera's frames and detection confidence (mean of the last 3 windows) are compared with its baseline (the same ~10-minute median as the anomaly features).
  - **Start:** when at least `ANOMALY_SYSTEM_MIN_CAMERAS` (3) cameras drop in the same check, the cause is taken to be the pipeline (cv_inference overloaded, a host or network problem), not traffic. A drop means frames below `ANOMALY_SYSTEM_FRAMES_RATIO` (0.85) × baseline, or confidence below `ANOMALY_SYSTEM_CONFIDENCE_RATIO` (0.9) × baseline. One `camera_degraded` anomaly is published with `scope: "system"`, no `camera_id` and the cameras in `affected_cameras`, instead of an alert per camera.
  - **While it lasts:** no new per-camera alerts start. `details.suppressed_checks` counts the checks that would otherwise have alarmed. An `update` is published when more cameras are affected.
  - **End:** after `ANOMALY_CLEAR_AFTER` (3) checks with fewer cameras affected. Per-camera alerts that started up to `ANOMALY_SYSTEM_LOOKBACK_SECONDS` (60) before it are closed at the same time, with `details.closed_by: "system_degradation"`. Alerts that started earlier carry on normally.
  - **Limit:** if throughput stays lower for good, the baseline adopts it after about 5 minutes and the episode ends.
  - **Reading it:** a system anomaly's `score` is the number of cameras that dropped and its `threshold` is the minimum, so "3 / 3" means three cameras degraded at once, not a score on the edge of its threshold.
- **Per-camera `camera_degraded`:** the anomaly model's `camera_degraded` hint only stands if the camera's own health check shows the same drop (frames below 0.85× or confidence below 0.9× its baseline).
  - **Without that drop the hint is removed.** cv_inference processes about 25 frames in almost every window, so the model's normal error for `frames_rel` is tiny, and a one-frame wobble used to top the score. The hint was also derived from fewer vehicles than usual, which is quiet traffic.
  - **Pipeline noise doesn't alert.** A check that crosses the threshold only through pipeline features (`frames_rel`, `mean_confidence`, `camera_speed_present`) on a healthy camera starts and holds no alert; the `health_noise` count in the stats line tracks these.
  - **Models from now on:** `ml/hints.py` no longer maps `vehicle_count` to `camera_degraded`, and needs the same minimum drop. Models already registered carry their own copy of `ml/` (MLflow `code_paths`), so the service check above is what applies to them.
- **What an alert contains:** `start`, `update` and `end` `Anomaly` messages are published to `anomalies` with the same `anomaly_id`. One row in the `anomalies` table tracks them (`severity`, `update_count`, and `details` with `fired_by`, `score_parts`, `hold`, `frozen_baseline`, `held_checks`). They carry:
  - hints (`speed_drop`, `stopped_vehicles`, `pedestrian_in_road`, `sensor_dropout`, `camera_degraded`);
  - the score, threshold and contributing features;
  - the congestion level;
  - both model versions.
- **Models:** champions are loaded from the MLflow registry at startup. The registry is checked every `MODEL_REFRESH_SECONDS` (300), and a model is reloaded when its `champion` alias moves. Versions in use are recorded in `model_versions`. Without a registered champion, the service uses rule-based detectors (`ml/models/rules.py`) and logs a WARNING saying so.

```sql
SELECT camera_id, anomaly_type, hints, round(score::numeric, 2) AS score, status, detected_at, ended_at
FROM anomalies ORDER BY detected_at DESC LIMIT 10;
SELECT * FROM camera_congestion;
```

## NLP service

nlp_service consumes `reports.text` as consumer group `nlp-service`.

- **Batching:** reports that are already waiting are handled together, up to `NLP_MAX_BATCH` (32). The consumer never waits to fill a batch. If a batch fails, its reports are retried one by one, with the usual dead-lettering.
- **Per batch:**
  - Each report is stored in `reports`.
  - The champion classifier runs once on the whole batch, and so does NER (`NER_MODEL`, default `dslim/bert-base-NER`).
  - Each report then gets rule-based entities and camera matching on its text.
  - The result is stored in `nlp_results` and published to `nlp.results` as an `NLPResult`. It carries the incident type and severity with their confidences, the full `probabilities`, entities, `camera_id` with `camera_confidence` (or none), `location_name`, `model_version`, `model_versions` (classifier, NER commit, matcher) and `details` (match details, batch size, the report's own camera tag).
- **Redelivery:** a report already analysed by the current model version is skipped.
- **Models:** the champion comes from the MLflow registry and is reloaded when `@champion` moves (`MODEL_REFRESH_SECONDS`). Each loaded version is recorded in `model_versions`, and so is the NER model. Until a classifier is registered, a keyword fallback (`incident-text-rules`) keeps the service running and says so in its logs.
- **Threads:** capped to 2, as in anomaly_service.
- **Downloads:** Hugging Face downloads live in the `hf_cache` volume, shared with `ml-trainer`, so restarts don't download again.
- **Speed:** about 90 ms per report on CPU with NER, most of it NER.
- **Settings:** `NLP_MAX_BATCH`, `NLP_NER_ENABLED`, `NER_MODEL`, `NLP_MATCH_MIN_SCORE` (0.85) and `NLP_MATCH_MIN_CONFIDENCE` (0.7).

## Fusion service

fusion_service turns per-modality signals into incidents. It consumes `anomalies` (camera- and system-scoped), `nlp.results` and `traffic.metrics` as group `fusion-service`. All settings below can be changed with `FUSION_<NAME>` variables (see `.env.example` and `fusion_service/config.py`).

### Correlation

- **Grouping:** evidence on a camera within a sliding window (`FUSION_WINDOW_SECONDS`, 600) joins the open incident it fits best: anomalies by how well their hints support the incident's type, reports by type compatibility. A camera can have several open incidents. A burst of reports that fits none of them starts its own incident instead of being dropped.
- **Ignored reports:** NLP results of type `none`, and those with classifier confidence below 0.5.
- **Unlocated reports:** reports with no matched camera wait in an unlocated pool. One joins an open incident later if one of its place entities fuzzy-matches (at least 0.88) a place named by that incident's reports. Its camera confidence is then 0.8 × the match score.
- **Equipment faults:** `camera_degraded` and `sensor_dropout` are not incidents. A camera reporting itself degraded also makes its other hints count as degraded vision.

### Incident type and severity

- **Common type:** the type space is `IncidentType`, and the reported type is preferred.
  - With reports, the type is the classifier-confidence vote of the attached reports. Congestion counts half, because it is usually a consequence of something else.
  - Without reports, it comes from the anomaly hints: stopped vehicles plus a speed drop → accident; stopped vehicles alone → breakdown; a speed drop alone → congestion; pedestrians in the road → police activity.
- **Severity:** the maximum of two estimates, kept as a running peak.
  - **Text:** the highest level at least two reports agree on.
  - **Vision:** the anomaly's own severity, capped at high unless traffic is at a standstill, and at least what the congestion level implies.

### Fused confidence

The fused confidence is stored as a per-component breakdown, in `incidents.confidence_components` and in every `Incident` message, so the dashboard can show where it came from:

| Part | How it's computed |
| --- | --- |
| vision | 0.3 + 0.7 × (0.6 × ratio score + 0.4 × duration score). The ratio score scales peak score ÷ threshold from 1× (0) to 4× (1); the duration score scales the anomaly's duration up to 120 s. × 0.3 if the evidence was seen during a system-wide degradation. |
| text | 0.4 × (1 − e^(−n/3)) + 0.25 × min(sources, 3)/3 + 0.2 × mean classifier confidence + 0.15 × mean camera-match confidence. *n* is the supporting reports of the last 3 minutes; congestion reports count half for other types, and unrelated reports count zero. |
| agreement | How well the anomaly hints support the reported type, for example stopped vehicles + accident = 1.0, or a speed drop + weather hazard = 0.8. It is 0 unless both modalities are present. |
| fused | 0.6 × the stronger modality + 0.25 × the weaker one + 0.15 × agreement |

### Lifecycle

- **Suspected:** when one modality fires.
  - Vision fires on a traffic anomaly that started recently, unless the camera is under a system-wide degradation.
  - Text fires on a burst: at least 4 same-type reports in 3 minutes, from at least 2 sources.
- **Confirmed:**
  - when both modalities are reasonably strong (vision at least 0.4, text at least 0.65) and agree (at least 0.5);
  - or when one modality alone reaches 0.9.
  - The breakdown at that moment is kept in `details.at_confirmation`.
- **Resolved:** when all three of these hold:
  - no attached anomaly is active;
  - the camera's speed is back to at least 0.8 × its pre-incident baseline;
  - either a clearing report has arrived (`stage: cleared` from nlp_service, after which only `FUSION_CLEAR_MIN_SECONDS`, 20, must pass since the anomaly ended), or the incident hasn't been firing for `FUSION_QUIET_SECONDS` (300). Stray background reports don't keep it firing.
- **System-wide degradation:** no vision-only incidents are created on the affected cameras. Existing incidents' vision evidence is marked degraded and down-weighted, and that change is published.

### Output and recovery

- **Storage:** incidents and their `incident_evidence` links (anomalies, reports, metric windows, each with a weight and details) are stored in Postgres. A link whose target row isn't written yet is retried for up to 5 minutes.
- **Publishing:** an `Incident` goes to `incidents` on creation and on every meaningful change: status, severity, type, degraded vision, or a confidence change of at least `FUSION_PUBLISH_CONFIDENCE_DELTA` (0.1). `version` counts these publications, and `change_reasons` says why each was sent.
- **Restart:** the full state of each open incident is kept in `incidents.details`. On startup, open incidents and the last window of anomalies and NLP results are rebuilt from Postgres. Evidence is keyed by id, so redelivered messages can't duplicate anything.

### Evaluation

```powershell
docker compose run --rm ml-trainer python -m ml.evaluation.fusion_eval --hours 2
```

The command compares incidents with `scenario_runs` and logs to the MLflow experiment `incident-fusion`.

- **Per scenario type:** detection rate, time from scenario start to suspected and to confirmed, type and severity accuracy, and time to resolve after the scenario's clearing phase.
- **False incidents:** incidents outside scenario windows, per camera-hour, with suspected and confirmed counted separately.

## Summarizer

The summarizer writes incident summaries for operators with a local model. Nothing is sent to an external LLM API.

### When it summarizes

- **Input:** it consumes `incidents` as consumer group `summarizer`.
- **Triggers:** a summary is written when an incident is created or confirmed, is resolved, or changes type or severity. A confidence change alone doesn't trigger one.
- **Debounce:** updates to one incident within `SUMMARY_DEBOUNCE_SECONDS` (10) are merged, and only the latest version is summarized. Each summary records what triggered it.
- **After a restart:** incident versions from the last hour that have no summary yet are summarized.

### Facts

- **Context:** a fact-only context is built from the `Incident` message and Postgres (`ml/summarize/context.py`):
  - camera name and area, incident type, severity, status, start time (in `DISPLAY_TIMEZONE`, with its abbreviation: "since 18:48 BST") and duration;
  - congestion level, stopped vehicles and pedestrians in the roadway;
  - **Speed:** loop-detector speed against its own pre-incident baseline (the 10 minutes before the start). Camera speed is stated only when its ratio to baseline agrees with the loops (within 25 points) or there is no loop data. The demo cameras play looped footage that never shows a simulated incident, which once produced "8 km/h against 6 km/h normally" during a standstill;
  - the number of reports and their sources, and the vehicles, lanes and places they mention;
  - the confidence breakdown, and whether vision was degraded.
- **Recommended action:** derived by rule from status and severity (dispatch and divert, divert and patrol, monitor, verify on camera), so the model can't make one up.

### Output

Each summary has a headline of under 15 words and 2–4 sentences: what is happening, the impact, the evidence, and the recommended action.

- **Model:** `google/flan-t5-base` by default, with the prompt template `ml/summarize/prompts/v1.txt`. Chat models such as Qwen2.5-Instruct use the chat prompt `v2.txt`, which asks for a headline and a summary in a single generation. The recorded prompt version includes a content hash. Weights come from the shared `hf_cache` volume, and threads are capped at 2.
- **Faithfulness guard:** the summarizer checks the model's output (`ml/summarize/faithfulness.py`) and uses the deterministic template instead when:
  - any number, time, place name or vehicle type in it isn't in the facts;
  - it contradicts the incident type, severity or status;
  - generation fails, or takes longer than `SUMMARY_TIMEOUT_SECONDS` (20);
  - the output is too short.

  The published summary says which path was used (`generation`), why (`faithfulness`), and what the model wrote when it was rejected.
- **Configuration:** the service loads the `incident-summarizer` @champion from MLflow (model name, prompt version, generation settings), reloads it when the alias moves, and records it in `model_versions`.
- **Storage:** summaries go to the `summaries` table with the incident version and the context used, and are published to `summaries`.

### Evaluation

```powershell
docker compose run --rm ml-trainer python -m ml.evaluation.summarizer_eval
```

The evaluation uses a reference set of 40 contexts: the stored incidents, topped up with contexts built from every phase of the scenario ground truth. Reference summaries are written from the same facts in a different style from the template (`ml/summarize/reference.py`).

It always uses the same pinned reference set (`data/summarizer/reference_set.json`). It compares the template with flan-t5-small and flan-t5-base (prompt v1) and Qwen2.5-0.5B-Instruct (prompt v2). Qwen2.5-1.5B-Instruct is included only if its p95 latency on a five-context probe is under 10 s. The comparison covers:
- ROUGE-L of the raw model output, and of what the service would publish after the guard;
- the guard's pass rate and the fallback rate;
- CPU latency with 2 threads.

The configuration with the best published ROUGE-L, among those with a guard pass rate of at least 0.9 that fit the time budget, is registered as `incident-summarizer` @champion. Everything is logged to the MLflow experiment `incident-summarizer`.

Results on the pinned reference set (40 contexts: 20 stored incidents and 20 scenario phases), CPU, 2 threads:

| Configuration | Prompt | ROUGE-L, raw | ROUGE-L, published | Guard pass rate | Fallback rate | Rejected as incomplete | Latency mean / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | – | 0.535 | **0.535** | 1.00 | 0 | 0 | < 1 ms |
| flan-t5-small | v1 | 0.218 | 0.452 | 0.98 | 0.65 | 0.63 | 1.4 s / 1.9 s |
| flan-t5-base | v1 | 0.297 | 0.432 | 1.00 | 0.40 | 0.40 | 5.0 s / 8.7 s |
| Qwen2.5-0.5B-Instruct | v2 (chat) | 0.257 | 0.510 | 0.13 | 0.90 | 0.03 | 14.1 s / 17.1 s |
| Qwen2.5-1.5B-Instruct | v2 (chat) | excluded: probe p95 121.5 s (over the 10 s limit) | | | | | |

**What the results show:**
- The flan-t5 models stay faithful but terse (20–27 words) and often incomplete.
- Qwen2.5-0.5B writes full briefings (about 80 words), but 35 of 40 failed the guard. It invents injuries, numbers of drivers, delays in hours and causes ("speed limit violated"). Its published score is high only because the template replaced most of its outputs.
- Qwen2.5-1.5B was far too slow for this CPU budget: about 6 GB in full precision in a 7.5 GB Docker VM.
- Only configurations with a guard pass rate of at least 0.9 can be promoted. Among those, the template has the best published ROUGE-L, so it stays champion (`incident-summarizer` v3).
- These numbers are lower than in earlier runs because the facts and references now use loop-detector speeds. Part of the template's lead comes from the references being written from the same facts. To try a model in the service anyway, set the `@champion` alias to a registered model configuration.

## API

`api` is a FastAPI service on port `API_PORT` (8001). It uses async SQLAlchemy with psycopg 3 against Postgres. Interactive documentation is at <http://localhost:8001/docs>. Records the platform already has a message schema for (incidents, summaries, anomalies, NLP results, reports, metrics) are returned in those shared schemas (`traffic_common.schemas`), so REST and the WebSocket use the same shapes. Lists are paginated with `?page=` and `?page_size=` (at most 200) and return `{items, total, page, page_size}`.

| Endpoint | What it returns |
| --- | --- |
| `GET /api/v1/cameras` | Every camera with location, current congestion level, latest metrics window, open incident count, and whether vision is degraded (with the reasons) |
| `GET /api/v1/cameras/{id}?minutes=60&resolution=60` | Zone config, the metrics history aggregated in SQL into `resolution`-second buckets (5 s to 1 h, so `resolution=60` gives 1-minute buckets), and recent anomalies, including system-wide ones |
| `GET /api/v1/incidents?status=&camera_id=&incident_type=&severity=&since=&until=` | Incidents, newest first |
| `GET /api/v1/incidents/{id}` | The incident with its confidence breakdown, every published version, linked evidence (anomalies, reports with their NLP results, metric windows, evidence weights) and all summaries with their generation path |
| `GET /api/v1/reports?camera_id=&incident_type=&unlocated=&since=` | Reports with their latest NLP result. `unlocated=true` gives those matched to no camera. The default range is the last 24 h. |
| `GET /api/v1/analytics/volume?hours=24` | Hourly vehicles (the sum of per-window unique tracks) and mean speed per camera |
| `GET /api/v1/analytics/congestion?hours=24&bucket_minutes=60` | Congestion level counts and shares per camera and time bucket, from `congestion_history` (written by anomaly_service on every check). Buckets go down to 1 minute; `minutes=` sets the range in minutes instead of `hours`. Over long ranges the bucket grows to keep at most 2000 per camera. |
| `GET /api/v1/analytics/incidents?hours=168` | Incident counts by type, severity and status |
| `GET /api/v1/models` | The version in service of every model (`model_versions`), with its MLflow champion, the champion run's metrics, and the latest run in the experiment of the same name (for example an online evaluation). MLflow is read over REST and cached for 60 s. If MLflow is unreachable, `mlflow` is null. |
| `GET /api/v1/scenarios`, `GET /api/v1/scenarios/runs` | The simulator's scenarios (from its YAML files) and recent runs |
| `POST /api/v1/scenarios/{name}/trigger` `{"camera_id": "...", "time_scale": 6}` | Validates the scenario and camera, publishes a `SimulatorCommand` to `simulator.control`, and returns `202` with the `scenario_id` |
| `GET /api/v1/frames/{camera}/latest.jpg` | The latest annotated frame, sent with no-cache headers and `X-Frame-Time` |
| `GET /api/v1/frames/{camera}/stream.mjpg?fps=5` | An MJPEG stream (`multipart/x-mixed-replace`) that pushes each new frame cv_inference writes. The rate is capped by `API_MJPEG_MAX_FPS`, and the last frame is resent every 10 s when nothing new arrives. |

**Live updates: `ws://localhost:8001/ws/live`.**

- **Consumer:** a background Kafka consumer runs with a group unique to the API process, starting at the latest offsets. It reads `incidents`, `summaries`, `anomalies`, `traffic.metrics` and `nlp.results`.
- **Envelopes:** each message is sent as `{"type": "<topic>", "camera_ids": [...], "ts": ..., "data": {...}}`, where `data` is the shared schema.
- **Subscribing:** connect with `?topics=incidents,summaries&cameras=cam-old-street`, or send `{"action": "subscribe", "topics": [...], "cameras": [...]}` later. Messages without a camera, such as unlocated reports, go to camera-filtered clients too.
- **Rate limits and heartbeats:** `traffic.metrics` is throttled per camera (`API_WS_METRICS_INTERVAL_SECONDS`, 2). Idle connections get `heartbeat` messages (`API_WS_HEARTBEAT_SECONDS`, 15).
- **Slow clients:** each client has its own queue (`API_WS_QUEUE_SIZE`, 500). A slow client loses its oldest messages and is told how many (`{"type": "dropped", "count": n}`). One that keeps falling behind is disconnected. Other clients are never held up.

**Health:**
- `/health` reports the process is up.
- `/ready` checks that Postgres and Kafka are reachable and the live consumer is running. It returns 503 otherwise, and it's the container healthcheck: the dashboard starts once the API is ready.

**CORS:** only the origins in `API_CORS_ORIGINS` are allowed (the dashboard on port 5173 by default).

**Errors:** every error has the same shape, whether it's a route's 404, a validation error or an unhandled exception:

```json
{"error": {"status": 422, "code": "validation_error", "message": "invalid request",
           "request_id": "8b81b8b0fe604003",
           "details": [{"location": ["query", "status"], "message": "Input should be ...", "type": "enum"}]}}
```

The codes are `not_found`, `validation_error`, `internal_error` and so on. `details` is only present for validation errors. A 500 never includes internal error text.

**Request logging:** every response carries `X-Request-ID` (yours, if you sent a well-formed one) and `Server-Timing: app;dur=<ms>`. Each request is logged as one JSON line on `api_service.access` with the request id, method, path, query, status, duration and client. `/health` and `/ready` log at DEBUG, so the healthcheck doesn't flood the log.

**Docs:** OpenAPI is at `/docs` (Swagger UI) and `/openapi.json`. Endpoints are grouped by tag, with descriptions, request and response examples, and the error schema.

## Dashboard

`dashboard` is a React 18 + Vite + TypeScript app at <http://localhost:5173>, run by the `dashboard` container (Vite dev server, `./dashboard` bind-mounted). It talks to the API only: REST through TanStack Query, live data through one `/ws/live` connection.

| Page | Shows |
| --- | --- |
| Overview `/` | KPI tiles (open incidents, confirmed today, cameras degraded, average speed), a Leaflet/OpenStreetMap map of the cameras coloured by congestion with an open-incident badge, the live incident feed, and the 4 annotated MJPEG streams with congestion, vehicle count and average speed overlaid |
| Camera `/cameras/:id` | The large stream, vehicle count, average and p85 speed, density and stopped-vehicle charts over 15 min / 1 h / 6 h / 24 h, the congestion level timeline (1-minute buckets on the 15-minute range), recent anomalies (type hints, score vs threshold) and the camera's incidents |
| Incidents `/incidents` | Filterable (status, type, severity, camera, time range), paginated table; filters live in the URL |
| Incident `/incidents/:id` | Latest summary with its generation path, the status timeline across versions, the confidence breakdown (vision, text, agreement; a note when vision was degraded), linked reports with NLP labels, anomalies and metric windows, and every summary with the version it was written for |
| Reports `/reports` | Live report feed with NLP type, severity, matched camera and confidences, including unlocated reports; filter by type and camera |
| Models `/models` | One card per model: version in service, champion, key MLflow metrics and links to MLflow; a placeholder for drift status |
| Demo control `/demo` | Trigger any scenario on any camera at a chosen time scale, see running and recent runs, and the incident each triggered run produces (also highlighted in the overview feed) |

**How live data flows:**
- **One connection.** `src/lib/liveSocket.ts` reconnects with capped exponential backoff and jitter. It drops a connection that has sent nothing (not even a heartbeat) for 45 s, and reconnects at once when the browser comes back online or the tab becomes visible.
- **Batching.** Messages are applied to a Zustand store (`src/store/live.ts`) in 250 ms batches, one store update per batch. Every collection in the store is capped.
- **Versions.** Incidents are keyed by id, and only a higher version replaces a row. So the feed updates rows in place and briefly highlights the one that changed.
- **Reconnects.** After a reconnect the store is re-seeded from REST and queries are refetched, so nothing missed during the gap is lost.
- **Streams.** MJPEG streams go to `127.0.0.1:8001` while REST uses `localhost:8001`. Browsers allow 6 connections per host, so four long-lived streams would otherwise starve REST calls. Streams pause while the tab is hidden.

**Times:** every clock time is shown in `DISPLAY_TIMEZONE` (default Europe/London), not the viewer's own zone, so the dashboard matches the summaries ("since 18:48 BST") and the timestamp burned into the video frames. The header shows the current time and zone.

**When the API is down:** the header shows API and WebSocket status. Views keep their last data and show an inline error with a retry. Requests time out after 10 s and nothing blocks rendering.

**Linking a demo run to its incident:** an incident doesn't carry a scenario id. The dashboard therefore links a run you trigger in two steps:
1. **Probable match:** the first incident at that camera after the trigger.
2. **Confirmed link:** once one of the incident's evidence reports carries the run's `scenario_id`.

Tracked runs are kept in the browser's localStorage and shared between the dashboard's open tabs. A link found in one tab is never overwritten by another.

**Colour scales** (`src/lib/scales.ts`) are always paired with a text label:

| Scale | Colours |
| --- | --- |
| Congestion | free flow green, moderate amber, heavy orange, standstill red |
| Severity | low blue, medium violet, high magenta, critical red-orange, plus 1–4 pips |

The severity colours never reuse a congestion colour, and they were checked for colour-vision-deficiency separation in both themes. Dark is the default theme; the header toggles light mode.

```powershell
cd dashboard
npm run gen:types     # regenerate src/api/schema.d.ts from http://localhost:8001/openapi.json (API must be up)
npm run typecheck     # tsc
npm run lint          # eslint, zero warnings allowed
npm test              # vitest + React Testing Library
npm run check         # all three
docker compose exec dashboard npm run check    # the same inside the container
```

The tests cover the WebSocket client (backoff, reconnect, stale detection, batching), the live store (version ordering, caps, single update per batch), the congestion and severity colour mapping, the incident feed's update logic and in-place rendering, demo-run matching, and the scenario trigger form.

After changing `dashboard/package.json`, rebuild and recreate the container with fresh `node_modules`: `docker compose build dashboard; docker compose up -d --no-deps --renew-anon-volumes dashboard`.

## Event simulator

`event_simulator` produces the non-video inputs for every camera in the `cameras` table, re-reading the table every `CAMERA_REFRESH_SECONDS`:

| Output | Topic | Rate |
| --- | --- | --- |
| Free-text reports (`TextReport`) | `reports.text` | One every `REPORT_INTERVAL_SECONDS` on average (default 10 s), across all cameras |
| Weather (`RawEvent`, `event_type=weather`) | `events.raw` | Every `SENSOR_INTERVAL_SECONDS` (15 s) per camera |
| Loop-detector readings (`RawEvent`, `event_type=loop_detector`) | `events.raw` | Every `SENSOR_INTERVAL_SECONDS` (15 s) per camera |

How the inputs are generated:

- **Reports** come from four source styles: formal road-operator bulletins, police updates, drivers (typos, lowercase, missing punctuation, abbreviations) and social media (emojis, hashtags, shouting). They cover 8 incident types (`accident`, `breakdown`, `road_works`, `congestion`, `weather_hazard`, `debris`, `police_activity`, and `none` for chatter) at 4 severities. The severity is written into the text, for example "minor bump, no injuries" versus "all lanes closed, casualties".
- **Location mentions** vary: official names, nearby streets, landmarks, abbreviations, misspellings, and "street near landmark" combinations. Operators and police never misspell.
- **Weather** changes condition rarely and moves gradually.
- **Traffic** follows a daily pattern with weekday peaks around 08:00 and 17:30, flatter at weekends. Speed falls as flow rises.

### Scenarios

Scenarios are YAML files in [services/event_simulator/scenarios/](services/event_simulator/scenarios/): `multi_vehicle_accident`, `broken_down_lorry`, `flooding` and `road_works`. Each is a timeline of phases. A phase sets:

- how many reports it emits, from which sources, with which incident types and severity;
- the traffic effect (speed and count factors, ramped in);
- optional weather.

Every scenario ends with a phase of "cleared" reports. Every message a run emits carries its `scenario_id`.

```powershell
docker compose exec event_simulator python -m event_simulator.trigger --list
docker compose exec event_simulator python -m event_simulator.trigger multi_vehicle_accident cam-hyde-park-corner
docker compose exec event_simulator python -m event_simulator.trigger flooding cam-old-street --time-scale 10   # 10x faster
docker compose exec event_simulator python -m event_simulator.trigger --stop <scenario_id>
```

The trigger CLI publishes a `SimulatorCommand` to `simulator.control`, and any other producer can do the same. Two more things to know:

- **Scheduled scenarios:** set `SCENARIO_SCHEDULE_MINUTES` to start a random scenario on a random idle camera every N minutes. It is off by default.
- **Limits:** each camera runs one scenario at a time, and commands older than 10 minutes are ignored.

### Ground truth

Every simulated report is stored with its labels in `simulated_reports`: incident type, severity, source style, `location_mention`, stage, scenario and phase. Every run is stored in `scenario_runs`: scenario, camera, status, trigger, and the timeline with actual start and end times. Both tables are kept separate from `reports` and `incidents`, which hold what the platform itself ingests and infers. Evaluation joins the two worlds on `report_id` and `scenario_id`.

```sql
SELECT scenario_name, camera_id, status, started_at, ended_at, reports_emitted FROM scenario_runs ORDER BY started_at DESC;
SELECT phase, incident_type, severity, style, text FROM simulated_reports WHERE scenario_run_id = '<scenario_id>';
```

### NLP training data

```powershell
docker compose exec event_simulator python -m event_simulator.dataset            # writes ./data/nlp/
```

The command writes `data/nlp/reports_train.csv` (5,000 rows) and `data/nlp/reports_test.csv` (1,000 rows), with columns `text, incident_type, severity, camera_id, location_mention, source`. It then prints the class distribution. The two splits differ in three ways:

- **Seeds:** each split uses a different random seed.
- **Phrasings:** every phrase pool (templates, incident and severity phrases, location forms) has held-out entries that only the test split uses, about half the time. A good test score therefore means the model generalises beyond memorised templates.
- **Duplicates:** texts are unique within each split, and no test text appears in train.

Options: `--train N`, `--test N`, `--train-seed`, `--test-seed`, `--out DIR`.

## Repository layout

```
services/<name>/     main.py, Dockerfile, requirements.txt per service
shared/              traffic-common package: settings, Kafka schemas + clients, DB models + migrations
dashboard/           React + Vite + TypeScript frontend
ml/                  features, labels, datasets, models, training CLIs, evaluation (+ ml-trainer image)
data/                local datasets and sample videos (git-ignored)
models/              trained model artifacts (git-ignored, mounted read-only)
tests/               repository-level tests (+ Dockerfile for the in-Docker test runner)
infra/               MLflow image, Postgres init scripts
scripts/             PowerShell helpers (up / down / logs / create_topics / test)
ruff.toml            lint settings for the whole repo
```

## Prerequisites (Windows)

- Docker Desktop with the WSL 2 backend
- Python 3.11 (for local development and tests; containers use `python:3.11-slim`)
- Node.js 20+ (optional, only for running the dashboard outside Docker)

If script execution is blocked, allow local scripts once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Quick start

```powershell
.\scripts\up.ps1                         # creates .env if missing, builds, starts everything, creates topics
.\scripts\logs.ps1                       # follow all logs
.\scripts\logs.ps1 -Services api         # follow one service
.\scripts\create_topics.ps1              # (re)create any missing Kafka topics; safe to re-run
.\scripts\down.ps1                       # stop; data in Kafka, Postgres and MLflow volumes is kept
.\scripts\down.ps1 -Volumes              # stop AND delete all Kafka, Postgres and MLflow data
```

| Script | Options |
| --- | --- |
| `up.ps1` | `-Services a,b` · `-NoBuild` · `-Attach` |
| `down.ps1` | `-Volumes` |
| `logs.ps1` | `-Services a,b` · `-Tail N` · `-NoFollow` |
| `create_topics.ps1` | `-Partitions N` (new topics only) · `-TimeoutSec N` |
| `test.ps1` | `-Local` · any pytest arguments (`-k`, `-x`, paths, ...) |

| Open | URL |
| --- | --- |
| Dashboard | <http://localhost:5173> |
| API | <http://localhost:8001> |
| Kafka UI | <http://localhost:8080> |
| MLflow | <http://localhost:5000> |
| Adminer | <http://localhost:8081>. Log in with System `PostgreSQL`, Server `postgres`, and the user, password and database from `.env` (`traffic` / `traffic` / `traffic` by default). |

## Local development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e "shared[dev]" -r services\video_ingest\requirements.txt -r services\event_simulator\requirements.txt
```

### Database migrations

Run these with the stack up. They use `POSTGRES_URL`, which defaults to `localhost:5432`:

```powershell
python -m traffic_common.db upgrade                            # apply all migrations
python -m traffic_common.db seed                               # 4 demo cameras (safe to re-run)
python -m traffic_common.db current                            # show the applied revision
python -m traffic_common.db downgrade -1                       # roll back one revision
python -m traffic_common.db revision -m "add x" --autogenerate # new migration from model changes
```

The same commands work inside any service container, for example `docker compose exec api python -m traffic_common.db upgrade`. See [shared/README.md](shared/README.md) for the models and the Kafka helpers.

### Tests and lint

```powershell
.\scripts\test.ps1                          # whole suite in a Linux container on the compose network
.\scripts\test.ps1 tests/test_schemas.py -x # any pytest arguments pass through
.\scripts\test.ps1 -Local                   # in .venv against localhost:9092 / localhost:5432
ruff check .
```

| File | Needs | What it covers |
| --- | --- | --- |
| `test_schemas.py` | nothing | Schema round-trips, validation rules, and the topic lists in `create_topics.ps1` |
| `test_messaging.py` | Kafka | Producer validation and keys, consumer commits, retries, dead-lettering, shutdown |
| `test_migrations.py` | Postgres | Upgrade, downgrade and upgrade again; models match the migration; constraints; idempotent seed |
| `test_skeleton.py` | nothing | Settings, logging, lifecycle, service layout |
| `test_video_ingest.py` | nothing | Frame sampling, resizing, JPEG encoding, stream_url parsing, file naming, retention, health metrics |
| `test_video_ingest_integration.py` | Kafka | A worker on a short synthetic video publishes to `video.frames`, plus retry when a source is missing |
| `test_event_simulator.py` | nothing | Report variety and label correctness, train/test holdout separation, dataset files, daily traffic pattern, weather, scenario YAML parsing and timelines, scenario runs, control commands |
| `test_cv_inference.py` | nothing | Zones (point-in-polygon, scaling, defaults), speed and smoothing, stopped detection, metric windows, latest-frame scheduling, message building |
| `test_cv_inference_integration.py` | Ultralytics (+ Kafka) | YOLO26n + ByteTrack on 12 frames of a sample video produce valid, tracked `Detection`s and metrics; the messages round-trip through `detections` |
| `test_ml_core.py` | pandas | Feature engineering, relative baselines, weak labels, window building, scenario labels, synthetic data, threshold calibration, alarm start, end, cooldown and persistence, evaluation metrics, hints, rule-based models |
| `test_ml_models.py` | torch (Docker) | The autoencoder and both congestion models round-trip through MLflow pyfunc packaging and score raw rows |
| `test_anomaly_service.py` | Postgres | Model reload when the champion alias moves, rule fallback, keeping the current model when the registry is down, the sensor join, start and end events, cooldown, stale metrics, sensor storage |
| `test_anomaly_integration.py` | Kafka + Postgres | A synthetic incident published to `traffic.metrics` produces an `Anomaly` on `anomalies` and a row in `anomalies` |
| `test_event_simulator_integration.py` | Kafka + Postgres | A scenario triggered through `simulator.control` emits tagged messages on `reports.text` and `events.raw`, and its ground truth lands in `scenario_runs` and `simulated_reports` |

Tests whose infrastructure isn't reachable are skipped, with the reason shown by `-rs`. The infrastructure tests work in throwaway topics, consumer groups and databases, which they delete afterwards. They never touch the `traffic` database. Four integration tests do publish to real topics:

- **video_ingest:** a few dozen frames on `video.frames`, under a `test-cam-<id>` camera id. Their frame files no longer exist.
- **cv_inference:** 4 `Detection`s on `detections` under a `test-cam-<id>` camera id.
- **anomaly_service:** about 36 `TrafficMetrics` on `traffic.metrics` and one `Anomaly`, for a `test-cam-<id>` camera.
- **event_simulator:** a scenario's reports and events on `reports.text` and `events.raw`, tagged with a `test-cam-<id>` camera and that run's `scenario_id`. The live simulator also receives the test's control command and logs `rejected start_scenario … unknown camera`, as intended.

Those messages stay until Kafka retention removes them.

On Windows, Smart App Control can block confluent-kafka's native library, so `import confluent_kafka` fails and the Kafka tests skip under `-Local`. The cv_inference integration test also skips under `-Local`, because PyTorch and Ultralytics are only installed in the Docker images. The default `test.ps1` mode runs in Linux and runs everything.

Run one service outside Docker: `python services\api\main.py`. Outside Docker, `traffic_common.get_settings()` defaults to the host-side addresses (`localhost:9092`, Postgres on `localhost:5432`, MLflow on `http://localhost:5000`), so a local process talks to the infrastructure running in Docker.

## Configuration

All settings come from environment variables; see [.env.example](.env.example).
`.env` is git-ignored.

`DISPLAY_TIMEZONE` (an IANA zone, default `Europe/London` because the demo cameras are in London) sets the zone for clock times everywhere a person reads them:
- **Summaries:** incident summaries name the zone, as in "since 18:48 BST".
- **Video:** the timestamp cv_inference writes on annotated frames.
- **Dashboard:** compose passes it to the dashboard as `VITE_DISPLAY_TIMEZONE`.

Stored timestamps and the API stay in UTC. After changing it, recreate the containers with `.\scripts\up.ps1`.

### Connection settings

`.env` holds host-side URLs for local development. Inside Docker, `docker-compose.yml` overrides them with in-network addresses:

| Variable | In `.env` (host) | In containers |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:${KAFKA_PORT}` | `kafka:29092` |
| `POSTGRES_URL` | `postgresql://traffic:traffic@localhost:${POSTGRES_PORT}/traffic` | `...@postgres:5432/traffic` |
| `MLFLOW_TRACKING_URI` | `http://localhost:${MLFLOW_PORT}` | `http://mlflow:5000` |

The Postgres credentials in `.env.example` are for development only.

### Ports

Each variable sets only the host side of a port mapping. Container ports are fixed.

| Variable | Default | Service | Notes |
| --- | --- | --- | --- |
| `API_PORT` | `8001` | api | The api listens on 8000 inside Docker. The host default is 8001 because 8000 is often taken by WSL processes. |
| `DASHBOARD_PORT` | `5173` | dashboard | |
| `KAFKA_PORT` | `9092` | kafka | Also the advertised external listener, so host clients must use this port |
| `KAFKA_UI_PORT` | `8080` | kafka-ui | |
| `POSTGRES_PORT` | `5432` | postgres | Change this if a local PostgreSQL install already uses 5432 |
| `MLFLOW_PORT` | `5000` | mlflow | |
| `ADMINER_PORT` | `8081` | adminer | |
| `VITE_API_BASE_URL` | `http://localhost:${API_PORT}` | dashboard | Only set this if the dashboard reaches the API through a different host or proxy |

The Kafka controller listener (29093) is never published. On this machine Windows reserves ports 9093 and 9095 (see `netsh interface ipv4 show excludedportrange protocol=tcp`), so avoid them as host ports.

After changing a port, re-run `.\scripts\up.ps1` so the containers are recreated.

To check whether a port is free, look on the Windows side and inside WSL:

```powershell
Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
wsl -e ss -ltn "sport = :5432"
```

## Video credits

The demo camera videos are not stored in git (`data/` is ignored); `scripts/download_sample_videos.py` fetches them from Wikimedia Commons. This section records their attribution, copied from the `data/videos/ATTRIBUTION.md` the script generates. If you redistribute these videos or material derived from them (for example extracted frames or screenshots), credit the author and keep the licence. CC BY-SA additionally requires sharing derivatives under the same licence.

| Local file (camera) | Commons file | Author | Licence |
| --- | --- | --- | --- |
| `cam-hyde-park-corner.webm` | [London roundabout.ogg](https://commons.wikimedia.org/wiki/File:London_roundabout.ogg) | Vcohen | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `cam-trafalgar-square.webm` | [Intersection Pie-IX-Sherbrooke.webm](https://commons.wikimedia.org/wiki/File:Intersection_Pie-IX-Sherbrooke.webm) | Thomas1313 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `cam-elephant-castle.webm` | [Road traffic on Stritarjeva street.webm](https://commons.wikimedia.org/wiki/File:Road_traffic_on_Stritarjeva_street.webm) | Sounds of Changes | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| `cam-old-street.webm` | [A intersection with traffic in Chiang Mai, Thailand.ogv](https://commons.wikimedia.org/wiki/File:A_intersection_with_traffic_in_Chiang_Mai,_Thailand.ogv) | Amada44 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

The local files are Wikimedia's 480p VP9 transcodes of these originals. The camera names are fictional placements: the footage was not filmed at the London junctions the demo cameras are named after.
