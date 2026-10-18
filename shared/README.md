# traffic-common

Shared Python package installed into every service image (and your `.venv` for local development).

| Module | Purpose |
| --- | --- |
| `traffic_common.config` | `Settings` from environment variables (log level, Kafka, Postgres, MLflow URLs) |
| `traffic_common.logging` | Consistent log format across services |
| `traffic_common.lifecycle` | Process-wide shutdown event, SIGINT/SIGTERM handling, `on_shutdown()` callbacks, `run_forever()` |
| `traffic_common.topics` | `Topic` and `ConsumerGroup` names, `dead_letter_topic()` |
| `traffic_common.enums` | Enums shared by schemas and database models (`Severity`, `IncidentStatus`, ...) |
| `traffic_common.schemas` | Pydantic v2 message schemas and `TOPIC_SCHEMAS` (topic -> schema) |
| `traffic_common.messaging` | `TypedProducer` / `TypedConsumer` on confluent-kafka |
| `traffic_common.db` | SQLAlchemy 2.0 models, engine helpers, Alembic migrations, demo seed data |
| `traffic_common.driftmode` | The demo drift mode: settings and presets, reading and switching it in Postgres, `DriftModeWatcher` for services, and a CLI (`python -m traffic_common.driftmode on\|off\|status`) |

`import traffic_common` only loads config, logging and lifecycle. Import `schemas`, `messaging` and `db` explicitly.

## Message schemas

| Topic | Schema | Key (`camera_id`) |
| --- | --- | --- |
| `video.frames` | `FrameMessage` | always |
| `events.raw` | `RawEvent` | optional |
| `reports.text` | `TextReport` | optional |
| `detections` | `Detection` | always |
| `traffic.metrics` | `TrafficMetrics` | always |
| `anomalies` | `Anomaly` | always |
| `nlp.results` | `NLPResult` | optional |
| `incidents` | `Incident` | always |
| `summaries` | `Summary` | optional |
| `drift.alerts` | `DriftAlert` | optional |
| `simulator.control` | `SimulatorCommand` | optional |

Every message has `schema_version` (currently `Literal[1]`), a `message_id` (UUID) and `timestamp`. The rules:

- **Timestamps:** `timestamp` is the event time. All datetimes must include a timezone and are converted to UTC; naive datetimes are rejected.
- **Unknown fields** are rejected, so producer typos fail fast.
- **Immutable:** messages can't be changed after creation.
- **Breaking changes:** bump `schema_version` and its `Literal` together. Consumers on the old version will then dead-letter the new messages instead of misreading them.

## Kafka

```python
from traffic_common import get_settings, run_forever
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.schemas import TrafficMetrics
from traffic_common.topics import ConsumerGroup, Topic

settings = get_settings("anomaly_service")
producer = TypedProducer(settings.kafka_bootstrap_servers, client_id="anomaly_service")
producer.send(Topic.TRAFFIC_METRICS, metrics)  # validated, keyed by metrics.camera_id


def handle(msg: TrafficMetrics) -> None:
    # raise to retry; offsets are committed only after this returns
    ...


TypedConsumer(
    [Topic.TRAFFIC_METRICS], ConsumerGroup.ANOMALY_SERVICE, handle, settings.kafka_bootstrap_servers
).run()  # blocks until SIGTERM
```

How the producer behaves:

- **Before sending:** it validates every message against the topic's schema. A model or a plain dict both work; a wrong model type raises `TypeError`.
- **Keys:** messages are keyed by `camera_id` when they have one, so each camera's messages stay in order on one partition.
- **Settings:** idempotent, with `acks=all`.
- **Shutdown:** by default the producer registers a flush to run at shutdown.

How the consumer behaves:

- **Delivery:** at-least-once. Offsets are committed manually, one message at a time, after the handler returns.
- **Bad messages:** messages that don't validate (including bad JSON) go straight to `dlq.<group>`.
- **Handler failures:** retried with exponential backoff (`max_retries`, default 3), then sent to `dlq.<group>`.
- **Dead-letter records:** they keep the original key, value and headers, plus `dlq.reason`, `dlq.error`, `dlq.original_topic`, `dlq.original_partition`, `dlq.original_offset`, `dlq.consumer_group`, `dlq.attempts` and `dlq.failed_at`.
- **If the dead-letter write fails:** the consumer stops without committing, so the message is redelivered.
- **SIGTERM during a retry wait:** the offset stays uncommitted and the message is redelivered on restart.
- **When `run()` returns:** it runs the registered shutdown callbacks, such as producer flushes.

Handlers must be idempotent, since redelivery is possible. Database rows keep the message UUID (`id` or `message_id`) so writes can use `ON CONFLICT DO NOTHING`.

## Database

```python
from traffic_common.db import create_db_engine, models, session_factory

Session = session_factory(create_db_engine())  # POSTGRES_URL
with Session() as s:
    cameras = s.query(models.Camera).all()
```

Tables: `cameras`, `traffic_metrics`, `detection_aggregates`, `reports`, `nlp_results`, `anomalies`, `incidents`, `incident_evidence`, `summaries`, `model_versions` (with `promotion`, the promotion gate's decision), `drift_reports` (grouped by `check_id`), `drift_mode` and `drift_mode_events`, `retraining_recommendations`, plus the simulator's ground truth: `scenario_runs` and `simulated_reports` (migration `0002`; `text_style` marks reports written in drift mode). `TextReport` and `RawEvent` carry an optional `scenario_id`, set on messages from simulated scenario runs. Every table with a `camera_id` has a composite index on `(camera_id, <time>)` and a separate index on the time column. Enums are stored as strings with CHECK constraints.

Migrations ship inside the package, so they run from any directory and inside any service image:

```powershell
python -m traffic_common.db upgrade               # to head
python -m traffic_common.db downgrade -1          # or: base
python -m traffic_common.db current
python -m traffic_common.db revision -m "add x" --autogenerate
python -m traffic_common.db seed                  # 4 demo cameras in central London (idempotent)
```

All commands take `--url`; otherwise they use `POSTGRES_URL`, falling back to `localhost:5432`. `alembic -c shared/alembic.ini ...` also works. After an autogenerated revision, review the file before applying it.
