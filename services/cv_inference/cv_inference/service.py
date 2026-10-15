"""Wires model loading, the frame consumer, inference workers and publishing together."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, insert, select
from sqlalchemy.exc import IntegrityError

from cv_inference.analytics import MetricsAggregator, TrackAnalyzer
from cv_inference.annotate import annotate, write_latest
from cv_inference.config import CvConfig
from cv_inference.detector import Detector, benchmark, export_openvino, make_tracker, track
from cv_inference.pipeline import (
    CameraContext,
    LatestFrames,
    frame_age_seconds,
    process_frame,
    read_frame,
)
from cv_inference.registry import ResolvedModel, record_model_version, resolve_model
from cv_inference.zones import ZoneConfig
from traffic_common.config import Settings
from traffic_common.db.models import Camera, TrafficMetric
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.schemas import FrameMessage, TrafficMetrics
from traffic_common.topics import ConsumerGroup, Topic

log = logging.getLogger(__name__)
_JOIN_TIMEOUT = 15.0


class CvService:
    def __init__(self, config: CvConfig, settings: Settings, engine: Engine, stop: threading.Event):
        self.cfg = config
        self.settings = settings
        self.engine = engine
        self.stop = stop
        self.slots = LatestFrames()
        self.contexts: dict[str, CameraContext] = {}
        self.zone_configs: dict[str, ZoneConfig] = {}
        self._contexts_lock = threading.Lock()
        self._metrics_db_warned: set[str] = set()
        self.model: ResolvedModel | None = None
        self.detectors: list[Detector] = []
        self.producer: TypedProducer | None = None

    # ------------------------------------------------------------------ startup

    def load_model(self) -> None:
        cfg = self.cfg
        self.model = resolve_model(
            self.settings.mlflow_tracking_uri,
            cfg.registry_model_name,
            cfg.registry_alias,
            cfg.model,
            cfg.imgsz,
            cfg.classes,
            cfg.weights_dir,
        )
        m = self.model
        log.info(
            "model in use: %s version %s (%s, weights %s, mlflow run %s)",
            m.name,
            m.version,
            m.source,
            m.weights.name,
            m.run_id or "-",
        )

        sample = self._sample_frame()
        common = {
            "imgsz": cfg.imgsz,
            "threads": cfg.threads,
            "classes": cfg.classes,
            "min_confidence": cfg.tracker_confidence,
        }
        pytorch = Detector(m.weights, "pytorch", **common)
        latency = {"pytorch": benchmark(pytorch, sample)}
        openvino, ov_dir = None, None
        # Ultralytics recognises OpenVINO models by the "_openvino_model" directory suffix.
        ov_name = f"{m.name}-v{m.version}-{m.weights.stem}-{cfg.imgsz}_openvino_model"
        try:
            ov_dir = export_openvino(m.weights, cfg.imgsz, cfg.weights_dir / "openvino" / ov_name)
            openvino = Detector(ov_dir, "openvino", **common)
            latency["openvino"] = benchmark(openvino, sample)
        except Exception as exc:
            log.warning("OpenVINO export/load failed, falling back to PyTorch: %s", exc)

        if openvino is None or cfg.backend == "pytorch":
            backend = "pytorch"
        elif cfg.backend == "openvino":
            backend = "openvino"
        else:  # auto
            backend = min(latency, key=lambda k: latency[k][0])
        comparison = " | ".join(
            f"{k} mean {v[0]:.1f} ms p95 {v[1]:.1f} ms" for k, v in latency.items()
        )
        log.info(
            "latency at imgsz=%d with %d threads (%dx%d frame): %s -> using %s (CV_BACKEND=%s)",
            cfg.imgsz,
            cfg.threads,
            sample.shape[1],
            sample.shape[0],
            comparison,
            backend,
            cfg.backend,
        )

        first = openvino if backend == "openvino" else pytorch
        self.detectors = [first]
        model_path = ov_dir if backend == "openvino" and ov_dir is not None else m.weights
        for _ in range(cfg.workers - 1):
            self.detectors.append(Detector(model_path, backend, **common))
        self.backend = backend
        try:
            record_model_version(
                self.engine,
                m,
                {
                    "backend": backend,
                    "imgsz": cfg.imgsz,
                    "threads": cfg.threads,
                    "weights": m.weights.name,
                    "classes": list(cfg.classes),
                    "latency_ms": {
                        k: {"mean": round(v[0], 2), "p95": round(v[1], 2)}
                        for k, v in latency.items()
                    },
                    "loaded_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as exc:
            log.warning("could not record model version: %s", exc)

    def _sample_frame(self) -> np.ndarray:
        """A recent real frame for benchmarking if one exists, else noise at a typical size."""
        frames_dir = Path("/data/frames")
        if frames_dir.is_dir():
            newest = max(frames_dir.glob("*/*.jpg"), key=lambda p: p.stat().st_mtime, default=None)
            if newest is not None and (image := read_frame(str(newest))) is not None:
                return image
        return np.random.default_rng(0).integers(0, 255, (480, 854, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        """Block until ``stop`` is set."""
        assert self.detectors, "call load_model() first"
        self.producer = TypedProducer(
            self.settings.kafka_bootstrap_servers, client_id="cv_inference"
        )
        self.refresh_zones()
        consumer = TypedConsumer(
            [Topic.VIDEO_FRAMES],
            ConsumerGroup.CV_INFERENCE,
            self.handle_frame,
            self.settings.kafka_bootstrap_servers,
            stop_event=self.stop,
            auto_offset_reset="latest",  # real time: a new group starts at the live edge
        )
        threads = [threading.Thread(target=consumer.run, name="frames-consumer", daemon=True)]
        threads += [
            threading.Thread(target=self._worker, args=(d,), name=f"inference-{i}", daemon=True)
            for i, d in enumerate(self.detectors)
        ]
        threads += [
            threading.Thread(
                target=self._every,
                args=(self.cfg.camera_refresh_seconds, self.refresh_zones),
                name="zones",
                daemon=True,
            ),
            threading.Thread(
                target=self._every,
                args=(1.0, self._close_idle_windows),
                name="metrics-poll",
                daemon=True,
            ),
            threading.Thread(
                target=self._every,
                args=(self.cfg.stats_interval_seconds, self._log_stats),
                name="stats",
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        log.info(
            "consuming %s with %d worker(s), backend %s",
            Topic.VIDEO_FRAMES,
            len(self.detectors),
            self.backend,
        )
        self.stop.wait()
        for t in threads:
            t.join(timeout=_JOIN_TIMEOUT)

    def _every(self, interval: float, fn) -> None:
        while not self.stop.wait(interval):
            try:
                fn()
            except Exception:
                log.exception("%s failed", fn.__name__)

    # ------------------------------------------------------------------ frames

    def context(self, camera_id: str) -> CameraContext:
        with self._contexts_lock:
            ctx = self.contexts.get(camera_id)
            if ctx is None:
                tracker = make_tracker(self.cfg.track_lost_seconds, self.cfg.expected_fps)
                ctx = CameraContext(
                    camera_id=camera_id,
                    tracker=lambda boxes, img, _t=tracker: track(
                        _t, boxes, img, self.cfg.confidence
                    ),
                    analyzer=TrackAnalyzer(
                        speed_window_s=self.cfg.speed_window_seconds,
                        stopped_speed_kmh=self.cfg.stopped_speed_kmh,
                        stopped_after_s=self.cfg.stopped_after_seconds,
                    ),
                    aggregator=MetricsAggregator(
                        camera_id,
                        self.cfg.metrics_window_seconds,
                        self.model.label if self.model else None,
                    ),
                    zone_config=self.zone_configs.get(camera_id, ZoneConfig()),
                )
                self.contexts[camera_id] = ctx
            return ctx

    def handle_frame(self, frame: FrameMessage) -> None:
        """Consumer handler: drop stale frames, otherwise park the frame for a worker."""
        ctx = self.context(frame.camera_id)
        if frame_age_seconds(frame) > self.cfg.max_frame_age_seconds:
            ctx.stats.stale += 1
            return
        if self.slots.put(frame) is not None:
            ctx.stats.superseded += 1

    def _worker(self, detector: Detector) -> None:
        while not self.stop.is_set():
            frame = self.slots.take(timeout=0.5)
            if frame is None:
                continue
            ctx = self.context(frame.camera_id)
            try:
                self._process(ctx, frame, detector)
            except Exception:
                ctx.stats.errors += 1
                log.exception("camera %s: frame %s failed", frame.camera_id, frame.frame_id)
            finally:
                self.slots.done(frame.camera_id)

    def _process(self, ctx: CameraContext, frame: FrameMessage, detector: Detector) -> None:
        if frame_age_seconds(frame) > self.cfg.max_frame_age_seconds:  # went stale while waiting
            ctx.stats.stale += 1
            return
        image = read_frame(frame.frame_path)
        if image is None:
            log.debug("camera %s: %s already deleted, skipping", frame.camera_id, frame.frame_path)
            ctx.stats.missing += 1
            return
        assert self.model is not None and self.producer is not None
        with ctx.lock:
            result = process_frame(
                ctx,
                frame,
                image,
                detector.detect,
                model_name=self.model.name,
                model_version=self.model.version,
                default_frame_width_metres=self.cfg.default_frame_width_metres,
            )
        self.producer.send(Topic.DETECTIONS, result.detection)
        ctx.stats.lag_ms.append(frame_age_seconds(frame) * 1000)
        ctx.stats.processed += 1
        if result.metrics is not None:
            self.publish_metrics(result.metrics)
        if self.cfg.annotate:
            header = f"{frame.camera_id}  {self.model.label} {self.backend}"
            write_latest(
                self.cfg.annotated_dir,
                frame.camera_id,
                annotate(image, result.analyses, result.zone, header, frame.timestamp),
            )

    # ------------------------------------------------------------------ metrics

    def _close_idle_windows(self) -> None:
        now = time.time()
        with self._contexts_lock:
            contexts = list(self.contexts.values())
        for ctx in contexts:
            with ctx.lock:
                metrics = ctx.aggregator.poll(now)
            if metrics is not None:
                self.publish_metrics(metrics)

    def publish_metrics(self, m: TrafficMetrics) -> None:
        assert self.producer is not None
        self.producer.send(Topic.TRAFFIC_METRICS, m)
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(TrafficMetric).values(
                        message_id=m.message_id,
                        camera_id=m.camera_id,
                        window_start=m.window_start,
                        window_end=m.window_end,
                        vehicle_count=m.vehicle_count,
                        avg_speed_kmh=m.avg_speed_kmh,
                        p85_speed_kmh=m.p85_speed_kmh,
                        occupancy=m.occupancy,
                        density=m.density,
                        stopped_vehicles=m.stopped_vehicles,
                        pedestrians_in_roadway=m.pedestrians_in_roadway,
                        mean_confidence=m.mean_confidence,
                        frames=m.frames,
                        model_version=m.model_version,
                        counts_by_class=m.counts_by_class,
                    )
                )
        except IntegrityError as exc:
            if m.camera_id not in self._metrics_db_warned:  # e.g. a camera not in the cameras table
                self._metrics_db_warned.add(m.camera_id)
                log.warning(
                    "camera %s: metrics not stored in traffic_metrics: %s",
                    m.camera_id,
                    str(exc.orig).splitlines()[0],
                )
        except Exception as exc:
            log.error("could not store traffic metrics: %s", exc)

    # ------------------------------------------------------------------ zones and stats

    def refresh_zones(self) -> None:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(select(Camera.id, Camera.zone_config)).all()
        except Exception as exc:
            log.error("could not load zone configs (keeping %d): %s", len(self.zone_configs), exc)
            return
        for camera_id, raw in rows:
            config, problem = ZoneConfig.parse(raw)
            if config != self.zone_configs.get(camera_id):
                if problem:
                    log.warning("camera %s: %s", camera_id, problem)
                log.info(
                    "camera %s zone: %s, %s",
                    camera_id,
                    f"roadway with {len(config.roadway)} points"
                    if config.roadway
                    else "whole frame (no roadway)",
                    f"{config.pixels_per_metre} px/m"
                    if config.pixels_per_metre
                    else f"default scale (frame = {self.cfg.default_frame_width_metres:g} m wide)",
                )
                self.zone_configs[camera_id] = config
                if camera_id in self.contexts:
                    with self.contexts[camera_id].lock:
                        self.contexts[camera_id].set_zone_config(config)

    def _log_stats(self) -> None:
        with self._contexts_lock:
            contexts = sorted(self.contexts.items())
        for camera_id, ctx in contexts:
            log.info(
                "stats camera=%s %s",
                camera_id,
                ctx.stats.snapshot_and_reset(self.cfg.stats_interval_seconds),
            )
