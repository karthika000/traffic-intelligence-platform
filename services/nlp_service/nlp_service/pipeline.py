"""Per batch of reports: classify (one call to the champion), run NER (one batched call), then
per report: rule entities, camera matching on the text, and assemble the NLPResult."""

import json
import logging
import time
from collections.abc import Callable

import pandas as pd

from ml.nlp.entities import location_spans, resolve_overlaps, rule_entities
from ml.nlp.matching import CameraMatcher
from ml.nlp.ner import NerExtractor
from ml.nlp.stage import report_stage
from ml.registry import LoadedModel
from traffic_common.schemas import Entity, NLPResult, TextReport
from traffic_common.telemetry import INFERENCE_SECONDS, timed

log = logging.getLogger(__name__)
MATCHER_VERSION = "gazetteer-fuzzy-1"


class NlpPipeline:
    def __init__(
        self,
        classifier: Callable[[], LoadedModel],
        matcher: Callable[[], CameraMatcher],
        ner: NerExtractor | None = None,
    ) -> None:
        self.classifier, self.matcher, self.ner = classifier, matcher, ner

    def model_versions(self, model: LoadedModel) -> dict[str, str]:
        versions = {model.name: model.version, "camera-matcher": MATCHER_VERSION}
        if self.ner is not None:
            versions[self.ner.model_name] = self.ner.version
        return versions

    def analyse(self, reports: list[TextReport]) -> list[NLPResult]:
        if not reports:
            return []
        started = time.perf_counter()
        model = self.classifier()
        texts = [r.text for r in reports]
        with timed(INFERENCE_SECONDS, model=model.name):  # one call per batch
            predictions = model.predictor.predict(pd.DataFrame({"text": texts}))
        if self.ner is not None:
            with timed(INFERENCE_SECONDS, model=self.ner.model_name):
                ner = self.ner.extract(texts)
        else:
            ner = [[] for _ in texts]
        matcher = self.matcher()
        versions = self.model_versions(model)
        per_report_ms = (time.perf_counter() - started) * 1000 / len(reports)
        results = []
        for report, (_, p), ner_entities in zip(reports, predictions.iterrows(), ner, strict=True):
            match = matcher.match(report.text)
            locations = location_spans(
                report.text, ner_entities, matcher.gazetteer_entities(report.text, match)
            )
            entities = resolve_overlaps(locations + rule_entities(report.text))
            location_name = match.matched_text if match.camera_id else None
            if location_name is None and locations:
                location_name = locations[0]["text"]
            results.append(
                NLPResult(
                    report_id=report.report_id,
                    source=report.source,
                    incident_type=p["incident_type"],
                    incident_confidence=round(float(p["incident_confidence"]), 4),
                    severity=p["severity"],
                    severity_confidence=round(float(p["severity_confidence"]), 4),
                    probabilities=json.loads(p["probabilities"]),
                    stage=report_stage(report.text),
                    entities=[Entity(**e) for e in entities],
                    camera_id=match.camera_id,
                    camera_confidence=match.confidence if match.camera_id else None,
                    location_name=location_name[:200] if location_name else None,
                    model_name=model.name,
                    model_version=model.version,
                    model_versions=versions,
                    details={
                        "model_source": model.source,
                        "camera_match": match.to_details(),
                        # The report's own camera tag (the simulator sets it): kept for
                        # evaluation only, never used for matching.
                        "reported_camera_id": report.camera_id,
                        "scenario_id": str(report.scenario_id) if report.scenario_id else None,
                        "batch_size": len(reports),
                        "ms_per_report": round(per_report_ms, 2),
                    },
                )
            )
        return results
