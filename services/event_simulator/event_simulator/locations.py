"""Camera gazetteers live in traffic_common.gazetteer (shared with nlp_service)."""

from traffic_common.gazetteer import (
    GAZETTEERS,
    CameraInfo,
    Gazetteer,
    derived_gazetteer,
    gazetteer_for,
)

__all__ = ["GAZETTEERS", "CameraInfo", "Gazetteer", "derived_gazetteer", "gazetteer_for"]
