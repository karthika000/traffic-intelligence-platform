"""Text models for traffic reports: incident type / severity classification, entity extraction
and camera matching. Shared by the ml-trainer (training, evaluation) and nlp_service.

Modules that need PyTorch or transformers (``transformer``, ``ner``) import them lazily, so
preprocessing, rules and matching stay usable (and testable) without them."""
