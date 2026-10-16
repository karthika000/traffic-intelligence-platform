"""The summarizer "model" is a configuration: the MLflow champion of "incident-summarizer"
(model name, prompt version, generation settings), or the default from the environment until
one is registered. Both expose ``.summarizer`` (ml.summarize.summarizer.IncidentSummarizer)."""

from ml.summarize.pyfunc import IncidentSummarizerModel
from ml.summarize.summarizer import SummarizerConfig
from summarizer_service.config import SummarizerServiceConfig


def mlflow_loader(uri: str) -> IncidentSummarizerModel:
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(uri).unwrap_python_model()


def default_factory(cfg: SummarizerServiceConfig):
    """Fallback until a configuration is registered: the configured default model and prompt."""

    def make() -> IncidentSummarizerModel:
        model = IncidentSummarizerModel().load_config(
            SummarizerConfig(
                model_name=cfg.default_model,
                prompt=cfg.default_prompt,
                timeout_seconds=cfg.timeout_seconds,
            ).to_dict()
        )
        model.name = "incident-summarizer-default"
        model.version = f"{cfg.default_model.split('/')[-1]}+{cfg.default_prompt}"
        return model

    return make


def warm_up(model) -> None:
    """Load the weights now, not on the first incident (which has a time budget)."""
    generator = getattr(model.predictor.summarizer, "generator", None)
    if generator is not None:
        generator.load()
