"""Incident summaries: a fact-only context, a versioned prompt for a local seq2seq model
(flan-t5), a deterministic template, and a faithfulness guard that falls back to the template
whenever the model's output isn't supported by the facts. Used by services/summarizer and the
evaluation CLI (ml.evaluation.summarizer_eval)."""
