// fusion_service confidence_components, as far as the dashboard reads them.
export type Components = {
  vision?: { score?: number; raw_score?: number; degraded?: boolean; present?: boolean; hints?: string[]; type?: string };
  text?: { score?: number; present?: boolean; type?: string; n_sources?: number; classifier?: number };
  agreement?: { score?: number; both_modalities?: boolean; text_type?: string; vision_type?: string };
  fusion?: {
    fused?: number;
    primary_modality?: "vision" | "text";
    weights?: { primary?: number; secondary?: number; agreement?: number };
    contributions?: { primary?: number; secondary?: number; agreement?: number };
  };
  severity?: { text?: string; vision?: string; congestion?: string };
};

/** Split the fused confidence into what vision, text and their agreement each contributed. */
export function confidenceParts(c: Components) {
  const f = c.fusion ?? {};
  const visionPrimary = (f.primary_modality ?? "vision") === "vision";
  const contrib = f.contributions ?? {};
  const weights = f.weights ?? {};
  return {
    vision: { contribution: (visionPrimary ? contrib.primary : contrib.secondary) ?? 0, weight: (visionPrimary ? weights.primary : weights.secondary) ?? null, score: c.vision?.score ?? null },
    text: { contribution: (visionPrimary ? contrib.secondary : contrib.primary) ?? 0, weight: (visionPrimary ? weights.secondary : weights.primary) ?? null, score: c.text?.score ?? null },
    agreement: { contribution: contrib.agreement ?? 0, weight: weights.agreement ?? null, score: c.agreement?.score ?? null },
    primary: visionPrimary ? "vision" : "text",
  };
}
