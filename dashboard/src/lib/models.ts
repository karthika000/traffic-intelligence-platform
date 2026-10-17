// Headline metrics, in order of preference; anything else is under "all metrics".
const KEY_METRICS = [
  /^test_macro_f1$/,
  /^test_accuracy$/,
  /macro_f1$/,
  /^rouge_l_published$/,
  /^faithfulness_pass_rate$/,
  /^fallback_rate$/,
  /^synthetic_recall$/,
  /^synthetic_precision$/,
  /false_alarms_per_hour$/,
  /^latency_p95_ms$/,
  /accuracy$/,
];

export function keyMetrics(metrics: Record<string, number> = {}, max = 6): [string, number][] {
  const out: [string, number][] = [];
  for (const re of KEY_METRICS) {
    for (const [k, v] of Object.entries(metrics)) {
      if (out.length >= max) return out;
      if (re.test(k) && !out.some(([o]) => o === k)) out.push([k, v]);
    }
  }
  return out;
}
