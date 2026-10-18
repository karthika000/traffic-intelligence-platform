"""Drift mode's visual degradation: make frames look like night or fog
(see traffic_common.driftmode). Applied after resizing, before JPEG encoding."""

import cv2
import numpy as np

HAZE_LEVEL = 200.0  # the light grey fog blends towards


def degrade(
    frame: np.ndarray,
    params: dict[str, float],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``params``: brightness (multiplier), noise_std (grey levels), haze (0-1 blend towards
    light grey, which also flattens contrast), blur (Gaussian kernel size in pixels)."""
    rng = rng or np.random.default_rng()
    out = frame.astype(np.float32)
    haze = float(params.get("haze", 0.0))
    if haze > 0:
        out = out * (1.0 - haze) + HAZE_LEVEL * haze
    blur = int(round(params.get("blur", 0.0)))
    if blur >= 2:
        k = blur | 1  # odd kernel size
        out = cv2.GaussianBlur(out, (k, k), 0)
    out *= float(params.get("brightness", 1.0))
    noise = float(params.get("noise_std", 0.0))
    if noise > 0:
        # Per-pixel luminance noise (sensor noise at night), the same on every channel.
        grain = rng.normal(0.0, noise, size=out.shape[:2]).astype(np.float32)
        out += grain[..., None] if out.ndim == 3 else grain
    return np.clip(out, 0, 255).astype(np.uint8)
