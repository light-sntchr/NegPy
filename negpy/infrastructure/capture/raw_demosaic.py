"""Linear sensor-RGB demosaic for calibration metering.

Follows NegPy's canonical RAW decode (`ImageProcessor._decode_sensor_rgb`): sensor-native
`output_color=raw`, no white balance, linear gamma, 16-bit — so calibration meters the film base
the same way the RGB-Scan merge later reads the channels. rawpy is imported lazily so the module
stays import-safe.

It deviates in one parameter, deliberately: `adjust_maximum_thr=0.0` (see `linear_demosaic`). The
canonical decode still runs LibRaw's default, where each frame is scaled by its own brightest
pixel — harmless for a single rendered image, fatal for a meter comparing frames. Whether the
canonical path wants the same fix is a separate question (it changes rendered output, so it needs
its own verification); it is NOT covered here.
"""

from __future__ import annotations

import numpy as np


def linear_demosaic(path: str, half_size: bool = False) -> np.ndarray:
    """Decode one RAW to a sensor-native, linear, 16-bit HxWx3 array (R=0, G=1, B=2).

    `half_size=True` bins each 2×2 Bayer quad straight into one RGB pixel (no interpolation)
    for a ~4× faster decode — used by calibration, which only meters a uniform base patch, so
    full resolution is wasted (and the raw-Bayer clip check reads full-res separately). Bayer
    only: X-Trans automatically falls back to a full-size decode because 2×2 binning aliases
    its 6×6 CFA.
    """
    import rawpy

    from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm, is_xtrans

    with rawpy.imread(path) as raw:
        algo = get_best_demosaic_algorithm(raw)
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            # Scale against the camera's white level ONLY, never the frame's own brightest pixel.
            # LibRaw's default (adjust_maximum_thr=0.75) switches the scaling reference to the image
            # maximum once that exceeds 75 % of the white level, so each frame is normalised by its
            # own content. That makes the decode non-linear in exposure: rig data showed the metered
            # base pinned across a range of LED levels, because the scaling grew exactly as fast as
            # the light, which reads as an LED or shutter defect and is neither. 0.0 disables the
            # substitution and makes the demosaiced scale a fixed multiple of the raw counts, which is
            # what a meter measuring absolute light requires and what CLIP_CEILING assumes.
            adjust_maximum_thr=0.0,
            use_camera_wb=False,
            user_wb=[1, 1, 1, 1],
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            demosaic_algorithm=algo,
            half_size=half_size and not is_xtrans(raw),
            user_flip=0,
        )
    return np.asarray(rgb)


def raw_channel_clip_fraction(path: str, channel_index: int, roi, saturation_margin: int = 16) -> float:
    """Fraction of *raw Bayer* photosites for one channel that are clipped, inside the ROI.

    A demosaiced channel can read below saturation while its source photosites are already at
    the sensor ceiling — interpolation averages a clipped site with clean neighbours and hides
    it. Metering the raw sites (before demosaic/color) catches that, which matters for ETTR
    where the base is deliberately exposed near the ceiling. `roi` is any object with a
    `.pixels(w, h)` method (duck-typed to avoid an infra→services import). channel_index: R=0,
    G=1, B=2. Returns 0.0 if the channel/white level can't be resolved."""
    import rawpy

    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible
        colors = raw.raw_colors_visible
        # No white level means no raw refinement, though the demosaiced clip guard still runs.
        # A frame's own maximum is never a level reference: on a uniform base it sits inside the
        # noise, so a fixed margin below it swallows most of a frame that clips nowhere (the
        # adjust_maximum_thr failure class). _plateau_clip_fraction reads that maximum as a count
        # instead, which noise leaves to a handful of photosites and saturation piles onto.
        white = int(raw.white_level or 0)
        if white <= 0:
            return 0.0
        letter = "RGB"[channel_index]
        desc = raw.color_desc.decode("ascii", errors="ignore")  # e.g. "RGBG": 0=R,1=G,2=B,3=G
        wanted = [j for j, c in enumerate(desc) if c.upper() == letter]
        if not wanted:
            return 0.0
        h, w = img.shape[:2]
        x0, y0, x1, y1 = roi.pixels(w, h)
        sub_img = img[y0:y1, x0:x1]
        mask = np.isin(colors[y0:y1, x0:x1], wanted)
        if not mask.any():
            return 0.0
        values = sub_img[mask]
        threshold = max(0, white - saturation_margin)
        by_white_level = float(np.mean(values >= threshold))
        return max(by_white_level, _plateau_clip_fraction(values, white))


# A plateau is a pile, never one photosite. The ROI maximum is present by definition, so
# without a floor a small ROI reports 1/n clipped — already over the caller's budget.
_MIN_PLATEAU_SITES = 4


def _plateau_clip_fraction(values: np.ndarray, white: int, tail: int = 8) -> float:
    """Fraction of photosites pinned on a saturation plateau, found without the white level.

    A sensor can saturate below the white level its metadata publishes, and no threshold derived
    from that number sees such clipping at all. Saturation has a shape instead: photosites pile up
    against the highest level they can reach, where an exposed surface's histogram is still falling.
    Only a band denser than the wider band below it counts, so noise near the top cannot qualify.
    """
    top = int(values.max())
    if top * 2 < white:  # a dark frame's narrow histogram is not a ceiling
        return 0.0
    pile = int(np.count_nonzero(values >= top - tail))
    below = int(np.count_nonzero((values >= top - 5 * tail) & (values < top - tail)))
    if pile <= below or pile < _MIN_PLATEAU_SITES:
        return 0.0
    return pile / values.size
