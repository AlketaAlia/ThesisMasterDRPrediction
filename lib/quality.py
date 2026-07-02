"""Image quality heuristics, calibrated for retinal fundus photography."""
import cv2
import numpy as np


def analyze_image_quality(image, tr):
    """Approximate image-quality heuristics.

    The thresholds below are loosened for retinal fundus images, which are
    naturally darker than ordinary photos: most of the frame is the dark
    background outside the optic disc. Generic photo-quality cutoffs were
    flagging perfectly usable fundus images as "Poor".

    The `tr` callable is the i18n translator — passed in so this module
    doesn't have to know about Streamlit / language state.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Compute brightness on the non-black region only — the dark border around
    # the retinal disc skews mean brightness downward.
    foreground_mask = gray > 20
    if foreground_mask.any():
        fg_brightness = float(gray[foreground_mask].mean() / 255.0)
    else:
        fg_brightness = brightness

    issues = []
    if fg_brightness < 0.10 or fg_brightness > 0.95:
        issues.append("brightness")
    if contrast < 0.07:
        issues.append("contrast")
    if blur_score < 30:
        issues.append("blur")

    if not issues:
        rating = tr("quality_good")
    elif len(issues) == 1:
        rating = tr("quality_fair")
    else:
        rating = tr("quality_poor")

    return {
        "rating": rating,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "issues": issues,
    }
