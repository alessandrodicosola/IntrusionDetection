from .BackgroundSubtractor import BackgroundSubtractor
from .BackgroundNoGaussian import BackgroundNoGaussian
from functools import partial

detectors = {
    "background_no_gaussian": BackgroundNoGaussian,
    "background": BackgroundSubtractor,
}


def detector_create(key, **kwargs):
    if key == "background":
        return detectors[key](kwargs.get("video"), kwargs.get("kind"))
    elif key == "background_no_gaussian":
        return detectors[key](kwargs.get("video"))
    else:
        raise KeyError(f"invalid key:{key}. Valid are: {','.join(detectors.keys())}")


__all__ = ["detector_create"]
