import colorsys
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def exact_div(x, y):
    assert x % y == 0
    return x // y


def confidence_to_color(confidence: float) -> str:
    hue = 120 * confidence
    rgb = colorsys.hsv_to_rgb(hue / 360, 1, 1)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )
