"""Reusable helpers for manufactured elliptic solutions."""
import numpy as np


def exponential_wave(points, decay_gradient, decay_laplacian, log_envelope, wave):
    phase = points @ wave
    envelope = np.exp(log_envelope(points))
    sine, cosine = np.sin(phase), np.cos(phase)
    gradient_log = decay_gradient(points)
    value = envelope * sine
    gradient = envelope[:, None] * (gradient_log * sine[:, None] + cosine[:, None] * wave)
    laplacian = envelope * (
        (decay_laplacian(points) + np.sum(gradient_log**2, axis=1) - np.sum(wave**2)) * sine
        + 2 * (gradient_log @ wave) * cosine
    )
    return value, gradient, laplacian
