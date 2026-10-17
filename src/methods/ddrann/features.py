import numpy as np


class GaussianFeatures:
    """Tensor-product Gaussian features with explicit first/second derivatives."""

    def __init__(self, weights, bias):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = np.asarray(bias, dtype=float)

    @classmethod
    def random(cls, dimension, neurons, radius, rng, center_radius=1.):
        weights = rng.uniform(-radius, radius, (dimension, neurons))
        bias = weights * rng.uniform(-center_radius, center_radius, (dimension, neurons))
        return cls(weights, bias)

    def evaluate(self, points, gradient=False, laplacian=False):
        z = np.asarray(points)[:, :, None] * self.weights[None] + self.bias[None]
        value = np.exp(-np.sum(z*z, axis=1))
        if not (gradient or laplacian):
            return value
        factors = -2 * z * self.weights[None]
        grad = value[:, None, :] * factors if gradient else None
        lap = value * np.sum(factors * factors - 2 * self.weights[None] ** 2, axis=1) if laplacian else None
        return value, grad, lap


class TanhFeatures:
    """One-dimensional tanh features used by the Petrov--Galerkin method.

    ``asymptotic_correction`` replaces each feature ``tanh(wx+b)`` by
    ``tanh(wx+b) - sign(w)``.  On the positive half-line this hard-codes the
    zero far-field limit while leaving the derivative unchanged.
    """

    def __init__(self, weight, bias, asymptotic_correction=False):
        self.weight = np.asarray(weight)
        self.bias = np.asarray(bias)
        self.asymptotic_correction = bool(asymptotic_correction)
        self.far_field_limit = np.sign(self.weight)
        self.asymptotic_shift = (self.far_field_limit
                                 if self.asymptotic_correction
                                 else np.zeros_like(self.weight))

    @classmethod
    def random(cls, count, radius, rng, asymptotic_correction=False):
        return cls(rng.uniform(-radius, radius, count),
                   rng.uniform(-radius, radius, count),
                   asymptotic_correction=asymptotic_correction)

    def evaluate(self, x, derivative=False):
        value = np.tanh(np.asarray(x)[:, None] * self.weight + self.bias)
        if derivative:
            # The constant asymptotic shift has zero derivative.
            return (1 - value * value) * self.weight
        return value - self.asymptotic_shift


class TanhHarmonicFeatures:
    """Complex space-time tanh-harmonic features."""

    def __init__(self, wx, bx, wt, bt):
        self.wx, self.bx, self.wt, self.bt = map(np.asarray, (wx, bx, wt, bt))

    @classmethod
    def random(cls, neurons, spatial_radius, temporal_radius, rng):
        values = [rng.uniform(-spatial_radius, spatial_radius, neurons) for _ in range(2)]
        values += [rng.uniform(-temporal_radius, temporal_radius, neurons) for _ in range(2)]
        return cls(*values)

    def evaluate(self, points, derivative=None):
        x, t = points[:, 0, None], points[:, 1, None]
        z = x * self.wx + self.bx
        value = np.tanh(z)
        phase = np.exp(1j * (t * self.wt + self.bt))
        if derivative == "x":
            return (1 - value * value) * self.wx * phase
        if derivative == "xx":
            return -2 * value * (1 - value * value) * self.wx**2 * phase
        if derivative == "t":
            return 1j * self.wt * value * phase
        return value * phase
