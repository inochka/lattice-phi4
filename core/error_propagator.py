from dataclasses import dataclass


@dataclass
class Estimate:
    """
        Class to propagate error in analu]yticak formulas. Implements the proper way of arithmetical operations on pairs of
        (value, error) to keep the second term as an error of the first one in a linear approximation.
    """
    value: float
    error: float

    def __add__(self, other):
        other = as_estimate(other)
        return Estimate(
            self.value + other.value,
            self.error + other.error,
        )

    __radd__ = __add__

    def __neg__(self):
        return Estimate(-self.value, self.error)

    def __sub__(self, other):
        return self + (-as_estimate(other))

    def __rsub__(self, other):
        return as_estimate(other) - self

    def __mul__(self, other):
        other = as_estimate(other)
        return Estimate(
            self.value * other.value,
            abs(other.value) * self.error
            + abs(self.value) * other.error,
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_estimate(other)

        value = self.value / other.value
        error = (
            self.error / abs(other.value)
            + abs(self.value) * other.error / other.value**2
        )

        return Estimate(value, error)

    def __rtruediv__(self, other):
        return as_estimate(other) / self

    def __pow__(self, power):
        return Estimate(
            self.value**power,
            abs(power * self.value**(power - 1)) * self.error,
        )

    def __str__(self):
        return str(self.value)


def as_estimate(x):
    if isinstance(x, Estimate):
        return x
    return Estimate(float(x), 0.0)
