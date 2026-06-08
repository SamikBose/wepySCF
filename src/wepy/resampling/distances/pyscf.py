"""Distance metrics for PySCF based simulations."""

# Third Party Library
import numpy as np

# First Party Library
from wepy.resampling.distances.distance import Distance


class QMGridDensityDistance(Distance):
    """Distance between walkers using electron density sampled on a grid.

    Expects walker states to include the ``density_grid`` field produced by
    :class:`wepy.runners.pyscf.PySCFRunner`.
    """

    def __init__(self, grid_key="density_grid", normalize=True):
        self.grid_key = grid_key
        self.normalize = normalize

    def image(self, state):
        rho = np.asarray(state[self.grid_key], dtype=float).ravel()

        if self.normalize:
            total = np.sum(np.abs(rho))
            if total > 0:
                rho = rho / total

        return rho

    def image_distance(self, image_a, image_b):
        if image_a.shape != image_b.shape:
            raise ValueError("Density images must have the same shape")

        return np.sqrt(np.mean((image_a - image_b) ** 2))


class BondBreakMakeDistance(Distance):
    """2D bond-breaking / bond-making geometric distance metric.

    For each walker an image is generated as::

        [d_break, d_make]

    where ``d_break`` is the interatomic distance for ``break_pair`` and
    ``d_make`` is the interatomic distance for ``make_pair``.

    The distance between two walker images is the RMS difference of this
    2-vector.
    """

    def __init__(self, break_pair, make_pair):
        self.break_pair = tuple(break_pair)
        self.make_pair = tuple(make_pair)

    def _pair_distance(self, positions, pair):
        i, j = pair
        disp = positions[i] - positions[j]
        return np.sqrt(np.sum(disp * disp))

    def image(self, state):
        positions = np.asarray(state["positions"], dtype=float)

        d_break = self._pair_distance(positions, self.break_pair)
        d_make = self._pair_distance(positions, self.make_pair)

        return np.array([d_break, d_make], dtype=float)

    def image_distance(self, image_a, image_b):
        return np.sqrt(np.mean((image_a - image_b) ** 2))


class ProtonTransferDistance(Distance):
    """1D proton-transfer reaction-coordinate metric.

    Defines a scalar coordinate:

        xi = d_break - d_make

    with ``d_break`` computed from ``break_pair`` and ``d_make`` from
    ``make_pair``.

    Images are stored as 1D arrays with one element for compatibility with
    generic ``Distance`` image handling.
    """

    def __init__(self, break_pair, make_pair):
        self.break_pair = tuple(break_pair)
        self.make_pair = tuple(make_pair)

    def _pair_distance(self, positions, pair):
        i, j = pair
        disp = positions[i] - positions[j]
        return np.sqrt(np.sum(disp * disp))

    def image(self, state):
        positions = np.asarray(state["positions"], dtype=float)

        d_break = self._pair_distance(positions, self.break_pair)
        d_make = self._pair_distance(positions, self.make_pair)

        xi = d_break - d_make
        return np.array([xi], dtype=float)

    def image_distance(self, image_a, image_b):
        return abs(float(image_a[0] - image_b[0]))
