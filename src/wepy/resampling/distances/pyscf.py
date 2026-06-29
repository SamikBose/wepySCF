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


class DihedralDistance(Distance):
    """Distance based on one or more dihedral (torsion) angles.

    Each dihedral is a 4-tuple of 0-indexed atom indices (i, j, k, l) defining
    the torsion about the j-k bond. Angles are mapped to the unit circle as
    (cos phi, sin phi), so the metric is smooth and inherently periodic -- no
    manual 2*pi wrapping, and gauche(+) / gauche(-) are correctly placed.

    image_distance is the RMS geodesic angular difference across the supplied
    dihedrals, in RADIANS (so set char_dist / merge_dist in radians).
    The metric is angle-based and therefore independent of the position units
    (Bohr or Angstrom).

    Examples
    --------
    Butane central torsion C0-C1-C2-C3:
        DihedralDistance((0, 1, 2, 3))
    Two torsions (e.g. a peptide phi/psi):
        DihedralDistance([(4, 6, 8, 14), (6, 8, 14, 16)])
    """

    def __init__(self, dihedrals):
        # accept either a single (i, j, k, l) or a list/tuple of them
        if len(dihedrals) == 4 and np.isscalar(dihedrals[0]):
            dihedrals = [dihedrals]
        self.dihedrals = [tuple(int(a) for a in quad) for quad in dihedrals]

    @staticmethod
    def _dihedral(positions, quad):
        p0, p1, p2, p3 = (positions[i] for i in quad)
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))

    def image(self, state):
        positions = np.asarray(state["positions"], dtype=float)
        feats = []
        for quad in self.dihedrals:
            phi = self._dihedral(positions, quad)
            feats.extend([np.cos(phi), np.sin(phi)])
        return np.array(feats, dtype=float)

    def image_distance(self, image_a, image_b):
        a = np.asarray(image_a, dtype=float).reshape(-1, 2)
        b = np.asarray(image_b, dtype=float).reshape(-1, 2)
        dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
        angles = np.arccos(dots)            # geodesic angle per dihedral, [0, pi]
        return float(np.sqrt(np.mean(angles ** 2)))
