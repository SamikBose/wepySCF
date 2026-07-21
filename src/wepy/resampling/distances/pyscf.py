"""Distance metrics for PySCF based simulations."""

# Standard Library
from typing import Literal

# Third Party Library
import numpy as np

# First Party Library
from wepy.resampling.distances.distance import Distance


class QMGridDensityDistance(Distance):
    """Distance between walkers using electron density sampled on a grid.

    Expects walker states to include the `density_grid` field produced by
    :class:`wepy.runners.pyscf.PySCFRunner`.
    """

    def __init__(self, grid_key: str = "density_grid", normalize: bool = True) -> None:
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

    where `d_break` is the interatomic distance for `break_pair` and
    `d_make` is the interatomic distance for `make_pair`.

    The distance between two walker images is the RMS difference of this
    2-vector.
    """

    def __init__(self, break_pair, make_pair) -> None:
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

    with `d_break` computed from `break_pair` and `d_make` from
    `make_pair`.

    Images are stored as 1D arrays with one element for compatibility with
    generic `Distance` image handling.
    """

    def __init__(self, break_pair, make_pair) -> None:
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

    def image_distance(self, image_a, image_b) -> float:
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

    def __init__(self, dihedrals) -> None:
        # accept either a single (i, j, k, l) or a list/tuple of them
        if len(dihedrals) == 4 and np.isscalar(dihedrals[0]):
            dihedrals = [dihedrals]
        self.dihedrals = [tuple(int(a) for a in quad) for quad in dihedrals]

    @staticmethod
    def _dihedral(positions, quad: tuple[int, int, int, int]):
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

    def image_distance(self, image_a, image_b) -> float:
        a = np.asarray(image_a, dtype=float).reshape(-1, 2)
        b = np.asarray(image_b, dtype=float).reshape(-1, 2)
        dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
        angles = np.arccos(dots)  # geodesic angle per dihedral, [0, pi]
        return float(np.sqrt(np.mean(angles**2)))


class ChargeDistance(Distance):
    """Distance between walkers based on per-atom partial charges.

    Uses the Mulliken partial charges computed by
    :class:`wepy.runners.pyscf.PySCFRunner` and stored in the walker state
    under `charge_key` (default `"charges"`), an array of shape
    `(n_atoms,)`.

    The image is the charge vector, optionally restricted to a subset of atoms
    via `atom_indices`. The distance between two images is the RMS difference
    of that vector, matching the convention of the other PySCF metrics.

    Restricting to the reacting atoms is usually what you want for a reaction
    progress coordinate. For an SN2 (X-CH3 + Y- -> X- + CH3-Y) charge flows
    onto the leaving group and off the nucleophile, so
    `atom_indices=[nucleophile, leaving_group]` tracks that transfer directly
    while ignoring spectator atoms.

    Note on units: partial charges are of order 1 (electrons), whereas the
    geometric metrics work in Bohr (order 2-5). Set `char_dist` / `merge_dist`
    in the resampler accordingly (charge units, typically ~0.01-0.1), not in the
    values you would use for a distance metric.

    Parameters
    ----------
    atom_indices : arraylike of int, optional
        0-based indices of the atoms whose charges define the image. If `None`
        (default) all atoms are used.
    charge_key : str
        State field holding the per-atom charges. Default `"charges"`.

    """

    def __init__(self, atom_indices=None, charge_key: str = "charges") -> None:
        self.atom_indices = (
            None if atom_indices is None else np.asarray(atom_indices, dtype=int)
        )
        self.charge_key = charge_key

    def image(self, state):
        charges = np.asarray(state[self.charge_key], dtype=float).ravel()

        if self.atom_indices is not None:
            charges = charges[self.atom_indices]

        return charges

    def image_distance(self, image_a, image_b):
        if image_a.shape != image_b.shape:
            raise ValueError("Charge images must have the same shape")

        return np.sqrt(np.mean((image_a - image_b) ** 2))


class ChargeTransferDistance(Distance):
    """1D charge-transfer reaction-coordinate metric.

    Defines a scalar coordinate from per-atom Mulliken charges:

        xi = q[acceptor] - q[donor]

    As negative charge transfers from `donor` to `acceptor` (e.g. from the
    nucleophile to the leaving group in an SN2), `xi` varies monotonically,
    giving a smooth 1D electronic progress coordinate that complements the
    geometric `ProtonTransferDistance` / `BondBreakMakeDistance` metrics.

    Images are stored as 1D arrays with one element for compatibility with the
    generic `Distance` image handling.

    Parameters
    ----------
    donor : int
        0-based index of the atom losing charge (e.g. the nucleophile).
    acceptor : int
        0-based index of the atom gaining charge (e.g. the leaving group).
    charge_key : str
        State field holding the per-atom charges. Default `"charges"`.

    """

    def __init__(self, donor: int, acceptor: int, charge_key: str = "charges") -> None:
        self.donor = int(donor)
        self.acceptor = int(acceptor)
        self.charge_key = charge_key

    def image(self, state):
        q = np.asarray(state[self.charge_key], dtype=float).ravel()
        xi = q[self.acceptor] - q[self.donor]
        return np.array([xi], dtype=float)

    def image_distance(self, image_a, image_b) -> float:
        return abs(float(image_a[0] - image_b[0]))


"""Cremer-Pople ring-puckering distance metric for wepy.

Collapses the out-of-plane displacements of an N-membered ring into puckering
coordinates (Cremer & Pople, J. Am. Chem. Soc. 1975, 97, 1354) and uses them as
the REVO resampling coordinate. For cyclohexane this turns the whole chair <->
twist-boat <-> inverted-chair inversion into a single smooth coordinate.

Drop in wepy/resampling/distances/ alongside the other Distance subclasses.
"""


class CremerPopleDistance(Distance):
    """Ring-puckering metric based on Cremer-Pople coordinates.

    Parameters
    ----------
    ring_indices : sequence of int
        0-based atom indices of the ring atoms, given IN CONNECTIVITY ORDER
        around the ring (e.g. the six cyclohexane carbons traversed
        0-1-2-3-4-5 and back to 0). Order matters: the puckering phases depend
        on the sequence, so a scrambled order gives meaningless coordinates.
    coordinate : {"theta", "full"}, default "theta"
        "theta" : 1-D inversion coordinate -- the polar angle theta (radians).
                  theta = 0 at one chair, pi at the inverted chair, pi/2 at the
                  equator (boats/twist-boats). image_distance is |d theta|, so
                  set char_dist / merge_dist in RADIANS.
        "full"  : the 3-D Cartesian puckering vector
                  (q2 cos phi, q2 sin phi, q3), a point on/inside the puckering
                  sphere with magnitude Q. image_distance is the Euclidean
                  distance in that space, in the SAME LENGTH UNITS as the input
                  coordinates (Bohr here). Resolves which boat/twist (phi) too,
                  and is naturally periodic in phi.

    Notes
    -----
    For a 6-ring: Q = sqrt(q2**2 + q3**2), cos(theta) = q3 / Q, phi = phi2.
    Even N only (the q_{N/2} term is a single amplitude); N = 6 is cyclohexane.
    The mean-plane normal is fixed by the ring traversal order, so the two chairs
    map reproducibly to theta = 0 and theta = pi.

    """

    def __init__(
        self,
        ring_indices,
        coordinate: Literal["theta", "full"] = "theta",
    ) -> None:
        if coordinate not in ("theta", "full"):
            raise ValueError("coordinate must be 'theta' or 'full'")
        self.ring_indices = list(ring_indices)
        self.N = len(self.ring_indices)
        if self.N < 4 or self.N % 2 != 0:
            raise ValueError(
                "ring must have an even number of atoms >= 4 (N=6 for cyclohexane)",
            )
        self.coordinate = coordinate

    def _puckering(self, coords):
        """coords: (N, 3) ring positions in ring order -> (Q, theta, phi, q2, q3)."""
        N = self.N
        r = coords - coords.mean(axis=0)  # shift to geometric center
        j = np.arange(N)

        # mean plane: normal from the two m=1 vectors (Cremer-Pople eqs 4-5)
        R1 = (r * np.sin(2 * np.pi * j / N)[:, None]).sum(axis=0)
        R2 = (r * np.cos(2 * np.pi * j / N)[:, None]).sum(axis=0)
        n = np.cross(R1, R2)
        n = n / np.linalg.norm(n)
        z = r @ n  # out-of-plane displacements

        # m = 2 puckering pair (q2, phi)
        q2cos = np.sqrt(2.0 / N) * (z * np.cos(2 * np.pi * 2 * j / N)).sum()
        q2sin = -np.sqrt(2.0 / N) * (z * np.sin(2 * np.pi * 2 * j / N)).sum()
        q2 = np.hypot(q2cos, q2sin)
        phi = np.arctan2(q2sin, q2cos) % (2 * np.pi)

        # m = N/2 single amplitude (even N): cos(pi*j) = (-1)**j
        q3 = np.sqrt(1.0 / N) * (z * np.cos(np.pi * j)).sum()

        Q = np.hypot(q2, q3)
        theta = np.arccos(np.clip(q3 / Q, -1.0, 1.0)) if Q > 1e-12 else 0.0
        return Q, theta, phi, q2, q3

    def image(self, state):
        coords = np.asarray(state["positions"], dtype=float)[self.ring_indices]
        Q, theta, phi, q2, q3 = self._puckering(coords)
        if self.coordinate == "theta":
            return np.array([theta])
        # 3-D puckering vector: phi periodicity handled by the embedding
        return np.array([q2 * np.cos(phi), q2 * np.sin(phi), q3])

    def image_distance(self, image_a, image_b) -> float:
        a = np.asarray(image_a, dtype=float)
        b = np.asarray(image_b, dtype=float)
        if self.coordinate == "theta":
            return float(abs(a[0] - b[0]))  # theta in [0, pi], not periodic
        return float(np.linalg.norm(a - b))  # Euclidean in puckering space


"""HOMO-LUMO gap distance metric for wepy.
Drives REVO along the frontier-orbital gap eps_LUMO - eps_HOMO, read from the
`mo_energy` array the PySCF runner writes into each walker state (ascending
orbital energies, in Hartree). No extra SCF is done -- it reuses what the
runner already computed.
Intended use: an electronic order parameter. As a sigma bond stretches toward
homolysis the sigma/sigma* gap collapses, so the gap is often a better driver
toward bond-breaking / near-degenerate geometries than a bond length. Caveat:
where the gap is smallest (near-degeneracy, diradical) single-reference KS-DFT
is least reliable, so the gap is excellent for *finding/driving toward* those
regions but treat the energetics there with care.
Drop this class into wepy/resampling/distances/pyscf.py alongside the others.
"""

HARTREE_TO_EV = 27.211386245988


class HOMOLUMOGapDistance(Distance):
    """Distance metric on the HOMO-LUMO gap.

    Parameters
    ----------
    n_occ : int or None, default None
        Number of doubly-occupied orbitals (closed-shell / RKS). If None it is
        derived per walker from the Mole as `mol.nelectron // 2`, which
        already accounts for charge and for any ECP core (validated for both
        anions and pseudopotential heavy atoms). Pass an explicit value only if
        you need to override that.
    units : {"hartree", "ev"}, default "hartree"
        Units of the returned gap -- and therefore of the char_dist / merge_dist
        you set in the resampler. Hartree keeps you consistent with pyscf's
        native mo_energy; "ev" is only for readability.
    mo_energy_key : str, default "mo_energy"
        State key holding the ascending MO energies written by the runner.

    Notes
    -----
    image() returns a length-1 array [gap]; image_distance is |gap_a - gap_b|.
    Closed-shell (RKS) only: mo_energy is expected 1-D. For UKS you would take
    the gap per spin channel (or min over channels) -- not handled here.
    Uninitialised states (cycle-0 NaN placeholder) return a gap of 0.0 so they
    don't poison the REVO distance matrix; by the time REVO resamples, the
    runner has filled in real energies.

    """

    def __init__(
        self,
        n_occ: int | None = None,
        units: Literal["hartree", "ev"] = "hartree",
        mo_energy_key: str = "mo_energy",
    ) -> None:
        if units not in ("hartree", "ev"):
            raise ValueError("units must be 'hartree' or 'ev'")
        self.n_occ = n_occ
        self.units = units
        self.mo_energy_key = mo_energy_key

    def _gap(self, state) -> float:
        mo = np.asarray(state[self.mo_energy_key], dtype=float).ravel()

        nocc = self.n_occ
        if nocc is None:
            nocc = state["mol"].nelectron // 2

        # guard: uninitialised placeholder, or nocc out of range
        if nocc < 1 or mo.size < nocc + 1:
            return 0.0
        homo, lumo = mo[nocc - 1], mo[nocc]
        if not (np.isfinite(homo) and np.isfinite(lumo)):
            return 0.0

        gap = lumo - homo
        return float(gap * HARTREE_TO_EV) if self.units == "ev" else float(gap)

    def image(self, state):
        return np.array([self._gap(state)])

    def image_distance(self, image_a, image_b) -> float:
        return float(
            abs(
                np.asarray(image_a, dtype=float)[0]
                - np.asarray(image_b, dtype=float)[0],
            ),
        )


class DielsAlderBondOrderLikeDistance(Distance):
    """Diels-Alder metric using smooth bond-formation variables."""

    def __init__(
        self,
        bond_pairs=((0, 11), (3, 10)),
        r0=4.0,      # midpoint in Bohr; ~2.1 Å
        k=2.0,       # steepness in 1/Bohr
        async_weight=1.0,
    ):
        self.bond_pairs = tuple(bond_pairs)
        self.r0 = float(r0)
        self.k = float(k)
        self.async_weight = float(async_weight)

    def _switch(self, r):
        return 1.0 / (1.0 + np.exp(self.k * (r - self.r0)))

    def image(self, state):
        pos = np.asarray(state["positions"], dtype=float)

        r1 = np.linalg.norm(pos[0] - pos[11])   # C1-C12
        r2 = np.linalg.norm(pos[3] - pos[10])   # C4-C11

        q1 = self._switch(r1)
        q2 = self._switch(r2)

        progress = 0.5 * (q1 + q2)              # 0 reactant-like, 1 product-like
        asynchronicity = abs(q1 - q2)

        return np.array(
            [
                progress,
                self.async_weight * asynchronicity,
            ],
            dtype=float,
        )

    def image_distance(self, image_a, image_b):
        return float(np.linalg.norm(np.asarray(image_a) - np.asarray(image_b)))
