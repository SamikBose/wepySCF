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


class NormalizedBondAngleDistance(Distance):
    """Dimensionless bond-progress and angle distance for reactions.

    The image is ``[q_bond, q_angle]``. Smooth logistic bond switches are
    combined so ``q_bond`` changes from approximately zero when the breaking
    bond is formed to approximately one when the making bond is formed.
    ``q_angle`` is either ``theta/pi`` (monotonic angular progress) or
    ``(1-cos(theta))/2`` (alignment, useful for SN2 backside attack). The
    distance is a weighted RMS in this normalized feature space.

    Parameters use the same coordinate units as walker positions (Bohr for
    :class:`~wepy.runners.pyscf.PySCFRunner`).
    """

    def __init__(
        self,
        break_pair,
        make_pair,
        angle_triplet,
        *,
        r0=3.0,
        k=3.0,
        angle_mode: Literal["progress", "alignment"] = "progress",
        weights=(1.0, 1.0),
    ) -> None:
        self.break_pair = tuple(int(i) for i in break_pair)
        self.make_pair = tuple(int(i) for i in make_pair)
        self.angle_triplet = tuple(int(i) for i in angle_triplet)
        if len(self.break_pair) != 2 or len(self.make_pair) != 2:
            raise ValueError("break_pair and make_pair must each contain two indices")
        if len(self.angle_triplet) != 3:
            raise ValueError("angle_triplet must be (outer, vertex, outer)")
        if r0 <= 0.0 or k <= 0.0:
            raise ValueError("r0 and k must be positive")
        if angle_mode not in ("progress", "alignment"):
            raise ValueError("angle_mode must be 'progress' or 'alignment'")
        self.r0 = float(r0)
        self.k = float(k)
        self.angle_mode = angle_mode
        self.weights = self._validate_weights(weights, 2)

    @staticmethod
    def _validate_weights(weights, size):
        values = np.asarray(weights, dtype=float)
        if (
            values.shape != (size,)
            or np.any(values < 0.0)
            or not np.any(values > 0.0)
        ):
            raise ValueError(
                f"weights must be {size} non-negative values with at least one positive"
            )
        return values

    def _switch(self, distance):
        exponent = np.clip(self.k * (distance - self.r0), -700.0, 700.0)
        return float(1.0 / (1.0 + np.exp(exponent)))

    @staticmethod
    def _pair_distance(positions, pair):
        displacement = positions[pair[0]] - positions[pair[1]]
        return float(np.linalg.norm(displacement))

    def _geometry_image(self, positions):
        positions = np.asarray(positions, dtype=float)
        s_break = self._switch(self._pair_distance(positions, self.break_pair))
        s_make = self._switch(self._pair_distance(positions, self.make_pair))
        denominator = s_break + s_make
        q_bond = 0.5 if denominator <= np.finfo(float).tiny else s_make / denominator

        outer_a, vertex, outer_b = self.angle_triplet
        vector_a = positions[outer_a] - positions[vertex]
        vector_b = positions[outer_b] - positions[vertex]
        norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if norm_product <= np.finfo(float).tiny:
            raise ValueError("Cannot define an angle for coincident atoms")
        cosine = float(np.clip(np.dot(vector_a, vector_b) / norm_product, -1.0, 1.0))
        q_angle = (
            float(np.arccos(cosine) / np.pi)
            if self.angle_mode == "progress"
            else 0.5 * (1.0 - cosine)
        )
        return np.asarray([q_bond, q_angle], dtype=float)

    def image(self, state):
        return self._geometry_image(state["positions"])

    def image_distance(self, image_a, image_b) -> float:
        delta = np.asarray(image_a, dtype=float) - np.asarray(image_b, dtype=float)
        if delta.shape != (2,):
            raise ValueError("Normalized bond-angle images must have shape (2,)")
        return float(np.sqrt(np.sum(self.weights * delta**2) / np.sum(self.weights)))


class NormalizedBondAngleChargeDistance(NormalizedBondAngleDistance):
    """Normalized bond-angle metric plus endpoint-projected atomic charges.

    The charge coordinate is the projection of the current atomic-charge
    vector onto ``reactant_charges -> product_charges``. It is zero and one at
    those references, respectively. Values are intentionally not clipped.
    """

    def __init__(
        self,
        break_pair,
        make_pair,
        angle_triplet,
        reactant_charges,
        product_charges,
        *,
        charge_key="charges",
        allow_initial_nan_charges=False,
        weights=(1.0, 1.0, 1.0),
        **geometry_kwargs,
    ) -> None:
        super().__init__(
            break_pair,
            make_pair,
            angle_triplet,
            weights=(1.0, 1.0),
            **geometry_kwargs,
        )
        self.reactant_charges = np.asarray(reactant_charges, dtype=float).ravel()
        self.product_charges = np.asarray(product_charges, dtype=float).ravel()
        if self.reactant_charges.shape != self.product_charges.shape:
            raise ValueError("Reactant and product charge arrays must have the same shape")
        self.charge_delta = self.product_charges - self.reactant_charges
        self.charge_denominator = float(np.dot(self.charge_delta, self.charge_delta))
        if self.charge_denominator <= np.finfo(float).eps:
            raise ValueError("Endpoint charge vectors are too similar to define a projection")
        self.charge_key = str(charge_key)
        self.allow_initial_nan_charges = bool(allow_initial_nan_charges)
        self.weights = self._validate_weights(weights, 3)

    def image(self, state):
        geometry = self._geometry_image(state["positions"])
        charges = np.asarray(state[self.charge_key], dtype=float).ravel()
        if (
            self.allow_initial_nan_charges
            and charges.size == 1
            and np.all(np.isnan(charges))
        ):
            charges = self.reactant_charges
        if charges.shape != self.reactant_charges.shape or not np.all(np.isfinite(charges)):
            raise ValueError(
                "Walker charges are missing, non-finite, or incompatible with references"
            )
        q_charge = float(
            np.dot(charges - self.reactant_charges, self.charge_delta)
            / self.charge_denominator
        )
        return np.asarray([geometry[0], geometry[1], q_charge], dtype=float)

    def image_distance(self, image_a, image_b) -> float:
        delta = np.asarray(image_a, dtype=float) - np.asarray(image_b, dtype=float)
        if delta.shape != (3,):
            raise ValueError("Normalized bond-angle-charge images must have shape (3,)")
        return float(np.sqrt(np.sum(self.weights * delta**2) / np.sum(self.weights)))


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


class DielsAlderCappedFormingBondDistance(Distance):
    """REVO metric based directly on the two forming Diels-Alder distances.

    The walker image is::

        [mean(capped_r1, capped_r2) / mean_scale,
         abs(capped_r1 - capped_r2) / async_scale]

    Positions are expected in Bohr. Capping each distance prevents increasingly
    separated reactants from gaining unlimited novelty in REVO image space.

    Parameters
    ----------
    bond_pairs : sequence of two atom-index pairs
        The two forming-bond pairs, using zero-based atom indices.
    r_cap : float
        Per-bond outward cap in Bohr. Distances larger than this value are
        represented by ``r_cap``.
    mean_scale : float
        Scale for the mean-distance coordinate, in Bohr.
    async_scale : float
        Scale for the asynchronicity coordinate, in Bohr.
    """

    def __init__(
        self,
        bond_pairs=((0, 11), (3, 10)),
        r_cap=9.0,
        mean_scale=1.0,
        async_scale=1.0,
    ) -> None:
        pairs = tuple(tuple(int(atom) for atom in pair) for pair in bond_pairs)
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError("bond_pairs must contain exactly two atom-index pairs")
        if r_cap <= 0:
            raise ValueError("r_cap must be positive")
        if mean_scale <= 0 or async_scale <= 0:
            raise ValueError("mean_scale and async_scale must be positive")

        self.bond_pairs = pairs
        self.r_cap = float(r_cap)
        self.mean_scale = float(mean_scale)
        self.async_scale = float(async_scale)

    def _forming_distances(self, state):
        positions = np.asarray(state["positions"], dtype=float)
        distances = np.asarray(
            [
                np.linalg.norm(positions[i] - positions[j])
                for i, j in self.bond_pairs
            ],
            dtype=float,
        )
        return np.minimum(distances, self.r_cap)

    def image(self, state):
        r1, r2 = self._forming_distances(state)
        mean_distance = 0.5 * (r1 + r2)
        asynchronicity = abs(r1 - r2)

        return np.asarray(
            [
                mean_distance / self.mean_scale,
                asynchronicity / self.async_scale,
            ],
            dtype=float,
        )

    def image_distance(self, image_a, image_b) -> float:
        return float(
            np.linalg.norm(
                np.asarray(image_a, dtype=float)
                - np.asarray(image_b, dtype=float),
            ),
        )


class DielsAlderSigmoidFormingBondDistance(Distance):
    """Smooth, bounded Diels-Alder forming-bond metric.

    Unlike the former ``DielsAlderBondOrderLikeDistance``, this implementation
    uses ``self.bond_pairs`` rather than hard-coded atom indices. Each forming
    distance is capped before applying the sigmoid.

    The walker image is::

        [0.5 * (q1 + q2), async_weight * abs(q1 - q2)]

    where ``q(r) = 1 / (1 + exp(k * (r - r0)))``.

    Positions, ``r0``, and ``r_cap`` are in Bohr; ``k`` is in Bohr**-1.
    """

    def __init__(
        self,
        bond_pairs=((0, 11), (3, 10)),
        r0=7.0,
        k=0.7,
        r_cap=9.0,
        async_weight=1.0,
    ) -> None:
        pairs = tuple(tuple(int(atom) for atom in pair) for pair in bond_pairs)
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError("bond_pairs must contain exactly two atom-index pairs")
        if r0 <= 0 or r_cap <= 0:
            raise ValueError("r0 and r_cap must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if async_weight < 0:
            raise ValueError("async_weight must be non-negative")

        self.bond_pairs = pairs
        self.r0 = float(r0)
        self.k = float(k)
        self.r_cap = float(r_cap)
        self.async_weight = float(async_weight)

    def _switch(self, distance):
        exponent = np.clip(self.k * (distance - self.r0), -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(exponent))

    def _forming_distances(self, state):
        positions = np.asarray(state["positions"], dtype=float)
        distances = np.asarray(
            [
                np.linalg.norm(positions[i] - positions[j])
                for i, j in self.bond_pairs
            ],
            dtype=float,
        )
        return np.minimum(distances, self.r_cap)

    def image(self, state):
        r1, r2 = self._forming_distances(state)
        q1 = self._switch(r1)
        q2 = self._switch(r2)

        progress = 0.5 * (q1 + q2)
        asynchronicity = abs(q1 - q2)

        return np.asarray(
            [
                progress,
                self.async_weight * asynchronicity,
            ],
            dtype=float,
        )

    def image_distance(self, image_a, image_b) -> float:
        return float(
            np.linalg.norm(
                np.asarray(image_a, dtype=float)
                - np.asarray(image_b, dtype=float),
            ),
        )


class DielsAlderTwoBondDistance(Distance):
    """Direct distance metric for the two forming Diels-Alder bonds.

    Each walker is represented as:

        [r_forming_1, r_forming_2]

    Positions and returned distances are in Bohr.

    Parameters
    ----------
    bond_pairs
        The two forming-bond atom pairs, using zero-based indexing.
    r_cap
        Optional upper cap in Bohr. Set to None for completely uncapped
        distances. A finite cap prevents continued reactant separation from
        producing unlimited novelty.
    """

    def __init__(
        self,
        bond_pairs=((0, 11), (3, 10)),
        r_cap=None,
    ):
        bond_pairs = tuple(
            tuple(int(index) for index in pair)
            for pair in bond_pairs
        )

        if len(bond_pairs) != 2:
            raise ValueError(
                "bond_pairs must contain exactly two forming-bond pairs"
            )

        if any(len(pair) != 2 for pair in bond_pairs):
            raise ValueError(
                "Each entry in bond_pairs must contain two atom indices"
            )

        if r_cap is not None and r_cap <= 0:
            raise ValueError("r_cap must be positive or None")

        self.bond_pairs = bond_pairs
        self.r_cap = None if r_cap is None else float(r_cap)

    @staticmethod
    def _pair_distance(positions, pair):
        atom_i, atom_j = pair
        return float(
            np.linalg.norm(
                positions[atom_i] - positions[atom_j]
            )
        )

    def image(self, state):
        positions = np.asarray(state["positions"], dtype=float)

        distances = np.asarray(
            [
                self._pair_distance(positions, pair)
                for pair in self.bond_pairs
            ],
            dtype=float,
        )

        if self.r_cap is not None:
            distances = np.minimum(distances, self.r_cap)

        return distances

    def image_distance(self, image_a, image_b):
        image_a = np.asarray(image_a, dtype=float)
        image_b = np.asarray(image_b, dtype=float)

        if image_a.shape != image_b.shape:
            raise ValueError(
                "Diels-Alder images must have the same shape"
            )

        return float(
            np.sqrt(
                np.mean((image_a - image_b) ** 2)
            )
        )


# Append these classes to:
#   src/wepy/resampling/distances/pyscf.py
#
# The current module already imports NumPy as ``np`` and ``Distance``.


class DielsAlderTwoBondDistance(Distance):
    """Direct two-coordinate metric for the two forming Diels-Alder bonds.

    The walker image is ``[r1, r2]`` in Bohr. Returning the two distances
    directly preserves which forming bond leads an asynchronous pathway.
    """

    def __init__(
        self,
        bond_pairs=((0, 11), (3, 10)),
        r_cap=None,
    ):
        pairs = tuple(
            tuple(int(atom_index) for atom_index in pair)
            for pair in bond_pairs
        )
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError(
                "bond_pairs must contain exactly two atom-index pairs",
            )
        if r_cap is not None and r_cap <= 0:
            raise ValueError("r_cap must be positive or None")

        self.bond_pairs = pairs
        self.r_cap = None if r_cap is None else float(r_cap)

    def image(self, state):
        positions = np.asarray(state["positions"], dtype=float)
        distances = np.asarray(
            [
                np.linalg.norm(
                    positions[atom_i] - positions[atom_j],
                )
                for atom_i, atom_j in self.bond_pairs
            ],
            dtype=float,
        )
        if self.r_cap is not None:
            distances = np.minimum(distances, self.r_cap)
        return distances

    def image_distance(self, image_a, image_b):
        image_a = np.asarray(image_a, dtype=float)
        image_b = np.asarray(image_b, dtype=float)
        if image_a.shape != (2,) or image_b.shape != (2,):
            raise ValueError(
                "DielsAlderTwoBondDistance images must have shape (2,)",
            )
        return float(np.sqrt(np.mean((image_a - image_b) ** 2)))


class DielsAlderTwoSigmoidBondDistance(Distance):
    """Bounded two-coordinate metric for the two forming Diels-Alder bonds.

    Unlike a ``[mean(q), abs(q1-q2)]`` representation, this class returns
    ``[q1, q2]`` directly and therefore preserves which forming bond leads an
    asynchronous pathway.

    Parameters
    ----------
    bond_pairs
        Exactly two zero-based forming-bond atom pairs.
    r0
        Sigmoid midpoint in Bohr.
    k
        Sigmoid steepness in 1/Bohr.
    r_cap
        Optional upper cap in Bohr. With a physical outer COM wall, ``None`` is
        recommended because the sigmoid already compresses outward motion.

    Notes
    -----
    ``q(r) = 1 / (1 + exp(k*(r-r0)))``.

    Product-like short distances have q near one; separated reactants have q
    near zero. ``image_distance`` is the RMS difference in the two q values.
    """

    def __init__(
        self,
        bond_pairs=((0, 11), (3, 10)),
        r0=5.0,
        k=1.0,
        r_cap=None,
    ):
        pairs = tuple(
            tuple(int(atom_index) for atom_index in pair)
            for pair in bond_pairs
        )
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError(
                "bond_pairs must contain exactly two atom-index pairs",
            )
        if r0 <= 0:
            raise ValueError("r0 must be positive (Bohr)")
        if k <= 0:
            raise ValueError("k must be positive (1/Bohr)")
        if r_cap is not None and r_cap <= 0:
            raise ValueError("r_cap must be positive or None")

        self.bond_pairs = pairs
        self.r0 = float(r0)
        self.k = float(k)
        self.r_cap = None if r_cap is None else float(r_cap)

    def _switch(self, distance):
        exponent = np.clip(
            self.k * (distance - self.r0),
            -700.0,
            700.0,
        )
        return 1.0 / (1.0 + np.exp(exponent))

    def image(self, state):
        positions = np.asarray(state["positions"], dtype=float)
        distances = np.asarray(
            [
                np.linalg.norm(
                    positions[atom_i] - positions[atom_j],
                )
                for atom_i, atom_j in self.bond_pairs
            ],
            dtype=float,
        )

        if self.r_cap is not None:
            distances = np.minimum(distances, self.r_cap)

        return np.asarray(
            [self._switch(distance) for distance in distances],
            dtype=float,
        )

    def image_distance(self, image_a, image_b):
        image_a = np.asarray(image_a, dtype=float)
        image_b = np.asarray(image_b, dtype=float)
        if image_a.shape != (2,) or image_b.shape != (2,):
            raise ValueError(
                "DielsAlderTwoSigmoidBondDistance images must have shape (2,)",
            )
        return float(np.sqrt(np.mean((image_a - image_b) ** 2)))

