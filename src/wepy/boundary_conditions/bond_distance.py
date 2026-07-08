# Standard Library
import logging

logger = logging.getLogger(__name__)
# Third Party Library
import numpy as np
from numpy import floating

# First Party Library
from wepy.boundary_conditions.boundary import WarpBC


class BondDistanceBC(WarpBC):
    """Boundary condition for bond breaking/forming distance.

    Walkers will be warped when all break distances are larger than their
    cutoffs and all make distances are smaller than their cutoffs.

    Break and make conditions are optional. If a condition type is not provided,
    it is ignored. There must be at least one condition type provided.
    """

    # TODO: Check if these are right

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ("break_cutoffs", "make_cutoffs")
    BC_SHAPES = (Ellipsis, Ellipsis)
    BC_DTYPES = (float, float)

    BC_RECORD_FIELDS = ("break_cutoffs", "make_cutoffs")

    # warping fields are directly inherited

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ("break_distances", "make_distances")
    PROGRESS_SHAPES = (Ellipsis, Ellipsis)
    PROGRESS_DTYPES = (float, float)

    PROGRESS_RECORD_FIELDS = ("break_distances", "make_distances")

    def __init__(
        self,
        initial_states=None,
        initial_weights=None,
        break_pairs=None,
        break_cutoffs=None,
        make_pairs=None,
        make_cutoffs=None,
        **kwargs,
    ) -> None:
        # FIXME: Remove periodic from all?
        """Constructor for BondDistanceBC class.

        Arguments:
        ---------
        initial_states : list of objects implementing the State interface
            The list of possible states that warped walkers will assume.

        initial_weights : list of float, optional
            List of normalized probabilities of the initial_states
            provided. If not given, uniform probabilities will be
            used.

        break_pairs : list of tuple, optional
            The atom pairs are given as indices.

        break_cutoffs : list of float, optional
            The respective cutoff distances for the break pairs.
            Condition: distance >= cutoff

        make_pairs : list of tuple, optional
            The atom pairs are given as indices.

        make_cutoffs : list of float, optional
            The respective cutoff distances for the make pairs.
            Condition: distance <= cutoff

        Raises:
        ------
        AssertionError
            If the lengths of the respective pairs and cutoffs do not match.

        AssertionError
            If no condition type is provided.
        """
        super().__init__(initial_states=initial_states, initial_weights=initial_weights, **kwargs)

        if break_pairs is None:
            break_pairs = []
            break_cutoffs = []

        if make_pairs is None:
            make_pairs = []
            make_cutoffs = []

        assert len(break_pairs) == len(break_cutoffs), "break_pairs and break_cutoffs must have the same length"
        assert len(make_pairs) == len(make_cutoffs), "make_pairs and make_cutoffs must have the same length"

        assert not (len(break_pairs) == 0 and len(make_pairs) == 0), "At least one condition type must be provided"

        self._break_pairs = break_pairs
        self._break_cutoffs = np.asarray(break_cutoffs)
        self._make_pairs = make_pairs
        self._make_cutoffs = np.asarray(make_cutoffs)

    @property
    def break_pairs(self):
        """The break pairs for the bond distance boundary condition."""
        return self._break_pairs

    @property
    def break_cutoffs(self):
        """The break cutoffs for the bond distance boundary condition."""
        return self._break_cutoffs

    @property
    def make_pairs(self):
        """The make pairs for the bond distance boundary condition."""
        return self._make_pairs

    @property
    def make_cutoffs(self):
        """The make cutoffs for the bond distance boundary condition."""
        return self._make_cutoffs

    def _calc_pair_distance(self, positions, pair) -> floating:
        """Calculate distance between an atom pair.

        Parameters
        ----------
        positions : ndarray
            Atom positions.

        pair : tuple of int
            The indices of the atoms in the pair.

        Returns
        -------
        distance : float
            The distance between the atom pair.

        """
        atom_i, atom_j = pair
        return np.linalg.norm(positions[atom_i] - positions[atom_j])

    def _calc_distances(self, walker):
        """Calculate all atom pair distances.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        distances : tuple of 2 arrays
            The first array contains the distances between the break pairs,
            and the second array contains the distances between the make pairs.

        """
        positions = walker.state["positions"]

        break_distances = [self._calc_pair_distance(positions, break_pair) for break_pair in self._break_pairs]
        make_distances = [self._calc_pair_distance(positions, make_pair) for make_pair in self._make_pairs]

        return (np.asarray(break_distances), np.asarray(make_distances))

    def _progress(self, walker):
        """Calculate if the walker satisfies the bond
        distance conditions and provide progress record.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        warped : bool
           Whether the walker is should be warped or not

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """
        break_distances, make_distances = self._calc_distances(walker)

        # All broken bonds must be longer than their cutoff
        break_satisfied = np.all(break_distances >= self._break_cutoffs)

        # All formed bonds must be shorter than their cutoff
        make_satisfied = np.all(make_distances <= self._make_cutoffs)

        warped = break_satisfied and make_satisfied

        progress_data = {
            "break_distances": np.squeeze(break_distances),
            "make_distances": np.squeeze(make_distances),
        }

        return warped, progress_data

    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):
        """Perform an update to the boundary conditions.

        This is only used on the first cycle to keep a record of the
        cutoff parameters.

        Parameters
        ----------
        new_walkers : list of walkers
            The walkers after warping.

        warp_data : list of dict

        progress_data : dict

        cycle : int

        Returns
        -------
        bc_data : list of dict
            The dictionary-style records for BC update events

        """
        if cycle == 0:
            return [
                {
                    "break_cutoffs": np.asarray(self._break_cutoffs),
                    "make_cutoffs": np.asarray(self._make_cutoffs),
                },
            ]
        return []
