# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import uuid

# Third Party Library
import numpy as np

# First Party Library
from wepy.boundary_conditions.bond_distance import BondDistanceBC
from wepy.boundary_conditions.multi_boundary import MultiBoundaryBC
from wepy.runners.pyscf import PySCFState


class _RegenerateWalkerIdMixin:
    """Regenerate walker IDs on warp to avoid scanner-cache collisions between unrelated trajectories.

    Use this as the first mixin in the inheritance list to override _warp method to work with PySCF runner.
    """

    def _warp(self, walker):
        """Warp walker and regenerate walker ID while preserving weight."""
        warped_walker, warp_data = super()._warp(walker)

        new_state = PySCFState(**{**warped_walker.state._data, "walker_id": str(uuid.uuid4())})  # noqa: SLF001
        warped_walker = type(warped_walker)(new_state, warped_walker.weight)

        return warped_walker, warp_data


class PySCFBondDistanceBC(_RegenerateWalkerIdMixin, BondDistanceBC):
    """Bond-distance BC with HDF5-safe fixed-shape progress records and PySCF walker-ID regeneration."""

    def __init__(self, *args, n_walkers: int, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_walkers = int(n_walkers)
        if self._n_walkers <= 0:
            raise ValueError("n_walkers must be positive")

        progress_fields = []
        progress_shapes = []
        progress_dtypes = []

        if len(self.break_pairs) > 0:
            progress_fields.append("break_distances")
            progress_shapes.append((self._n_walkers, len(self.break_pairs)))
            progress_dtypes.append(float)

        if len(self.make_pairs) > 0:
            progress_fields.append("make_distances")
            progress_shapes.append((self._n_walkers, len(self.make_pairs)))
            progress_dtypes.append(float)

        self.PROGRESS_FIELDS = tuple(progress_fields)
        self.PROGRESS_SHAPES = tuple(progress_shapes)
        self.PROGRESS_DTYPES = tuple(progress_dtypes)
        self.PROGRESS_RECORD_FIELDS = tuple(progress_fields)

    def _progress(self, walker):
        """Delegate distance-condition logic to the parent, then reshape for HDF5."""
        warped, progress_data = super()._progress(walker)

        reshaped = {}
        if len(self.break_pairs) > 0:
            reshaped["break_distances"] = np.asarray(progress_data["break_distances"], dtype=float).reshape(-1)

        if len(self.make_pairs) > 0:
            reshaped["make_distances"] = np.asarray(progress_data["make_distances"], dtype=float).reshape(-1)

        return warped, reshaped


class PySCFMultiBoundaryBC(_RegenerateWalkerIdMixin, MultiBoundaryBC):
    """Generic multi-boundary BC with PySCF/HDF5 integration and walker-ID regeneration."""

    def __init__(self, *args, n_walkers: int, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_walkers = int(n_walkers)
        if self._n_walkers <= 0:
            raise ValueError("n_walkers must be positive")

        # Parent __init__ already sets self.PROGRESS_SHAPES for a single walker's
        # worth of distances; here we widen the leading dim to n_walkers for the
        # HDF5-fixed-shape records.
        self.PROGRESS_SHAPES = (
            (self._n_walkers, len(self.tracked_pairs)),
            (self._n_walkers, 1),
            (self._n_walkers, 1),
            (self._n_walkers, 1),
        )
