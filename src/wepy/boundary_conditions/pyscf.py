# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import uuid

# First Party Library
from wepy.boundary_conditions.bond_distance import BondDistanceBC
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
    pass


# TODO: Implement/use MultiBoundaryBC
class PySCFMultiBoundaryBC(_RegenerateWalkerIdMixin, MultiBoundaryBC):
    """Generic multi-boundary BC with PySCF/HDF5 integration."""

    def __init__(self, *args, n_walkers: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_walkers = int(n_walkers)
        if self._n_walkers <= 0:
            raise ValueError("n_walkers must be positive")
        self.PROGRESS_SHAPES = (
            (self._n_walkers, len(self.tracked_pairs)),
            (self._n_walkers, 1),
            (self._n_walkers, 1),
            (self._n_walkers, 1),
        )
