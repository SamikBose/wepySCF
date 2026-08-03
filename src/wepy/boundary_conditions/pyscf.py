# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import uuid

# First Party Library
from wepy.boundary_conditions.bond_distance import BondDistanceBC
from wepy.runners.pyscf import PySCFState


class PySCFBondDistanceBC(BondDistanceBC):
    """Regenerate walker IDs on warp to avoid scanner-cache collisions between unrelated trajectories."""

    def _warp(self, walker):
        """Warp walker and regenerate walker ID."""
        warped_walker, warp_data = super()._warp(walker)

        new_state = PySCFState(**{**warped_walker.state._data, "walker_id": str(uuid.uuid4())})  # noqa: SLF001
        warped_walker = type(warped_walker)(new_state, warped_walker.weight)

        return warped_walker, warp_data
