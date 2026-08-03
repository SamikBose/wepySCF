# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import uuid

# First Party Library
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFState, PySCFWalker


class PySCFREVOResampler(REVOResampler):
    """Regenerate walker IDs on resample to avoid scanner-cache collisions between unrelated trajectories."""

    def resample(self, walkers):
        """Resample walkers using REVO and regenerate duplicate IDs."""
        resampled_walkers, resampling_data, resampler_data = super().resample(walkers)

        seen_ids = set()
        fixed = []
        for walker in resampled_walkers:
            walker_id = walker.state.get("walker_id")
            if walker_id in seen_ids:
                # If duplicate ID, give fresh
                new_state = PySCFState(**{**walker.state._data, "walker_id": str(uuid.uuid4())})  # noqa: SLF001
                fixed.append(PySCFWalker(new_state, walker.weight))
            else:
                seen_ids.add(walker_id)
                fixed.append(PySCFWalker(walker.state, walker.weight))

        return fixed, resampling_data, resampler_data
