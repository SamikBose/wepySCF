# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import uuid

# First Party Library
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.resampling.resamplers.wexplore import WExploreResampler
from wepy.runners.pyscf import PySCFState, PySCFWalker


class _RegenerateWalkerIdMixin:
    """Regenerate duplicate walker IDs on resample to avoid scanner-cache collisions between unrelated trajectories.

    Use this as the first mixin in the inheritance list to override the resample method to work with the PySCF runner.
    """

    def resample(self, walkers):
        """Resample walkers and regenerate any duplicate IDs, preserving weight."""
        resampled_walkers, resampling_data, resampler_data = super().resample(walkers)

        seen_ids = set()
        fixed = []
        for walker in resampled_walkers:
            walker_id = walker.state.get("walker_id")
            if walker_id in seen_ids:
                # If duplicate ID, give fresh one
                new_state = PySCFState(**{**walker.state._data, "walker_id": str(uuid.uuid4())})  # noqa: SLF001
                fixed.append(PySCFWalker(new_state, walker.weight))
            else:
                seen_ids.add(walker_id)
                fixed.append(PySCFWalker(walker.state, walker.weight))

        return fixed, resampling_data, resampler_data


class PySCFREVOResampler(_RegenerateWalkerIdMixin, REVOResampler):
    pass


# TODO: (W)Explore this further...
# class WExploreResampler(_RegenerateWalkerIdMixin, WExploreResampler):
#     pass
