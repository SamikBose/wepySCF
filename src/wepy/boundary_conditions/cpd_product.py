"""Boundary conditions for competing cyclopentadiene product channels."""

import numpy as np

from wepy.boundary_conditions.bond_distance import BondDistanceBC


class CPDProductBC(BondDistanceBC):
    """Warp upon forming either CPD product channel.

    The three ``make_pairs`` must be ordered as ``(shared, competing_A,
    competing_B)``.  A walker reaches product A when the shared and A bonds
    are within their cutoffs, and product B when the shared and B bonds are
    within their cutoffs. Optional break-pair conditions are combined with
    either product channel, so all configured break bonds must also exceed
    their cutoffs.

    ``product_channel`` is a bit mask: 0 means no product, 1 means A, 2 means
    B, and 3 means both product definitions are satisfied.
    """

    PROGRESS_FIELDS = ("break_distances", "make_distances", "product_channel")
    PROGRESS_SHAPES = (Ellipsis, Ellipsis, (1,))
    PROGRESS_DTYPES = (float, float, int)
    PROGRESS_RECORD_FIELDS = PROGRESS_FIELDS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.make_pairs) != 3:
            raise ValueError(
                "CPDProductBC requires make_pairs=(shared, competing_A, "
                "competing_B)"
            )

    def _progress(self, walker):
        break_distances, make_distances = self._calc_distances(walker)

        break_satisfied = bool(
            len(self.break_pairs) == 0
            or np.all(break_distances >= self.break_cutoffs)
        )
        formed = np.asarray(make_distances <= self.make_cutoffs, dtype=bool)
        shared_formed, channel_a_formed, channel_b_formed = formed

        channel = 0
        if break_satisfied and shared_formed and channel_a_formed:
            channel |= 1
        if break_satisfied and shared_formed and channel_b_formed:
            channel |= 2

        progress_data = {
            "break_distances": np.asarray(break_distances, dtype=float),
            "make_distances": np.asarray(make_distances, dtype=float),
            "product_channel": np.asarray([channel], dtype=int),
        }
        return channel != 0, progress_data
