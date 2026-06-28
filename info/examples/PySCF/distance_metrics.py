# First Party Library
from wepy.resampling.distances.pyscf import ProtonTransferDistance, QMGridDensityDistance


@staticmethod
def qm_grid_density():
    return QMGridDensityDistance(grid_key="density_grid", normalize=True)


@staticmethod
def proton_transfer(break_pair: tuple[int, int], make_pair: tuple[int, int]):
    return ProtonTransferDistance(break_pair=break_pair, make_pair=make_pair)
