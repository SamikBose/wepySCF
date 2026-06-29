# First Party Library
from wepy.resampling.distances.pyscf import (
    DihedralDistance,
    ProtonTransferDistance,
    QMGridDensityDistance,
)


@staticmethod
def qm_grid_density():
    return QMGridDensityDistance(grid_key="density_grid", normalize=True)


@staticmethod
def proton_transfer(break_pair: tuple[int, int], make_pair: tuple[int, int]):
    return ProtonTransferDistance(break_pair=break_pair, make_pair=make_pair)


@staticmethod
def dihedral(dihedrals):
    """Return a DihedralDistance for the given dihedral(s).

    Parameters
    ----------
    dihedrals : tuple or list of tuples
        A single 4-tuple of atom indices (i, j, k, l) or a list of such
        tuples for multi-dihedral sampling.

    Examples
    --------
    Single torsion (butane C0-C1-C2-C3):
        dihedral((0, 1, 2, 3))
    Two torsions (peptide phi/psi):
        dihedral([(4, 6, 8, 14), (6, 8, 14, 16)])
    """
    return DihedralDistance(dihedrals)
