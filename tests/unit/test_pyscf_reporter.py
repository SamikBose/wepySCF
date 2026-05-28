"""Tests for PySCF-specific reporters."""

# Third Party
import numpy as np
import pytest

pytest.importorskip("mdtraj")

# First Party
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.runners.pyscf import PySCFState, PySCFWalker


# FIXME: Not updated for new runner
def test_dashboard_section_tracks_energy():
    section = PySCFRunnerDashboardSection(
        step_size=1e-3,
        backend="cpu",
        dynamics_mode="langevin",
        temperature_kelvin=300.0,
    )

    walkers = [
        PySCFWalker(
            PySCFState(
                symbols=["H"],
                positions=np.zeros((1, 3)),
                energy=np.array([-1.0]),
            ),
            1.0,
        )
    ]

    section.update_values(new_walkers=walkers)
    fields = section.gen_fields()

    assert fields["avg_energy"] == -1.0
    assert fields["backend"] == "cpu"
    assert fields["dynamics_mode"] == "langevin"
    assert fields["temperature_kelvin"] == 300.0


def test_hdf5_reporter_defaults():
    reporter = PySCFHDF5Reporter(wepy_hdf5_path="tmp.wepy.h5", topology="{}")

    assert "density_matrix" in reporter.save_fields
    assert "positions_unit" in reporter.units
