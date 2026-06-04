"""Reporter helpers for PySCF based simulations."""

# Third Party Library
import numpy as np

# First Party Library
from wepy.reporter.dashboard import RunnerDashboardSection
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.runners.pyscf import UNIT_NAMES


class PySCFRunnerDashboardSection(RunnerDashboardSection):
    RUNNER_SECTION_TEMPLATE = """
Runner: {{ name }}

Backend: {{ backend }}
Integrator: {{ integrator }}
dt (a.u.): {{ dt }}
Temperature (K): {{ temperature_kelvin }}
Average Potential (Hartree): {{ avg_potential }}
Average Kinetic (Hartree): {{ avg_kinetic }}
"""

    def __init__(self, runner=None, backend="cpu", integrator=None, dt=None, temperature_kelvin=None, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "PySCFRunner"

        super().__init__(runner=runner, **kwargs)

        if runner is None:
            self.backend = backend
            self.integrator = integrator
            self.dt = dt
            self.temperature_kelvin = temperature_kelvin
        else:
            self.backend = getattr(runner, "backend", backend)
            integrator_cls = getattr(runner, "integrator_cls", None)
            self.integrator = getattr(integrator_cls, "__name__", None)
            self.dt = getattr(runner, "dt", None)
            self.temperature_kelvin = getattr(runner, "temperature_kelvin", None)

        self._potentials = []
        self._kinetics = []

    def update_values(self, **kwargs):
        potentials = []
        kinetics = []

        for walker in kwargs.get("new_walkers", []):
            state_d = walker.state.dict()

            pot = state_d.get("potential", None)
            if pot is not None:
                pot_val = float(np.asarray(pot).ravel()[0])
                if np.isfinite(pot_val):
                    potentials.append(pot_val)

            kin = state_d.get("kinetic", None)
            if kin is not None:
                kin_val = float(np.asarray(kin).ravel()[0])
                if np.isfinite(kin_val):
                    kinetics.append(kin_val)

        if potentials:
            self._potentials.extend(potentials)
        if kinetics:
            self._kinetics.extend(kinetics)

    def gen_fields(self, **kwargs):
        fields = super().gen_fields(**kwargs)

        avg_potential = float(np.mean(self._potentials)) if self._potentials else np.nan
        avg_kinetic = float(np.mean(self._kinetics)) if self._kinetics else np.nan

        fields.update(
            {
                "backend": self.backend,
                "integrator": self.integrator,
                "dt": self.dt,
                "temperature_kelvin": self.temperature_kelvin,
                "avg_potential": avg_potential,
                "avg_kinetic": avg_kinetic,
            }
        )

        return fields


class PySCFHDF5Reporter(WepyHDF5Reporter):
    """HDF5 reporter preconfigured for PySCF MD walker state fields."""

    DEFAULT_SAVE_FIELDS = (
        "positions",
        "velocities",
        "accelerations",
        "potential",
        "kinetic",
    )

    def __init__(
        self,
        save_fields=None,
        units=None,
        wepy_hdf5_path=None,
        file_paths=None,
        **kwargs,
    ):
        if save_fields is None:
            save_fields = self.DEFAULT_SAVE_FIELDS

        if units is None:
            units = dict(UNIT_NAMES)

        # Work around explicit-path handling in FileReporter by always
        # normalizing to file_paths for this single-file reporter.
        if file_paths is None and wepy_hdf5_path is not None:
            file_paths = [wepy_hdf5_path]

        super().__init__(
            save_fields=save_fields,
            units=units,
            file_paths=file_paths,
            **kwargs,
        )
