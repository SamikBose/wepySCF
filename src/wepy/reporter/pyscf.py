"""Reporter helpers for PySCF based simulations."""

# Third Party Library
import numpy as np

# First Party Library
from wepy.reporter.dashboard import RunnerDashboardSection
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.runners.pyscf import UNIT_NAMES, PySCFRunner


class PySCFRunnerDashboardSection(RunnerDashboardSection):
    RUNNER_SECTION_TEMPLATE = """
Runner: {{ name }}

Backend: {{ backend }}
Integrator: {{ integrator }}
dt (a.u.): {{ dt }}
Target Temperature (K): {{ target_temperature }}
Average Potential (Hartree): {{ avg_potential }}
Average Kinetic (Hartree): {{ avg_kinetic }}
"""

    def __init__(self, runner: PySCFRunner, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "PySCFRunner"

        super().__init__(runner=runner, **kwargs)

        self.backend = runner.backend
        self.integrator_name = runner.integrator_cls.__name__
        self.dt = runner.dt
        self.target_temperature = runner._integrator_temperature_kelvin  # noqa: SLF001

        self._potentials = []
        self._kinetics = []

    def update_values(self, **kwargs):
        potentials = []
        kinetics = []

        for walker in kwargs.get("new_walkers", []):
            state = walker.state

            pot = state.get("potential")
            if pot is not None:
                pot_val = float(np.asarray(pot).ravel()[0])
                if np.isfinite(pot_val):
                    potentials.append(pot_val)

            kin = state.get("kinetic")
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
                "integrator": self.integrator_name,
                "dt": self.dt,
                "target_temperature": self.target_temperature,
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
        "temperature",
        "total_energy",
        "potential",
        "kinetic",
        "mo_energy",
        "charges",
    )

    def __init__(
        self,
        save_fields=None,
        units=None,
        **kwargs,
    ):
        if save_fields is None:
            save_fields = self.DEFAULT_SAVE_FIELDS

        if units is None:
            units = dict(UNIT_NAMES)

        super().__init__(
            save_fields=save_fields,
            units=units,
            **kwargs,
        )
