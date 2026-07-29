"""Unit tests for the PySCF runner stack using mocked pyscf modules."""

# Standard Library
from types import SimpleNamespace

# Third Party Library
import numpy as np
import pytest

# First Party Library
from wepy.runners.pyscf import (
    PySCFCPUWalkerTaskProcess,
    PySCFCPUWorker,
    PySCFGPUWalkerTaskProcess,
    PySCFGPUWorker,
    PySCFCPUTaskMapper,
    PySCFGPUTaskMapper,
    PySCFRunner,
    PySCFState,
    PySCFWalker,
)
from wepy.walker import Walker, WalkerState


class _FakeNumInt:
    @staticmethod
    def eval_ao(_mol, grid_coords):
        return np.ones((len(grid_coords), 1), dtype=float)

    @staticmethod
    def eval_rho(_mol, _ao_values, _density_matrix):
        return np.ones((_ao_values.shape[0],), dtype=float)


class _FakeGradients:
    def __init__(self, gradients, energy, mf):
        self._gradients = gradients
        self._energy = energy
        self._mf = mf

    def kernel(self):
        return self._gradients

    def as_scanner(self):
        mf = self._mf

        def _scanner(_mol):
            if getattr(mf, "_on_gpu", False) and getattr(mf, "_gpu_runtime_fail", False):
                raise RuntimeError("failed in block_diag kernel")
            return self._energy, self._gradients

        _scanner.base = mf
        return _scanner


class _FakeMF:
    def __init__(self, gradients, energy, supports_gpu=True, missing_cupy=False, gpu_runtime_fail=False):
        self._gradients = gradients
        self._energy = energy
        self._supports_gpu = supports_gpu
        self._missing_cupy = missing_cupy
        self._gpu_runtime_fail = gpu_runtime_fail
        self.xc = None
        self._gpu_runtime_fail = gpu_runtime_fail
        self._on_gpu = False
        self.e_tot = energy

    def kernel(self):
        return self._energy

    def nuc_grad_method(self):
        return _FakeGradients(self._gradients, self._energy, self)

    def make_rdm1(self):
        return np.eye(1, dtype=float)

    def to_gpu(self):
        if self._missing_cupy:
            raise ModuleNotFoundError("No module named cupy", name="cupy")
        if not self._supports_gpu:
            raise AttributeError("no GPU support")
        self._on_gpu = True
        return self




class _FakePostHF:
    def __init__(self, mf):
        self._mf = mf
        self.e_tot = mf._energy - 0.01

    def kernel(self):
        return self.e_tot

    def nuc_grad_method(self):
        return _FakeGradients(self._mf._gradients, self.e_tot, self)

    def make_rdm1(self):
        return np.eye(1, dtype=float)


class _FakeMP2(_FakePostHF):
    def density_fit(self):
        return self


class _FakeCCSD(_FakePostHF):
    pass


class _FakeModuleFactory:
    def __init__(self, gradients, energy, supports_gpu=True, missing_cupy=False, gpu_runtime_fail=False):
        self._gradients = gradients
        self._energy = energy
        self._supports_gpu = supports_gpu
        self._missing_cupy = missing_cupy
        self._gpu_runtime_fail = gpu_runtime_fail

    def gto(self):
        return SimpleNamespace(M=lambda **kwargs: kwargs)

    def scf(self):
        return SimpleNamespace(
            RHF=lambda mol: _FakeMF(self._gradients, self._energy, self._supports_gpu, self._missing_cupy, self._gpu_runtime_fail),
            UHF=lambda mol: _FakeMF(self._gradients, self._energy, self._supports_gpu, self._missing_cupy, self._gpu_runtime_fail),
        )

    def dft(self):
        return SimpleNamespace(
            RKS=lambda mol: _FakeMF(self._gradients, self._energy, self._supports_gpu, self._missing_cupy, self._gpu_runtime_fail),
            UKS=lambda mol: _FakeMF(self._gradients, self._energy, self._supports_gpu, self._missing_cupy, self._gpu_runtime_fail),
        )

    def mp(self):
        return SimpleNamespace(MP2=lambda mf: _FakeMP2(mf))

    def cc(self):
        return SimpleNamespace(CCSD=lambda mf: _FakeCCSD(mf))

    def numint(self):
        return _FakeNumInt()


def _patch_imports(monkeypatch, factory):
    def _fake_import(name):
        if name == "pyscf.gto":
            return factory.gto()
        if name == "pyscf.scf":
            return factory.scf()
        if name == "pyscf.dft":
            return factory.dft()
        if name == "pyscf.dft.numint":
            return factory.numint()
        if name == "pyscf.mp":
            return factory.mp()
        if name == "pyscf.cc":
            return factory.cc()
        raise ValueError(name)

    monkeypatch.setattr("wepy.runners.pyscf.importlib.import_module", _fake_import)


def test_run_segment_updates_positions_and_quantum_fields(monkeypatch):
    gradients = np.array([[1.0, -2.0, 0.5]])
    energy = -1.23
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=energy))

    runner = PySCFRunner(step_size=0.1, density_grid_shape=(2, 2, 2))
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        0.5,
    )

    new_walker = runner.run_segment(walker, 2)

    np.testing.assert_allclose(new_walker.state["positions"], np.array([[-0.2, 0.4, -0.1]]))
    assert float(new_walker.state["energy"][0]) == energy
    np.testing.assert_allclose(new_walker.state["gradients"], gradients)
    assert int(new_walker.state["segment_step_idx"][0]) == 2
    assert new_walker.state["density_matrix"].shape == (1, 1)
    assert new_walker.state["density_grid"].shape == (2, 2, 2)
    assert new_walker.weight == walker.weight


def test_runner_pre_cycle_backend_override(monkeypatch):
    gradients = np.array([[0.0, 0.0, 0.0]])
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=-0.5))

    runner = PySCFRunner(step_size=0.1)
    runner.pre_cycle(backend="gpu", platform_kwargs={"DeviceIndex": "0"})

    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )
    new_walker = runner.run_segment(walker, 1)
    assert int(new_walker.state["segment_step_idx"][0]) == 1

    runner.post_cycle()
    assert runner._cycle_backend is None
    assert runner._cycle_platform_kwargs is None


def test_rks_requires_xc(monkeypatch):
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=np.zeros((1, 3)), energy=0.0))

    runner = PySCFRunner(method="RKS")
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    with pytest.raises(ValueError, match="requires an xc functional"):
        runner.run_segment(walker, 1)


def test_zero_segment_length_has_zero_step_index(monkeypatch):
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=np.zeros((1, 3)), energy=0.0))

    runner = PySCFRunner()
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    new_walker = runner.run_segment(walker, 0)

    assert int(new_walker.state["segment_step_idx"][0]) == 0


def test_pyscf_state_and_walker():
    state = PySCFState(symbols=["He"], positions=np.array([[0.0, 0.0, 0.0]]))
    walker = PySCFWalker(state, 1.0)

    assert walker.state["symbols"] == ["He"]

    with pytest.raises(AssertionError):
        PySCFWalker(WalkerState(symbols=["He"], positions=np.array([[0.0, 0.0, 0.0]])), 1.0)


def test_gpu_backend_requires_to_gpu(monkeypatch):
    _patch_imports(
        monkeypatch,
        _FakeModuleFactory(gradients=np.zeros((1, 3)), energy=0.0, supports_gpu=False),
    )

    runner = PySCFRunner(backend="gpu")
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    with pytest.raises(RuntimeError, match="does not support to_gpu"):
        runner.run_segment(walker, 1)




def test_gpu_backend_missing_cupy_gives_actionable_error(monkeypatch):
    _patch_imports(
        monkeypatch,
        _FakeModuleFactory(
            gradients=np.zeros((1, 3)),
            energy=0.0,
            supports_gpu=True,
            missing_cupy=True,
        ),
    )

    runner = PySCFRunner(backend="gpu")
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    with pytest.raises(RuntimeError, match="CuPy"):
        runner.run_segment(walker, 1)




def test_gpu_runtime_error_can_fallback_to_cpu(monkeypatch):
    _patch_imports(
        monkeypatch,
        _FakeModuleFactory(
            gradients=np.zeros((1, 3)),
            energy=-0.1,
            supports_gpu=True,
            gpu_runtime_fail=True,
        ),
    )

    runner = PySCFRunner(backend="gpu", gpu_fallback_cpu_on_error=True)
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    new_walker = runner.run_segment(walker, 1)
    assert float(new_walker.state["energy"][0]) == -0.1


def test_gpu_runtime_error_without_fallback_raises(monkeypatch):
    _patch_imports(
        monkeypatch,
        _FakeModuleFactory(
            gradients=np.zeros((1, 3)),
            energy=-0.1,
            supports_gpu=True,
            gpu_runtime_fail=True,
        ),
    )

    runner = PySCFRunner(backend="gpu", gpu_fallback_cpu_on_error=False)
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    with pytest.raises(RuntimeError, match="failed in block_diag kernel"):
        runner.run_segment(walker, 1)


def test_worker_and_task_process_inject_hardware_kwargs():
    cpu_worker = object.__new__(PySCFCPUWorker)
    cpu_worker._attributes = {"num_threads": 4}
    cpu_worker._mapper_attributes = {}
    cpu_worker._worker_idx = 0

    gpu_worker = object.__new__(PySCFGPUWorker)
    gpu_worker._attributes = {}
    gpu_worker._mapper_attributes = {"device_ids": [2, 3]}
    gpu_worker._worker_idx = 1

    cpu_task = object.__new__(PySCFCPUWalkerTaskProcess)
    cpu_task.mapper_attributes = {"num_threads": 8}
    cpu_task._worker_idx = 0

    gpu_task = object.__new__(PySCFGPUWalkerTaskProcess)
    gpu_task.mapper_attributes = {"device_ids": [5, 6]}
    gpu_task._worker_idx = 0

    def echo_kwargs(**kwargs):
        return kwargs

    assert cpu_worker.run_task(echo_kwargs) == {
        "backend": "cpu",
        "platform_kwargs": {"Threads": "4"},
    }
    assert gpu_worker.run_task(echo_kwargs) == {
        "backend": "gpu",
        "platform_kwargs": {"DeviceIndex": "3"},
    }
    assert cpu_task.run_task(echo_kwargs) == {
        "backend": "cpu",
        "platform_kwargs": {"Threads": "8"},
    }
    assert gpu_task.run_task(echo_kwargs) == {
        "backend": "gpu",
        "platform_kwargs": {"DeviceIndex": "5"},
    }


def test_run_segment_without_scanner(monkeypatch):
    gradients = np.array([[0.5, 0.0, -0.5]])
    energy = -2.5
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=energy))

    runner = PySCFRunner(step_size=0.2, use_scf_scanner=False)
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    new_walker = runner.run_segment(walker, 1)

    np.testing.assert_allclose(new_walker.state["positions"], np.array([[-0.1, 0.0, 0.1]]))
    assert float(new_walker.state["energy"][0]) == energy


def test_task_mapper_convenience_classes():
    cpu_mapper = PySCFCPUTaskMapper(num_workers=2, num_threads=3)
    assert cpu_mapper.walker_task_type.__name__ == "PySCFCPUWalkerTaskProcess"
    assert cpu_mapper.num_workers == 2
    assert cpu_mapper._attributes["num_threads"] == 3

    gpu_mapper = PySCFGPUTaskMapper(num_workers=2, platform="CUDA", device_ids=[0, 1])
    assert gpu_mapper.walker_task_type.__name__ == "PySCFGPUWalkerTaskProcess"
    assert gpu_mapper.num_workers == 2
    assert gpu_mapper._attributes["platform"] == "CUDA"
    assert gpu_mapper._attributes["device_ids"] == [0, 1]

    try:
        PySCFGPUTaskMapper(num_workers=1)
    except ValueError as exc:
        assert "device_ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no device_ids are provided")




def test_steepest_descent_remains_deterministic(monkeypatch):
    gradients = np.array([[1.0, -2.0, 0.5]])
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=-1.0))

    runner = PySCFRunner(step_size=0.1, dynamics_mode="steepest_descent", temperature_kelvin=1200.0)
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    new_walker = runner.run_segment(walker, 1)
    np.testing.assert_allclose(new_walker.state["positions"], np.array([[-0.1, 0.2, -0.05]]))

def test_langevin_mode_is_stochastic_and_seeded(monkeypatch):
    gradients = np.array([[0.0, 0.0, 0.0]])
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=-1.0))

    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    runner_a = PySCFRunner(
        step_size=0.1,
        dynamics_mode="langevin",
        temperature_kelvin=300.0,
        random_seed=7,
    )
    runner_b = PySCFRunner(
        step_size=0.1,
        dynamics_mode="langevin",
        temperature_kelvin=300.0,
        random_seed=7,
    )

    new_walker_a = runner_a.run_segment(walker, 1)
    new_walker_b = runner_b.run_segment(walker, 1)

    np.testing.assert_allclose(new_walker_a.state["positions"], new_walker_b.state["positions"])
    assert not np.allclose(new_walker_a.state["positions"], np.zeros((1, 3)))


def test_invalid_dynamics_mode_raises():
    with pytest.raises(ValueError, match="Unsupported PySCF dynamics mode"):
        PySCFRunner(dynamics_mode="not-a-mode")


def test_mp2_method_runs_with_reference_hf(monkeypatch):
    gradients = np.array([[0.1, 0.0, 0.0]])
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=-1.0))

    runner = PySCFRunner(method="MP2", step_size=0.1)
    walker = Walker(WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])), 1.0)

    new_walker = runner.run_segment(walker, 1)
    assert float(new_walker.state["energy"][0]) < -1.0


def test_ccsd_method_runs_with_reference_hf(monkeypatch):
    gradients = np.array([[0.1, 0.0, 0.0]])
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=-1.0))

    runner = PySCFRunner(method="CCSD", step_size=0.1)
    walker = Walker(WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])), 1.0)

    new_walker = runner.run_segment(walker, 1)
    assert float(new_walker.state["energy"][0]) < -1.0


def test_dfmp2_method_runs(monkeypatch):
    gradients = np.array([[0.1, 0.0, 0.0]])
    _patch_imports(monkeypatch, _FakeModuleFactory(gradients=gradients, energy=-1.0))

    runner = PySCFRunner(method="DFMP2", step_size=0.1)
    walker = Walker(WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])), 1.0)

    new_walker = runner.run_segment(walker, 1)
    assert float(new_walker.state["energy"][0]) < -1.0
