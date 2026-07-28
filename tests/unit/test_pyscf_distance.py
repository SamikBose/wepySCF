"""Tests for PySCF-specific distance metrics."""

# Third Party
import numpy as np

# First Party
from wepy.resampling.distances.pyscf import (
    BondBreakMakeDistance,
    ProtonTransfer,
    ProtonTransferDistance,
    QMGridDensityDistance,
)


def test_qm_grid_density_distance_normalize() -> None:
    metric = QMGridDensityDistance(normalize=True)

    image_a = metric.image({"density_grid": np.array([1.0, -1.0, 2.0])})
    image_b = metric.image({"density_grid": np.array([2.0, -2.0, 4.0])})

    np.testing.assert_allclose(image_a, image_b)
    assert metric.image_distance(image_a, image_b) == 0.0


def test_bond_break_make_distance_image_and_distance() -> None:
    metric = BondBreakMakeDistance(break_pair=(0, 1), make_pair=(0, 2))

    state_a = {
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
    }
    state_b = {
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    }

    image_a = metric.image(state_a)
    image_b = metric.image(state_b)

    np.testing.assert_allclose(image_a, np.array([1.0, 2.0]))
    np.testing.assert_allclose(image_b, np.array([2.0, 1.0]))

    expected = np.sqrt(np.mean((np.array([1.0, 2.0]) - np.array([2.0, 1.0])) ** 2))
    assert metric.image_distance(image_a, image_b) == expected


def test_proton_transfer_distance_image_and_alias() -> None:
    metric = ProtonTransferDistance(break_pair=(0, 1), make_pair=(0, 2))
    alias_metric = ProtonTransfer(break_pair=(0, 1), make_pair=(0, 2))

    state = {
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    }

    image = metric.image(state)
    alias_image = alias_metric.image(state)

    np.testing.assert_allclose(image, np.array([1.0]))
    np.testing.assert_allclose(alias_image, image)

    image_other = np.array([-0.5])
    assert metric.image_distance(image, image_other) == 1.5
    assert alias_metric.image_distance(image, image_other) == 1.5
