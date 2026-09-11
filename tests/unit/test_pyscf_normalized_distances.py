import numpy as np

from wepy.resampling.distances.pyscf import (
    NormalizedBondAngleChargeDistance,
    NormalizedBondAngleDistance,
)


def test_normalized_bond_angle_sn2_directionality():
    metric = NormalizedBondAngleDistance(
        (0, 1), (0, 2), (2, 0, 1), angle_mode="alignment"
    )
    backside = np.array([[0, 0, 0], [0, 0, 3.5], [0, 0, -4.5]], dtype=float)
    frontside = backside.copy()
    frontside[2, 2] = 4.5
    image_back = metric.image({"positions": backside})
    image_front = metric.image({"positions": frontside})
    assert 0.0 <= image_back[0] <= 1.0
    assert np.isclose(image_back[1], 1.0)
    assert np.isclose(image_front[1], 0.0)


def test_normalized_bond_angle_charge_endpoints_and_nan_placeholder():
    reactant_q = np.array([0.2, -0.3, 0.1])
    product_q = np.array([0.1, 0.2, -0.3])
    metric = NormalizedBondAngleChargeDistance(
        (0, 1),
        (0, 2),
        (2, 0, 1),
        reactant_q,
        product_q,
        angle_mode="alignment",
        allow_initial_nan_charges=True,
    )
    positions = np.array([[0, 0, 0], [0, 0, 3.5], [0, 0, -4.5]], dtype=float)
    initial = metric.image({"positions": positions, "charges": np.array([np.nan])})
    product = metric.image({"positions": positions, "charges": product_q})
    assert np.isclose(initial[2], 0.0)
    assert np.isclose(product[2], 1.0)
    assert metric.image_distance(initial, product) > 0.0
