from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from elisa.experimental.area_reflection import _Surface
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.node_reemission import NodeReemissionReflection, emission_rows


def test_mirrors_use_physical_receiver_and_excluded_node_is_dark():
    points = np.array([[0.9, .1, .2], [.91, .2, .1], [1.1, .3, .15]])
    mirrored = points[[2, 0, 1]] * [1, -1, -1]
    surface = SimpleNamespace(receiver=points[:2], emitter=np.vstack((points, mirrored)),
                              parent=np.array([0, 0]), visible_base=np.zeros(6, int),
                              visible_area=np.array([1., 2, 3, 3, 1, 2]), fraction=np.array([1/6, 2/6]))
    rows = emission_rows(surface, 1.)
    heat = np.array([5., 7., 0.])
    np.testing.assert_array_equal(rows, [0, 1, 2, 2, 0, 1])
    weights = surface.visible_area.reshape(-1, 3)/6
    np.testing.assert_allclose((weights * heat[rows].reshape(-1, 3)).sum(axis=1), surface.fraction @ heat[:2])
    surface.emitter[3, 0] = .91
    with pytest.raises(UnsupportedReflectionError, match="physical receiver"):
        emission_rows(surface, 1.)
    surface.emitter[3, 0] = 1.1
    surface.visible_area[-1] += .1
    with pytest.raises(UnsupportedReflectionError, match="energy weights"):
        emission_rows(surface, 1.)


@pytest.mark.parametrize("secondary_first", [True, False])
def test_local_deposits_from_latest_half_iteration(secondary_first):
    solver = object.__new__(NodeReemissionReflection)
    solver.mode = "reference"
    solver._emission_rows = np.array([1, 2, 0])
    s = _Surface(*([None] * 7), visible_base=np.zeros(3, int), visible_area=np.ones(3),
                 prefix_ld=np.ones((1, 3)), intrinsic=np.array([99.]), absorption=np.ones(1))
    step = 0
    seen = []

    def sampled(receiver, donor, deposited):
        nonlocal step
        step += 1
        seen.append(deposited.copy())
        return np.array([step, 2*step]), np.array([1.5*step])

    solver._sampled_update = sampled
    order = ["secondary", "primary"] if secondary_first else ["primary", "secondary"]
    update = solver._sweep_update(order, deque(maxlen=2))
    heat = np.array([0.])
    for _ in range(4):
        heat = update(s, s, heat)
    expected = ([[0.], [2, 0, 1], [3.], [6, 0, 3]] if secondary_first else
                [[0, 0, 0], [1.5], [4, 0, 2], [4.5]])
    for actual, wanted in zip(seen, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
