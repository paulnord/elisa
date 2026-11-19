from types import SimpleNamespace

import numpy as np

from elisa.experimental.primary_cubic import CUBIC_NODES
from scripts.weight_projected_heat import disk_weights


def test_projected_heat_response_against_cubic_intensity():
    tri = np.eye(3)
    field = SimpleNamespace(vertex=np.zeros((10, 1, 2)), triangles=np.stack((tri, -tri)),
                            arrays=(None, np.ones(2)), faces=np.tile(np.arange(10), (2, 1)))
    # The two angular charts share the same optical coefficients by mirror symmetry.
    bary = np.array([[0.22,0.51,0.27],[0.1,0.65,0.25],[0.7,0.2,0.1],[1/3,1/3,1/3]])
    parents = np.array([0, 1, 0, 1])
    rays = (1-2*parents[:,None]) * bary
    mu = np.array([0.05, 0.24, 0.91, 0.0])
    area = np.array([0.9, 0.3, 0.1, 0.4])

    def intensity_terms(coords):
        x, y, z = coords.T
        return np.column_stack((2 + x*x*y - 0.4*y*z, 0.6 + 0.2*x*z*z))

    actual = (disk_weights(field, parents, rays, mu, area) * intensity_terms(CUBIC_NODES)).sum(axis=0)
    direct = intensity_terms(bary)
    expected = [(area*mu) @ direct[:,0], (area*mu*(mu-1)) @ direct[:,1]]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-16)
