"""Independent angular patches for explicit geometry-stage controls."""

# ruff: noqa: INP001 - standalone benchmark fixture
from __future__ import annotations

import numpy as np

from elisa.experimental.curved_geometry import split_triangles


def triangles(level: int = 0) -> np.ndarray:
    eye = np.eye(3)
    result = np.array(
        [[sx * eye[0], sy * eye[1], sz * eye[2]] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    )
    for _ in range(level):
        result = split_triangles(result)
        result /= np.linalg.norm(result, axis=-1)[..., None]
    return result
