from __future__ import annotations

import numpy as np
from scipy.constants import mu_0

from sources.circular_loop import CircularLoop
from fields.biot_savart import biot_savart_field
from analysis.validation import analytical_bz_on_axis


def test_biot_savart_field_shape():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=200)
    obs = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    B = biot_savart_field(
        observation_points=obs,
        source_points=loop.midpoints(),
        segment_vectors=loop.segment_vectors(),
        current=loop.current,
    )
    assert B.shape == (2, 3)


def test_on_axis_field_matches_analytical_center():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=400)
    obs = np.array([[0.0, 0.0, 0.0]])
    B = biot_savart_field(
        observation_points=obs,
        source_points=loop.midpoints(),
        segment_vectors=loop.segment_vectors(),
        current=loop.current,
    )
    Bz_num = B[0, 2]
    Bz_an = analytical_bz_on_axis(loop.radius, loop.current, np.array([0.0]))[0]
    rel_err = abs(Bz_num - Bz_an) / abs(Bz_an)
    assert rel_err < 1e-2