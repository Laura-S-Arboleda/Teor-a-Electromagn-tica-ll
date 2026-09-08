from __future__ import annotations

import numpy as np

from sources.circular_loop import CircularLoop


def test_loop_points_shape():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=100)
    pts = loop.points()
    assert pts.shape == (100, 3)


def test_loop_closure_error_small():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=100)
    err = loop.closure_error()
    assert err < 1e-12


def test_radial_deviation_small():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=100)
    dev = loop.radial_deviation()
    assert np.max(dev) < 1e-12