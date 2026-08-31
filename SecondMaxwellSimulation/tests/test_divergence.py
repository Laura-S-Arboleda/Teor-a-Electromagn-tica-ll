from __future__ import annotations

import numpy as np

from sources.circular_loop import CircularLoop
from simulations.gauss_magnetism import compute_field_on_grid_3d
from numerical.divergence import divergence_from_grid


def test_divergence_output_shape():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=100)
    X, Y, Z, Bx, By, Bz, dx, dy, dz = compute_field_on_grid_3d(loop, extent=1.0, n=9)
    divB = divergence_from_grid(Bx, By, Bz, dx, dy, dz)
    assert divB.shape == Bx.shape


def test_divergence_not_nan():
    loop = CircularLoop(radius=1.0, current=1.0, num_segments=100)
    X, Y, Z, Bx, By, Bz, dx, dy, dz = compute_field_on_grid_3d(loop, extent=1.0, n=9)
    divB = divergence_from_grid(Bx, By, Bz, dx, dy, dz)
    assert not np.isnan(divB).any()