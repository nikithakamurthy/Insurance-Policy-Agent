import unittest

import numpy as np

from pipeline.reconstruct import estimate_scale
from pipeline.room import measure_room
from pipeline.spines import is_spine_frame

RNG = np.random.default_rng(1)


def _plane(origin, u, v, nu, nv, noise=0.005):
    a, b = RNG.random((nu * nv, 1)), RNG.random((nu * nv, 1))
    pts = np.asarray(origin) + a * np.asarray(u) + b * np.asarray(v)
    return pts + RNG.normal(0, noise, pts.shape)


def box_room(L, W, H, density=400):
    n = lambda area: int(area * density)
    return np.concatenate([
        _plane([0, 0, 0], [L, 0, 0], [0, W, 0], 1, n(L * W)),        # floor (z up)
        _plane([0, 0, H], [L, 0, 0], [0, W, 0], 1, n(L * W)),        # ceiling
        _plane([0, 0, 0], [L, 0, 0], [0, 0, H], 1, n(L * H)),        # walls
        _plane([0, W, 0], [L, 0, 0], [0, 0, H], 1, n(L * H)),
        _plane([0, 0, 0], [0, W, 0], [0, 0, H], 1, n(W * H)),
        _plane([L, 0, 0], [0, W, 0], [0, 0, H], 1, n(W * H)),
    ])


def rotate(points, yaw_deg, tilt_deg):
    y, t = np.radians(yaw_deg), np.radians(tilt_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    r = rx @ rz
    return points @ r.T, r


class RoomGeometryTest(unittest.TestCase):
    def test_rectangular_room_with_bookshelf_front(self):
        L, W, H = 5.0, 4.0, 2.7
        shelf = _plane([1.0, 0.35, 0], [2.0, 0, 0], [0, 0, 2.0], 1, 1600)  # tall decoy plane inside the room
        points, r = rotate(np.concatenate([box_room(L, W, H), shelf]), 33, 8)
        room = measure_room(points, up_hint=r @ np.array([0.1, 0.0, 1.0]))
        self.assertAlmostEqual(room["length_m"], L, delta=0.1)
        self.assertAlmostEqual(room["width_m"], W, delta=0.1)
        self.assertAlmostEqual(room["height_m"], H, delta=0.06)
        self.assertEqual(room["shape"], "rectangle")
        self.assertAlmostEqual(room["floor_area_m2"], L * W, delta=0.05 * L * W)
        self.assertAlmostEqual(room["wall_area_m2"], 2 * (L + W) * H, delta=0.05 * 2 * (L + W) * H)
        self.assertEqual(room["walls_found"], 4)

    def test_l_shaped_room_uses_polygon_area(self):
        H = 2.6
        # 6 x 5 room missing a 3 x 2.5 corner: true floor area 30 - 7.5 = 22.5 m2.
        d = 400
        parts = [
            _plane([0, 0, 0], [6, 0, 0], [0, 2.5, 0], 1, 6000), _plane([0, 2.5, 0], [3, 0, 0], [0, 2.5, 0], 1, 3000),
            _plane([0, 0, H], [6, 0, 0], [0, 2.5, 0], 1, 6000), _plane([0, 2.5, H], [3, 0, 0], [0, 2.5, 0], 1, 3000),
            _plane([0, 0, 0], [6, 0, 0], [0, 0, H], 1, int(6 * H * d)), _plane([0, 0, 0], [0, 5, 0], [0, 0, H], 1, int(5 * H * d)),
            _plane([6, 0, 0], [0, 2.5, 0], [0, 0, H], 1, int(2.5 * H * d)), _plane([3, 2.5, 0], [3, 0, 0], [0, 0, H], 1, int(3 * H * d)),
            _plane([3, 2.5, 0], [0, 2.5, 0], [0, 0, H], 1, int(2.5 * H * d)), _plane([0, 5, 0], [3, 0, 0], [0, 0, H], 1, int(3 * H * d)),
        ]
        room = measure_room(np.concatenate(parts), up_hint=np.array([0, 0, 1.0]))
        self.assertEqual(room["shape"], "polygon")
        self.assertAlmostEqual(room["floor_area_m2"], 22.5, delta=0.1 * 22.5)
        self.assertAlmostEqual(room["height_m"], H, delta=0.06)


class ScaleTest(unittest.TestCase):
    def test_scale_is_ratio_of_metric_to_relative_depth(self):
        rel = RNG.uniform(0.5, 3.0, (4, 60, 80))
        metric = rel * 1.8 * RNG.normal(1.0, 0.02, rel.shape)
        result = estimate_scale(rel, np.ones_like(rel), metric)
        self.assertAlmostEqual(result["scale"], 1.8, delta=0.02)
        self.assertLess(result["relative_iqr"], 0.02)

    def test_close_up_frames_are_left_out_of_scale(self):
        rel = RNG.uniform(1.0, 3.0, (8, 60, 80))
        metric = rel * 1.8
        metric[:2] = rel[:2] * 0.25   # two close-ups (< 1 m) with a wrong ratio
        result = estimate_scale(rel, np.ones_like(rel), metric)
        self.assertAlmostEqual(result["scale"], 1.8, delta=0.01)
        self.assertEqual(result["frames_excluded_close_up"], [0, 1])


class SpineFilterTest(unittest.TestCase):
    def test_spine_frame_rules(self):
        book = {"category": "book", "area_frac": 0.01}
        self.assertTrue(is_spine_frame([book] * 3))
        self.assertFalse(is_spine_frame([book, {"category": "chair", "area_frac": 0.2}]))
        self.assertTrue(is_spine_frame([book, {"category": "bookshelf", "area_frac": 0.3}]))


if __name__ == "__main__":
    unittest.main()
