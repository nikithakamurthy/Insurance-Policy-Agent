import unittest

import cv2
import numpy as np

from pipeline.frames import _thumb, is_duplicate

RNG = np.random.default_rng(3)
# A textured "scene" wider than one frame, so crops at different x positions simulate a pan.
SCENE = cv2.GaussianBlur((RNG.random((480, 1400, 3)) * 255).astype(np.uint8), (0, 0), 3)


def view(x: int, y: int = 0, w: int = 360, h: int = 480) -> np.ndarray:
    return SCENE[y:y + h, x:x + w].copy()


class DuplicateFrameTest(unittest.TestCase):
    def test_same_view_with_sensor_noise_is_duplicate(self):
        a = view(200)
        b = np.clip(a.astype(int) + RNG.normal(0, 3, a.shape), 0, 255).astype(np.uint8)
        self.assertTrue(is_duplicate(_thumb(a), _thumb(b)))

    def test_tiny_hand_shake_is_duplicate(self):
        self.assertTrue(is_duplicate(_thumb(view(200)), _thumb(view(204))))

    def test_pan_is_not_duplicate(self):
        self.assertFalse(is_duplicate(_thumb(view(200)), _thumb(view(250))))

    def test_stepping_closer_is_not_duplicate(self):
        a = view(200)
        zoomed = cv2.resize(a[60:420, 45:315], (360, 480))  # same centre, 1.33x closer
        self.assertFalse(is_duplicate(_thumb(a), _thumb(zoomed)))

    def test_new_object_on_blank_wall_is_not_duplicate(self):
        wall = np.full((480, 360, 3), 235, np.uint8)
        with_frame = wall.copy()
        cv2.rectangle(with_frame, (100, 120), (260, 300), (40, 40, 160), -1)
        self.assertFalse(is_duplicate(_thumb(wall), _thumb(with_frame)))


if __name__ == "__main__":
    unittest.main()
