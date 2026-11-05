import unittest

import cv2
import numpy as np
from PIL import Image

from pipeline.room_corner import measure_frame, plain_paper, sheet_length_px, summarise

A4 = 0.297
W, H, F = 1280, 960, 900.0      # image size and focal length (px), about a phone camera


def synthetic_view(wall_distance: float, yaw_deg: float, camera_height: float = 1.3, sheet_at=(0.0, 0.0)):
    """Depth image of a flat floor and one wall, seen from a level camera turned yaw_deg away from the
    wall's normal; the camera is wall_distance from the wall. Returns depth, the sheet's mask and its
    long side in pixels (A4 upright on the wall, centred at sheet_at = (along the wall, height above camera))."""
    yaw = np.radians(yaw_deg)
    n = np.array([np.sin(yaw), 0.0, np.cos(yaw)])               # wall normal in camera coordinates (x right, y down, z fwd)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u, dtype=float)], axis=-1)
    t_wall = wall_distance / np.clip(rays @ n, 1e-6, None)
    t_floor = np.where(rays[..., 1] > 1e-6, camera_height / np.clip(rays[..., 1], 1e-6, None), np.inf)
    t = np.minimum(np.where(rays @ n > 1e-6, t_wall, np.inf), t_floor)
    depth = t * rays[..., 2]                                       # depth = z of the hit point
    # The sheet: a 0.210 x 0.297 m rectangle on the wall, drawn by projecting its corners.
    along = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    centre = n * wall_distance + along * sheet_at[0] + np.array([0, -sheet_at[1], 0])
    corners = [centre + along * dx + np.array([0, dy, 0]) for dx, dy in ((-0.105, -0.1485), (0.105, -0.1485), (0.105, 0.1485), (-0.105, 0.1485))]
    px = np.array([[W / 2 + F * c[0] / c[2], H / 2 + F * c[1] / c[2]] for c in corners], dtype=np.float32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
    long_px = (np.linalg.norm(px[1] - px[2]) + np.linalg.norm(px[0] - px[3])) / 2
    image = np.full((H, W, 3), 120, np.uint8)
    image[mask.astype(bool)] = 245
    return Image.fromarray(image), depth.astype(np.float32), mask.astype(bool), float(long_px)


class GeometryTest(unittest.TestCase):
    def test_distance_to_a_wall_faced_straight_on(self):
        image, depth, mask, long_px = synthetic_view(4.2, 0)
        m = measure_frame(image, mask, long_px, depth, F, A4)
        self.assertAlmostEqual(m["wall_distance_m"], 4.2, delta=0.05)

    def test_distance_is_perpendicular_whatever_the_viewing_angle(self):
        # Turned 35 degrees away from the wall: the sheet is further away along the view, but the
        # wall's perpendicular distance, the room dimension, stays 3.1 m.
        image, depth, mask, long_px = synthetic_view(3.1, 35, sheet_at=(-1.5, 0.0))
        m = measure_frame(image, mask, long_px, depth, F, A4)
        self.assertAlmostEqual(m["wall_distance_m"], 3.1, delta=0.06)

    def test_depth_model_scale_error_is_corrected_by_the_sheet(self):
        # Depth Pro 40% too far everywhere (as seen close up on spines): the sheet scales it back.
        image, depth, mask, long_px = synthetic_view(5.5, 10)
        m = measure_frame(image, mask, long_px, depth * 1.4, F, A4)
        self.assertAlmostEqual(m["scale_factor"], 1 / 1.4, delta=0.02)
        self.assertAlmostEqual(m["wall_distance_m"], 5.5, delta=0.08)

    def test_an_empty_sheet_outline_is_refused_with_a_reason(self):
        image, depth, _, long_px = synthetic_view(4.0, 0)
        m = measure_frame(image, np.zeros_like(depth, dtype=bool), long_px, depth, F, A4)
        self.assertIsNone(m["wall_distance_m"])

    def test_without_the_floor_the_frame_is_not_used(self):
        image, depth, mask, long_px = synthetic_view(4.0, 0, camera_height=50.0)     # floor far out of view
        self.assertIsNone(measure_frame(image, mask, long_px, depth, F, A4)["wall_distance_m"])

    def test_a_sheet_too_far_away_is_refused(self):
        image, depth, mask, long_px = synthetic_view(9.0, 0)                         # ~30 px at this focal
        m = measure_frame(image, mask, long_px, depth, F, A4)
        self.assertIsNone(m["wall_distance_m"])
        self.assertIn("px long", m["reason"])


class SheetTest(unittest.TestCase):
    def test_blurred_edge_does_not_lengthen_the_sheet(self):
        # A 60 px sheet, blurred; the segmentation outline grown by 2 px on every side (as SAM does).
        image = np.full((200, 200), 90, np.uint8)
        image[70:130, 79:121] = 240                                   # 60 px long, 42 px wide
        image = cv2.GaussianBlur(image, (0, 0), 1.5)
        grown = np.zeros((200, 200), bool)
        grown[68:132, 77:123] = True                                  # 64 px: 6.7% too long
        length, how = sheet_length_px(Image.fromarray(image).convert("RGB"), grown, 64.0)
        self.assertEqual(how, "sub-pixel corners")
        self.assertLess(abs(length - 60), 2.0)

    def test_white_paper_passes_and_a_coloured_print_does_not(self):
        mask = np.ones((10, 10), bool)
        white = Image.new("RGB", (10, 10), (235, 232, 225))
        pink = Image.new("RGB", (10, 10), (230, 60, 140))
        self.assertTrue(plain_paper(white, mask))
        self.assertFalse(plain_paper(pink, mask))


class SummaryTest(unittest.TestCase):
    def frames(self, t0, distance, n=4):
        return [{"t": t0 + 0.5 * i, "wall_distance_m": distance} for i in range(n)]

    def test_two_walls_give_length_and_width_with_the_corner_offset(self):
        out = summarise(self.frames(0, 3.7) + self.frames(6, 4.2), 0.5)
        self.assertEqual((out["length_m"], out["width_m"]), (4.7, 4.2))

    def test_one_wall_leaves_the_room_empty(self):
        out = summarise(self.frames(0, 3.7), 0.5)
        self.assertNotIn("length_m", out)
        self.assertTrue(out["flags"])

    def test_extra_sheet_views_are_flagged(self):
        out = summarise(self.frames(0, 3.7) + self.frames(6, 4.2) + self.frames(12, 3.7, 2), 0.5)
        self.assertTrue(any("sheet views" in f for f in out["flags"]))


if __name__ == "__main__":
    unittest.main()


class CornerRoomJobTest(unittest.TestCase):
    def observation(self, distance, yaw, ref):
        image, depth, mask, long_px = synthetic_view(distance, yaw)
        return {"image": image, "mask": mask, "long_px": long_px, "how": "test", "depth": depth, "focal": F, "frame_ref": ref}

    def test_two_held_walls_give_length_and_width(self):
        from pipeline.room_corner import CornerRoomJob
        job = CornerRoomJob(corner_offset_m=0.5)
        job.observations = {"1": [self.observation(3.7, 0, "wall1_0.jpg"), self.observation(3.7, 10, "wall1_1.jpg")],
                            "2": [self.observation(4.2, 0, "wall2_0.jpg"), self.observation(4.2, 5, "wall2_1.jpg")]}
        result = job._compute()
        self.assertEqual((result["length_m"], result["width_m"]), (4.7, 4.2))
        self.assertEqual(result["frames_used"][0], "room_frames/wall1_0.jpg")
        self.assertIn("A4 sheet", result["method"])

    def test_one_wall_only_gives_no_size(self):
        from pipeline.room_corner import CornerRoomJob
        job = CornerRoomJob()
        job.observations = {"1": [self.observation(3.7, 0, "wall1_0.jpg")]}
        self.assertNotIn("length_m", job._compute())


class CornerRoomJobStateTest(unittest.TestCase):
    def test_a_wall_being_measured_shows_as_measuring_then_done(self):
        import threading
        from pathlib import Path
        from unittest import mock
        from pipeline import room_corner
        job = room_corner.CornerRoomJob()
        started, release, seen = threading.Event(), threading.Event(), {}

        def slow_observe(*args):
            started.set()
            release.wait(5)
            return None
        with mock.patch.object(room_corner, "observe", slow_observe), mock.patch.object(room_corner.Image, "open") as opened:
            opened.return_value.convert.return_value = None
            worker = threading.Thread(target=job.add_wall, args=("1", "A4", [("wall1_0.jpg", Path("x.jpg"))], None, None, None))
            worker.start()
            started.wait(5)
            seen["during"] = job.status()["state"]
            release.set()
            worker.join(5)
        self.assertEqual((seen["during"], job.status()["state"]), ("measuring", "done"))
