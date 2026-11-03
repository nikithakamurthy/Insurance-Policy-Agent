import unittest

import numpy as np

from pipeline.live_room import card_scale_factor, rescale


def recon(depth_at_card: float, fx: float = 500.0, size=(518, 294)):
    w, h = size
    return {"size": np.array([w, h]), "intrinsic": np.array([[[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]]]),
            "depth": np.full((1, h, w), depth_at_card, dtype=np.float32)}


class CardScaleTest(unittest.TestCase):
    def test_card_sets_the_true_distance(self):
        # At VGGT's 518 px width with fx = 500 px, a card 0.0856 m long that is 1.0 m away is 42.8 px long;
        # in the 1920 px original that is 158.6 px. The reconstruction put it at 0.8 m: scale up by 1.25.
        mask = np.zeros((1080, 1920), dtype=bool)
        mask[500:600, 900:1060] = True
        factor = card_scale_factor(42.8 * 1920 / 518, 1920, recon(0.8), 0, mask)
        self.assertAlmostEqual(factor, 1.25, places=3)

    def test_card_too_small_to_trust_is_ignored(self):
        mask = np.zeros((1080, 1920), dtype=bool)
        mask[500:510, 900:920] = True
        self.assertIsNone(card_scale_factor(20, 1920, recon(1.0), 0, mask))      # ~5 px at VGGT's resolution

    def test_rescale_moves_points_depths_and_cameras_together(self):
        r = {"points": np.ones((1, 2, 2, 3)), "depth": np.ones((1, 2, 2)),
             "extrinsic": np.concatenate([np.eye(3)[None], np.array([[[1.0], [2.0], [3.0]]])], axis=2)}
        out = rescale(r, 2.0)
        self.assertEqual((out["points"].max(), out["depth"].max()), (2.0, 2.0))
        self.assertEqual(out["extrinsic"][0, :, 3].tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(out["extrinsic"][0, 0, 0], 1.0)                              # rotation unchanged


if __name__ == "__main__":
    unittest.main()
