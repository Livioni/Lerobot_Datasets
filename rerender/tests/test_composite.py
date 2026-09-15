import unittest

import numpy as np

from rerender._composite import compose


class CompositeTests(unittest.TestCase):
    def test_occlusion_cases(self):
        # old depth, new depth, scene depth, expected RGB, erase, draw
        cases = [
            (0, 0, 2, 80, False, False),   # background
            (1, 0, 1, 0, True, False),    # exposed old robot -> black
            (2, 0, 1, 80, False, False),  # foreground hides old robot
            (0, 1, 2, 190, False, True),  # new robot in front of scene
            (0, 2, 1, 80, False, False),  # scene hides new robot
            (1, 2, 1, 190, True, True),   # removed old depth cannot hide new robot
            (3, 1, 2, 190, False, True),  # old hidden, new visible: independent tests
            (3, 4, 2, 80, False, False),  # both hidden
            (1.004, 0, 1, 0, True, False),
            (1.006, 0, 1, 80, False, False),
            (0, 1.004, 1, 190, False, True),
            (0, 1.006, 1, 80, False, False),
            (1, 1, 0, 80, False, False),
            (1, 1, float('nan'), 80, False, False),
            (1, 1, float('inf'), 80, False, False),
            (float('nan'), 0, 2, 80, False, False),
            (0, float('inf'), 2, 80, False, False),
        ]
        values = np.array(cases)
        shape = (1, len(cases), 3)
        result = compose(np.full(shape, 80, np.uint8), values[None, :, 0],
                         np.full(shape, 190, np.uint8), values[None, :, 1], values[None, :, 2])
        np.testing.assert_array_equal(result.rgb[0, :, 0], values[:, 3])
        np.testing.assert_array_equal(result.erased[0], values[:, 4])
        np.testing.assert_array_equal(result.drawn[0], values[:, 5])

    def test_no_depth_ignores_scene_and_does_not_mutate_inputs(self):
        rgb = np.full((1, 3, 3), 80, np.uint8)
        old = np.array([[1., 2, 0]])
        new = np.array([[0., 8, 0]])
        result = compose(rgb, old, np.full_like(rgb, 190), new)
        np.testing.assert_array_equal(result.rgb[0, :, 0], [0, 190, 80])
        self.assertTrue(np.all(rgb == 80))
        np.testing.assert_array_equal(old, [[1, 2, 0]])

    def test_untouched_pixels_are_bit_exact(self):
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        old = rng.choice([0., 1., 3.], (20, 30))
        new = rng.choice([0., 1., 3.], (20, 30))
        result = compose(rgb, old, np.zeros_like(rgb), new, np.full((20, 30), 2.))
        untouched = ~(result.erased | result.drawn)
        np.testing.assert_array_equal(result.rgb[untouched], rgb[untouched])


if __name__ == '__main__':
    unittest.main()
