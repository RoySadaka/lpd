import unittest

from lpd.enums import MonitorMode


class TestCallbacks(unittest.TestCase):

    def test_absolute_threshold_checker__true(self):
        from lpd.utils.threshold_checker import AbsoluteThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.899), (0.1, 0.9, 0.799)]:
            min_checker = AbsoluteThresholdChecker(MonitorMode.MIN, threshold)
            with self.subTest():
                self.assertTrue(min_checker(new_value=lower_value, old_value=higher_value))

            max_checker = AbsoluteThresholdChecker(MonitorMode.MAX, threshold)
            with self.subTest():
                self.assertTrue(max_checker(new_value=higher_value, old_value=lower_value))

    def test_absolute_threshold_checker__false(self):
        from lpd.utils.threshold_checker import AbsoluteThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.9), (0.1, 0.9, 0.81)]:
            min_checker = AbsoluteThresholdChecker(MonitorMode.MIN, threshold)
            with self.subTest():
                self.assertFalse(min_checker(new_value=lower_value, old_value=higher_value))

            max_checker = AbsoluteThresholdChecker(MonitorMode.MAX, threshold)
            with self.subTest():
                self.assertFalse(max_checker(new_value=higher_value, old_value=lower_value))

    def test_relative_threshold_checker__true(self):
        from lpd.utils.threshold_checker import RelativeThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.899), (0.1, 120.1, 100.0)]:
            min_checker = RelativeThresholdChecker(MonitorMode.MIN, threshold)
            with self.subTest():
                self.assertTrue(min_checker(new_value=lower_value, old_value=higher_value))
            max_checker = RelativeThresholdChecker(MonitorMode.MAX, threshold)
            with self.subTest():
                self.assertTrue(max_checker(new_value=higher_value, old_value=lower_value))

    def test_relative_threshold_checker__false(self):
        from lpd.utils.threshold_checker import RelativeThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.9), (0.1, 109.99, 100.0)]:
            min_checker = RelativeThresholdChecker(MonitorMode.MIN, threshold)
            with self.subTest():
                self.assertFalse(min_checker(new_value=lower_value, old_value=higher_value))
            max_checker = RelativeThresholdChecker(MonitorMode.MAX, threshold)
            with self.subTest():
                self.assertFalse(max_checker(new_value=higher_value, old_value=lower_value))
