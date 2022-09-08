import unittest

import pytest as pytest


class TestCallbacks(unittest.TestCase):

    def test_absolute_threshold_checker__true(self):
        from lpd.utils.threshold_checker import AbsoluteThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.899), (0.1, 0.9, 0.799)]:
            checker = AbsoluteThresholdChecker(threshold)
            with self.subTest():
                self.assertTrue(checker.is_new_value_higher(new_value=higher_value, old_value=lower_value))

                self.assertTrue(checker.is_new_value_lower(new_value=lower_value, old_value=higher_value))

    def test_absolute_threshold_checker__false(self):
        from lpd.utils.threshold_checker import AbsoluteThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.9), (0.1, 0.9, 0.81)]:
            checker = AbsoluteThresholdChecker(threshold)
            with self.subTest():
                self.assertFalse(checker.is_new_value_higher(new_value=higher_value, old_value=lower_value))

                self.assertFalse(checker.is_new_value_lower(new_value=lower_value, old_value=higher_value))

    def test_relative_threshold_checker__true(self):
        from lpd.utils.threshold_checker import RelativeThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.899), (0.1, 120.1, 100.0)]:
            checker = RelativeThresholdChecker(threshold)
            with self.subTest():
                self.assertTrue(checker.is_new_value_higher(new_value=higher_value, old_value=lower_value))

                self.assertTrue(checker.is_new_value_lower(new_value=lower_value, old_value=higher_value))

    def test_relative_threshold_checker__false(self):
        from lpd.utils.threshold_checker import RelativeThresholdChecker
        for (threshold, higher_value, lower_value) in [(0.0, 0.9, 0.9), (0.1, 109.99, 100.0)]:
            checker = RelativeThresholdChecker(threshold)
            with self.subTest():
                self.assertFalse(checker.is_new_value_higher(new_value=higher_value, old_value=lower_value))

                self.assertFalse(checker.is_new_value_lower(new_value=lower_value, old_value=higher_value))