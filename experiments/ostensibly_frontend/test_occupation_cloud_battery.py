import unittest

from experiments.ostensibly_frontend.run_occupation_cloud_battery import (
    ReferenceWindow,
    select_duration_matched_windows,
)


class OccupationCloudBatteryTests(unittest.TestCase):
    def test_selection_is_duration_based_and_forced_positive_is_retained(self):
        windows = [
            ReferenceWindow("R", "bdl", "u1", 0.0, 0.10, "r"),
            ReferenceWindow("R", "bdl", "u2", 0.0, 0.15, "r"),
            ReferenceWindow("R", "bdl", "u3", 0.0, 0.30, "r"),
            ReferenceWindow("L", "bdl", "u1", 0.2, 0.34, "l"),
            ReferenceWindow("L", "bdl", "u2", 0.2, 0.40, "l"),
        ]
        selected = select_duration_matched_windows(
            windows,
            target_duration=0.16,
            witnesses_per_label=1,
            forced=("R", "u3", 0.0, 0.30),
        )
        keys = {item.key for item in selected}
        self.assertIn(windows[1].key, keys)
        self.assertIn(windows[3].key, keys)
        self.assertIn(windows[2].key, keys)
        self.assertEqual(len(selected), 3)

    def test_missing_forced_window_fails(self):
        windows = [ReferenceWindow("L", "bdl", "u1", 0.0, 0.1, "l")]
        with self.assertRaises(ValueError):
            select_duration_matched_windows(
                windows,
                target_duration=0.1,
                witnesses_per_label=1,
                forced=("R", "u1", 0.0, 0.1),
            )


if __name__ == "__main__":
    unittest.main()
