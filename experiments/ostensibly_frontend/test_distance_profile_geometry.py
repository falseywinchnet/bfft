from __future__ import annotations

import unittest

import numpy as np

from .distance_profile_geometry import fit_distance_profile_atlas


class DistanceProfileGeometryTests(unittest.TestCase):
    def test_centroid_atlas_recovers_two_profile_classes(self) -> None:
        labels = ("A", "B")
        channels = ("left", "right")
        profiles = []
        targets = []
        profile_channels = []
        for channel, offset in (("left", 0.0), ("right", 10.0)):
            for label, center in (("A", (0.0, 2.0)), ("B", (2.0, 0.0))):
                for jitter in (-0.1, 0.1):
                    profiles.append(np.asarray(center) + offset + jitter)
                    targets.append(label)
                    profile_channels.append(channel)
        atlas = fit_distance_profile_atlas(
            np.stack(profiles),
            np.asarray(targets),
            np.asarray(profile_channels),
            channels,
            labels,
        )
        ranking = atlas.rank(
            {"left": np.asarray((0.05, 2.05)), "right": np.asarray((10.05, 12.05))}
        )
        self.assertEqual(ranking[0]["phone"], "A")


if __name__ == "__main__":
    unittest.main()
