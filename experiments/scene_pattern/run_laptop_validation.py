"""Run focused checks before the expensive native laptop comparison."""
import unittest
from .laptop_alignment import main

if __name__=='__main__':
    names=['test_distortion_basis','test_registration_audit','test_structure_alignment','test_floor_affine','test_average_alignment','test_panorama','test_fusion','test_residual']
    suite=unittest.defaultTestLoader.loadTestsFromNames(['experiments.scene_pattern.'+name for name in names])
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():raise SystemExit(1)
    main()
