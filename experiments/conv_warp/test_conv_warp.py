from __future__ import annotations

import unittest

import numpy as np

from .geometry import (
    ProjectiveMap,
    affine_about_center,
    perspective_about_center,
    pullback_spd,
)
from .operators import (
    adaptive_positive_warped_basin_average,
    direct_warp,
    positive_warped_basin_average,
)
from .synthetic import ConvexPolygonField, FourierField
from .warped_pixels import (
    certified_projective_warped_pixel_average,
    exact_affine_warped_pixel_average,
)


class WarpGeometryTests(unittest.TestCase):
    def test_affine_jacobian_is_exact(self) -> None:
        transform = affine_about_center(
            angle_degrees=17.0, scale_x=0.8, scale_y=1.1, shear_x=0.2
        )
        x = np.array([0.2, 0.7])
        y = np.array([0.3, 0.8])
        jacobian = transform.jacobian(x, y)
        linear, _ = transform.affine_parts()
        self.assertTrue(np.array_equal(jacobian, np.broadcast_to(linear, jacobian.shape)))

    def test_projective_jacobian_matches_central_difference(self) -> None:
        transform = ProjectiveMap(np.array(
            ((0.9, 0.1, 0.02), (-0.05, 1.1, -0.01), (0.08, -0.04, 1.0))
        ))
        x, y, h = np.array([0.4]), np.array([0.6]), 1e-6
        jacobian = transform.jacobian(x, y)[0]
        pxp, pyp = transform.map(x + h, y)
        pxm, pym = transform.map(x - h, y)
        qxp, qyp = transform.map(x, y + h)
        qxm, qym = transform.map(x, y - h)
        numerical = np.array((
            (((pxp-pxm)/(2*h))[0], ((qxp-qxm)/(2*h))[0]),
            (((pyp-pym)/(2*h))[0], ((qyp-qym)/(2*h))[0]),
        ))
        self.assertTrue(np.allclose(jacobian, numerical, rtol=2e-10, atol=2e-10))

    def test_pullback_shape_has_unit_determinant(self) -> None:
        transform = perspective_about_center(
            perspective_x=0.18, perspective_y=-0.11,
            angle_degrees=12.0, scale_x=0.7, scale_y=0.9,
        )
        x, y = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.2, 0.8, 3))
        jacobian = transform.jacobian(x, y)
        metric = np.broadcast_to(np.array(((2.0, 0.3), (0.3, 0.545))), jacobian.shape)
        _, shape = pullback_spd(metric, jacobian)
        self.assertTrue(np.allclose(np.linalg.det(shape), 1.0, rtol=2e-13, atol=2e-13))


class SyntheticTruthTests(unittest.TestCase):
    def test_affine_fourier_basin_average_matches_dense_reference(self) -> None:
        field = FourierField(
            np.array(((2.0, 1.0),)), np.array((0.3,)), np.array((0.2,))
        )
        transform = affine_about_center(angle_degrees=13.0, shear_x=0.1)
        exact = field.affine_basin_average(transform, (5, 6))
        # A deliberately dense midpoint rule is only a cross-check of the
        # closed-form expression, not the benchmark truth definition.
        from .synthetic import endpoint_basin_bounds
        xb, yb = endpoint_basin_bounds(6), endpoint_basin_bounds(5)
        reference = np.empty_like(exact)
        count = 600
        phase = (np.arange(count) + 0.5) / count
        for j in range(5):
            for i in range(6):
                x = xb[i] + phase * (xb[i+1] - xb[i])
                y = yb[j] + phase * (yb[j+1] - yb[j])
                yy, xx = np.meshgrid(y, x, indexing="ij")
                px, py = transform.map(xx, yy)
                reference[j, i] = np.mean(field.evaluate(px, py))
        self.assertLess(float(np.max(np.abs(exact-reference))), 2e-6)

    def test_identity_direct_warp_is_cardinal(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.random((7, 8), dtype=np.float32)
        identity = ProjectiveMap(np.eye(3))
        for method in ("conv", "bilinear", "lanczos3"):
            estimate = direct_warp(source, identity, source.shape, method=method)
            self.assertLess(float(np.max(np.abs(estimate-source))), 2e-6)

    def test_positive_basin_operator_preserves_constants_and_range(self) -> None:
        source = np.full((7, 8), 0.37, dtype=np.float32)
        transform = affine_about_center(angle_degrees=9.0, scale_x=0.8, scale_y=0.9)
        for method in ("conv", "bilinear"):
            estimate = positive_warped_basin_average(
                source, transform, (5, 6), method=method, quadrature_order=3
            )
            self.assertLess(float(np.max(np.abs(estimate-0.37))), 2e-6)
            self.assertGreaterEqual(float(np.min(estimate)), 0.37 - 2e-6)
            self.assertLessEqual(float(np.max(estimate)), 0.37 + 2e-6)

    def test_adaptive_projective_basin_preserves_constants(self) -> None:
        source = np.full((7, 8), 0.41, dtype=np.float32)
        transform = perspective_about_center(
            perspective_x=0.25, perspective_y=-0.15,
            angle_degrees=8.0, scale_x=0.75, scale_y=0.8,
        )
        estimate, diagnostic = adaptive_positive_warped_basin_average(
            source, transform, (4, 5), method="conv", tolerance=1e-7,
            maximum_depth=2,
        )
        self.assertLess(float(np.max(np.abs(estimate-0.41))), 2e-6)
        self.assertLessEqual(float(diagnostic["maximum_estimator"]), 2e-6)

    def test_projective_polygon_basin_coverage_is_exact_on_axis_split(self) -> None:
        polygon=np.array(((-2,-2),(0.5,-2),(0.5,2),(-2,2)),dtype=np.float64)
        field=ConvexPolygonField((polygon,),np.array((1.0,)))
        identity=ProjectiveMap(np.eye(3))
        average=field.warped_basin_average(identity,(5,5))
        self.assertTrue(np.array_equal(average[:,0:2],np.ones((5,2))))
        self.assertTrue(np.array_equal(average[:,3:5],np.zeros((5,2))))
        self.assertTrue(np.array_equal(average[:,2],np.full(5,0.5)))

    def test_exact_affine_warped_pixels_reproduce_plane_basin_means(self)->None:
        y,x=np.meshgrid(np.arange(6.0),np.arange(7.0),indexing="ij")
        source=.17+.09*x-.04*y
        transform=affine_about_center(
            angle_degrees=17.0,scale_x=.62,scale_y=.71,shear_x=.13
        )
        estimate,diagnostic=exact_affine_warped_pixel_average(
            source,transform,(5,6)
        )
        from .synthetic import endpoint_basin_bounds
        xb=endpoint_basin_bounds(6);yb=endpoint_basin_bounds(5)
        xx=.5*(xb[:-1]+xb[1:]);yy=.5*(yb[:-1]+yb[1:])
        qy,qx=np.meshgrid(yy,xx,indexing="ij")
        px,py=transform.map(qx,qy)
        truth=.17+.09*(source.shape[1]-1)*px-.04*(source.shape[0]-1)*py
        self.assertLess(float(np.max(np.abs(estimate-truth))),2e-10)
        self.assertEqual(int(diagnostic["outside_basins"]),0)
        self.assertLess(float(diagnostic["maximum_source_area_partition_error"]),2e-12)

    def test_exact_affine_warped_pixels_preserve_multichannel_constants(self)->None:
        source=np.broadcast_to(np.array((.25,.5,.75)),(6,7,3)).copy()
        transform=affine_about_center(angle_degrees=-11.0,scale_x=.7,scale_y=.8)
        estimate,diagnostic=exact_affine_warped_pixel_average(source,transform,(4,5))
        self.assertLess(float(np.max(np.abs(estimate-source[0,0]))),2e-11)
        self.assertEqual(int(diagnostic["outside_basins"]),0)

    def test_certified_projective_warped_pixels_preserve_constants(self)->None:
        source=np.full((6,7),.375)
        transform=perspective_about_center(
            perspective_x=.24,perspective_y=-.13,
            angle_degrees=7.0,scale_x=.67,scale_y=.74,
        )
        estimate,diagnostic=certified_projective_warped_pixel_average(
            source,transform,(3,4),tolerance=2e-5,maximum_depth=5
        )
        self.assertLess(float(np.max(np.abs(estimate-.375))),2e-12)
        self.assertLess(float(diagnostic["maximum_certified_error"]),2e-12)
        self.assertEqual(int(diagnostic["outside_basins"]),0)

    def test_projective_certificate_contains_independent_plane_integral(self)->None:
        y,x=np.meshgrid(np.arange(6.0),np.arange(7.0),indexing="ij")
        source=.17+.09*x-.04*y
        transform=perspective_about_center(
            perspective_x=.24,perspective_y=-.13,
            angle_degrees=7.0,scale_x=.67,scale_y=.74,
        )
        estimate,diagnostic=certified_projective_warped_pixel_average(
            source,transform,(3,4),tolerance=2e-4,maximum_depth=3
        )
        from .synthetic import endpoint_basin_bounds
        node,weight=np.polynomial.legendre.leggauss(24)
        node=.5*(node+1.0);weight=.5*weight
        xb=endpoint_basin_bounds(4);yb=endpoint_basin_bounds(3)
        reference=np.empty_like(estimate)
        for row in range(3):
            for column in range(4):
                qx=xb[column]+node*(xb[column+1]-xb[column])
                qy=yb[row]+node*(yb[row+1]-yb[row])
                yy,xx=np.meshgrid(qy,qx,indexing="ij")
                px,py=transform.map(xx,yy)
                value=.17+.09*6.0*px-.04*5.0*py
                reference[row,column]=np.einsum("i,j,ij",weight,weight,value)
        self.assertLessEqual(
            float(np.max(np.abs(estimate-reference))),
            float(diagnostic["maximum_certified_error"])+3e-13,
        )

    def test_affine_basin_analysis_conserves_the_admitted_atlas_integral(self)->None:
        rng=np.random.default_rng(113);source=rng.random((7,8))
        identity=ProjectiveMap(np.eye(3))
        estimate,_=exact_affine_warped_pixel_average(source,identity,(5,6))
        from .joint_reference import finite_joint_control_nets
        from .synthetic import endpoint_basin_bounds
        from .warped_pixels import _integrate_patch_polygon
        control,_=finite_joint_control_nets(source)
        atlas_integral=0.0
        square=np.array(((0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)))
        for cy in range(control.shape[0]):
            for cx in range(control.shape[1]):
                atlas_integral+=float(_integrate_patch_polygon(control[cy,cx],square,6))
        xb=endpoint_basin_bounds(6);yb=endpoint_basin_bounds(5)
        weights=np.outer(np.diff(yb),np.diff(xb))
        target_integral=float(np.sum(weights*estimate))
        source_normalized_integral=atlas_integral/((source.shape[0]-1)*(source.shape[1]-1))
        self.assertLess(abs(target_integral-source_normalized_integral),3e-12)


if __name__ == "__main__":
    unittest.main()
