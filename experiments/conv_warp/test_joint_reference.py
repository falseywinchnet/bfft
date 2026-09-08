from __future__ import annotations

import unittest

import numpy as np

from .joint_reference import (
    evaluate_joint_atlas,
    finite_joint_control_nets,
    global_joint_control_nets,
    joint_control_nets,
)


class JointReferenceTests(unittest.TestCase):
    def test_affine_plane_is_reproduced(self) -> None:
        y,x = np.meshgrid(np.arange(6.0),np.arange(7.0),indexing="ij")
        source = 0.17 + 0.09*x - 0.04*y
        control, diagnostic = joint_control_nets(source,enforce_range=True)
        qy,qx = np.meshgrid(np.linspace(0,5,19),np.linspace(0,6,23),indexing="ij")
        estimate = evaluate_joint_atlas(control,qx,qy)
        truth = 0.17 + 0.09*qx - 0.04*qy
        self.assertLess(float(np.max(np.abs(estimate-truth))),2e-10)
        self.assertLessEqual(float(diagnostic["face_violation"]),2e-12)

    def test_patch_controls_stay_in_corner_range_when_witnessed_monotone(self) -> None:
        y,x = np.meshgrid(np.arange(6.0),np.arange(6.0),indexing="ij")
        source = np.tanh((x+0.3*y-2.4)*1.8)
        control, diagnostic = joint_control_nets(source,enforce_range=True)
        for j in range(5):
            for i in range(5):
                low = np.min(source[j:j+2,i:i+2])
                high = np.max(source[j:j+2,i:i+2])
                self.assertGreaterEqual(float(np.min(control[j,i])),low-3e-11)
                self.assertLessEqual(float(np.max(control[j,i])),high+3e-11)
        self.assertLessEqual(float(diagnostic["range_violation"]),2e-12)
        self.assertLessEqual(float(diagnostic["fixed_error"]),2e-12)

    def test_adjacent_patches_share_the_entire_boundary_curve(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.random((5,6))
        control,_ = joint_control_nets(source)
        self.assertLess(float(np.max(np.abs(control[:,:-1,:,-1]-control[:,1:,:,0]))),2e-12)
        self.assertLess(float(np.max(np.abs(control[:-1,:,-1,:]-control[1:,:,0,:]))),2e-12)

    def test_sampled_canonical_atlas_matches_its_cell_collocation_nodes(self)->None:
        y,x=np.meshgrid(np.arange(5.0),np.arange(6.0),indexing="ij")
        source=np.sin(.3*x+.2*y)
        control,_=joint_control_nets(source,proposal="canonical_sampled")
        phase=np.linspace(0.0,1.0,6)
        # Shared boundary admission can only alter the collocation result by
        # its float32 evaluation error; for this monotone patch the current
        # inequalities are already feasible.
        qy,qx=np.meshgrid(1+phase,2+phase,indexing="ij")
        estimate=evaluate_joint_atlas(control,qx,qy)
        from .operators import _sample_conv
        truth=_sample_conv(source.astype(np.float32),qx/(source.shape[1]-1),qy/(source.shape[0]-1))
        self.assertLess(float(np.max(np.abs(estimate-truth))),2e-5)

    def test_global_joint_atlas_is_cardinal_and_c0(self)->None:
        y,x=np.meshgrid(np.arange(5.0),np.arange(6.0),indexing="ij")
        source=np.tanh(x+.2*y-2.2)
        control,diagnostic=global_joint_control_nets(source)
        estimate=evaluate_joint_atlas(control,x,y)
        self.assertLess(float(np.max(np.abs(estimate-source))),2e-12)
        self.assertLess(float(np.max(np.abs(control[:,:-1,:,-1]-control[:,1:,:,0]))),2e-12)
        self.assertLess(float(np.max(np.abs(control[:-1,:,-1,:]-control[1:,:,0,:]))),2e-12)
        self.assertLessEqual(float(diagnostic["maximum_violation"]),3e-10)

    def test_support_sum_transports_a_step_normal_into_the_stencil_plateau(self)->None:
        source=np.zeros((9,9));source[:,4:]=1.0
        from experiments.conv_distilled_core import _variation_jet
        from .joint_reference import _support_sum_directions
        gx,gy=_variation_jet(source[...,None]);direction=_support_sum_directions(gx,gy)
        # Every interior cell whose declared six-sample support intersects the
        # step has a positive horizontal current and no vertical current.
        active=direction[2:6,1:7,0]
        self.assertTrue(np.all(active[...,0]>0.0))
        self.assertLess(float(np.max(np.abs(active[...,1]))),2e-12)

    def test_phase_gradient_constraints_have_the_expected_affine_orientation(self)->None:
        from .joint_reference import _phase_gradient_constraints
        phase=np.array(((0.0,1.0),(0.0,1.0)))
        constraints=_phase_gradient_constraints(phase)
        increasing=np.broadcast_to(np.linspace(0.0,1.0,6),(6,6))
        decreasing=1.0-increasing
        self.assertGreaterEqual(min(float(np.sum(a*increasing)) for a in constraints),-2e-12)
        self.assertLess(min(float(np.sum(a*decreasing)) for a in constraints),-0.9)

    def test_completed_direction_polytope_contains_the_q1_baseline(self)->None:
        from .joint_reference import _completed_direction_constraints
        rng=np.random.default_rng(12)
        phase=np.linspace(0.0,1.0,6)
        for _ in range(30):
            corners=rng.normal(size=(2,2));transported=rng.normal(size=2)
            control=np.empty((6,6))
            for j,v in enumerate(phase):
                for i,u in enumerate(phase):
                    control[j,i]=(
                        (1-u)*(1-v)*corners[0,0]+u*(1-v)*corners[0,1]
                        +(1-u)*v*corners[1,0]+u*v*corners[1,1]
                    )
            constraints=_completed_direction_constraints(corners,transported)
            if constraints:
                self.assertGreaterEqual(
                    min(float(np.sum(a*control)) for a in constraints),-3e-12
                )

    def test_finite_entity_admission_is_cardinal_c0_and_feasible(self)->None:
        y,x=np.meshgrid(np.arange(7.0),np.arange(8.0),indexing="ij")
        source=np.tanh((x-.37*y-2.8)*1.7)+.08*np.sin(.8*x+.4*y)
        control,diagnostic=finite_joint_control_nets(source)
        estimate=evaluate_joint_atlas(control,x,y)
        self.assertLess(float(np.max(np.abs(estimate-source))),2e-12)
        self.assertLess(float(np.max(np.abs(control[:,:-1,:,-1]-control[:,1:,:,0]))),2e-12)
        self.assertLess(float(np.max(np.abs(control[:-1,:,-1,:]-control[1:,:,0,:]))),2e-12)
        self.assertGreaterEqual(float(diagnostic["minimum_baseline_margin"]),-5e-12)
        self.assertGreaterEqual(float(diagnostic["minimum_admitted_margin"]),-5e-12)

    def test_finite_entity_admission_reproduces_an_affine_plane(self)->None:
        y,x=np.meshgrid(np.arange(6.0),np.arange(7.0),indexing="ij")
        source=.17+.09*x-.04*y
        control,_=finite_joint_control_nets(source)
        qy,qx=np.meshgrid(np.linspace(0,5,19),np.linspace(0,6,23),indexing="ij")
        estimate=evaluate_joint_atlas(control,qx,qy)
        self.assertLess(float(np.max(np.abs(estimate-(.17+.09*qx-.04*qy)))),3e-6)

    def test_planar_gradient_cone_has_expected_duals(self)->None:
        from .joint_reference import _dual_generators_of_planar_cone
        east_north=np.array(((1.0,0.0),(0.0,1.0)))
        dual=_dual_generators_of_planar_cone(east_north,1.0)
        self.assertEqual(len(dual),2)
        probes=np.array(((1.0,0.0),(0.0,1.0),(.3,.7)))
        self.assertGreaterEqual(min(float(n@p) for n in dual for p in probes),-2e-12)
        full=_dual_generators_of_planar_cone(
            np.array(((1.0,0.0),(-.7,.7),(-.7,-.7))),1.0
        )
        self.assertEqual(full,[])
        line=_dual_generators_of_planar_cone(
            np.array(((1.0,0.0),(-1.0,0.0))),1.0
        )
        self.assertEqual(len(line),2)
        self.assertGreaterEqual(
            min(float(n@v) for n in line for v in ((1.0,0.0),(-1.0,0.0))),-2e-12
        )

    def test_planar_dual_generators_contain_every_witness_current(self)->None:
        from .joint_reference import _dual_generators_of_planar_cone
        rng=np.random.default_rng(120)
        for _ in range(200):
            centre=rng.uniform(-np.pi,np.pi);width=rng.uniform(0.0,.99*np.pi)
            angle=centre+rng.uniform(-.5*width,.5*width,size=12)
            vectors=rng.uniform(.05,2.0,size=(12,1))*np.stack(
                (np.cos(angle),np.sin(angle)),axis=1
            )
            dual=_dual_generators_of_planar_cone(vectors,2.0)
            self.assertTrue(dual)
            self.assertGreaterEqual(
                min(float(normal@vector) for normal in dual for vector in vectors),
                -3e-12,
            )

    def test_q1_is_feasible_for_its_local_gradient_cone(self)->None:
        from .joint_reference import _gradient_cone_constraints
        rng=np.random.default_rng(27);phase=np.linspace(0.0,1.0,6)
        for _ in range(20):
            source=rng.normal(size=(5,5,1));corners=source[2:4,2:4,0]
            control=np.empty((6,6))
            for j,v in enumerate(phase):
                for i,u in enumerate(phase):
                    control[j,i]=(
                        (1-u)*(1-v)*corners[0,0]+u*(1-v)*corners[0,1]
                        +(1-u)*v*corners[1,0]+u*v*corners[1,1]
                    )
            constraints=_gradient_cone_constraints(source,2,2,0,support=False)
            if constraints:
                self.assertGreaterEqual(
                    min(float(np.sum(a*control)) for a in constraints),-8e-12
                )

    def test_finite_support_cone_is_equivariant_under_axis_exchange(self)->None:
        y,x=np.indices((9,10));source=(x-.37*y>2.8).astype(np.float64)
        control,_=finite_joint_control_nets(source)
        exchanged,_=finite_joint_control_nets(source.T)
        restored=exchanged.transpose(1,0,3,2)
        # The compiled proposal uses two native FP32 factor passes.  Exchanging
        # those passes changes only their rounding order; the finite admission
        # itself is axis-equivariant.
        self.assertLess(float(np.max(np.abs(control-restored))),1e-6)

    def test_finite_controls_stay_in_every_declared_support_range(self)->None:
        rng=np.random.default_rng(91);source=rng.normal(size=(8,9,3))
        control,_=finite_joint_control_nets(source)
        for cy in range(source.shape[0]-1):
            top=max(0,cy-2);bottom=min(source.shape[0],cy+4)
            for cx in range(source.shape[1]-1):
                left=max(0,cx-2);right=min(source.shape[1],cx+4)
                low=np.min(source[top:bottom,left:right],axis=(0,1))
                high=np.max(source[top:bottom,left:right],axis=(0,1))
                self.assertTrue(np.all(control[cy,cx]>=low-3e-12))
                self.assertTrue(np.all(control[cy,cx]<=high+3e-12))

    def test_finite_atlas_satisfies_every_support_cone_coefficient(self)->None:
        y,x=np.indices((9,10));source=(x-.41*y>2.9).astype(np.float64)
        control,diagnostic=finite_joint_control_nets(source)
        from .joint_reference import _gradient_cone_constraints
        minimum=float("inf")
        for cy in range(source.shape[0]-1):
            for cx in range(source.shape[1]-1):
                constraints=_gradient_cone_constraints(
                    source[...,None],cy,cx,0,support=True
                )
                for coefficient in constraints:
                    minimum=min(minimum,float(np.sum(coefficient*control[cy,cx])))
        self.assertGreaterEqual(minimum,-3e-12)
        self.assertGreaterEqual(float(diagnostic["minimum_admitted_margin"]),-3e-12)


if __name__ == "__main__":
    unittest.main()
