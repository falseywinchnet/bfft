"""Focused exact tests for the non-root A2 weight-current attack."""

from fractions import Fraction

from run_weight_current_attack import (
    adjacent_root_carrier_root,
    adjacent_root_shape,
    adjacent_root_shape_residual,
    build_adjacent_root_candidate,
    build_candidate,
    build_doubled_candidate,
    build_mixed_candidate,
    build_root_lattice_candidate,
    dilation_minimize_generalized_model,
    doubled_omega2_quotient,
    doubled_omega2_shape_residual,
    doubled_quotient_chord,
    first_doubled_lifted_parameters,
    first_lifted_parameters,
    lower_omega2_quotient,
    lower_omega2_shape_residual,
    mixed_weight_quotient,
    mixed_weight_shape_residual,
    pythagorean_lift,
    quotient_chord,
    root_lattice_carrier_root,
    root_lattice_shape,
    root_lattice_shape_residual,
)
from run_weight_lift_quotient import (
    DOUBLED,
    FOUR_OMEGA,
    MIXED,
    NEXT_PRIMITIVE,
    THREE_TWO,
    on_curve,
    split_lattice_shell,
    split_multiples,
)
from run_hidden_half_root_attack import first_descendant
from run_fixed_bad_support_recurrence import recurrent_hits
from run_carrier_lattice_attack import (
    GENERATORS as HALF_ROOT_GENERATORS,
    build_carrier_candidate,
    carrier_coordinates,
)
from run_r_minus_5s_carrier_attack import (
    build_candidate as build_r_minus_5s_candidate,
    shell as r_minus_5s_shell,
)
from run_wall_squareclass_sieve import local_obstructions, sub_23_survivors
from run_fixed_23_recurrence import (
    conductor_divisor as fixed_23_conductor_divisor,
    fourth_multiple_conductor_floor,
    quartic_point as fixed_23_quartic_point,
    multiply as fixed_23_multiply,
)
from run_four_split_torus_attack import (
    build_four_split_fiber,
    conductor_norm_currents,
    conductor_quartic_cores,
    current_and_split_root,
    cube_second_current_jacobian,
    doubled_support,
    integral_short_model,
    first_nonunit_norm_obstructions,
    norm_unit_step,
    square_current_jacobians,
    square_current_torsion_support_polynomials,
    three_neighbor,
    three_neighbor_weyl_fixed_discriminants,
    unit_shape_hits,
    verify_three_neighbor_identity,
)
from run_four_split_norm_class_sieve import (
    NormOrbit,
    admissible_norms,
    parse_bnfisintnorm_output,
    split_support_image,
    unit_orbit_obstruction,
)
from run_asymmetric_three_split_attack import (
    build_asymmetric_three_split_fiber,
    build_asymmetric_torus_multiplier_fiber,
    cusp_sextic_coefficients,
    fourth_current_boundary_support_polynomials,
    fourth_current_mod_5_boundary_classes,
    fifth_current_projection_jacobian,
    sixth_current_mod_11_obstruction,
    square_discriminant_projection_jacobian,
    square_discriminant_recurrence_jacobian,
    triple_current_cancellation_mod_11,
    torus_current_multiplier,
)
from run_asymmetric_sextic_norm_sieve import (
    NormOrbit as SexticNormOrbit,
    admissible_norms as admissible_sextic_norms,
    certify_orbits as certify_sextic_orbits,
)
from run_three_current_height_conic import (
    RATIONAL_91_TRIPLES,
    RANK_FIVE_217_TRIPLES,
    SELF_COLLISION_TRIPLES,
    SHELL_637_TRIPLES,
    binary_quartic_invariants,
    build_three_current_height_fiber,
    first_four_height_slopes,
    four_height_jacobian_model,
    four_height_quartic_coefficients,
    rank_five_217_distinguished_slopes,
    rank_five_217_jacobian_model,
    rank_five_217_quartic_coefficients,
    self_collision_binary_currents,
    self_collision_discriminant_current,
    shortest_farey_neighbors,
)


Q = Fraction


def main() -> None:
    # Doubling the Eisenstein norm-one torus point multiplies its cubic
    # current by the split-discriminant root.  The resulting universal
    # Legendre point supplies four complete horizontal triples, so the full
    # height compatibility is an identity rather than a coefficient search.
    four_split = build_four_split_fiber(Q(1, 2))
    current, split_root = current_and_split_root(Q(1, 2))
    doubled = doubled_support(Q(1, 2))
    doubled_current = -doubled[0] * doubled[1] * (doubled[0] + doubled[1])
    assert doubled_current == split_root * current
    assert four_split.verify()
    assert len(four_split.points) == 12
    assert len(set(four_split.currents)) == 4
    assert integral_short_model(Q(1, 2)) == (
        0,
        -21_609,
        1_350_441,
    )
    assert conductor_norm_currents(Q(1, 2)) == (-1_251_788, 155_668)
    assert conductor_quartic_cores(Q(1, 2)) == (-312_947, 38_917)
    assert three_neighbor(Q(1, 2)) == Q(-1, 5)
    assert verify_three_neighbor_identity(Q(1, 2))
    assert verify_three_neighbor_identity(Q(1, 4))
    assert set(three_neighbor_weyl_fixed_discriminants()) == {-12, -3, 12}
    assert norm_unit_step(686, 324) == (2344, 1334)
    assert norm_unit_step(686, 324, -1) == (400, -38)
    assert unit_shape_hits(Q(1, 2), 64) == ((0, 7, 6),)
    assert first_nonunit_norm_obstructions() == (
        (-11, (1, 2), 9, 18),
        (-11, (1, -2), 8, 4),
        (13, (4, 1), 7, 8),
        (13, (4, -1), 7, 8),
    )
    assert square_current_jacobians() == (
        (0, 54, 0, 972, 52_488),
        (0, -162, 0, 8_748, -1_417_176),
    )
    assert square_current_torsion_support_polynomials() == (
        (1, 0, -3, -1),
        (1, 3, 0, -1),
    )
    assert cube_second_current_jacobian() == (0, 0, 0, 34_992, 0)
    # The complete local image of the split-support map closes norm classes,
    # not merely the separate cube and square coefficient conditions.  The
    # hard c=73 orbit is killed modulo 5, while the first genuine fiber at
    # c=38917 meets the local image (343,162) after one unit step.
    assert len(admissible_norms(1_000)) == 55
    assert parse_bnfisintnorm_output("73,-10,3\n") == (
        NormOrbit(73, -10, 3),
        NormOrbit(73, 10, -3),
    )
    support_mod_5 = split_support_image(5)
    assert unit_orbit_obstruction(
        NormOrbit(73, -10, 3), 5, support_mod_5
    ) is not None
    assert unit_orbit_obstruction(
        NormOrbit(38_917, 200, -19), 5, support_mod_5
    ) is None
    assert (343 % 5, 162 % 5) in support_mod_5

    # Three rather than four horizontal cubics remove the final height gate
    # while retaining a free conic.  The exceptional ell=4 chord is a true
    # rank-bearing cusp blow-up, and its first conductor current is a
    # Q(sqrt(7)) norm rather than the former determinant-three pair.
    asymmetric = build_asymmetric_three_split_fiber(Q(1, 2), Q(4))
    assert asymmetric.verify()
    assert len(asymmetric.points) == 9
    multiplier_three = build_asymmetric_torus_multiplier_fiber(
        Q(1, 2), 3, Q(6)
    )
    assert multiplier_three.verify()
    assert torus_current_multiplier(Q(286, 343), 3) == Q(-35_853, 117_649)
    assert cusp_sextic_coefficients(1, 2) == (-521, 216, -55_151)
    assert square_discriminant_recurrence_jacobian() == (0, 96, 0, 576, 0)
    assert square_discriminant_projection_jacobian() == (0, 0, 0, -39, 70)
    assert triple_current_cancellation_mod_11()
    assert fourth_current_mod_5_boundary_classes() == (
        (1, 2, 4, 2, 0, 0),
        (1, 3, 4, 2, 0, 0),
        (4, 2, 1, 3, 0, 0),
        (4, 3, 1, 3, 0, 0),
    )
    assert fourth_current_boundary_support_polynomials() == (
        (1, 0, -3, -1),
        (1, 3, 0, -1),
        (1, -3, -6, -1),
        (1, 6, 3, -1),
    )
    assert fifth_current_projection_jacobian() == (0, 0, 0, -1, 0)
    assert sixth_current_mod_11_obstruction()
    assert len(admissible_sextic_norms(55_150)) == 348
    sextic_certificates, sextic_survivors = certify_sextic_orbits(
        (
            SexticNormOrbit(73, -10, 1),
            SexticNormOrbit(1, 1, 0),
        )
    )
    assert sextic_certificates
    assert sextic_survivors == (SexticNormOrbit(1, 1, 0),)

    # Three arbitrary equal-norm currents admit a universal height conic.
    # The first genuinely three-orbit shell proves generic independence,
    # while its self-collision lift splits the residual degree-eight
    # discriminant current into eight rational linear branches.
    first_three_orbit = build_three_current_height_fiber(
        SHELL_637_TRIPLES, Q(3, 2)
    )
    assert first_three_orbit.verify()
    assert first_three_orbit.norm == 637
    assert first_three_orbit.currents == (2484, 4116, 5916)
    assert first_three_orbit.short_model == (
        0,
        Q(-1_070_797, 640_000),
        Q(-42_463_741, 128_000_000),
    )

    self_collision = build_three_current_height_fiber(
        SELF_COLLISION_TRIPLES, Q(2)
    )
    assert self_collision.verify()
    assert self_collision.norm == 1_217_307
    assert self_collision.currents == (
        -428_024_806,
        59_530_394,
        350_352_794,
    )
    k1 = self_collision.currents[0]
    lam = self_collision.dilation
    residual = (
        27
        - 54 * k1 * lam
        + (27 * k1 * k1 - 4 * self_collision.norm**3) * lam * lam
    )
    assert residual == self_collision_discriminant_current(Q(2))
    assert self_collision_binary_currents(2, 1) == (
        -8,
        14,
        63,
        7,
        395,
        175,
        243,
        457,
    )
    assert shortest_farey_neighbors() == (
        Q(5),
        Q(-3),
        Q(1, 3),
        Q(61, 32),
        Q(-3, 5),
        Q(50, 43),
        Q(36, 43),
        Q(92, 67),
    )
    assert tuple(
        sum(root * root for root in support) / 2
        for support in RATIONAL_91_TRIPLES
    ) == (91, 91, 91, 91)
    assert binary_quartic_invariants(four_height_quartic_coefficients()) == (
        30_118_645_593,
        -6_610_138_671_777_930,
    )
    assert four_height_jacobian_model() == (
        0,
        0,
        0,
        -10_039_548_531,
        244_819_950_806_590,
    )
    assert first_four_height_slopes() == (Q(77, 5), Q(392, 139))
    four_height = build_three_current_height_fiber(
        (
            RATIONAL_91_TRIPLES[0],
            RATIONAL_91_TRIPLES[1],
            RATIONAL_91_TRIPLES[3],
        ),
        first_four_height_slopes()[0],
    )
    fourth_height_square = 1 + four_height.dilation * (
        Q(87_450, 343) - Q(90)
    )
    assert fourth_height_square == Q(10_133**2, 12_663**2)

    assert tuple(
        sum(root * root for root in support) / 2
        for support in RANK_FIVE_217_TRIPLES
    ) == (217, 217, 217, 217)
    assert rank_five_217_jacobian_model() == (
        0,
        -1,
        0,
        -10_612_680,
        12_761_798_400,
    )
    assert binary_quartic_invariants(rank_five_217_quartic_coefficients()) == (
        509_408_656,
        -22_046_274_731_392,
    )
    wall_slope, subrecord_slope, independent_slope = (
        rank_five_217_distinguished_slopes()
    )
    rank_five_wall = build_three_current_height_fiber(
        (
            RANK_FIVE_217_TRIPLES[0],
            RANK_FIVE_217_TRIPLES[1],
            RANK_FIVE_217_TRIPLES[3],
        ),
        wall_slope,
        Q(19),
    )
    assert rank_five_wall.short_model == (0, -217, 1585)
    subrecord = build_three_current_height_fiber(
        (
            RANK_FIVE_217_TRIPLES[0],
            RANK_FIVE_217_TRIPLES[1],
            RANK_FIVE_217_TRIPLES[3],
        ),
        subrecord_slope,
        Q(19),
    )
    subrecord_extra_square = 19**2 + subrecord.dilation * (624 + 1224)
    assert subrecord_extra_square == Q(1615**2, 41**2)
    independent = build_three_current_height_fiber(
        (
            RANK_FIVE_217_TRIPLES[0],
            RANK_FIVE_217_TRIPLES[1],
            RANK_FIVE_217_TRIPLES[3],
        ),
        independent_slope,
        Q(19),
    )
    independent_extra_square = 19**2 + independent.dilation * (624 + 1224)
    assert independent.short_model == (
        0,
        -40_835_824_708,
        3_159_825_870_904_132,
    )
    assert independent_extra_square == 5035**2

    first = (Q(-3, 2), Q(49, 4))
    assert first[1] ** 2 == lower_omega2_quotient(first[0])
    assert pythagorean_lift(first[0]) == (Q(1, 2), Q(-2))
    second = quotient_chord(first, (Q(0), Q(20)))
    assert second == (Q(-8, 3), Q(196, 9))
    assert pythagorean_lift(second[0]) == (Q(1, 3), Q(-3))
    assert first_lifted_parameters() == (Q(1, 2), Q(1, 3))

    for t in first_lifted_parameters():
        assert lower_omega2_shape_residual(Q(-9), t) == 0
        assert lower_omega2_shape_residual(Q(4, 5), t) == 0
        assert build_candidate(Q(-9), t).saturation_status == "dependent"
        assert build_candidate(Q(4, 5), t).saturation_status == "dependent"

    doubled_first = doubled_quotient_chord((Q(-4), Q(-92)), (Q(0), Q(-16)))
    assert doubled_first == (Q(3, 2), Q(91, 2))
    assert doubled_first[1] ** 2 == doubled_omega2_quotient(doubled_first[0])
    assert first_doubled_lifted_parameters() == (Q(2), Q(3))
    for t in first_doubled_lifted_parameters():
        for c in (Q(-9, 7), Q(-5, 14)):
            assert doubled_omega2_shape_residual(c, t) == 0
            assert build_doubled_candidate(c, t).saturation_status == "dependent"

    for u, v in ((Q(3, 2), Q(49, 4)), (Q(8, 3), Q(196, 9))):
        assert v * v == mixed_weight_quotient(u)
    for t in (Q(2), Q(3)):
        for c in (Q(-9, 7), Q(-16, 7)):
            assert mixed_weight_shape_residual(c, t) == 0
            assert build_mixed_candidate(c, t).saturation_status == "dependent"

    for spec in (DOUBLED, MIXED):
        assert on_curve(spec, spec.generator)
        assert split_multiples(spec, 28) == (
            (1, Q(25, 6), (Q(8, 3), Q(3, 2))),
        )
        assert split_multiples(spec, 28, torsion_coset=True) == (
            (0, Q(-4), (Q(-2), Q(-2))),
            (1, Q(25, 6), (Q(8, 3), Q(3, 2))),
        )
    assert on_curve(NEXT_PRIMITIVE, NEXT_PRIMITIVE.generator)
    assert split_multiples(NEXT_PRIMITIVE, 64) == ()
    assert split_multiples(NEXT_PRIMITIVE, 64, torsion_coset=True) == (
        (0, Q(-4), (Q(-2), Q(-2))),
    )
    known_shell_hits = (
        ((-1, 0), False, Q(25, 6), (Q(8, 3), Q(3, 2))),
        ((-1, 0), True, Q(25, 6), (Q(8, 3), Q(3, 2))),
    )
    assert split_lattice_shell(DOUBLED, 12) == known_shell_hits
    assert split_lattice_shell(MIXED, 12) == known_shell_hits
    assert split_lattice_shell(NEXT_PRIMITIVE, 12) == ()
    assert on_curve(FOUR_OMEGA, FOUR_OMEGA.generator)
    assert on_curve(FOUR_OMEGA, FOUR_OMEGA.second_generator)
    assert on_curve(FOUR_OMEGA, FOUR_OMEGA.third_generator)
    assert split_lattice_shell(FOUR_OMEGA, 8) == ()
    assert on_curve(THREE_TWO, THREE_TWO.generator)
    assert on_curve(THREE_TWO, THREE_TWO.second_generator)
    assert split_lattice_shell(THREE_TWO, 16) == ()

    # The next A2 root-lattice orbit is a rational rank-six family rather
    # than another genus-one lift condition.
    assert root_lattice_shape(Q(2)) == Q(-100, 49)
    assert root_lattice_shape(Q(3)) == Q(-100, 49)
    for t in (Q(1, 2), Q(1, 3), Q(2), Q(3), Q(1, 4), Q(4)):
        c = root_lattice_shape(t)
        assert root_lattice_shape_residual(c, t) == 0
        assert root_lattice_carrier_root(c, t) ** 2 > 0
        assert build_root_lattice_candidate(t).saturation_status == "rank6"
    root_candidate = build_root_lattice_candidate(Q(2))
    assert root_candidate.model == (
        0,
        0,
        -202_300,
        -8_671_156,
        0,
    )
    assert root_candidate.discriminant_factors == {
        2: 8,
        17: 6,
        73: 1,
        7_748_769_683: 1,
    }
    assert root_candidate.tame_conductor_lower_bound == 326_951_588_004_502
    assert dilation_minimize_generalized_model(
        (0, 0, -5_462_100_000, -7_023_636_360_000, 0)
    ) == (root_candidate.model, 30)

    # The norm-7 component is the first member of a complete adjacent-root
    # ray.  Norm 13 keeps rank six; the denominator-resonant norm 21 point
    # collapses to exact rank four (the upper bound is certified by PARI in
    # jackpot_curve_pari.gp).
    for index, expected_c in ((Q(3), Q(-100, 49)), (Q(4), Q(-81, 49)), (Q(5), Q(-8, 5))):
        c = adjacent_root_shape(index, Q(2))
        assert c == expected_c
        assert adjacent_root_shape_residual(index, c, Q(2)) == 0
        assert adjacent_root_carrier_root(index, c, Q(2)) ** 2 > 0
    norm13 = build_adjacent_root_candidate(Q(4), Q(2))
    assert norm13.model == (0, 0, -3780, -44937, 0)
    assert norm13.saturation_status == "rank6"
    assert norm13.discriminant_factors == {2: 6, 3: 6, 6_328_527_157: 1}
    jackpot = build_adjacent_root_candidate(Q(5), Q(2))
    assert jackpot.model == (0, 0, -250, -1225, 0)
    assert jackpot.saturation_status == "dependent"
    assert jackpot.discriminant_factors == {2: 4, 5: 6, 83: 1, 587: 1}

    # The hidden x=3r/2 carrier is an exact rank-three lattice.  Its first
    # generator is the Weyl mate of the c=1/8 near miss; the canonical tangent
    # leaves that rank-loss node but immediately incurs a huge denominator.
    assert tuple(carrier_coordinates(point)[0] for point in HALF_ROOT_GENERATORS) == (
        Q(-9, 8),
        Q(3, 2),
        Q(1, 2),
    )
    assert first_descendant() == (
        Q(1215, 23434),
        Q(-58874215, 274576178),
    )
    near_c, near_k = carrier_coordinates(HALF_ROOT_GENERATORS[0])
    assert build_carrier_candidate(near_c, near_k).model == (
        0,
        0,
        820,
        -7300,
        0,
    )
    assert recurrent_hits(32) == ((1, Q(9, 64), Q(5, 4), Q(909, 64)),)

    # The rank-four r-5s carrier needs its rational two-torsion coset.  That
    # coset recovers the known exact rank-six seed; the non-torsion shell
    # exposes the primitive rank-jump generator and the deceptive -9/4 screen.
    torsion_shell = {
        coefficients: (c, root)
        for coefficients, c, root in r_minus_5s_shell(1, torsion_coset=True)
    }
    seed_c, seed_root = torsion_shell[(-1, -1, -1, 0)]
    assert seed_c == Q(6, 49)
    assert build_r_minus_5s_candidate(seed_c, seed_root).model == (
        0,
        0,
        427,
        -2731,
        0,
    )
    ordinary_shell = {
        coefficients: (c, root)
        for coefficients, c, root in r_minus_5s_shell(1, torsion_coset=False)
    }
    jump_c, jump_root = ordinary_shell[(0, 0, -1, 0)]
    assert jump_c == Q(17, 88)
    assert build_r_minus_5s_candidate(jump_c, jump_root).model == (
        0,
        0,
        1_727_642,
        -134_940_169,
        0,
    )
    deceptive_c, deceptive_root = ordinary_shell[(0, 0, -1, -1)]
    assert deceptive_c == Q(-9, 4)
    deceptive = build_r_minus_5s_candidate(deceptive_c, deceptive_root)
    assert deceptive.model == (0, 0, 24_156, -265_716, 0)
    assert deceptive.tame_conductor_lower_bound == 4_973_362_086

    # The factorized angle wall has no smaller nontrivial squareclass: every
    # squarefree |D|<23 except D=1 is locally obstructed, while D=-23 is the
    # first seeded nondegenerate class.
    assert sub_23_survivors() == (1,)
    assert local_obstructions(-23) == ()
    assert fixed_23_quartic_point(fixed_23_multiply(1))[0] == Q(1363, 279)
    assert fixed_23_conductor_divisor(1) == (
        (7, 373, 4931, 31957),
        411_441_293_837,
    )
    assert fixed_23_conductor_divisor(2) == (
        (61, 431, 1559, 2351),
        96_362_009_819,
    )
    assert fixed_23_conductor_divisor(3) == (
        (37, 431, 1559, 5231, 44497, 99409),
        575_262_767_427_105_436_499,
    )
    assert fourth_multiple_conductor_floor() == (
        2,
        0,
        400_000_040_000_001,
    )
    print("A2 weight-current attack tests passed")


if __name__ == "__main__":
    main()
