"""Exact structural tests for the seven-section collision family."""

from fractions import Fraction
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "elliptic_rank_support"
sys.path.insert(0, str(SUPPORT))

from elliptic_rank_support import (  # noqa: E402
    add_points,
    invariants,
    reduction_dependency_masks,
)
from elliptic_collision_family import (  # noqa: E402
    RECORD_GENERALIZED_MODEL,
    base_point_to_fiber,
    congruent_base_add,
    generalized_weierstrass_invariants,
    integral_right_triangles_with_area,
    is_prime_by_trial_division,
    pythagorean_rank_five_family,
    record_has_no_rational_two_torsion,
)
from seven_section_surface import (  # noqa: E402
    a2_weight_carriers,
    base_height_shell_forms,
    base_second_height_forms,
    cross_relation_forms,
    essential_discriminant,
    exposed_base_height_forms,
    extra_section_r_minus_s,
    extra_section_r_minus_s_conic,
    extra_section_r_minus_s_parameter,
    extra_section_2r_plus_s,
    extra_section_2r_plus_s_carrier,
    extra_section_r_plus_2s,
    extra_section_r_plus_2s_carrier,
    extra_section_r_plus_2s_parameter,
    homogeneous_discriminant,
    is_integer_square,
    linear_abscissa_carrier,
    linear_abscissa_section,
    nodal_center_unit,
    nodal_height_coordinate,
    positive_chamber_wall_offsets,
    primitive_support_square_carrier,
    quadratic6_norm,
    rank_six_extra_section_example,
    rank_safe_contour_t,
    short_discriminant,
    short_integral_scale,
    sixth_height_shell_current,
    sixth_height_shell_forms,
    surface_fiber,
    unit_offset_control_rhs,
)
from run_control_curve_descent import descent_report  # noqa: E402
from run_single_recurrence_attack import (  # noqa: E402
    CURRENT_DETERMINANT_MULTIPLIERS,
    SQRT2_NORM_SEEDS,
    CORE_SQUARECLASS_ROOTS,
    admissible_defect_residues,
    biquadratic_relative_norms,
    biquadratic_square,
    compressed_negative_pell_orbit,
    core_defect_covers,
    core_cover_level,
    descent_defect,
    defect_cover_quartic,
    first_counterexample_floor,
    first_forward,
    first_index_gate,
    first_orbit,
    first_predecessor,
    level5_currents_have_two_adic_contradiction,
    level5_first_current_two_adic_balance,
    level5_two_adic_balance,
    mod5_defect_covers,
    negative_pell_unit_shadow,
    negative_pell_orbit,
    second_forward,
    second_index_gate,
    second_orbit,
    second_predecessor,
    shadow_from_negative_pell,
    squareclass_radicand,
    quadratic_unit_power,
    reduced_vanishing_current,
    two_adic_valuation,
    vanishing_currents,
    vanishing_current_determinant,
    universal_quartic,
)


def five_generators(points):
    return (points[0], points[1], points[2], points[4], points[5])


def test_record_specialization_and_generic_rank_certificate() -> None:
    fiber = pythagorean_rank_five_family(1)
    assert (fiber.r, fiber.s) == (3, 7)
    assert fiber.center_height == 37
    model, points = fiber.integral_presentation(2)
    assert model == (0, -1264, 21904)
    assert invariants(model)[2] == -(2**12) * 19_047_851
    assert not reduction_dependency_masks(model, five_generators(points))
    c4, _, discriminant = generalized_weierstrass_invariants(RECORD_GENERALIZED_MODEL)
    assert (c4, discriminant) == (3_792, -19_047_851)
    assert is_prime_by_trial_division(-discriminant)
    assert c4 % -discriminant
    assert record_has_no_rational_two_torsion()


def test_two_horizontal_principal_divisors() -> None:
    fiber = pythagorean_rank_five_family(1)
    model, points = fiber.integral_presentation(2)
    for triple in (points[1:4], points[4:7]):
        total = None
        for point in triple:
            total = add_points(total, point, model)
        assert total is None


def test_integral_orbit_classification_and_rank_four_trap() -> None:
    assert integral_right_triangles_with_area(840) == (
        (15, 112, 113),
        (24, 70, 74),
        (40, 42, 58),
    )
    trapped = base_point_to_fiber((Fraction(1960), Fraction(78400)))
    model, points = trapped.integral_presentation(2)
    generators = five_generators(points)
    # P0 + P_s + P_-r is the extra cross-line relation.
    total = None
    for point in (generators[0], generators[2], generators[3]):
        total = add_points(total, point, model)
    assert total is None


def test_first_two_generator_orbit_fiber_has_rank_five() -> None:
    area = 840
    record_base = (Fraction(1176), Fraction(28224))
    trapped_base = (Fraction(1960), Fraction(78400))
    combined = congruent_base_add(record_base, trapped_base, area)
    fiber = base_point_to_fiber(combined)
    assert fiber.center_height == Fraction(-113, 2)
    model, points = fiber.integral_presentation(2)
    assert model == (0, -1264, 51076)
    assert not reduction_dependency_masks(model, five_generators(points))


def test_canonical_adjacent_chart_fiber_keeps_rank_five() -> None:
    fiber = pythagorean_rank_five_family(Fraction(5, 4))
    model, points = fiber.integral_presentation(8)
    assert model == (0, -199600, 28090000)
    assert not reduction_dependency_masks(model, five_generators(points))


def test_surface_record_and_discriminant_factorization() -> None:
    c, t, w = Fraction(7, 3), Fraction(1, 6), Fraction(54, 35)
    fiber = surface_fiber(c, t, w)
    assert (fiber.r, fiber.s) == (3, 7)
    assert (fiber.lower_height, fiber.center_height, fiber.upper_height) == (
        23,
        37,
        47,
    )
    transport = t * (1 - t * t)
    expected = (
        transport**4
        * c**6
        * (1 + c) ** 6
        * w**12
        * essential_discriminant(c, t)
    )
    assert short_discriminant(fiber) == expected == -19_047_851


def test_rank_loss_wall_and_safe_contour() -> None:
    record_c, record_t = Fraction(7, 3), Fraction(1, 6)
    trapped_t = Fraction(2, 5)
    assert cross_relation_forms(record_c, record_t)[4] == Fraction(-7, 9)
    assert cross_relation_forms(record_c, trapped_t)[4] == 0
    assert rank_safe_contour_t(record_c) == record_t
    for c in (Fraction(7, 3), Fraction(5, 2), Fraction(12, 5)):
        t = rank_safe_contour_t(c)
        assert cross_relation_forms(c, t)[4] == Fraction(-7, 9)


def test_first_simple_safe_contour_fiber() -> None:
    c = Fraction(5, 2)
    t = rank_safe_contour_t(c)
    assert t == Fraction(13, 63)
    fiber = surface_fiber(c, t)
    scale = short_integral_scale(fiber)
    assert scale == 1134
    model, points = fiber.integral_presentation(scale)
    assert model == (
        0,
        -48_181_857_750_000,
        132_214_754_848_556_250_000,
    )
    assert not reduction_dependency_masks(model, five_generators(points))


def test_nodal_center_is_fixed_norm_one_support() -> None:
    unit = nodal_center_unit()
    assert unit == (Fraction(-5), Fraction(-2))
    assert quadratic6_norm(unit) == 1
    # H(U0)=1/(6*sqrt(6))=sqrt(6)/36.
    assert nodal_height_coordinate(unit) == (0, Fraction(1, 36))


def test_arithmetic_eikonal_carrier_and_complete_wall_offsets() -> None:
    carrier = homogeneous_discriminant(7, 3, 1, 6)
    assert carrier == -(210**2) * 19_047_851
    assert positive_chamber_wall_offsets(7, 3, 1, 6) == (
        -1,
        -8,
        -11,
        -29,
        -32,
        -14,
    )
    # The formerly hidden rank-four relation is the first signed wall.
    assert positive_chamber_wall_offsets(5, 2, 1, 6)[0] == 0
    # The opposite simple neighbor lies on the sixth wall.
    assert positive_chamber_wall_offsets(3, 2, 1, 5)[5] == 0


def test_primitive_support_gate_and_control_curve() -> None:
    record_square = primitive_support_square_carrier(7, 3, 1, 6)
    assert record_square == 210**2
    assert is_integer_square(record_square)
    assert not is_integer_square(primitive_support_square_carrier(5, 2, 1, 7))
    assert unit_offset_control_rhs(3) == 420**2


def test_hidden_signed_wall_and_safe_even_neighbor() -> None:
    trapped = surface_fiber(Fraction(5, 2), Fraction(1, 6))
    scale = short_integral_scale(trapped)
    model, points = trapped.integral_presentation(scale)
    # P0 - lower_s - upper_(r+s) is the formerly hidden third relation.
    relation = None
    for point in (points[0], (points[2][0], -points[2][1]), (points[6][0], -points[6][1])):
        relation = add_points(relation, point, model)
    assert relation is None

    safe = surface_fiber(Fraction(5, 2), Fraction(1, 7))
    scale = short_integral_scale(safe)
    model, points = safe.integral_presentation(scale)
    assert model == (0, -35_100, 3_515_625)
    assert not reduction_dependency_masks(model, five_generators(points))


def test_control_curve_local_two_descent() -> None:
    report = descent_report()
    assert tuple((modulus, before, len(after)) for modulus, before, after in report) == (
        (8, 54, 14),
        (3, 14, 5),
        (5, 5, 2),
        (7, 2, 1),
    )
    assert report[-1][2] == ((3, 7, 10),)


def test_single_recurrence_unit_flow_and_strict_predecessors() -> None:
    first = (1, 1)
    second = (1, 1)
    for index in range(8):
        assert first == first_orbit(index)
        assert second == second_orbit(index)
        assert 15 * first[0] ** 2 - 14 * first[1] ** 2 == 1
        assert 21 * second[0] ** 2 - 20 * second[1] ** 2 == 1
        if index:
            assert first_predecessor(*first) == first_orbit(index - 1)
            assert second_predecessor(*second) == second_orbit(index - 1)
        first = first_forward(*first)
        second = second_forward(*second)


def test_negative_pell_sqrt2_compression() -> None:
    for index in range(12):
        v, w = negative_pell_orbit(index)
        assert (v, w) == compressed_negative_pell_orbit(index)
        assert 49 * v * v - 50 * w * w == -1


def test_index_gates_defect_and_counterexample_floor() -> None:
    assert [index for index in range(20) if first_index_gate(index)] == [0, 9, 10, 19]
    assert [index for index in range(28) if second_index_gate(index)] == [0, 13, 14, 27]
    assert admissible_defect_residues() == (0, 14, 50, 64)
    certificate = descent_defect(1, 1, 1)
    assert certificate.h == 0
    assert certificate.half_h == certificate.half_gap == 0
    assert certificate.factor_gcd == 3
    assert certificate.first_u == certificate.second_u == 1
    assert first_counterexample_floor() == 7_472_659_988_568_811_532_839_281
    assert mod5_defect_covers() == (
        (1, 2),
        (1, 7),
        (1, 10),
        (1, 35),
        (3, 1),
        (3, 5),
        (3, 14),
        (3, 70),
    )
    assert core_defect_covers() == (
        (1, 2),
        (1, 7),
        (1, 35),
        (3, 5),
        (3, 14),
        (3, 70),
    )
    assert tuple(core_cover_level(*cover) for cover in core_defect_covers()) == (
        6,
        21,
        105,
        5,
        14,
        70,
    )
    # The quartic is the Pell norm after removing the common factor g/3.
    assert defect_cover_quartic(1, 0, 1) == 70**2


def test_universal_biquadratic_norm() -> None:
    for A, b in ((1, 1), (2, 3), (35, 17), (123, 1_019)):
        value = universal_quartic(A, b)
        for discriminant, (rational, radical) in biquadratic_relative_norms(A, b).items():
            assert rational * rational - discriminant * radical * radical == value
    for g, d in core_defect_covers():
        for a, b in ((1, 1), (3, 2), (7, 11)):
            assert d * d * defect_cover_quartic(d, a, b) == universal_quartic(d * a, b)
    assert {
        level: tuple(x * x - 2 * y * y for x, y in seeds)
        for level, seeds in SQRT2_NORM_SEEDS.items()
    } == {
        5: (25,),
        6: (36,),
        14: (196, 196, 196),
        21: (441, 441, 441),
        70: (4_900, 4_900, 4_900),
        105: (11_025, 11_025, 11_025),
    }
    for index in range(20):
        assert shadow_from_negative_pell(*negative_pell_orbit(index)) == negative_pell_unit_shadow(index)
    for level, (_, _, root) in CORE_SQUARECLASS_ROOTS.items():
        assert biquadratic_square(root) == squareclass_radicand(level)


def test_vanishing_current_descent() -> None:
    def ratio_polynomials(a: int, b: int, c: int, d: int) -> tuple[Fraction, Fraction]:
        x = Fraction(b, a)
        y = Fraction(d, c)
        p = (
            42 * x * x * y
            - 7 * x * x
            + 420 * x * y * y
            - 2 * x
            + 735 * y * y
            - 21 * y
        )
        q = (
            26_460 * x * x * y * y
            + 3_444 * x * x * y
            + 112 * x * x
            + 840 * x * y * y
            - 4 * x
            - 11_760 * y * y
            - 1_722 * y
            - 63
        )
        return p, q

    samples = {5: 2, 6: 1, 14: 2, 21: 1, 70: 2, 105: 1}
    i, ell = 5, 7
    c, d = quadratic_unit_power(210, 29, 2, i)
    e, f = quadratic_unit_power(105, 41, 4, ell)
    for level, n in samples.items():
        a, b = quadratic_unit_power(2, 7, 5, n)
        p, q = ratio_polynomials(a, b, c, d)
        selected = p if level in (5, 21, 105) else q
        assert selected * a * a * c * c == reduced_vanishing_current(
            level, a, b, c, d
        )
        assert vanishing_current_determinant(level, n, i) == (
            CURRENT_DETERMINANT_MULTIPLIERS[level]
            * reduced_vanishing_current(level, a, b, c, d)
        )

    # The second currents expose a product law on the two surviving signs.
    a5, b5 = quadratic_unit_power(2, 7, 5, 2)
    assert vanishing_currents(5, 2, i, ell)[1] == 105 * a5 * d * f - b5 * c * e
    a6, b6 = quadratic_unit_power(2, 7, 5, 1)
    level6_second = vanishing_currents(6, 1, i, ell)[1]
    assert level6_second == 3 * (
        a6 * e * (c + 14 * d) - 14 * b6 * f * (c + 15 * d)
    )
    assert level6_second % 2 == 1

    # The other paired levels are excluded by a strict reduced-current sign.
    assert reduced_vanishing_current(14, a5, b5, c, d) < 0
    assert reduced_vanishing_current(21, a6, b6, c, d) < 0
    assert reduced_vanishing_current(70, a5, b5, c, d) < 0
    assert reduced_vanishing_current(105, a6, b6, c, d) < 0

    for exponent in (2, 4, 6, 8, 10, 12):
        _, coefficient = quadratic_unit_power(2, 7, 5, exponent)
        assert two_adic_valuation(coefficient) == two_adic_valuation(exponent)
    for exponent in (5, 10, 15, 20):
        _, coefficient = quadratic_unit_power(210, 29, 2, exponent)
        assert two_adic_valuation(coefficient) == two_adic_valuation(exponent) + 1
    for exponent in (7, 14, 21, 28):
        _, coefficient = quadratic_unit_power(105, 41, 4, exponent)
        assert two_adic_valuation(coefficient) == two_adic_valuation(exponent) + 2
    assert level5_two_adic_balance(8, 5, 7) == (3, 3)
    assert level5_first_current_two_adic_balance(8, 5) == (4, 1)
    for n, i, ell in ((2, 5, 7), (8, 5, 7), (16, 10, 14), (64, 20, 28)):
        assert level5_currents_have_two_adic_contradiction(n, i, ell)


def test_parametrized_extra_section_rank_six_example() -> None:
    t = Fraction(1, 3)
    slope = -2 * (1 + t * t)
    c, z = extra_section_r_minus_s_parameter(t, slope)
    assert (c, z) == (149, Fraction(-980, 3))
    assert z * z == extra_section_r_minus_s_conic(c, t)
    fiber = surface_fiber(c, t)
    x, y = extra_section_r_minus_s(c, t, z)
    assert (x, y) == (Fraction(-8_820_800, 9), Fraction(-4_351_396_000, 27))
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4

    model, points = rank_six_extra_section_example()
    assert invariants(model)[2] == (
        2**12 * 149**6 * 14_627 * 48_839_731_007
    )
    assert not reduction_dependency_masks(model, points, count=6)
    # No rational 2-torsion: the monic 2-division cubic has no root mod 3.
    assert all((x**3 + model[1] * x + model[2]) % 3 for x in range(3))


def test_second_edge_difference_current() -> None:
    t = Fraction(1, 7)
    root = Fraction(10, 7)
    c = extra_section_r_plus_2s_parameter(t, root)
    assert c == Fraction(-17, 84)
    assert extra_section_r_plus_2s_carrier(c, t) == root * root
    fiber = surface_fiber(c, t)
    x, y = extra_section_r_plus_2s(c, t, root)
    assert x == (1 + 2 * c) * fiber.r
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4


def test_factored_sixth_height_loss_current() -> None:
    # Direct sixth-section relations from the directed wall descent.
    quarter = sixth_height_shell_forms(Fraction(-12, 7), Fraction(1, 4))
    fifth = sixth_height_shell_forms(Fraction(-10, 7), Fraction(1, 5))
    ninth = sixth_height_shell_forms(Fraction(-49, 4), Fraction(1, 9))
    eleventh = sixth_height_shell_forms(Fraction(6, 49), Fraction(1, 11))
    assert [index for index, value in enumerate(quarter) if value == 0] == [3]
    assert [index for index, value in enumerate(fifth) if value == 0] == [3]
    assert [index for index, value in enumerate(ninth) if value == 0] == [4]
    assert [index for index, value in enumerate(eleventh) if value == 0] == [2]
    assert sixth_height_shell_current(Fraction(-12, 7), Fraction(1, 4)) == 0

    # The fifth is a transverse intersection of sixth loss with base loss;
    # the seventh is a base-lattice collapse but misses all eight sixth forms.
    assert exposed_base_height_forms(Fraction(-10, 7), Fraction(1, 5))[0] == 0
    assert exposed_base_height_forms(Fraction(21, 4), Fraction(1, 7))[1] == 0
    assert all(
        sixth_height_shell_forms(Fraction(21, 4), Fraction(1, 7))
    )
    assert base_height_shell_forms(Fraction(21, 4), Fraction(3, 4))[1] == 0
    assert base_second_height_forms(Fraction(-9), Fraction(1, 2))[0] == 0
    assert base_second_height_forms(Fraction(-9), Fraction(1, 3))[1] == 0


def test_reciprocal_edge_difference_current() -> None:
    c, t = Fraction(-12, 7), Fraction(1, 4)
    root = Fraction(39, 28)
    assert extra_section_2r_plus_s_carrier(c, t) == root * root
    fiber = surface_fiber(c, t)
    x, y = extra_section_2r_plus_s(c, t, root)
    assert x == (2 + c) * fiber.r
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4


def test_universal_linear_abscissa_and_weight_currents() -> None:
    c, t = Fraction(-12, 7), Fraction(1, 4)
    root = c * (1 + c) * (1 - 2 * t - t * t)
    assert linear_abscissa_carrier(c, t, 1, 0) == root * root
    assert linear_abscissa_section(c, t, 1, 0, root) == surface_fiber(
        c, t
    ).seven_points()[1]
    assert a2_weight_carriers(c, t) == (
        Fraction(1_170_125, 806_736),
        Fraction(2_541_925, 806_736),
    )


if __name__ == "__main__":
    test_record_specialization_and_generic_rank_certificate()
    test_two_horizontal_principal_divisors()
    test_integral_orbit_classification_and_rank_four_trap()
    test_first_two_generator_orbit_fiber_has_rank_five()
    test_canonical_adjacent_chart_fiber_keeps_rank_five()
    test_surface_record_and_discriminant_factorization()
    test_rank_loss_wall_and_safe_contour()
    test_first_simple_safe_contour_fiber()
    test_nodal_center_is_fixed_norm_one_support()
    test_arithmetic_eikonal_carrier_and_complete_wall_offsets()
    test_primitive_support_gate_and_control_curve()
    test_hidden_signed_wall_and_safe_even_neighbor()
    test_control_curve_local_two_descent()
    test_single_recurrence_unit_flow_and_strict_predecessors()
    test_negative_pell_sqrt2_compression()
    test_index_gates_defect_and_counterexample_floor()
    test_universal_biquadratic_norm()
    test_vanishing_current_descent()
    test_parametrized_extra_section_rank_six_example()
    test_second_edge_difference_current()
    test_factored_sixth_height_loss_current()
    test_reciprocal_edge_difference_current()
    test_universal_linear_abscissa_and_weight_currents()
    print("elliptic collision-family tests passed")
