\\ Exact arithmetic certificates for the four-horizontal-triple family.

E6 = ellinit([0, 0, 0, -21609, 1350441]);
print("rank6_model=", ellminimalmodel(E6)[1..5]);
print("rank6_conductor=", ellglobalred(E6)[1]);
print("rank6_conductor_factorization=", factor(ellglobalred(E6)[1]));
print("rank6_bounds=", ellrank(E6, 8)[1..2]);

E7 = ellinit([0, 0, 0, -257049, 57729681]);
print("rank7_model=", ellminimalmodel(E7)[1..5]);
print("rank7_conductor=", ellglobalred(E7)[1]);
print("rank7_conductor_factorization=", factor(ellglobalred(E7)[1]));
print("rank7_bounds=", ellrank(E7, 8)[1..2]);

\\ If F2=4, coprimality of (D^3-1)/2 and (D^3+1)/2 reduces
\\ the nondegenerate support gate to one of
\\
\\     y^4 - 3*x^4 = 1,     3*y^4 - x^4 = 1.
\\
\\ If F1=4 the same reduction gives coefficient 27.  PARI's exact
\\ Thue solver proves that all four equations have only the boundary
\\ solution (or no solution).
T3plus = thueinit(x^4 - 3);
T3minus = thueinit(3*x^4 - 1);
T27plus = thueinit(x^4 - 27);
T27minus = thueinit(27*x^4 - 1);
print("norm4_F2_plus=", thue(T3plus, 1));
print("norm4_F2_minus=", thue(T3minus, 1));
print("norm4_F1_plus=", thue(T27plus, 1));
print("norm4_F1_minus=", thue(T27minus, 1));

\\ Perfect-square conductor cores are genus-one curves.  Their Jacobians
\\ are rank-zero j=0 twists.  The first torsion orbit maps back to
\\ g/r=+/-3; the second has only the two points at infinity.
J2 = ellinit(ellfromeqn(z^2 - (x^4 + 54*x^2 - 243)));
J1 = ellinit(ellfromeqn(z^2 - (x^4 - 162*x^2 - 2187)));
print("F2_square_jacobian=", ellminimalmodel(J2)[1..5]);
print("F2_square_rank=", ellrank(J2, 8)[1..2]);
print("F2_square_torsion=", elltors(J2));
print("F1_square_jacobian=", ellminimalmodel(J1)[1..5]);
print("F1_square_rank=", ellrank(J1, 8)[1..2]);
print("F1_square_torsion=", elltors(J1));

\\ If F2/4 is a cube, coprime factor allocation reduces the nonboundary
\\ case to w^2=x^4-8748.  Its rank-zero Jacobian has only boundary torsion.
J2cube = ellinit(ellfromeqn(z^2 - (x^4 - 8748)));
print("F2_cube_jacobian=", ellminimalmodel(J2cube)[1..5]);
print("F2_cube_rank=", ellrank(J2cube, 8)[1..2]);
print("F2_cube_torsion=", elltors(J2cube));

quit;
