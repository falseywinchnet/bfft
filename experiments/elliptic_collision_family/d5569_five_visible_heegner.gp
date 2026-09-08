\\ Heegner generator for the five-visible-section d=5569 parameter quotient.
\\ The cyclic-4 member of the 2-isogeny class is used because its descent
\\ has trivial unresolved 2-Selmer contribution.

allocatemem(4000000000);
E = ellinit([-27198708469013520391681/3,8971238718991075021130040693454658/27]);
r = ellrank(E);
if (r[1] != 1 || r[2] != 1, error("parameter rank is not exactly one"));
P = ellheegner(E);
if (!ellisoncurve(E,P), error("Heegner point is not on the isogenous curve"));
print("rank = ", r);
print("Heegner generator = ", P);
