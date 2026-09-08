\\ Exact rank-zero certificate for the norm-square / rational-shape boundary.
\\
\\ Parametrizing a^2=4*s^2-1 and imposing 1-3*s^2=h^2 gives
\\
\\     Y^2=-3*q^4+10*q^2-3.
\\
\\ The only rational classes on this genus-one curve are torsion/boundary.
E = ellinit(ellfromeqn(y^2 - (-3*x^4 + 10*x^2 - 3)));
M = ellminimalmodel(E);
print("model=", E[1..5]);
print("minimal=", M[1..5]);
print("conductor=", ellglobalred(M)[1]);
print("rank=", ellrank(E, 8));
print("torsion=", elltors(E));
quit;
