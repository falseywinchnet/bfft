\\ Exact certificate for the first-shell c=1/8 constant half-root curve.
E = ellinit([0, 0, 820, -7300, 0]);
print("global_reduction=", ellglobalred(E));
print("root_number=", ellrootno(E));
print("algebraic_rank=", ellrank(E));
print("analytic_rank=", ellanalyticrank(E));
quit;
