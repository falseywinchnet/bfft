\\ Exact arithmetic certificate for the denominator-resonant jackpot curve.
E = ellinit([0, 0, -250, -1225, 0]);
print("jackpot_global_reduction=", ellglobalred(E));
print("jackpot_root_number=", ellrootno(E));
print("jackpot_algebraic_rank_bounds=", ellrank(E));
print("jackpot_analytic_rank=", ellanalyticrank(E));

E13 = ellinit([0, 0, -3780, -44937, 0]);
print("norm13_global_reduction=", ellglobalred(E13));
print("norm13_root_number=", ellrootno(E13));
print("norm13_algebraic_rank_bounds=", ellrank(E13));

E7 = ellinit([0, 0, -202300, -8671156, 0]);
print("norm7_global_reduction=", ellglobalred(E7));
print("norm7_root_number=", ellrootno(E7));
print("norm7_algebraic_rank_bounds=", ellrank(E7));
quit;
