\\ Exact rank/conductor audit of the finite theory-derived dependent fibers.
names = ["record", "edge_quarter", "edge_fifth", "edge_seventh", "edge_ninth", "edge_eleventh", "first_rank6_capacity", "omega_half_minus9", "omega_half_4over5", "omega_third_minus9", "omega_third_4over5", "doubled_weight", "mixed_weight"];
models = [[0,0,1,-79,342], [0,0,833,-5341,0], [0,0,637,-3871,0], [0,0,125,-541,0], [0,0,287,-2221,0], [0,0,427,-2731,0], [0,0,184900,-7856401,0], [0,0,2430,-53217,0], [0,0,121500,-4446900,0], [0,0,46080,-2691072,0], [0,0,288000,-14054400,0], [0,0,-476280,-38292912,0], [0,0,-17640,-340452,0]];
for(i = 1, #models, E = ellinit(models[i]); print(names[i], " conductor=", ellglobalred(E)[1], " root=", ellrootno(E), " rank=", ellrank(E)[1..2]));
Ed = ellinit([0,0,-476280,-38292912,0]);
print("doubled_weight_global_reduction=", ellglobalred(Ed));
print("doubled_weight_conductor_factorization=", factor(ellglobalred(Ed)[1]));
E9 = ellinit([0,0,287,-2221,0]);
E11 = ellinit([0,0,427,-2731,0]);
print("edge_ninth_global_reduction=", ellglobalred(E9));
print("edge_eleventh_global_reduction=", ellglobalred(E11));
quit;
