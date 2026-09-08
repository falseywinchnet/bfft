\\ Exact PARI certificates for the m=1083 fixed-current recurrence.

allocatemem(1000000000);

parameter_raw = [0,0,0,-226419708672,38857935652724736];
parameter_model = ellminimalmodel(ellinit(parameter_raw));
parameter_coefficients = [parameter_model.a1,parameter_model.a2,parameter_model.a3,parameter_model.a4,parameter_model.a6];
if (parameter_coefficients != [0,1,0,-10919160,13009804308], error("unexpected parameter minimal model"));
if (ellglobalred(parameter_model)[1] != 8888880, error("unexpected parameter conductor"));
parameter_rank = ellrank(parameter_model);
if (parameter_rank[1] != 2 || parameter_rank[2] != 2, error("parameter rank is not exactly two"));

fourth_pair_raw = [0,0,0,-53112205872,-3570486866315136];
fourth_pair_model = ellminimalmodel(ellinit(fourth_pair_raw));
fourth_pair_coefficients = [fourth_pair_model.a1,fourth_pair_model.a2,fourth_pair_model.a3,fourth_pair_model.a4,fourth_pair_model.a6];
if (fourth_pair_coefficients != [0,-1,0,-40981640,-76514264400], error("unexpected fourth-height quotient minimal model"));
if (ellglobalred(fourth_pair_model)[1] != 31920, error("unexpected fourth-height quotient conductor"));
fourth_pair_rank = ellrank(fourth_pair_model);
if (fourth_pair_rank[1] != 0 || fourth_pair_rank[2] != 0, error("fourth-height quotient rank is not exactly zero"));
fourth_pair_torsion = elltors(fourth_pair_model);
if (fourth_pair_torsion[1] != 2 || fourth_pair_torsion[2] != [2], error("fourth-height quotient torsion is not Z/2"));

print("parameter model = ", parameter_coefficients);
print("parameter conductor = ", ellglobalred(parameter_model)[1]);
print("parameter rank = ", parameter_rank);
print("parameter generators = ", parameter_rank[4]);
print("fourth-height model = ", fourth_pair_coefficients);
print("fourth-height conductor = ", ellglobalred(fourth_pair_model)[1]);
print("fourth-height rank = ", fourth_pair_rank);
print("fourth-height torsion = ", fourth_pair_torsion);
