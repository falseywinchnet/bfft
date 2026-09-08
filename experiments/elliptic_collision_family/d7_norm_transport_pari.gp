\\ Exact PARI certificates for the d=7 norm-current transport.

allocatemem(1000000000);

parameter_model = ellinit([0,1,0,-14916,205884]);
parameter_rank = ellrank(parameter_model);
if (ellglobalred(parameter_model)[1] != 34320, error("unexpected parameter conductor"));
if (parameter_rank[1] != 1 || parameter_rank[2] != 1, error("parameter rank is not exactly one"));
if (elltors(parameter_model)[1] != 4 || elltors(parameter_model)[2] != [2,2], error("parameter torsion is not (Z/2)^2"));

first_model = ellminimalmodel(ellinit([0,0,0,-147,2825/4]));
first_rank = ellrank(first_model);
if ([first_model.a1,first_model.a2,first_model.a3,first_model.a4,first_model.a6] != [0,0,1,-147,706], error("unexpected first target model"));
if (ellglobalred(first_model)[1] != 50121, error("unexpected first target conductor"));
if (first_rank[1] != 3 || first_rank[2] != 3, error("first target rank is not exactly three"));

second_height = 76719/1148;
second_rational = ellinit([0,0,0,-147,686+second_height^2]);
second_model = ellminimalmodel(second_rational,&second_change);
second_rank = ellrank(second_model);
second_coefficients = [second_model.a1,second_model.a2,second_model.a3,second_model.a4,second_model.a6];
if (second_coefficients != [0,0,0,-15957501882672,184268088879537135620], error("unexpected second target model"));
if (ellglobalred(second_model)[1] != 1750328941213617099804, error("unexpected second target conductor"));
if (second_rank[1] != 6 || second_rank[2] != 6, error("second target rank is not exactly six"));
if (second_change != [1/574,0,0,0], error("unexpected second target change of variables"));
if (setsearch(Set(second_rank[4]),[4749784,14684367774]) == 0, error("invisible point is absent from exact rank-six basis"));

missing_raw = ellinit([0,0,0,-262590768,-1590029889408]);
missing_model = ellminimalmodel(missing_raw);
missing_rank = ellrank(missing_model);
if ([missing_model.a1,missing_model.a2,missing_model.a3,missing_model.a4,missing_model.a6] != [0,-1,0,-202616,-34012320], error("unexpected missing-height quotient model"));
if (ellglobalred(missing_model)[1] != 1680, error("unexpected missing-height quotient conductor"));
if (missing_rank[1] != 0 || missing_rank[2] != 0, error("missing-height quotient rank is not exactly zero"));
if (elltors(missing_model)[1] != 2 || elltors(missing_model)[2] != [2], error("missing-height quotient torsion is not Z/2"));

print("parameter rank = ", parameter_rank);
print("first target = ", [first_model.a1,first_model.a2,first_model.a3,first_model.a4,first_model.a6]);
print("first conductor/rank = ", [ellglobalred(first_model)[1],first_rank]);
print("second target = ", second_coefficients);
print("second conductor/rank = ", [ellglobalred(second_model)[1],second_rank]);
print("missing-height quotient/rank = ", [[missing_model.a1,missing_model.a2,missing_model.a3,missing_model.a4,missing_model.a6],missing_rank]);
