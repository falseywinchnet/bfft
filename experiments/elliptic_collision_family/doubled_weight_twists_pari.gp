\\ Bad-prime-supported quadratic twists of the closest rank-five near miss.
E = ellinit([0, 0, -476280, -38292912, 0]);
radicands = [-21, -7, -3, -1, 1, 3, 7, 21];
discriminants = [-84, -7, -3, -4, 1, 12, 28, 21];
for(i = 1, #radicands, d = radicands[i]; T = ellinit(elltwist(E, discriminants[i])); r = ellrank(T); print("d=", d, " conductor=", ellglobalred(T)[1], " root=", ellrootno(T), " rank=", r[1..2], " model=", ellminimalmodel(T)));
quit;
