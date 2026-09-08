\\ Existing-bad-prime quadratic characters of the c=1/8 half-root curve.
target = 5187563742;
E = ellinit([0, 0, 820, -7300, 0]);
support = 2 * 5 * 7 * 419 * 3863;
ds = divisors(support);
hits = 0;
for(i = 1, #ds, forstep(sign = -1, 1, 2, d = sign * ds[i]; D = quaddisc(d); T = ellinit(elltwist(E, D)); N = ellglobalred(T)[1]; if(N < target, hits++; r = ellrank(T); M = ellminimalmodel(T); print("d=", d, " conductor=", N, " root=", ellrootno(T), " rank=", r[1..2], " model=", M[1..5]))));
print("subtarget_twists=", hits);
quit;
