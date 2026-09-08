\\ Exact PARI certificates for the asymmetric three-split attack.

E8=ellinit([0,0,0,-225/20449,22519670625/49196294196049]);
print("ell0_model=",ellminimalmodel(E8)[1..5]);
print("ell0_rank=",ellrank(E8)[1..2]);
print("ell0_conductor=",ellglobalred(E8)[1]);

E6=ellinit([0,0,0,-256000000/13841287201,173961318400000000/191581231380566414401]);
print("ell4_model=",ellminimalmodel(E6)[1..5]);
print("ell4_rank=",ellrank(E6)[1..2]);
print("ell4_conductor=",ellglobalred(E6)[1]);
print("ell4_conductor_factorization=",factor(ellglobalred(E6)[1]));

E3=ellinit([0,0,0,-34012224000000/1628413597910449,2851614040049658086400000000/2651730845859653471779023381601]);
print("multiplier3_model=",ellminimalmodel(E3)[1..5]);
print("multiplier3_rank=",ellrank(E3)[1..2]);
print("multiplier3_conductor=",ellglobalred(E3)[1]);

\\ The symmetric (-K,0,+K) arithmetic-progression cover has rank zero.
Ezero=ellinit([0,-48,0,2304,0]);
print("zero_current_jacobian=",ellminimalmodel(Ezero)[1..5]);
print("zero_current_rank=",ellrank(Ezero)[1..2]);
print("zero_current_torsion=",elltors(Ezero));

\\ The torus-double square-discriminant quotient is rank one.
Er=ellinit([0,96,0,576,0]);
G=[-72,288];
T=[24,288];
print("square_discriminant_rank=",ellrank(Er)[1..2]);
print("square_discriminant_torsion=",elltors(Er));

Eprojection=ellinit(ellfromeqn(y^2+3*(x^4-14*x^2+1)));
print("square_projection_model=",ellminimalmodel(Eprojection)[1..5]);
print("square_projection_rank=",ellrank(Eprojection)[1..2]);
print("square_projection_torsion=",elltors(Eprojection));

Efifth=ellinit([0,0,0,-1,0]);
print("fifth_projection_rank=",ellrank(Efifth)[1..2]);
print("fifth_projection_torsion=",elltors(Efifth));

lift_hits=List();
check_lift(P,n,k)={
  if(#P==0,return());
  my(eta2=-P[1]/3);
  if(eta2<0 || !issquare(numerator(eta2)) || !issquare(denominator(eta2)),return());
  my(x2=eta2/2-1);
  if(x2>=0 && issquare(numerator(x2)) && issquare(denominator(x2)),
    listput(lift_hits,[n,k,eta2,x2]));
};
for(n=-128,128,for(k=0,3,check_lift(elladd(Er,ellmul(Er,G,n),ellmul(Er,T,k)),n,k)));
print("square_discriminant_double_lifts=",Vec(lift_hits));

\\ Exact fixed-sextic-current Thue closures.
f=x^6+6*x^5-33*x^4-220*x^3-93*x^2+966*x+1201;
Tf=thueinit(f);
print("sextic_norm_plus1=",thue(Tf,1));
print("sextic_norm_minus1=",thue(Tf,-1));
print("sextic_norm_729=",thue(Tf,729));
print("sextic_norm_minus55151=",thue(Tf,-55151));
