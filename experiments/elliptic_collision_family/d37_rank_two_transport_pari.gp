\\ Exact PARI certificates for the d=37 rank-two transport.
default(parisizemax, 4000000000);

Epar = ellinit([0,1,0,-711167436,7193953224864]);
print("parameter N=", ellglobalred(Epar)[1]);
print("parameter rank=", ellrank(Epar,2));
print("parameter torsion=", elltors(Epar));

targets=[[0,0,1,-4107,114646],[0,0,1,-2566875,2467225156],[0,0,0,-5488331952,1011501404522660],[0,0,1,-640605825251723952,201869167863825640662818726]];
for(i=1,#targets,E=ellinit(targets[i]);print("target ",i," N=",ellglobalred(E)[1]," root=",ellrootno(E)," rank=",ellrank(E,2)));

Ejoint = ellinit([0,457380,0,-28860573456,-13200249087305280]);
print("joint N=", ellglobalred(Ejoint)[1]);
print("joint rank=", ellrank(Ejoint,2));
print("joint torsion=", elltors(Ejoint));
print("joint base height matrix=",ellheightmatrix(Ejoint,[[-387684,91998720],[-339768,100911096],[405108,341545248]]));

Etwist = elltwist(ellinit(targets[1]),-3);
print("base -3 twist N=",ellglobalred(Etwist)[1]," root=",ellrootno(Etwist)," rank=",ellrank(Etwist,2));

quit;
