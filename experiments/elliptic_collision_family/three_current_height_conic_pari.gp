\\ Exact rank and conductor witnesses for the universal height conic.

default(parisizemax, 4000000000);

models = [[0,0,0,-1070797/640000,-42463741/128000000],[0,0,0,-214651801/134239063615699680000,22347505019/33395264473949746495776000000],[0,0,0,-10144225/5415893679197092608,195322025/270627366325237457093167104],[0,0,0,-701103611827128601/4558121093515815624068943552,53371736382630588438115229/72683861022931995737003072064946978904832],[0,0,0,-41697732076/96382187058249,597921192477532/105136291962188581923],[0,-1,0,-10612680,12761798400],[0,0,0,-1953,48177],[0,0,0,-313348,67164772]];

labels = ["m637-s3/2", "self-s2", "self-s1/2", "self-s92/67", "m91-four-height-first", "m217-four-height-jacobian", "m217-subrecord-rank5", "m217-best-rank6"];

for (i = 1, #models, E = ellinit(models[i]); M = ellminimalmodel(E, &v); r = ellrank(M); print(labels[i], " model=", [M.a1,M.a2,M.a3,M.a4,M.a6], " conductor=", ellglobalred(M)[1], " root=", ellrootno(M), " rank=[", r[1], ",", r[2], "]"));
