\\ Exact elliptic quotient ranks for the m=481 norm-preserving cover.

default(parisizemax, 4000000000);

models = [[1,0,0,-111875661590346504143900445378398523831926,-10355448163609740688374980453845901878151493030077446949812119],[0,1,0,-13094440062408185166824739951030301220016,-570246715811019126330712559235587493560601024232935455432880],[0,1,0,-1088332527181202692852209675776,-357197621008566738848718735836593172099301360]];
labels = ["B-times-norm", "A-times-norm", "Prym-A-B-times-norm"];
efforts = [0, 0, 2];

for(i = 1, #models, E = ellinit(models[i]); r = ellrank(E, efforts[i]); print(labels[i], " conductor=", ellglobalred(E)[1], " root=", ellrootno(E), " rank=[", r[1], ",", r[2], "]", " points=", r[4]));
