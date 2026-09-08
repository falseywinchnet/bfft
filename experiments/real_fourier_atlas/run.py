"""Exhaustive small real-atlas traversal and increasing-size random walks."""
import argparse
import json
import random
from collections import deque
import sympy as sp
from atlas import Atlas,canonical,partitions,neighbors


def exhaustive():
    a=Atlas(8)
    states=sorted(set(partitions(list(a.atoms))))
    edge_count=0
    # Every coordinate basis vector, every directed elementary edge.
    for p in states:
        for q in neighbors(p):
            assert q in states and p in neighbors(q)
            edge_count+=1
            for j in range(a.n):
                x=[int(i==j) for i in range(a.n)]
                assert a.move(a.oracle(x,p),q)==a.oracle(x,q)
    # Graph reachability; edges are all unit-cost here, no performance weights.
    distance={a.start:0};queue=deque([a.start])
    while queue:
        p=queue.popleft()
        for q in neighbors(p):
            if q not in distance:distance[q]=distance[p]+1;queue.append(q)
    assert len(distance)==len(states)
    return {"N":8,"states":len(states),"directed_edges":edge_count,
            "exact_basis_edge_checks":edge_count*8,
            "all_states_reachable":True,"root_to_leaves_edges":distance[a.end]}


def random_walk(n,steps,seed):
    a=Atlas(n);rng=random.Random(seed)
    x=[rng.randrange(-5,6) for _ in range(n)]
    state=a.input(x);visited=[]
    for step in range(steps):
        # General lateral repartition, not merely a next level of a chosen tree.
        count=rng.randrange(2,len(a.atoms)+1)
        bins=[[] for _ in range(count)]
        for k in a.atoms:bins[rng.randrange(count)].append(k)
        p=canonical(b for b in bins if b)
        state=a.move(state,p)
        assert state==a.oracle(x,p)
        visited.append(p)
    state=a.move(state,a.end)
    packed=a.readout(state)
    # Independent real cosine/sine evaluation, no FFT and no complex vector.
    target=[sum(x)]
    for k in range(1,n//2):
        target.extend([sum(x[j]*sp.chebyshevt((k*j)%n,a.c) for j in range(n)),
                       sum(x[j]*sp.chebyshevt(abs(n//4-(k*j)%n),a.c) for j in range(n))])
    target.append(sum(v*(-1)**j for j,v in enumerate(x)))
    assert all(a.domain.from_sympy(v-w)==a.domain.zero for v,w in zip(packed,target))
    # Close the complete loop back into the input chart.
    assert a.move(state,a.start)==a.input(x)
    return {"N":n,"arbitrary_repartitions":steps,"real_state_dimension":n,
            "endpoint_exact":True,"closed_loop_exact":True,
            "partition_path":[[list(b) for b in p] for p in visited]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result={"arithmetic":"exact real algebraic polynomial coefficients",
            "exhaustive":exhaustive()}
    print(json.dumps(result),flush=True)
    result["walks"]=[random_walk(n,12,7043+n) for n in (8,16)]
    with open(args.out,"w") as f:json.dump(result,f,indent=2);f.write("\n")
    print("All exact traversal, endpoint, and loop checks passed",flush=True)
