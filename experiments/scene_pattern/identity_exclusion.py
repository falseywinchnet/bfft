"""Transport reserved image footprints before adding mutual pair factors.

The split is frozen before fitting. Every retained identity alternative is used
for exclusion, so changing the selected identity cannot unlock reserved data.
This remains conditional on image discovery; it is not an unseen-image test.
"""
import numpy as np
from .identity_geometry import roles

RADIUS = 14.


def source_box(p):
    p=np.asarray(p);return np.r_[p-RADIUS,p+RADIUS]


def target_box(row):
    radius=np.abs(np.asarray(row['A'])).sum(axis=1)*RADIUS
    q=np.asarray(row['q']);return np.r_[q-radius,q+radius]


def touches(box, protected):
    return bool(len(protected) and np.any(np.all(box[:2] <= protected[:,2:],axis=1)
                &np.all(box[2:] >= protected[:,:2],axis=1)))


def freeze_split(groups, rows_bc, source_a, source_b):
    train_a,held_a,blocks_a=roles([dict(p=g['p']) for g in groups],source_a)
    train_b,held_b,blocks_b=roles(rows_bc,source_b)
    protected_b=[];protected_c=[]
    for held,g in zip(held_a,groups):
        if not held:continue
        protected_b.extend(target_box(r) for r in g['b'])
        protected_c.extend(target_box(r) for r in g['c'])
    for held,row in zip(held_b,rows_bc):
        if not held:continue
        protected_b.append(source_box(row['p']))
        protected_c.append(target_box(row))
    pb=np.asarray(protected_b).reshape(-1,4);pc=np.asarray(protected_c).reshape(-1,4)
    initial_a=int(train_a.sum());initial_b=len({r['site'] for k,r in enumerate(rows_bc) if train_b[k]})
    for i,g in enumerate(groups):
        if not train_a[i]:continue
        if any(touches(target_box(r),pb) for r in g['b']) or any(touches(target_box(r),pc) for r in g['c']):train_a[i]=False
    # Exclude a whole physical source group when any alternative touches a
    # reserved footprint. No pose or identity score influences the partition.
    blocked={r['site'] for i,r in enumerate(rows_bc) if train_b[i] and
             (touches(source_box(r['p']),pb) or touches(target_box(r),pc))}
    for i,r in enumerate(rows_bc):
        if r['site'] in blocked:train_b[i]=False
    return dict(training_a=train_a.tolist(),held_a=held_a.tolist(),
                training_b=train_b.tolist(),held_b=held_b.tolist(),
                blocks_a=blocks_a,blocks_b=blocks_b,
                receipt=dict(radius=RADIUS,source_training_before=initial_a,
                             source_training_after=int(train_a.sum()),
                             third_pair_training_before=initial_b,
                             third_pair_training_after=len({r['site'] for i,r in enumerate(rows_bc) if train_b[i]}),
                             protected_b_footprints=len(pb),protected_c_footprints=len(pc),
                             policy='all alternative footprints; symmetric across source and third-pair checks; frozen before fitting'))
