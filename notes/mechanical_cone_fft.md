# Mechanical cone FFT: dual gauges and reciprocal realizations

This note records the exact mathematics and the `N=8` computational mechanics
that grew out of the positive-cone BFFT work. It asks when a signed local FFT
stage can be represented by passive nonnegative transport or a reciprocal
mechanical linkage. It is not a fabrication claim.

Companion programs are in `experiments/mechanical_cone_fft/`.

## 1. Keep the three walks separate

| walk | traversal | coefficient locality | role here |
| --- | --- | --- | --- |
| DIF | factor-tree descent | one angle per node/span | primary two-rail flexure candidate |
| DIT | spectral ascent | angle varies with position | digital comparison form |
| DIP | third diagonal/phase walk | phase packet and span-diagonal angle law | algorithmic form; a phase-delay machine is optional and precision-intensive |

The mechanically simple object in this note is the real, two-rail,
cone-lifted DIF.

Claims below are labelled by their evidence: algebraic proof, machine-precision
certificate, or linear Euler--Bernoulli model. The last category is not
continuum FEA.

## 2. Cone lift is exact after projection, not before it

For a signed matrix `M`, let `M+ = max(M,0)` and `M- = max(-M,0)` and define

```text
         [ M+  M- ]
C(M)  =  [         ],          Pi = [I  -I].
         [ M-  M+ ]
```

Then

```text
Pi C(M) = M Pi,
Pi C(A) C(B) = AB Pi.                                  (2.1)
```

This is the exact property required by the physical circuit.

A correction matters: the canonical lift is not generally multiplicative
before projection. With `A=[1 1]` and `B=[1,-1]^T`, `AB=0` but `C(A)C(B)` has
equal nonzero mass on both rails. Thus

```text
C(A)C(B) != C(AB)
```

in general. They agree in the quotient by `ker(Pi)={(u,u)}`. Composition can
create common mode even when the signed result cancels to zero. Annihilation
changes the representative in this kernel without changing the answer.

## 3. Backward mass-potential gauge

For signed stages `x_(s+1)=M_s x_s`, set `g_S=1` and sweep backward:

```text
g_s = |M_s|^T g_(s+1),
A_s = diag(g_(s+1)) M_s diag(g_s)^-1.                  (3.1)
```

Every absolute column sum of `A_s` is one. Therefore `C(A_s)` is
column-stochastic and needs no active gain. The gauges telescope, so presenting
`diag(g_0)x` at the input and choosing `g_S=1` recovers the original signed map.

## 4. Forward kinematic gauge

There is a distinct diagonal gauge for displacement. Let `C_s=C(M_s)>=0`,
choose a positive `d_0`, and sweep forward componentwise:

```text
d_(s+1) = 1 / (C_s (1/d_s)),
B_s = diag(d_(s+1)) C_s diag(d_s)^-1.                  (4.1)
```

Then `B_s 1=1`; every output row is a convex combination. The stages telescope:

```text
B_(S-1)...B_0 = diag(d_S) C_(S-1)...C_0 diag(d_0)^-1. (4.2)
```

A fixed output calibration `diag(d_S)^-1`, followed by rail subtraction,
recovers the signed transform. No internal gain lever is required.

The two gauges are dual local normalizations:

| gauge | sweep | stochastic object | natural variable |
| --- | --- | --- | --- |
| mass potential | backward | columns | force / flow |
| kinematic | forward | rows | displacement |

## 5. Reciprocal transpose theorem

Let `A>=0` be a column-stochastic cone stage. Suppose an ideal reciprocal
linkage imposes

```text
q_in = A^T q_out.                                      (5.1)
```

For an admissible virtual displacement, `delta q_in=A^T delta q_out`. Virtual
work gives

```text
f_in^T delta q_in = (A f_in)^T delta q_out,
```

so the transmitted force is

```text
f_out = A f_in.                                       (5.2)
```

This yields a force-domain cone-DIF: implement the transpose as a displacement
constraint and the desired forward stage appears in generalized forces.
Because `A^T` is row-stochastic, every kinematic row is a convex interpolation.

## 6. Convex interpolation trees

A row-stochastic row

```text
y = sum_i w_i x_i,       w_i>=0,       sum_i w_i=1
```

can be built recursively from binary bars. For two inputs, the displacement at
a tap fraction `beta` is `(1-beta)x_0+beta x_1`. More inputs use a binary tree.
Under ideal pins and rigid interpolation members the construction is exact.

This gives two forms of the same cone-DIF:

1. Force form: mass-gauge the signed stages, cone-lift them, build
   `q_s=A_s^T q_(s+1)`, excite forces, and read forces.
2. Displacement form: cone-lift the original stages, apply the kinematic gauge,
   and build `q_(s+1)=B_s q_s` directly.

## 7. Dense reciprocal spring existence and the cascade failure

For a nonnegative doubly-stochastic matrix `B`,

```text
E(x,y)=1/2 sum_(j,i) B[j,i](y_j-x_i)^2                 (7.1)
```

has free-output equilibrium `y=Bx`. Its stiffness matrix is the symmetric graph
Laplacian

```text
K = [ diag(B^T 1)  -B^T ]
    [ -B             diag(B 1) ].                      (7.2)
```

Two-sided balancing of the cone-lifted `N=8` Fourier matrix gives an exact
dense reciprocal realization with 96 positive springs. It is an existence
result, not an FFT architecture, because it is `O(N^2)`.

Naively cascading isolated spring averages also fails: downstream stages load
upstream equilibria. In the lumped `N=8` control model, after one scalar gain
calibration:

| total stiffness span | transform shape error |
| ---: | ---: |
| 1 | 19.82% |
| 10 | 5.62% |
| 100 | 1.86% |
| 1,000 | 0.594% |
| 10,000 | 0.188% |
| 1,000,000 | 0.0189% |

The kinematic and force-whiffletree forms remove this back-loading rather than
hiding it behind an extreme stiffness hierarchy.

## 8. Exact normalized `N=8` Bruun form

Use the orthonormal packed order

```text
[DC, cos1, -sin1, cos2, -sin2, cos3, -sin3, Nyquist].
```

Define

```text
H_2m = (1/sqrt(2)) [I_m  I_m; I_m -I_m]
```

and the normalized Bruun cell

```text
R(theta)=(1/sqrt(2))*[ 1  0   c  -s
                       0  1   s   c
                       1  0  -c   s
                       0 -1   s   c ].
```

The certificate uses

```text
S0 = H8,
S1 = H4 direct_sum (diag(1,-1,1,-1) R(pi/4) P_[0,2,1,3]),
S2 = E1 direct_sum I4,
```

where

```text
E1=[1/sqrt(2)  1/sqrt(2)  0  0
    0          0          1  0
    0          0          0 -1
    1/sqrt(2) -1/sqrt(2)  0  0].
```

The natural output is
`[DC,cos2,-sin2,Nyquist,cos1,-sin1,cos3,-sin3]`; wire relabeling
`[0,4,5,1,2,6,7,3]` gives the packed Fourier matrix. The certified Frobenius
error is about `1e-15`.

## 9. Exact convex vocabulary at `N=8`

The forward kinematic gauge produces:

- 44 binary interpolation bars, split `[16,24,4]` by stage;
- 36 midpoint taps, `beta=1/2`;
- 8 taps at `beta=sqrt(2)-1`;
- 12 pass-through rows;
- no tap arm shorter than `(sqrt(2)-1)L`.

The non-midpoint tap retains about 94.2% of midpoint point-load compliance and
97.1% of midpoint peak bending moment in the local simply-supported surrogate.
Its relative placement sensitivity is 1.207 times the midpoint case.

## 10. Linear beam-model results

The companion beam model represents each binary bar by two Euler--Bernoulli
elements with independent connection rotations.

Force form census:

- 44 binary bars and 88 beam elements;
- 60 vertical displacement DOFs and 132 rotational DOFs;
- ideal fixed-output transform error about `2e-15`;
- worst error about `1.36e-10` in 100 seeded trials randomizing all stage `EI`
  and all 16 output-sensor stiffnesses over many orders of magnitude.

First-order modeled error laws are approximately

```text
e_hinge ~= 0.1697 (k_theta L/EI),
e_tap   ~= 3.08 sigma_x/L.
```

At `k_theta L/EI=0.01`, shape error is about `5.85e-4`. A `0.1%` one-sigma tap
error gives about `3.15e-3` median and `3.93e-3` 95th-percentile shape error.

For the displacement form, differential output compliance is isotropic when

```text
EI_1/EI_0 = (-338+248 sqrt(2))/7   = 1.817851924075...,
EI_2/EI_0 = (482-124 sqrt(2))/161  = 1.904580858793....
```

The computed isotropy defect is `8.2e-17`. Equal output loading then causes
only scalar attenuation

```text
gain^-1 = 1 + 0.021877079397... k_L
```

in the script's dimensionless units, with Fourier shape unchanged to numerical
precision after one global gain calibration.

## 11. Boundary of the result

Established algebraically or by machine-precision certificate:

1. projected cone composition;
2. the common-mode quotient correction;
3. both diagonal gauges;
4. reciprocal transpose force transport;
5. finite convex-tree realization;
6. the stated `N=8` factorization and tap inventory.

Established only in the linear model:

1. rigidity/load insensitivity of the ideal force divider;
2. hinge and tap sensitivity laws;
3. the `N=8` compliance-matching ratios;
4. the spring-cascade back-loading table.

Not established: collision-free layout, continuum stress, large-displacement
nonlinearity, fatigue, friction, drift, scalable routing/readout, or a general
proof of compliance matching for all powers of two.

The next mathematical target is a recursive compliance operator for the dyadic
Bruun convex tree. The next mechanical target is a two-input and three-input
coupon whose measured hinge parasitics replace the ideal-pin model.
