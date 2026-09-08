# Local real Fourier exchanges and an executable real FFT

2026-09-05. This follows the arbitrary residue-chart atlas. The previous state
made changing one frequency location depend on all packets. The new state
stores successive quotient coefficients, which permits strictly local exchanges.
The experiment includes both an unrestricted local walk and a balanced
O(N log N) real-to-packed-Fourier transform. Their distinction matters:
the unrestricted walk is not yet a new fast FFT; the balanced normalized
route recovers Bruun-type arithmetic and provides a verified Fourier endpoint.

## A local state, with no coprimality restriction

Choose an ordered list of monic real quadratic factors p_i and write

    f = a_0 + p_0(a_1 + p_1(a_2 + ...)),   degree(a_i)<2.

This is a block Newton expansion. Successive monic division makes it unique
regardless of whether factors share roots. With N/2 factors it contains N
real scalar coefficients. Unlike separate CRT residues, this representation
remains valid when factors coincide: repeated factors encode successive jets.

The ordinary real sample vector already is this state with every p_i=t²:

    a_i = x_(2i) + x_(2i+1)t.

No input FFT, change to a complex vector, or initial numerical solve is needed.
The current factor order and the N real coefficients are the complete state.
The polynomial reconstruction method in the test code is an oracle only;
exchange, replacement, and Fourier extraction do not call it.

The Newton basis and polynomial interpolation foundations are established:
https://dlmf.nist.gov/3.3
The purpose here is to derive and test the real quadratic exchange as a
candidate transition for the user's Fourier walk, not to assert priority
for Newton interpolation or its reorderings.

## The four-coordinate exchange

Locally, the expansion has

    a + p b + p q (remaining tail).

Exchange p and q by finding

    a + p b = a' + q b'.

Write p=t²+p1*t+p0, q=t²+q1*t+q0, a=a0+a1*t, b=b0+b1*t.
With d0=p0-q0, d1=p1-q1, g=d1*b1, the exact update is

    a0' = a0 + d0*b0 - q0*g
    a1' = a1 + d1*b0 + d0*b1 - q1*g
    b0' = b0 + g
    b1' = b1.

Only these four real coordinates change. The earlier prefix and later tail
are unaffected because pq=qp. There are no divisions by root separations,
so this rule is defined at collisions as well as at distinct factors.

For two unit-circle conjugate factors, p0=q0=1, it simplifies to

    g=(p1-q1)*b1
    a0'=a0-g
    a1'=a1+(p1-q1)*b0-q1*g
    b0'=b0+g, b1'=b1.

This simplified arithmetic uses three real multiplications and four additions.
The generic Python oracle intentionally uses the general formula.

The exchange is reversible. It also obeys the braid/path relation

    S_i S_(i+1) S_i = S_(i+1) S_i S_(i+1),

with factor metadata transported alongside coefficients. Both paths preserve
f and end in the same ordered factor list; uniqueness of the Newton expansion
therefore proves equality for arbitrary coefficients. Disjoint exchanges
commute. The tests include a symbolic general inverse, a concrete braid
identity, and 80 exact arbitrary exchanges/replacements.

## Moving locations through local steps

The last factor does not occur in the degree-bounded expansion: it multiplies
the identically zero remaining tail. It can therefore change without changing
any data. To replace an arbitrary factor, move it to the last position by
adjacent exchanges, change it, and move it back. Every data-changing operation
still touches only four neighboring real coordinates. This is history-free
local transport, but the number of local steps grows with travel distance.

This is a real improvement over the previous residue atlas's all-packet
transition formula. It is not a claim that arbitrary movement has constant
total cost. The relation to the rest of the polynomial is carried by quotient
coefficients and the ordered prefix factors.

## An actual real-to-Fourier walk

`walk_rfft` starts directly from real samples, changes its quadratic factors to

    t²-1  (DC and Nyquist together),
    t²-2 cos(2πk/N)t+1,  k=1,...,N/2-1,

and exposes each factor at the first position using local exchanges. The first
Newton pair is then f mod that factor. For a unit-circle factor, residue u+vt
gives (Re,-Im)=(u+v cos(theta),v sin(theta)); the t²-1 packet gives (u+v,u-v).
All computation is real. A separate N-real output buffer receives the result.

For M=N/2 factors, the implemented conversion/extraction path takes exactly
3M(M-1)/2 exchanges. It is a quadratic DFT algorithm, not an FFT. This explicit
count prevents arbitrary local motion from being mistaken for a short route.

Its floating-point conditioning depends strongly on factor order. For the
seeded N=128 case, natural order has relative L2 error about 1.10e14 and sampled
coordinate magnitudes up to 1.16e29. Bit-reversed order has relative L2 error
about 1.41e-12 and sampled magnitudes around 347. Both paths are exact over
exact arithmetic. Avoiding division by root differences does not, by itself,
guarantee stable coordinates. These measurements are diagnostic, not passing
accuracy claims for the unrestricted walk.

## The same exchange at whole-packet scale

Replace t by t^h in the factor parameters, and let a0,a1,b0,b1 each be a
polynomial of degree<h. The SAME four-stream rule exchanges

    p=t^(2h)+p1*t^h+p0,   q=t^(2h)+q1*t^h+q0.

The rule now updates 4h real coordinates in O(h) work. No new dense carry
matrix appears for this sparse factor family. Exact polynomial tests cover
h=1,2,4,8; the substitution identity proves the extension for arbitrary h.

This provides the connection from a chain of small exchanges to hierarchical
packet movement. At general equal-degree factors A,B, the rule is

    a' = a + rem((A-B)b, B)
    b' = b + quo((A-B)b, B).

Sparse A-B makes the exchange cheap; an arbitrary dense factor pair need not
have that property. The state space is broad, but its short hierarchical
routes depend on the geometry of the factor groups.

## From the local move to an FFT split

For f=a+A*b, the two child residues are a=f mod A and a'=f mod B.
An FFT split can therefore be viewed as exposing the first packet before
and after the exchange. This identifies the branch operation within the
same real polynomial geometry; it does not require a complex row transform.

`quotient_fft.py` implements a balanced split hierarchy. For a parent

    t^(4h) - 2c*t^(2h) + 1,

put C=cos(theta/2), c=cos(theta)=2C²-1. Its children are
t^(2h)-2C*t^h+1 and t^(2h)+2C*t^h+1. If the parent residue has blocks
r0+r1*t^h+r2*t^(2h)+r3*t^(3h), its child residues are

    low0 = r0-r2-2C*r3
    low1 = r1+2C*r2+(2c+1)*r3
    high0 = r0-r2+2C*r3
    high1 = r1-2C*r2+(2c+1)*r3.

The real DC/Nyquist ridge uses the binomial split t^L-1 into t^(L/2)±1.
Each level performs O(N) real arithmetic and halves packet sizes. The default
depth-first stack implementation therefore has O(N log N) work and O(N)
working memory. It computes ordinary real-input Fourier output in natural
bin order. No FFT is called inside the transform; NumPy FFT appears only
in the independent tests.

## The coordinate normalization matters

For a packet r=u(t)+t^w v(t) modulo t^(2w)-2c*t^w+1, use real coordinates

    A=u+c*v,   B=s*v,   c=cos(theta), s=sin(theta).

In these coordinates the split is

    r=C*A1-S*B1;   j=S*A1+C*B1
    low=(A0+r, B0+j)
    high=(A0-r, j-B0),

where C=cos(theta/2), S=sin(theta/2). This is Bruun-type real-pair arithmetic,
derived here as a stable frame for the quotient packets. It is not claimed
as a new FFT primitive. At a leaf, (A,B) already is (Re,-Im), avoiding a poorly
conditioned final polynomial evaluation.

The unnormalized and normalized constructions both reach the Fourier endpoint.
At N=16384, the tested normalized route has maximum relative L2 error 3.65e-16
and maximum absolute error 1.43e-13. The raw residue route has relative error
1.16e-14 and absolute error 4.54e-11. These maxima cover seeded random,
constant, and impulse inputs, not a worst-case stability theorem.

## Scheduling, storage, and limits

Depth-first, breadth-first, and an interleaved task order produce bit-identical
answers. They reorder independent completed-packet tasks; this alone is NOT
the arbitrary noncommuting traversal demonstrated by the Newton exchange.
The alternate Python queues are correctness controls, not fast schedulers.

The active packet frontier has at most N real coordinates. Output storage and
NumPy expression temporaries are additional; the experiment does not claim
an in-place production implementation. Trigonometric plan constants are
currently computed during traversal and excluded from the arithmetic counters;
there is no native speed comparison in this experiment.

The completed bridge is: real samples -> quotient states with local arbitrary
exchanges -> hierarchical real Fourier splitting -> actual Fourier output.
The subsequent bounded quartic construction in `QUARTIC_CANDIDATE.md` now
implements O(N log N) refinement with selectable cross-parent grouping. The
small-factor chain remains too long. The quartic candidate still needs a better
small real frame, native comparison, and algorithmic prior-art assessment.

## Reproduction and evidence

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_walk/run.py \
  --out /tmp/real_fourier_walk.json
```

Copy the JSON back immediately with the route selected by `m4host`.
`results.json` retains symbolic identities, exact N=8 Fourier-residue checks,
80 arbitrary exact moves, whole-packet identities, 26 real FFT sweep rows
through N=16384, and 10 unrestricted-walk conditioning probes. Catastrophic
conditioning cases are deliberately retained. `initial_results.json` records
the earlier, smaller certificate before whole-packet and conditioning probes.

## Native quartic follow-up

The quartic packet transition is now factored into normalized real coordinates
and implemented with BFFT NEON primitives. See [the factorization](NATIVE_FACTORIZATION.md)
and [the measured comparison with production DIF](NATIVE_RESULTS.md). Run
`experiments/real_fourier_walk/run_native.sh` for the exact certificate, native
checks, and the opt-in comparison in the existing benchmark program.
