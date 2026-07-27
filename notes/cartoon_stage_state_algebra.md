# Why the cartoon stage holds 17 planes and five spectra, and what it needs

Verified by `experiments/cartoon_state_algebra.py`. Nothing was modified.

## 1. The inventory, exactly

From `engine::init` in `src/detail/meyer_kernel.hpp`, in image-plane
equivalents (one plane = `H*W` doubles; a `spectrum` is two planes of
`2*WB*HB` doubles, which is `~H*W` doubles in total, so one plane-equivalent):

| planes | what | used by |
|---:|---|---|
| 2 | `u`, `w` | split |
| 4 | `bux, buy, dbux, dbuy` | split, u-subproblem |
| 4 | `bvx, bvy, dbvx, dbvy` | split, v-subproblem |
| 4 | `rbx, rby, rdbx, rdby` | **`rof_from_spec` only** |
| 2 | `xit`, `prev` | **`rof_from_spec` only** |
| 1 | `vplane` | split |
| 1 | `reT`, `imT` | column transform stage |
| 4 | `f_spec, u_spec, w_spec, d_spec` | split |
| 1 | `v_spec` | **`decompose` ladder only** |
| 1 | `s_u`, `s_v` | split |
| 2 | `s_r0, s_r1, s_r2, s_gen` | **ladder / ROF only** |
| **26** | | |

At 2048² that is **832 MB**, which is why the per-pixel cost climbs from
13.63 ns at 512² to 39.40 ns at 2048². The stage is bandwidth-bound long
before it is flop-bound.

Three separate things are going on, and they have different fixes.

## 2. Nine planes are allocated but never touched by `meyer_split`

`rbx, rby, rdbx, rdby, xit, prev` belong to `rof_from_spec`; `v_spec` and
`s_r0..s_gen` belong to the ladder. `meyer_split` — the fast path the
transport model calls — reads none of them, yet `init` allocates and
first-touch-zeroes all of them. That is 9 of 26 plane-equivalents, ~35%, and
it is an allocation-policy question rather than a mathematical one: allocate
the ROF and ladder working sets on first use.

## 3. Four more are redundant, exactly

This part is algebra, not policy.

Split Bregman for `min_u TV(u) + (c/2)||u - g||^2` splits `d = grad(u)` and
alternates

```
u   <- (c - eta*Lap)^-1 ( c*g - eta*div(d - b) )
d   <- shrink(grad(u) + b, theta),      theta = 1/eta
b   <- (grad(u) + b) - d
```

The engine stores `b` as `(bx, by)` and the fused `d - b` as `(dbx, dby)`:
four planes per subproblem, eight in total.

Write `t = grad(u) + b`, the field entering the shrink. Isotropic soft
shrinkage and projection onto the ball of radius `theta` **partition their
argument**:

```
shrink(t) = t * (1 - theta/|t|)_+
proj(t)   = t * min(1, theta/|t|)
shrink(t) + proj(t) = t          for every t
```

(|t| > theta: `t(1 - theta/|t|) + t*theta/|t| = t`. |t| <= theta: `0 + t = t`.
Measured residual 1.8e-15 over 4096 random vectors at three `theta`.)

Therefore `d = shrink(t)` and `b_new = t - d = proj(t)`, and the quantity the
linear solve actually consumes is

```
p = d - b_new = t - 2*proj(t, theta)
```

a pointwise function of `t` alone. It is the Douglas-Rachford reflection
`(2P - I)` of `t` about the ball, negated. The whole recursion closes on `t`:

```
p_k     = t_k - 2*proj(t_k)                      (consumed, never stored)
u_{k+1} = (c - eta*Lap)^-1 ( c*g - eta*div p_k )
t_{k+1} = grad(u_{k+1}) + proj(t_k)
```

`p_k` is formed inside the forward divergence transform and never lands in
memory; `proj(t_k)` is recomputed in the shrink pass from the still-stored
`t_k` and written back in place. **Two planes per subproblem instead of four.**

There is no ordering constraint: `p_k` depends on `t_k` only, so it does not
matter that `fwd2d_div` runs before `inv2d` overwrites `u`.

**Verified.** Both recursions run side by side at two sizes and three
`(c, eta)` pairs: every one of 24 iterates agrees to **7e-16 relative**.

### That two is the floor

Douglas-Rachford and ADMM carry exactly one dual field per split constraint.
The constraint here is `d = grad(u)`, an `R^2`-valued field, so one
`R^2`-valued dual is the state dimension of the algorithm. The kernel carries
two such fields where the second is an explicit pointwise function of the
first. Below two you are no longer running this algorithm.

## 4. `vplane` need not exist

`run_passes` fills it with `f - u - w` in a final fused pass, and
`split()` then copies it to the caller. Write into the caller's buffer.

## 5. What is genuinely irreducible

- `u`, `w`: the two iterates. 2.
- `t_u`, `t_v`: one dual field per subproblem. 4.
- `f_spec`: referenced by both subproblems every pass.
- `u_spec`: the v-step needs it (`solve_diff`).
- `w_spec`: the u-step needs it (`solve_sum_inplace`).
- `d_spec`: divergence scratch, already shared by both steps.
- `reT`, `imT`, `s_u`, `s_v`.

There is no rearrangement that drops one of the three persistent spectra: the
u-step needs `u + w` and the v-step needs `f - u`, so all three appear in
different combinations and none is recoverable from the others without an
extra transform, which trades 1 plane for 2 transforms — a bad trade in a
bandwidth-bound stage.

## 6. Result

| | plane-equivalents | 1024² | 2048² | 4096² |
|---|---:|---:|---:|---:|
| allocated now | 26 | 208 MB | 832 MB | 3328 MB |
| needed by `meyer_split` | 12 | 96 MB | 384 MB | 1536 MB |

**2.17x less resident state**, from three independent changes: 9 planes of
lazy allocation, 4 planes of the reflection identity, 1 plane of writing
through to the caller.

Traffic per pass falls less dramatically — 20 plane traversals to 16, 1.25x —
because the dual fields are still read and written every pass. The footprint
reduction is the larger effect at 2048² and above, where the working set is
the reason the per-pixel cost nearly triples.

## 7. Alternatives considered and rejected

**Chambolle's dual algorithm.** State is one field `p` with `|p| <= 1`, no
`u`, no `d`, no transform at all — fewer planes than anything above. Rejected:
its convergence is `O(1/k)` with a step bounded by `1/8`, and the measured
tail here is already the expensive part (still moving at pass 48). Trading a
2x footprint win for a slower tail is the wrong direction.

**Chambolle-Pock / PDHG.** State `u`, `u_bar`, `p` — four planes per
subproblem, no transform needed. Already measured in
`notes/meyer_accel_theory.md` at 50x worse without a spectral preconditioner.
The reflection identity above gets the same footprint while keeping the exact
spectral solve.

## 8. Order of work

1. Lazy-allocate the ROF and ladder working sets. 9 planes, no math, no
   behaviour change.
2. Write `f - u - w` through to the caller. 1 plane.
3. The `t`-form reflection identity. 4 planes, and it also removes two plane
   writes per subproblem per pass. Needs `shrink` and `fwd2d_div` rewritten as
   described in §3; both are already single-purpose functions.

All three are independent of `include/bfft/meyer_facr_plan.md` and can land in
either order.
