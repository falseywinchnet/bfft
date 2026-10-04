// Wrench-transport solver.
//
// The contact impulses of a frame are found as a retained field. Each contact
// point keeps the impulse it carried last (lambda), and a frame runs a few
// passes. One pass minimises, over the twists V of all awake bodies,
//
//   l(V) = sum_b 1/2 (V_b - V*_b)^T M_b (V_b - V*_b)
//        + sum_c R_c/2 | P_K( lambda_c - (J_c V - vhat_c) / R_c ) |^2,
//
// where V* is the free (contact-less) twist, P_K projects onto the friction
// cone |g_t| <= mu g_n, and R_c is a compliance. The projected vector is the
// pass's impulse; it becomes lambda for the next pass. At the fixed point
// lambda no longer changes, the compliance has no effect, and the impulses
// satisfy rigid unilateral contact with Coulomb friction (the normal target
// vhat carries de Saxce's mu |v_t| shift, so sliding does not lift a body).
// This is the proximal method of multipliers around the compliant convex
// contact potential of Castro et al. (2022).
//
// A resting group is a fixed point: its first pass starts with a zero
// gradient, costs one gradient evaluation, and moves nothing.
//
// Macro-frames enter twice.
//
// 1. Every contact between two bodies acts on them only through their twelve
//    twist coordinates, so the Hessian is M plus one 6x6-block relation per
//    touching body pair. Each Newton system is solved exactly by macro-frame
//    elimination (block_ldl.mjs): load and response cross the whole contact
//    group inside one solve, with no relaxation sweep per contact.
//
// 2. A contact's compliance R_c is set against the mass of everything it
//    carries, not against the two bodies that touch. The passes then contract
//    at a rate that does not collapse with stack height or mass ratio: the
//    slowest error mode of a column of N equal bodies decays like
//    1 / (1 + 1 / (epsilon N)) per pass instead of 1 / (1 + 1 / (epsilon N^2)).

import {
    v3, add, sub, scale, addScaled, dot, cross,
    quatIntegrate, quatToMat3, mat3MulVec, mat3Rotated,
} from "./math3.mjs";
import { tangentBasis } from "./solver_soft_step.mjs";
import {
    createFactor, analyze, clearBlocks, diagonalBlock, couplingBlock, factorize, solve,
} from "./block_ldl.mjs";

export const MODE_OPEN = 0;
export const MODE_STICK = 1;
export const MODE_SLIDE = 2;

const KIND_CONTACT = 0;   // friction cone, retained
const KIND_BALL = 1;      // 3-vector bounded in length
const KIND_HOLD = 2;      // crane linear spring

export const defaultWrenchOptions = {
    regularization: 0.03,           // per-pass contact compliance, as a fraction of the inverse carried mass
    carriedMass: true,              // scale compliance by the supported load, not only the touching pair
    loadMass: true,                 // and by the load each point carried last frame
    relaxationTime: 0.1,            // seconds; overlap decay time
    relativeTolerance: 1.0e-9,      // Newton gradient tolerance against the momentum scale
    absoluteTolerance: 1.0e-14,
    maxNewtonIterations: 30,        // per pass
    looseTolerance: 1.0e-3,         // Newton tolerance while the retained field is still changing
    maxPasses: 6,
    passTolerance: 1.0e-3,          // relative impulse change that ends the passes
    shiftTolerance: 1.0e-3,         // m/s change of the sliding shift that ends the passes
};

export function createWrenchState() {
    return {
        factor: createFactor(),
        graphKey: "",
        rolling: new Map(),   // pair key -> retained rolling-resistance impulse
        trace: null,          // optional array receiving one record per Newton iteration
        passTrace: null,      // optional array receiving one record per pass
        stats: {
            frames: 0, passes: 0, newtonIterations: 0, factorizations: 0, factorReuses: 0,
            lineSearchEvaluations: 0, maxNewtonIterations: 0, notConverged: 0, stalled: 0, passLimit: 0,
            analyses: 0,
        },
    };
}

// Projection of a 3-vector onto a ball of radius `limit`: rolling resistance
// and the crane's angular spring. Same outputs as evaluateContact.
function evaluateBall(row, c0, c1, c2, withHessian) {
    const inverseR = 1 / row.R;
    const y0 = row.lambda0 - (c0 - row.vhat0) * inverseR;
    const y1 = row.lambda1 - (c1 - row.vhat1) * inverseR;
    const y2 = row.lambdaN - (c2 - row.vhatN) * inverseR;
    const size = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
    const G = row.G;
    if (size <= row.limit) {
        row.mode = row.limit > 0 ? MODE_STICK : MODE_OPEN;
        row.gamma0 = y0;
        row.gamma1 = y1;
        row.gammaN = y2;
        if (withHessian) {
            G[0] = inverseR; G[1] = 0; G[2] = 0;
            G[3] = 0; G[4] = inverseR; G[5] = 0;
            G[6] = 0; G[7] = 0; G[8] = inverseR;
        }
        return;
    }
    const shrink = row.limit / size;
    row.mode = row.limit > 0 ? MODE_SLIDE : MODE_OPEN;
    row.gamma0 = y0 * shrink;
    row.gamma1 = y1 * shrink;
    row.gammaN = y2 * shrink;
    if (withHessian) {
        const p = shrink * inverseR;
        const u0 = y0 / size;
        const u1 = y1 / size;
        const u2 = y2 / size;
        G[0] = p * (1 - u0 * u0); G[1] = -p * u0 * u1; G[2] = -p * u0 * u2;
        G[3] = G[1]; G[4] = p * (1 - u1 * u1); G[5] = -p * u1 * u2;
        G[6] = G[2]; G[7] = G[5]; G[8] = p * (1 - u2 * u2);
    }
}

// The crane's linear spring: the two horizontal components are bounded
// together by `limit`; the vertical one (the wires) lies in [0, limitLift].
function evaluateHold(row, c0, c1, c2, withHessian) {
    const inverseR = 1 / row.R;
    const y0 = row.lambda0 - (c0 - row.vhat0) * inverseR;
    const y1 = row.lambda1 - (c1 - row.vhat1) * inverseR;
    const y2 = row.lambdaN - (c2 - row.vhatN) * inverseR;
    const G = row.G;
    if (withHessian) {
        G.fill(0);
    }
    row.mode = MODE_STICK;
    const size = Math.sqrt(y0 * y0 + y1 * y1);
    if (size <= row.limit) {
        row.gamma0 = y0;
        row.gamma1 = y1;
        if (withHessian) {
            G[0] = inverseR;
            G[4] = inverseR;
        }
    } else {
        const shrink = row.limit / size;
        row.mode = MODE_SLIDE;
        row.gamma0 = y0 * shrink;
        row.gamma1 = y1 * shrink;
        if (withHessian) {
            const p = shrink * inverseR;
            const u0 = y0 / size;
            const u1 = y1 / size;
            G[0] = p * (1 - u0 * u0); G[1] = -p * u0 * u1;
            G[3] = G[1]; G[4] = p * (1 - u1 * u1);
        }
    }
    if (y2 <= 0) {
        row.gammaN = 0;
        row.mode = MODE_SLIDE;
    } else if (y2 >= row.limitLift) {
        row.gammaN = row.limitLift;
        row.mode = MODE_SLIDE;
    } else {
        row.gammaN = y2;
        if (withHessian) {
            G[8] = inverseR;
        }
    }
}

function evaluateRow(row, c0, c1, c2, withHessian) {
    if (row.kind === KIND_CONTACT) {
        evaluateContact(row, c0, c1, c2, withHessian);
    } else if (row.kind === KIND_BALL) {
        evaluateBall(row, c0, c1, c2, withHessian);
    } else {
        evaluateHold(row, c0, c1, c2, withHessian);
    }
}

// Impulse, mode and (optionally) Hessian of one contact for contact-frame
// velocity (c0, c1, c2) = (tangent 1, tangent 2, normal). Writes row.gamma*,
// row.mode and, when `withHessian`, row.G (symmetric 3x3, row-major).
function evaluateContact(row, c0, c1, c2, withHessian) {
    const inverseR = 1 / row.R;
    const y0 = row.lambda0 - (c0 - row.vhat0) * inverseR;
    const y1 = row.lambda1 - (c1 - row.vhat1) * inverseR;
    const yn = row.lambdaN - (c2 - row.vhatN) * inverseR;
    const yr = Math.sqrt(y0 * y0 + y1 * y1);
    const mu = row.friction;
    const G = row.G;
    if (yr <= mu * yn) {
        row.mode = MODE_STICK;
        row.gamma0 = y0;
        row.gamma1 = y1;
        row.gammaN = yn;
        if (withHessian) {
            G[0] = inverseR; G[1] = 0; G[2] = 0;
            G[3] = 0; G[4] = inverseR; G[5] = 0;
            G[6] = 0; G[7] = 0; G[8] = inverseR;
        }
        return;
    }
    if (yn <= -mu * yr) {
        row.mode = MODE_OPEN;
        row.gamma0 = 0;
        row.gamma1 = 0;
        row.gammaN = 0;
        if (withHessian) {
            G.fill(0);
        }
        return;
    }
    // Projection onto the cone's surface.
    row.mode = MODE_SLIDE;
    const factor = 1 / (1 + mu * mu);
    const gammaN = (yn + mu * yr) * factor;
    const t0 = y0 / yr;
    const t1 = y1 / yr;
    row.gammaN = gammaN;
    row.gamma0 = mu * gammaN * t0;
    row.gamma1 = mu * gammaN * t1;
    if (withHessian) {
        // G = (factor/R) [mu t; 1][mu t; 1]^T + (mu gammaN / (yr R)) (1 - t t^T on the tangent block)
        const a = factor * inverseR;
        const p = mu * gammaN * inverseR / yr;
        G[0] = a * mu * mu * t0 * t0 + p * (1 - t0 * t0);
        G[1] = a * mu * mu * t0 * t1 - p * t0 * t1;
        G[2] = a * mu * t0;
        G[3] = G[1];
        G[4] = a * mu * mu * t1 * t1 + p * (1 - t1 * t1);
        G[5] = a * mu * t1;
        G[6] = G[2];
        G[7] = G[5];
        G[8] = a;
    }
}

// Contact-frame velocity of every row for twist vector `twist`.
function contactVelocities(rows, twist, out) {
    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        const Ja = row.Ja;
        const a = row.indexA * 6;
        let c0 = 0;
        let c1 = 0;
        let c2 = 0;
        for (let j = 0; j < 6; j += 1) {
            const value = twist[a + j];
            c0 += Ja[j] * value;
            c1 += Ja[6 + j] * value;
            c2 += Ja[12 + j] * value;
        }
        if (row.indexB >= 0) {
            const Jb = row.Jb;
            const b = row.indexB * 6;
            for (let j = 0; j < 6; j += 1) {
                const value = twist[b + j];
                c0 += Jb[j] * value;
                c1 += Jb[6 + j] * value;
                c2 += Jb[12 + j] * value;
            }
        }
        out[k * 3] = c0;
        out[k * 3 + 1] = c1;
        out[k * 3 + 2] = c2;
    }
}

// y = M x for the block-diagonal mass matrix.
function applyMass(bodies, x, y) {
    for (let i = 0; i < bodies.length; i += 1) {
        const body = bodies[i];
        const o = i * 6;
        const inertia = body.inertiaWorldFull;
        y[o] = body.mass * x[o];
        y[o + 1] = body.mass * x[o + 1];
        y[o + 2] = body.mass * x[o + 2];
        y[o + 3] = inertia[0] * x[o + 3] + inertia[1] * x[o + 4] + inertia[2] * x[o + 5];
        y[o + 4] = inertia[3] * x[o + 3] + inertia[4] * x[o + 4] + inertia[5] * x[o + 5];
        y[o + 5] = inertia[6] * x[o + 3] + inertia[7] * x[o + 4] + inertia[8] * x[o + 5];
    }
}

// Energy norm sqrt(x^T M^-1 x) of a generalised momentum vector.
function inverseMassNorm(bodies, x) {
    let sum = 0;
    for (let i = 0; i < bodies.length; i += 1) {
        const body = bodies[i];
        const o = i * 6;
        const inv = body.inertiaWorld;
        sum += body.invMass * (x[o] * x[o] + x[o + 1] * x[o + 1] + x[o + 2] * x[o + 2]);
        const ax = x[o + 3];
        const ay = x[o + 4];
        const az = x[o + 5];
        sum += ax * (inv[0] * ax + inv[1] * ay + inv[2] * az) +
            ay * (inv[3] * ax + inv[4] * ay + inv[5] * az) +
            az * (inv[6] * ax + inv[7] * ay + inv[8] * az);
    }
    return Math.sqrt(sum);
}

function buildRow(world, contact, bodyA, bodyB, h, options) {
    const basis = tangentBasis(contact.normal);
    const directions = [basis.t1, basis.t2, contact.normal];
    const rA = sub(contact.pointA, bodyA.position);
    const Ja = new Float64Array(18);
    let traceW = 0;
    for (let d = 0; d < 3; d += 1) {
        const arm = cross(rA, directions[d]);
        Ja[d * 6] = directions[d].x;
        Ja[d * 6 + 1] = directions[d].y;
        Ja[d * 6 + 2] = directions[d].z;
        Ja[d * 6 + 3] = arm.x;
        Ja[d * 6 + 4] = arm.y;
        Ja[d * 6 + 5] = arm.z;
        traceW += bodyA.invMass + dot(arm, mat3MulVec(bodyA.inertiaWorld, arm));
    }
    let Jb = null;
    if (bodyB !== null) {
        const rB = sub(contact.pointB, bodyB.position);
        Jb = new Float64Array(18);
        for (let d = 0; d < 3; d += 1) {
            const arm = cross(rB, directions[d]);
            Jb[d * 6] = -directions[d].x;
            Jb[d * 6 + 1] = -directions[d].y;
            Jb[d * 6 + 2] = -directions[d].z;
            Jb[d * 6 + 3] = -arm.x;
            Jb[d * 6 + 4] = -arm.y;
            Jb[d * 6 + 5] = -arm.z;
            traceW += bodyB.invMass + dot(arm, mat3MulVec(bodyB.inertiaWorld, arm));
        }
    }
    const params = world.params;
    // Normal target: a gap may close exactly; an overlap decays, never pops.
    let vhatN = 0;
    if (contact.separation > 0) {
        vhatN = -contact.separation / h;
    } else {
        vhatN = Math.min(-contact.separation / (h + options.relaxationTime), params.pushMaxVelocity);
    }
    return {
        kind: KIND_CONTACT,
        retained: true,
        limit: 0,
        limitLift: 0,
        contact: contact,
        indexA: bodyA.solverIndex,
        indexB: bodyB === null ? -1 : bodyB.solverIndex,
        Ja: Ja,
        Jb: Jb,
        t1: basis.t1,
        t2: basis.t2,
        friction: contact.friction,
        localInverseMass: traceW / 3,
        R: options.regularization * traceW / 3,
        vhat0: 0,
        vhat1: 0,
        vhatN: vhatN,
        vhatNBase: vhatN,
        shift: 0,
        lambda0: dot(contact.warmTangent, basis.t1),
        lambda1: dot(contact.warmTangent, basis.t2),
        lambdaN: contact.warmNormal,
        gamma0: 0,
        gamma1: 0,
        gammaN: 0,
        mode: MODE_OPEN,
        G: new Float64Array(9),
    };
}

// Adds J_x^T G J_y into a 6x6 block (optionally into its transpose).
function addRelation(block, Jx, Jy, G, transposed) {
    for (let i = 0; i < 6; i += 1) {
        const g0 = Jx[i] * G[0] + Jx[6 + i] * G[3] + Jx[12 + i] * G[6];
        const g1 = Jx[i] * G[1] + Jx[6 + i] * G[4] + Jx[12 + i] * G[7];
        const g2 = Jx[i] * G[2] + Jx[6 + i] * G[5] + Jx[12 + i] * G[8];
        for (let j = 0; j < 6; j += 1) {
            const value = g0 * Jy[j] + g1 * Jy[6 + j] + g2 * Jy[12 + j];
            if (transposed) {
                block[j * 6 + i] += value;
            } else {
                block[i * 6 + j] += value;
            }
        }
    }
}

function ascendingNumber(a, b) {
    return a - b;
}

// First and second derivative of the cost along `step` at parameter alpha.
function lineDerivatives(rows, contactVelocity, contactStep, slopeLinear, curvatureMass, alpha, out) {
    let slope = slopeLinear + alpha * curvatureMass;
    let curvature = curvatureMass;
    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        const d0 = contactStep[k * 3];
        const d1 = contactStep[k * 3 + 1];
        const d2 = contactStep[k * 3 + 2];
        evaluateRow(row,
            contactVelocity[k * 3] + alpha * d0,
            contactVelocity[k * 3 + 1] + alpha * d1,
            contactVelocity[k * 3 + 2] + alpha * d2, true);
        slope -= d0 * row.gamma0 + d1 * row.gamma1 + d2 * row.gammaN;
        const G = row.G;
        curvature += d0 * (G[0] * d0 + G[1] * d1 + G[2] * d2) +
            d1 * (G[3] * d0 + G[4] * d1 + G[5] * d2) +
            d2 * (G[6] * d0 + G[7] * d1 + G[8] * d2);
    }
    out.slope = slope;
    out.curvature = curvature;
}

// The step length that minimises the convex cost along the Newton direction:
// a bracketed Newton iteration on its derivative.
function exactLineSearch(rows, contactVelocity, contactStep, slopeLinear, curvatureMass, slope0, stats) {
    const search = { slope: 0, curvature: 0 };
    const slopeTolerance = 1.0e-10 * Math.abs(slope0);
    lineDerivatives(rows, contactVelocity, contactStep, slopeLinear, curvatureMass, 1, search);
    stats.lineSearchEvaluations += 1;
    if (Math.abs(search.slope) <= slopeTolerance) {
        return 1;
    }
    let low = 0;
    let high = 1;
    if (search.slope < 0) {
        low = 1;
        high = 2;
        for (let grow = 0; grow < 20; grow += 1) {
            lineDerivatives(rows, contactVelocity, contactStep, slopeLinear, curvatureMass, high, search);
            stats.lineSearchEvaluations += 1;
            if (search.slope >= 0) {
                break;
            }
            low = high;
            high *= 2;
        }
    }
    let alpha = 0.5 * (low + high);
    for (let refine = 0; refine < 60; refine += 1) {
        lineDerivatives(rows, contactVelocity, contactStep, slopeLinear, curvatureMass, alpha, search);
        stats.lineSearchEvaluations += 1;
        if (Math.abs(search.slope) <= slopeTolerance) {
            break;
        }
        if (search.slope > 0) {
            high = alpha;
        } else {
            low = alpha;
        }
        let next = alpha - search.slope / search.curvature;
        if (!(next > low && next < high)) {
            next = 0.5 * (low + high);
        }
        if (Math.abs(next - alpha) <= 1.0e-15 * Math.max(1, alpha)) {
            alpha = next;
            break;
        }
        alpha = next;
    }
    return alpha;
}

// Evaluates every contact at `twist` and fills the gradient
// M (V - V*) - J^T gamma.
function evaluateState(bodies, rows, twist, free, contactVelocity, difference, gradient, impulse) {
    const size = bodies.length * 6;
    contactVelocities(rows, twist, contactVelocity);
    for (let i = 0; i < size; i += 1) {
        difference[i] = twist[i] - free[i];
    }
    applyMass(bodies, difference, gradient);
    impulse.fill(0);
    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        evaluateRow(row, contactVelocity[k * 3], contactVelocity[k * 3 + 1], contactVelocity[k * 3 + 2], true);
        const a = row.indexA * 6;
        for (let j = 0; j < 6; j += 1) {
            impulse[a + j] += row.Ja[j] * row.gamma0 + row.Ja[6 + j] * row.gamma1 + row.Ja[12 + j] * row.gammaN;
        }
        if (row.indexB >= 0) {
            const b = row.indexB * 6;
            for (let j = 0; j < 6; j += 1) {
                impulse[b + j] += row.Jb[j] * row.gamma0 + row.Jb[6 + j] * row.gamma1 + row.Jb[12 + j] * row.gammaN;
            }
        }
    }
    for (let i = 0; i < size; i += 1) {
        gradient[i] -= impulse[i];
    }
}

// Newton's method with an exact line search on the current cost, from
// ctx.twist. Stops at `relativeTolerance`, but a start that misses the tight
// tolerance always takes at least one step, so a nearly resting group keeps
// converging quadratically instead of being accepted as it is. Returns true
// when a tolerance was met.
function newtonSolve(ctx, relativeTolerance, tightTolerance, maxIterations) {
    const bodies = ctx.bodies;
    const rows = ctx.rows;
    const factor = ctx.factor;
    const stats = ctx.stats;
    const count = bodies.length;
    const twist = ctx.twist;
    const gradient = ctx.gradient;
    const direction = ctx.direction;
    const search = ctx.search;
    let previousResidual = Infinity;
    let stalls = 0;
    for (let iteration = 0; iteration <= maxIterations; iteration += 1) {
        evaluateState(bodies, rows, twist, ctx.free, ctx.contactVelocity, ctx.difference, gradient, ctx.impulse);
        applyMass(bodies, twist, ctx.momentum);
        ctx.residual = inverseMassNorm(bodies, gradient);
        ctx.scaleNorm = Math.max(inverseMassNorm(bodies, ctx.momentum), inverseMassNorm(bodies, ctx.impulse));
        if (ctx.residual <= ctx.absoluteTolerance + tightTolerance * ctx.scaleNorm) {
            return true;
        }
        if (iteration > 0 && ctx.residual <= relativeTolerance * ctx.scaleNorm) {
            return true;
        }
        if (iteration === maxIterations) {
            return false;
        }
        // Round-off floor: accept once the residual stops falling near zero.
        if (ctx.residual > 0.5 * previousResidual && ctx.residual <= 1.0e-6 * ctx.scaleNorm) {
            stalls += 1;
            if (stalls >= 3) {
                stats.stalled += 1;
                return true;
            }
        } else {
            stalls = 0;
        }
        previousResidual = ctx.residual;

        // Hessian: M + sum over contacts of J^T G J, assembled per body pair.
        // While no point slides, G depends only on which points press, so the
        // retained factor is reused until that set changes.
        let signature = 0;
        let sliding = false;
        for (let k = 0; k < rows.length; k += 1) {
            const mode = rows[k].mode;
            if (mode === MODE_SLIDE) {
                sliding = true;
                break;
            }
            signature = (Math.imul(signature, 16777619) ^ (mode + 1 + k)) | 0;
        }
        if (sliding || !ctx.factorValid || signature !== ctx.factorSignature) {
            clearBlocks(factor);
            for (let i = 0; i < count; i += 1) {
                const body = bodies[i];
                const block = diagonalBlock(factor, i);
                block[0] = body.mass;
                block[7] = body.mass;
                block[14] = body.mass;
                const inertia = body.inertiaWorldFull;
                for (let r = 0; r < 3; r += 1) {
                    for (let c = 0; c < 3; c += 1) {
                        block[(3 + r) * 6 + 3 + c] = inertia[r * 3 + c];
                    }
                }
            }
            for (let k = 0; k < rows.length; k += 1) {
                const row = rows[k];
                if (row.mode === MODE_OPEN) {
                    continue;
                }
                addRelation(diagonalBlock(factor, row.indexA), row.Ja, row.Ja, row.G, false);
                if (row.indexB >= 0) {
                    addRelation(diagonalBlock(factor, row.indexB), row.Jb, row.Jb, row.G, false);
                    const coupling = couplingBlock(factor, row.indexA, row.indexB);
                    addRelation(coupling.block, row.Ja, row.Jb, row.G, coupling.transposed);
                }
            }
            if (!factorize(factor)) {
                throw new Error("wrench transport: Hessian lost positive definiteness");
            }
            stats.factorizations += 1;
            ctx.factorValid = !sliding;
            ctx.factorSignature = signature;
        } else {
            stats.factorReuses += 1;
        }
        for (let i = 0; i < count * 6; i += 1) {
            gradient[i] = -gradient[i];
        }
        solve(factor, gradient, direction);
        ctx.iterations += 1;

        contactVelocities(rows, direction, ctx.contactStep);
        applyMass(bodies, direction, ctx.massDirection);
        let slopeLinear = 0;
        let curvatureMass = 0;
        for (let i = 0; i < count * 6; i += 1) {
            slopeLinear += ctx.massDirection[i] * ctx.difference[i];
            curvatureMass += ctx.massDirection[i] * direction[i];
        }
        lineDerivatives(rows, ctx.contactVelocity, ctx.contactStep, slopeLinear, curvatureMass, 0, search);
        const slope0 = search.slope;
        if (!(slope0 < 0)) {
            stats.stalled += 1;
            return ctx.residual <= 1.0e-6 * ctx.scaleNorm;
        }
        const alpha = exactLineSearch(rows, ctx.contactVelocity, ctx.contactStep, slopeLinear, curvatureMass, slope0, stats);
        for (let i = 0; i < count * 6; i += 1) {
            twist[i] += alpha * direction[i];
        }
        if (ctx.trace !== null) {
            ctx.trace.push({
                frame: stats.frames, pass: ctx.pass, iteration: iteration,
                residual: ctx.residual, scale: ctx.scaleNorm, alpha: alpha, rows: rows.length,
            });
        }
    }
    return false;
}

function blankRow(kind, indexA, indexB, R) {
    return {
        kind: kind,
        retained: false,
        limit: 0,
        limitLift: 0,
        contact: null,
        indexA: indexA,
        indexB: indexB,
        Ja: new Float64Array(18),
        Jb: indexB >= 0 ? new Float64Array(18) : null,
        t1: null,
        t2: null,
        friction: 0,
        localInverseMass: 0,
        R: R,
        vhat0: 0,
        vhat1: 0,
        vhatN: 0,
        vhatNBase: 0,
        shift: 0,
        lambda0: 0,
        lambda1: 0,
        lambdaN: 0,
        gamma0: 0,
        gamma1: 0,
        gammaN: 0,
        mode: MODE_OPEN,
        G: new Float64Array(9),
        members: null,
        resistance: 0,
        pairKey: 0,
    };
}

// Rolling resistance of one touching pair: a torque opposing their relative
// angular velocity, bounded by a lever arm times the pair's normal impulse.
// It is a row of the solve, so the translation that goes with a rotation is
// resisted together with it.
function rollingRow(bodyA, bodyB, members, resistance, pairKey, state, options) {
    let traceInverse = bodyA.inertiaWorld[0] + bodyA.inertiaWorld[4] + bodyA.inertiaWorld[8];
    if (bodyB !== null) {
        traceInverse += bodyB.inertiaWorld[0] + bodyB.inertiaWorld[4] + bodyB.inertiaWorld[8];
    }
    const row = blankRow(KIND_BALL, bodyA.solverIndex, bodyB === null ? -1 : bodyB.solverIndex,
        options.regularization * traceInverse / 3);
    for (let d = 0; d < 3; d += 1) {
        row.Ja[d * 6 + 3 + d] = 1;
        if (row.Jb !== null) {
            row.Jb[d * 6 + 3 + d] = -1;
        }
    }
    row.retained = true;
    row.members = members;
    row.resistance = resistance;
    row.pairKey = pairKey;
    const old = state.rolling.get(pairKey);
    if (old !== undefined) {
        row.lambda0 = old.x;
        row.lambda1 = old.y;
        row.lambdaN = old.z;
    }
    let load = 0;
    for (let k = 0; k < members.length; k += 1) {
        load += members[k].lambdaN;
    }
    row.limit = resistance * load;
    return row;
}

// Soft-constraint coefficients of an implicit spring (frequency, damping
// ratio) over one step h, for unit effective mass.
function springCoefficients(hertz, zeta, h) {
    const omega = 6.283185307179586 * hertz;
    const a1 = 2 * zeta + h * omega;
    return { biasRate: omega / a1, stiffness: h * omega * a1 };
}

// The crane's two rows on the held body: a linear spring with gravity
// feed-forward, capped wires and capped side force, and an angular spring
// with capped torque.
function holdRows(world, body, h, rows) {
    const hold = world.hold;
    const params = hold.params;
    const gravity = Math.sqrt(dot(world.params.gravity, world.params.gravity));
    const weightImpulse = body.mass * gravity * h;
    const linear = springCoefficients(params.linHertz, params.linZeta, h);
    const lift = blankRow(KIND_HOLD, body.solverIndex, -1, 1 / (body.mass * linear.stiffness));
    for (let d = 0; d < 3; d += 1) {
        lift.Ja[d * 6 + d] = 1;
    }
    const error = sub(hold.target.position, body.position);
    lift.vhat0 = linear.biasRate * error.x;
    lift.vhat1 = linear.biasRate * error.y;
    lift.vhatN = linear.biasRate * error.z;
    lift.lambdaN = weightImpulse;              // feed-forward: a free-hanging rock does not droop
    lift.limit = params.maxLateral * weightImpulse;
    lift.limitLift = params.maxLift * weightImpulse;
    rows.push(lift);

    const angularSpring = springCoefficients(params.angHertz, params.angZeta, h);
    const meanInertia = (body.inertiaWorldFull[0] + body.inertiaWorldFull[4] + body.inertiaWorldFull[8]) / 3;
    const turn = blankRow(KIND_BALL, body.solverIndex, -1, 1 / (meanInertia * angularSpring.stiffness));
    for (let d = 0; d < 3; d += 1) {
        turn.Ja[d * 6 + 3 + d] = 1;
    }
    // Twice the vector part of target * conj(current), on the short side.
    const q = body.orientation;
    const t = hold.target.orientation;
    let w = t.w * q.w + t.x * q.x + t.y * q.y + t.z * q.z;
    let x = -t.w * q.x + t.x * q.w - t.y * q.z + t.z * q.y;
    let y = -t.w * q.y + t.x * q.z + t.y * q.w - t.z * q.x;
    let z = -t.w * q.z - t.x * q.y + t.y * q.x + t.z * q.w;
    if (w < 0) {
        w = -w;
        x = -x;
        y = -y;
        z = -z;
    }
    turn.vhat0 = angularSpring.biasRate * 2 * x;
    turn.vhat1 = angularSpring.biasRate * 2 * y;
    turn.vhatN = angularSpring.biasRate * 2 * z;
    turn.limit = params.maxTorque * weightImpulse * body.shape.radius;
    rows.push(turn);
    return lift;
}

// Mass carried through each contact row: the body resting on a contact
// brings its own mass and the mass resting on it. Bodies are visited from the
// highest centre of mass down; a body's load is split among the pairs that
// support it in proportion to how upward-facing they are.
function assignCarriedMass(bodies, rows, gravity, options) {
    const count = bodies.length;
    const gravityLength = Math.sqrt(dot(gravity, gravity));
    if (gravityLength === 0) {
        return;
    }
    const up = v3(-gravity.x / gravityLength, -gravity.y / gravityLength, -gravity.z / gravityLength);
    // Per body: its supporting pairs as (other body index or -1, weight, rows).
    const supports = [];
    const load = new Float64Array(count);
    const order = [];
    for (let i = 0; i < count; i += 1) {
        supports.push(new Map());
        load[i] = bodies[i].mass;
        order.push(i);
    }
    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        if (row.kind !== KIND_CONTACT) {
            continue;
        }
        const lift = dot(row.contact.normal, up);   // normal points from B to A
        let upper = -1;
        let lower = -1;
        if (lift > 0.1) {
            upper = row.indexA;
            lower = row.indexB;
        } else if (lift < -0.1 && row.indexB >= 0) {
            upper = row.indexB;
            lower = row.indexA;
        } else {
            continue;
        }
        let entry = supports[upper].get(lower);
        if (entry === undefined) {
            entry = { weight: 0, rows: [] };
            supports[upper].set(lower, entry);
        }
        entry.weight += Math.abs(lift);
        entry.rows.push(row);
    }
    order.sort(byHeightDescending.bind(null, bodies, up));
    const visited = new Uint8Array(count);
    for (let n = 0; n < count; n += 1) {
        const i = order[n];
        visited[i] = 1;
        let total = 0;
        const iterator = supports[i].entries();
        for (let item = iterator.next(); !item.done; item = iterator.next()) {
            total += item.value[1].weight;
        }
        const again = supports[i].entries();
        for (let item = again.next(); !item.done; item = again.next()) {
            const lower = item.value[0];
            const entry = item.value[1];
            const share = load[i] * entry.weight / total;
            for (let k = 0; k < entry.rows.length; k += 1) {
                const row = entry.rows[k];
                row.R = options.regularization / Math.max(1 / row.localInverseMass, share);
            }
            if (lower >= 0 && !visited[lower]) {
                load[lower] += share;
            }
        }
    }
}

function byHeightDescending(bodies, up, i, j) {
    const difference = dot(bodies[j].position, up) - dot(bodies[i].position, up);
    if (difference !== 0) {
        return difference;
    }
    return i - j;
}

// Advances every awake body by one frame.
export function solveWrenchTransport(world, contacts, state, options) {
    const params = world.params;
    const h = params.frameDt;
    const bodies = [];
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        if (body.alive && !body.asleep && !body.isStatic) {
            body.solverIndex = bodies.length;
            bodies.push(body);
            const rotation = quatToMat3(body.orientation);
            body.inertiaWorld = mat3Rotated(rotation, body.invInertiaLocal);
            body.inertiaWorldFull = mat3Rotated(rotation, body.inertiaLocal);
            body.contactImpulse = 0;
        }
    }
    const count = bodies.length;
    const stats = state.stats;
    stats.frames += 1;
    if (count === 0) {
        return { iterations: 0, residual: 0, passes: 0 };
    }

    const twist = new Float64Array(count * 6);
    const free = new Float64Array(count * 6);
    const linearDecay = 1 / (1 + h * params.linearDamping);
    const angularDecay = 1 / (1 + h * params.angularDamping);
    for (let i = 0; i < count; i += 1) {
        const body = bodies[i];
        const o = i * 6;
        twist[o] = body.velocity.x;
        twist[o + 1] = body.velocity.y;
        twist[o + 2] = body.velocity.z;
        twist[o + 3] = body.angular.x;
        twist[o + 4] = body.angular.y;
        twist[o + 5] = body.angular.z;
        free[o] = (body.velocity.x + h * params.gravity.x) * linearDecay;
        free[o + 1] = (body.velocity.y + h * params.gravity.y) * linearDecay;
        free[o + 2] = (body.velocity.z + h * params.gravity.z) * linearDecay;
        free[o + 3] = body.angular.x * angularDecay;
        free[o + 4] = body.angular.y * angularDecay;
        free[o + 5] = body.angular.z * angularDecay;
    }

    const rows = [];
    const pairs = [];
    const pairMembers = new Map();
    for (let k = 0; k < contacts.length; k += 1) {
        const contact = contacts[k];
        const bodyA = world.bodies[contact.a];
        const bodyB = contact.fixedB ? null : world.bodies[contact.b];
        const row = buildRow(world, contact, bodyA, bodyB, h, options);
        rows.push(row);
        if (row.indexB >= 0) {
            pairs.push(row.indexA);
            pairs.push(row.indexB);
        }
        const key = contact.a * 65536 + (contact.b + 1);
        let entry = pairMembers.get(key);
        if (entry === undefined) {
            entry = { bodyA: bodyA, bodyB: bodyB, other: contact.b, members: [] };
            pairMembers.set(key, entry);
        }
        entry.members.push(row);
    }
    const contactRowCount = rows.length;
    const pairIterator = pairMembers.entries();
    for (let item = pairIterator.next(); !item.done; item = pairIterator.next()) {
        const entry = item.value[1];
        // The larger lever arm of the two surfaces; the ground has its own.
        let resistance = entry.bodyA.shape.rollingResistance;
        if (entry.other >= 0) {
            resistance = Math.max(resistance, world.bodies[entry.other].shape.rollingResistance);
        } else {
            resistance = Math.max(resistance, params.groundRollingResistance);
        }
        if (resistance > 0) {
            rows.push(rollingRow(entry.bodyA, entry.bodyB, entry.members, resistance, item.value[0], state, options));
        }
    }
    let liftRow = null;
    if (world.hold !== null) {
        const heldBody = world.bodies[world.hold.id];
        if (heldBody.alive && !heldBody.asleep && !heldBody.isStatic) {
            liftRow = holdRows(world, heldBody, h, rows);
        }
    }

    // The elimination order is reused while the contact graph is unchanged.
    const sortedPairs = [];
    for (let k = 0; k + 1 < pairs.length; k += 2) {
        const low = Math.min(pairs[k], pairs[k + 1]);
        const high = Math.max(pairs[k], pairs[k + 1]);
        sortedPairs.push(low * count + high);
    }
    sortedPairs.sort(ascendingNumber);
    let graphKey = "" + count;
    let previousPair = -1;
    for (let k = 0; k < sortedPairs.length; k += 1) {
        if (sortedPairs[k] !== previousPair) {
            graphKey += "," + sortedPairs[k];
            previousPair = sortedPairs[k];
        }
    }
    for (let i = 0; i < count; i += 1) {
        graphKey += ";" + bodies[i].id;
    }
    const factor = state.factor;
    if (graphKey !== state.graphKey) {
        analyze(factor, count, pairs);
        state.graphKey = graphKey;
        stats.analyses += 1;
    }

    const ctx = {
        bodies: bodies,
        rows: rows,
        factor: factor,
        stats: stats,
        trace: state.trace,
        twist: twist,
        free: free,
        contactVelocity: new Float64Array(rows.length * 3),
        contactStep: new Float64Array(rows.length * 3),
        gradient: new Float64Array(count * 6),
        momentum: new Float64Array(count * 6),
        impulse: new Float64Array(count * 6),
        direction: new Float64Array(count * 6),
        difference: new Float64Array(count * 6),
        massDirection: new Float64Array(count * 6),
        search: { slope: 0, curvature: 0 },
        absoluteTolerance: options.absoluteTolerance,
        residual: 0,
        scaleNorm: 0,
        iterations: 0,
        pass: 0,
        factorValid: false,
        factorSignature: 0,
    };

    if (options.carriedMass) {
        assignCarriedMass(bodies, rows, params.gravity, options);
    }
    if (options.loadMass) {
        // A point that carried normal impulse g last frame was holding up a
        // mass of g / (|gravity| h); its compliance is set against that too.
        const gravityImpulse = Math.sqrt(dot(params.gravity, params.gravity)) * h;
        if (gravityImpulse > 0) {
            for (let k = 0; k < contactRowCount; k += 1) {
                const row = rows[k];
                const held = row.lambdaN / gravityImpulse;
                if (held * row.R > options.regularization) {
                    row.R = options.regularization / held;
                }
            }
        }
    }
    // The sliding shift starts from the incoming velocities.
    contactVelocities(rows, twist, ctx.contactVelocity);
    for (let k = 0; k < contactRowCount; k += 1) {
        const row = rows[k];
        const s0 = ctx.contactVelocity[k * 3];
        const s1 = ctx.contactVelocity[k * 3 + 1];
        // Restitution. A point closing faster than the threshold and due to
        // touch within this frame first closes its gap; the rebound, e times
        // the closing speed, is asked for on the next frame, at the surface.
        const closing = ctx.contactVelocity[k * 3 + 2];
        const contact = row.contact;
        contact.bounce = 0;
        if (contact.pendingBounce > 0) {
            row.vhatNBase = Math.max(row.vhatNBase, contact.pendingBounce);
        } else if (closing < -params.restitutionThreshold && contact.separation + h * closing < 0) {
            if (contact.separation > params.linearSlop) {
                contact.bounce = -contact.restitution * closing;
            } else {
                row.vhatNBase = Math.max(row.vhatNBase, -contact.restitution * closing);
            }
        }
        row.shift = row.friction * Math.sqrt(s0 * s0 + s1 * s1);
        row.vhatN = row.vhatNBase - row.shift;
    }

    let passesUsed = 0;
    // A pass is solved only as accurately as the field is still changing.
    let passAccuracy = options.looseTolerance;
    for (let pass = 0; pass < options.maxPasses; pass += 1) {
        passesUsed += 1;
        ctx.pass = pass;
        if (!newtonSolve(ctx, passAccuracy, options.relativeTolerance, options.maxNewtonIterations)) {
            stats.notConverged += 1;
        }
        // The pass's impulses become the retained field. The passes end when
        // the field and the sliding shifts have stopped changing.
        let change = 0;
        let magnitude = 0;
        let shiftChange = 0;
        for (let k = 0; k < rows.length; k += 1) {
            const row = rows[k];
            if (!row.retained) {
                continue;
            }
            const d0 = row.gamma0 - row.lambda0;
            const d1 = row.gamma1 - row.lambda1;
            const dn = row.gammaN - row.lambdaN;
            change += row.R * (d0 * d0 + d1 * d1 + dn * dn);
            magnitude += row.R * (row.gamma0 * row.gamma0 + row.gamma1 * row.gamma1 + row.gammaN * row.gammaN);
            row.lambda0 = row.gamma0;
            row.lambda1 = row.gamma1;
            row.lambdaN = row.gammaN;
            if (row.kind !== KIND_CONTACT) {
                continue;
            }
            const s0 = ctx.contactVelocity[k * 3];
            const s1 = ctx.contactVelocity[k * 3 + 1];
            const shift = row.friction * Math.sqrt(s0 * s0 + s1 * s1);
            shiftChange = Math.max(shiftChange, Math.abs(shift - row.shift));
            row.shift = shift;
            row.vhatN = row.vhatNBase - shift;
        }
        // Rolling bounds follow the pair's normal impulse of this pass.
        for (let k = contactRowCount; k < rows.length; k += 1) {
            const row = rows[k];
            if (row.members === null) {
                continue;
            }
            let load = 0;
            for (let m = 0; m < row.members.length; m += 1) {
                load += row.members[m].gammaN;
            }
            row.limit = row.resistance * load;
        }
        if (state.passTrace !== null) {
            state.passTrace.push({
                frame: stats.frames, pass: pass, change: Math.sqrt(change / Math.max(magnitude, 1.0e-300)),
                shiftChange: shiftChange, iterations: ctx.iterations,
            });
        }
        const relativeChange = Math.sqrt(change) / Math.max(Math.sqrt(magnitude), 1.0e-300);
        if (relativeChange <= options.passTolerance && shiftChange <= options.shiftTolerance) {
            break;
        }
        passAccuracy = Math.min(options.looseTolerance, Math.max(options.relativeTolerance, 0.1 * relativeChange));
        if (pass === options.maxPasses - 1) {
            stats.passLimit += 1;
        }
    }
    stats.passes += passesUsed;
    stats.newtonIterations += ctx.iterations;
    stats.maxNewtonIterations = Math.max(stats.maxNewtonIterations, ctx.iterations);

    for (let k = 0; k < contactRowCount; k += 1) {
        const row = rows[k];
        const contact = row.contact;
        contact.normalImpulse = row.gammaN;
        contact.warmNormal = row.gammaN;
        contact.warmTangent = add(scale(row.t1, row.gamma0), scale(row.t2, row.gamma1));
        contact.mode = row.mode;
        bodies[row.indexA].contactImpulse += row.gammaN;
        if (row.indexB >= 0) {
            bodies[row.indexB].contactImpulse += row.gammaN;
        }
    }
    const rolling = new Map();
    for (let k = contactRowCount; k < rows.length; k += 1) {
        const row = rows[k];
        if (row.members !== null) {
            rolling.set(row.pairKey, v3(row.gamma0, row.gamma1, row.gammaN));
        }
    }
    state.rolling = rolling;
    if (liftRow !== null) {
        const heldBody = world.bodies[world.hold.id];
        const weightImpulse = heldBody.mass * Math.sqrt(dot(params.gravity, params.gravity)) * h;
        world.holdState = {
            tension: liftRow.gammaN / weightImpulse,
            positionError: sub(world.hold.target.position, heldBody.position),
            lateralImpulse: Math.sqrt(liftRow.gamma0 * liftRow.gamma0 + liftRow.gamma1 * liftRow.gamma1),
        };
    }
    for (let i = 0; i < count; i += 1) {
        const body = bodies[i];
        const o = i * 6;
        body.velocity = v3(twist[o], twist[o + 1], twist[o + 2]);
        body.angular = v3(twist[o + 3], twist[o + 4], twist[o + 5]);
        body.position = addScaled(body.position, body.velocity, h);
        body.orientation = quatIntegrate(body.orientation, body.angular, h);
    }
    return { iterations: ctx.iterations, residual: ctx.residual, passes: passesUsed };
}
