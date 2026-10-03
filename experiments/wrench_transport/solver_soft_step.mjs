// Baseline solver: substepped soft-step sequential impulses (Catto, Box2D v3
// "Solver2D"), extended to 3-D as the Zen Construction specification section 7
// describes. This is the comparator, not the experiment.

import {
    v3, add, sub, scale, addScaled, dot, cross, length,
    quatRotate, quatRotateInverse, quatIntegrate, quatToMat3, mat3MulVec, mat3Rotated,
} from "./math3.mjs";

const TWO_PI = 6.283185307179586;

export function tangentBasis(normal) {
    let first = null;
    if (Math.abs(normal.x) < 0.57735) {
        first = cross(v3(1, 0, 0), normal);
    } else {
        first = cross(v3(0, 1, 0), normal);
    }
    const firstLength = length(first);
    first = scale(first, 1 / firstLength);
    return { t1: first, t2: cross(normal, first) };
}

function worldInverseInertia(body) {
    return mat3Rotated(quatToMat3(body.orientation), body.invInertiaLocal);
}

function effectiveMass(bodyA, bodyB, inertiaA, inertiaB, rA, rB, axis) {
    const armA = cross(rA, axis);
    let k = bodyA.invMass + dot(armA, mat3MulVec(inertiaA, armA));
    if (bodyB !== null) {
        const armB = cross(rB, axis);
        k += bodyB.invMass + dot(armB, mat3MulVec(inertiaB, armB));
    }
    return 1 / k;
}

function applyImpulse(bodyA, bodyB, inertiaA, inertiaB, rA, rB, impulse) {
    bodyA.velocity = addScaled(bodyA.velocity, impulse, bodyA.invMass);
    bodyA.angular = add(bodyA.angular, mat3MulVec(inertiaA, cross(rA, impulse)));
    if (bodyB !== null) {
        bodyB.velocity = addScaled(bodyB.velocity, impulse, -bodyB.invMass);
        bodyB.angular = sub(bodyB.angular, mat3MulVec(inertiaB, cross(rB, impulse)));
    }
}

function relativeVelocity(bodyA, bodyB, rA, rB) {
    let velocity = add(bodyA.velocity, cross(bodyA.angular, rA));
    if (bodyB !== null) {
        velocity = sub(velocity, add(bodyB.velocity, cross(bodyB.angular, rB)));
    }
    return velocity;
}

// One pass over all contacts. `rows` carries per-contact prepared data.
function solvePass(rows, soft, useBias, h, params) {
    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        const bodyA = row.bodyA;
        const bodyB = row.bodyB;
        // Separation tracked through the frame with current rotations.
        row.rA = quatRotate(bodyA.orientation, row.localA);
        let travel = add(sub(bodyA.position, row.startA), sub(row.rA, row.rA0));
        if (bodyB !== null) {
            row.rB = quatRotate(bodyB.orientation, row.localB);
            travel = sub(travel, add(sub(bodyB.position, row.startB), sub(row.rB, row.rB0)));
        }
        const s = row.s0 + dot(travel, row.normal);
        let bias = 0;
        let massScale = 1;
        let impulseScale = 0;
        if (s > 0) {
            bias = s / h;
        } else if (useBias) {
            bias = Math.max(soft.biasRate * Math.min(s + params.linearSlop, 0), -params.pushMaxVelocity);
            massScale = soft.massScale;
            impulseScale = soft.impulseScale;
        }
        const vn = dot(relativeVelocity(bodyA, bodyB, row.rA, row.rB), row.normal);
        const delta = -row.normalMass * massScale * (vn + bias) - impulseScale * row.jn;
        const updated = Math.max(row.jn + delta, 0);
        const applied = updated - row.jn;
        row.jn = updated;
        applyImpulse(bodyA, bodyB, row.inertiaA, row.inertiaB, row.rA, row.rB, scale(row.normal, applied));

        const slip = relativeVelocity(bodyA, bodyB, row.rA, row.rB);
        let jt1 = row.jt1 - row.tangentMass1 * dot(slip, row.t1);
        let jt2 = row.jt2 - row.tangentMass2 * dot(slip, row.t2);
        const limit = row.friction * row.jn;
        const magnitude = Math.sqrt(jt1 * jt1 + jt2 * jt2);
        if (magnitude > limit) {
            const shrink = magnitude > 0 ? limit / magnitude : 0;
            jt1 *= shrink;
            jt2 *= shrink;
        }
        const tangentImpulse = add(scale(row.t1, jt1 - row.jt1), scale(row.t2, jt2 - row.jt2));
        row.jt1 = jt1;
        row.jt2 = jt2;
        applyImpulse(bodyA, bodyB, row.inertiaA, row.inertiaB, row.rA, row.rB, tangentImpulse);
    }
}

// Advances every awake body by one frame. Contacts carry warm-start impulses
// in and total frame impulses out.
export function solveSoftStep(world, contacts) {
    const params = world.params;
    const substeps = params.substeps;
    const h = params.frameDt / substeps;
    const omega = TWO_PI * Math.min(params.contactHertz, 0.25 / h);
    const zeta = params.contactDamping;
    const a1 = 2 * zeta + h * omega;
    const a2 = h * omega * a1;
    const a3 = 1 / (1 + a2);
    const soft = { biasRate: omega / a1, massScale: a2 * a3, impulseScale: a3 };

    const awake = [];
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        if (body.alive && !body.asleep && !body.isStatic) {
            awake.push(body);
            body.inertiaWorld = worldInverseInertia(body);
            body.contactImpulse = 0;
        }
    }

    const rows = [];
    for (let k = 0; k < contacts.length; k += 1) {
        const contact = contacts[k];
        const bodyA = world.bodies[contact.a];
        const bodyB = contact.fixedB ? null : world.bodies[contact.b];
        const basis = tangentBasis(contact.normal);
        const rA = sub(contact.pointA, bodyA.position);
        const rB = bodyB === null ? v3(0, 0, 0) : sub(contact.pointB, bodyB.position);
        const inertiaA = bodyA.inertiaWorld;
        const inertiaB = bodyB === null ? null : bodyB.inertiaWorld;
        const row = {
            contact: contact,
            bodyA: bodyA,
            bodyB: bodyB,
            inertiaA: inertiaA,
            inertiaB: inertiaB,
            normal: contact.normal,
            t1: basis.t1,
            t2: basis.t2,
            localA: quatRotateInverse(bodyA.orientation, rA),
            localB: bodyB === null ? v3(0, 0, 0) : quatRotateInverse(bodyB.orientation, rB),
            rA0: rA,
            rB0: rB,
            rA: rA,
            rB: rB,
            startA: bodyA.position,
            startB: bodyB === null ? v3(0, 0, 0) : bodyB.position,
            s0: contact.separation,
            friction: contact.friction,
            normalMass: effectiveMass(bodyA, bodyB, inertiaA, inertiaB, rA, rB, contact.normal),
            tangentMass1: effectiveMass(bodyA, bodyB, inertiaA, inertiaB, rA, rB, basis.t1),
            tangentMass2: effectiveMass(bodyA, bodyB, inertiaA, inertiaB, rA, rB, basis.t2),
            jn: contact.warmNormal,
            jt1: dot(contact.warmTangent, basis.t1),
            jt2: dot(contact.warmTangent, basis.t2),
            totalNormal: 0,
            vn0: 0,
        };
        row.vn0 = dot(relativeVelocity(bodyA, bodyB, rA, rB), contact.normal);
        rows.push(row);
    }

    const linearDecay = 1 / (1 + h * params.linearDamping);
    const angularDecay = 1 / (1 + h * params.angularDamping);
    for (let step = 0; step < substeps; step += 1) {
        for (let i = 0; i < awake.length; i += 1) {
            const body = awake[i];
            body.velocity = scale(addScaled(body.velocity, params.gravity, h), linearDecay);
            body.angular = scale(body.angular, angularDecay);
            body.substepImpulse = 0;
        }
        for (let k = 0; k < rows.length; k += 1) {
            const row = rows[k];
            const impulse = add(scale(row.normal, row.jn), add(scale(row.t1, row.jt1), scale(row.t2, row.jt2)));
            applyImpulse(row.bodyA, row.bodyB, row.inertiaA, row.inertiaB, row.rA, row.rB, impulse);
        }
        solvePass(rows, soft, true, h, params);
        for (let i = 0; i < awake.length; i += 1) {
            const body = awake[i];
            body.position = addScaled(body.position, body.velocity, h);
            body.orientation = quatIntegrate(body.orientation, body.angular, h);
        }
        solvePass(rows, soft, false, h, params);
        for (let k = 0; k < rows.length; k += 1) {
            const row = rows[k];
            row.totalNormal += row.jn;
            row.bodyA.substepImpulse += row.jn;
            if (row.bodyB !== null) {
                row.bodyB.substepImpulse += row.jn;
            }
        }
        // Rolling resistance: an angular impulse opposing w, bounded by the
        // lever arm times this substep's normal impulse, never reversing w.
        for (let i = 0; i < awake.length; i += 1) {
            const body = awake[i];
            const speed = length(body.angular);
            if (!params.rollingResistance || speed === 0 || body.substepImpulse === 0) {
                continue;
            }
            const axis = scale(body.angular, 1 / speed);
            const response = dot(axis, mat3MulVec(body.inertiaWorld, axis));
            const reduction = body.shape.rollingResistance * body.substepImpulse * response;
            if (reduction >= speed) {
                body.angular = v3(0, 0, 0);
            } else {
                body.angular = sub(body.angular, scale(mat3MulVec(body.inertiaWorld, axis),
                    body.shape.rollingResistance * body.substepImpulse));
            }
        }
    }

    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        if (row.vn0 < -params.restitutionThreshold && row.totalNormal > 0) {
            const vn = dot(relativeVelocity(row.bodyA, row.bodyB, row.rA, row.rB), row.normal);
            const target = -row.contact.restitution * row.vn0;
            const delta = Math.max(row.jn - row.normalMass * (vn - target), 0) - row.jn;
            row.jn += delta;
            applyImpulse(row.bodyA, row.bodyB, row.inertiaA, row.inertiaB, row.rA, row.rB, scale(row.normal, delta));
        }
    }

    for (let k = 0; k < rows.length; k += 1) {
        const row = rows[k];
        const contact = row.contact;
        contact.normalImpulse = row.totalNormal;
        contact.warmNormal = row.jn;
        contact.warmTangent = add(scale(row.t1, row.jt1), scale(row.t2, row.jt2));
    }
    return { iterations: substeps * 2, residual: 0 };
}
