// World: bodies, once-per-frame collision, manifold persistence, islands and
// sleeping. The contact solver is selected by name so both run on identical
// contacts and bookkeeping.

import {
    v3, add, sub, scale, dot, length, quatIdentity, quatRotateInverse, mat3Inverse,
} from "./math3.mjs";
import { transformHull, boundsOverlap, collideHulls, collideGround, reduceManifold } from "./collide.mjs";
import { solveSoftStep } from "./solver_soft_step.mjs";
import { solveWrenchTransport, createWrenchState, defaultWrenchOptions } from "./solver_wrench.mjs";

export function defaultWorldParams() {
    return {
        gravity: v3(0, 0, -9.81),
        frameDt: 1 / 60,
        substeps: 8,
        contactHertz: 30,
        contactDamping: 10,
        pushMaxVelocity: 0.5,
        linearSlop: 0.0005,
        speculativeDistance: 0.002,
        linearDamping: 0.02,
        angularDamping: 0.05,
        sleepLinear: 0.004,
        sleepAngular: 0.02,
        sleepTime: 0.5,
        groundZ: 0,
        groundFriction: 0.9,
        groundRollingResistance: 0.004,   // metres; the sand bed absorbs rocking and spin
        restitutionThreshold: 1.0,
        sleeping: true,
        reduceManifolds: true,
        rollingResistance: true,
    };
}

export function createWorld(solverName, params, wrenchOptions) {
    const options = {};
    const names = Object.keys(defaultWrenchOptions);
    for (let k = 0; k < names.length; k += 1) {
        options[names[k]] = defaultWrenchOptions[names[k]];
    }
    if (wrenchOptions !== undefined) {
        const given = Object.keys(wrenchOptions);
        for (let k = 0; k < given.length; k += 1) {
            options[given[k]] = wrenchOptions[given[k]];
        }
    }
    return {
        solverName: solverName,
        params: params === undefined ? defaultWorldParams() : params,
        wrenchOptions: options,
        wrenchState: createWrenchState(),
        bodies: [],
        contacts: [],
        persisted: new Map(),   // manifold key -> last frame's points, for warm starting
        frame: 0,
        lastSolve: { iterations: 0, residual: 0 },
        nextIsland: 1,
        hold: null,             // { id, target: {position, orientation}, params } while the crane holds a body
        holdState: { tension: 0, positionError: v3(0, 0, 0) },
        rest: { quiet: false, quietSeconds: 0, kineticEnergy: 0, maxSpeed: 0, awake: 0 },
    };
}

export function defaultHoldParams() {
    return {
        linHertz: 3.0, linZeta: 1.0,
        angHertz: 2.0, angZeta: 1.0,
        maxLift: 1.5, maxLateral: 0.35, maxTorque: 0.35,
    };
}

// The crane: at most one held body. The target is {position, orientation}.
export function holdBody(world, id, target, params) {
    wakeBody(world, id);
    world.hold = { id: id, target: target, params: params === undefined ? defaultHoldParams() : params };
    world.holdState = { tension: 0, positionError: v3(0, 0, 0) };
}

export function releaseHold(world) {
    if (world.hold !== null) {
        wakeBody(world, world.hold.id);
    }
    world.hold = null;
    world.holdState = { tension: 0, positionError: v3(0, 0, 0) };
}

// `isStatic` bodies never move: they collide like the ground and carry load
// without entering the solve.
export function addBody(world, shape, position, orientation, velocity, angular, isStatic) {
    const body = {
        id: world.bodies.length,
        isStatic: isStatic === true,
        shape: shape,
        position: position,
        orientation: orientation === undefined ? quatIdentity() : orientation,
        velocity: velocity === undefined ? v3(0, 0, 0) : velocity,
        angular: angular === undefined ? v3(0, 0, 0) : angular,
        mass: shape.mass,
        invMass: isStatic === true ? 0 : 1 / shape.mass,
        inertiaLocal: shape.inertia,
        invInertiaLocal: isStatic === true ? [0, 0, 0, 0, 0, 0, 0, 0, 0] : mat3Inverse(shape.inertia),
        inertiaWorld: null,
        inertiaWorldFull: null,
        alive: true,
        asleep: false,
        island: 0,
        quietTime: 0,
        solverIndex: -1,
        contactImpulse: 0,
        substepImpulse: 0,
        worldHulls: null,
        hullsFresh: false,
    };
    world.bodies.push(body);
    return body.id;
}

export function wakeBody(world, id) {
    const body = world.bodies[id];
    if (!body.asleep) {
        body.quietTime = 0;
        return;
    }
    const island = body.island;
    for (let i = 0; i < world.bodies.length; i += 1) {
        const other = world.bodies[i];
        if (other.alive && other.asleep && other.island === island) {
            other.asleep = false;
            other.quietTime = 0;
        }
    }
}

function refreshHulls(world, body) {
    // A sleeping body's world hulls stay valid once rebuilt at its final pose.
    if (body.asleep && body.hullsFresh) {
        return;
    }
    const hulls = [];
    for (let h = 0; h < body.shape.hulls.length; h += 1) {
        hulls.push(transformHull(body.shape.hulls[h], body.position, body.orientation));
    }
    body.worldHulls = hulls;
    body.hullsFresh = body.asleep;
}

function motionBound(body, dt) {
    return (length(body.velocity) + length(body.angular) * body.shape.radius) * dt;
}

function makeContact(world, a, b, hullA, hullB, point, friction, restitution) {
    const bodyA = world.bodies[a];
    return {
        a: a,
        b: b,
        hullA: hullA,
        hullB: hullB,
        fixedB: b < 0 || world.bodies[b].isStatic,   // side B is the ground or a static body
        pointA: point.pointA,
        pointB: point.pointB,
        normal: point.normal,
        separation: point.separation,
        friction: friction,
        restitution: restitution,
        normalImpulse: 0,
        warmNormal: 0,
        warmTangent: v3(0, 0, 0),
        matched: false,           // true when last frame's impulses were inherited
        bounce: 0,                // rebound speed owed on the next frame
        pendingBounce: 0,         // rebound speed inherited from the last frame
        localA: quatRotateInverse(bodyA.orientation, sub(point.pointA, bodyA.position)),
        mode: 0,
    };
}

// Inherits last frame's impulses for points that stayed within 2 mm on body A
// with a normal within about 25 degrees.
function warmStart(world, key, bodyA, contacts) {
    const old = world.persisted.get(key);
    if (old === undefined) {
        return;
    }
    for (let k = 0; k < contacts.length; k += 1) {
        const contact = contacts[k];
        const local = quatRotateInverse(bodyA.orientation, sub(contact.pointA, bodyA.position));
        let best = -1;
        let bestDistance = 0.002;
        for (let j = 0; j < old.length; j += 1) {
            if (old[j].used) {
                continue;
            }
            const distance = length(sub(local, old[j].localA));
            if (distance < bestDistance && dot(contact.normal, old[j].normal) > 0.9) {
                bestDistance = distance;
                best = j;
            }
        }
        if (best >= 0) {
            old[best].used = true;
            contact.warmNormal = old[best].warmNormal;
            contact.warmTangent = old[best].warmTangent;
            contact.matched = true;
            contact.pendingBounce = old[best].bounce;
        }
    }
}

function collide(world) {
    const params = world.params;
    const bodies = world.bodies;
    for (let restart = 0; restart < 64; restart += 1) {
        let woke = false;
        const contacts = [];
        for (let i = 0; i < bodies.length; i += 1) {
            if (bodies[i].alive) {
                refreshHulls(world, bodies[i]);
            }
        }
        for (let i = 0; i < bodies.length && !woke; i += 1) {
            const bodyA = bodies[i];
            if (!bodyA.alive) {
                continue;
            }
            const motionA = motionBound(bodyA, params.frameDt);
            if (!bodyA.asleep && !bodyA.isStatic) {
                const margin = Math.min(params.speculativeDistance + motionA, 0.05);
                const friction = Math.sqrt(bodyA.shape.friction * params.groundFriction);
                for (let h = 0; h < bodyA.worldHulls.length; h += 1) {
                    let points = collideGround(bodyA.worldHulls[h], params.groundZ, margin);
                    if (params.reduceManifolds) {
                        points = reduceManifold(points);
                    }
                    const group = [];
                    for (let k = 0; k < points.length; k += 1) {
                        group.push(makeContact(world, i, -1, h, 0, points[k], friction, bodyA.shape.restitution));
                    }
                    warmStart(world, manifoldKey(i, -1, h, 0), bodyA, group);
                    for (let k = 0; k < group.length; k += 1) {
                        contacts.push(group[k]);
                    }
                }
            }
            for (let j = i + 1; j < bodies.length && !woke; j += 1) {
                const bodyB = bodies[j];
                if (!bodyB.alive) {
                    continue;
                }
                // A pair matters only if one of its bodies is awake and free.
                const restingA = bodyA.asleep || bodyA.isStatic;
                const restingB = bodyB.asleep || bodyB.isStatic;
                if (restingA && restingB) {
                    continue;
                }
                const margin = Math.min(
                    params.speculativeDistance + motionA + motionBound(bodyB, params.frameDt), 0.05);
                const reach = bodyA.shape.radius + bodyB.shape.radius + margin;
                if (length(sub(bodyA.position, bodyB.position)) > reach) {
                    continue;
                }
                // The contact's first body is always a free one.
                let first = i;
                let second = j;
                if (bodyA.isStatic) {
                    first = j;
                    second = i;
                }
                const bodyFirst = bodies[first];
                const bodySecond = bodies[second];
                const friction = Math.sqrt(bodyA.shape.friction * bodyB.shape.friction);
                const restitution = Math.max(bodyA.shape.restitution, bodyB.shape.restitution);
                for (let ha = 0; ha < bodyFirst.worldHulls.length && !woke; ha += 1) {
                    for (let hb = 0; hb < bodySecond.worldHulls.length; hb += 1) {
                        const hullA = bodyFirst.worldHulls[ha];
                        const hullB = bodySecond.worldHulls[hb];
                        if (!boundsOverlap(hullA, hullB, margin)) {
                            continue;
                        }
                        let points = collideHulls(hullA, hullB, margin);
                        if (points.length === 0) {
                            continue;
                        }
                        if (bodyA.asleep || bodyB.asleep) {
                            // An awake body touching a sleeping one wakes its island.
                            wakeBody(world, bodyA.asleep ? i : j);
                            woke = true;
                            break;
                        }
                        if (params.reduceManifolds) {
                            points = reduceManifold(points);
                        }
                        const group = [];
                        for (let k = 0; k < points.length; k += 1) {
                            group.push(makeContact(world, first, second, ha, hb, points[k], friction, restitution));
                        }
                        warmStart(world, manifoldKey(first, second, ha, hb), bodyFirst, group);
                        for (let k = 0; k < group.length; k += 1) {
                            contacts.push(group[k]);
                        }
                    }
                }
            }
        }
        if (!woke) {
            world.contacts = contacts;
            return;
        }
        // Clear warm-start claims and collide again with the woken island.
        const iterator = world.persisted.values();
        for (let item = iterator.next(); !item.done; item = iterator.next()) {
            for (let k = 0; k < item.value.length; k += 1) {
                item.value[k].used = false;
            }
        }
    }
}

function manifoldKey(a, b, hullA, hullB) {
    return ((a * 4096 + (b + 1)) * 4 + hullA) * 4 + hullB;
}

function persistContacts(world) {
    // Manifolds of sleeping bodies keep their stored impulses.
    const kept = new Map();
    const iterator = world.persisted.entries();
    for (let item = iterator.next(); !item.done; item = iterator.next()) {
        const points = item.value[1];
        if (points.length > 0 && world.bodies[points[0].a].asleep) {
            for (let k = 0; k < points.length; k += 1) {
                points[k].used = false;
            }
            kept.set(item.value[0], points);
        }
    }
    for (let k = 0; k < world.contacts.length; k += 1) {
        const contact = world.contacts[k];
        const key = manifoldKey(contact.a, contact.b, contact.hullA, contact.hullB);
        let list = kept.get(key);
        if (list === undefined || list.stamp !== world.frame) {
            list = [];
            list.stamp = world.frame;
            kept.set(key, list);
        }
        list.push({
            a: contact.a,
            b: contact.b,
            fixedB: contact.fixedB,
            localA: contact.localA,
            normal: contact.normal,
            warmNormal: contact.warmNormal,
            warmTangent: contact.warmTangent,
            bounce: contact.bounce,
            used: false,
        });
    }
    world.persisted = kept;
}

function findRoot(parent, index) {
    let root = index;
    while (parent[root] !== root) {
        root = parent[root];
    }
    while (parent[index] !== root) {
        const next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

function updateSleep(world) {
    const params = world.params;
    const bodies = world.bodies;
    const parent = [];
    for (let i = 0; i < bodies.length; i += 1) {
        parent.push(i);
    }
    for (let k = 0; k < world.contacts.length; k += 1) {
        const contact = world.contacts[k];
        if (!contact.fixedB && contact.normalImpulse > 0) {
            const rootA = findRoot(parent, contact.a);
            const rootB = findRoot(parent, contact.b);
            if (rootA !== rootB) {
                parent[Math.max(rootA, rootB)] = Math.min(rootA, rootB);
            }
        }
    }
    const islandQuiet = new Array(bodies.length).fill(true);
    const held = world.hold === null ? -1 : world.hold.id;
    let maxSpeed = 0;
    let awake = 0;
    for (let i = 0; i < bodies.length; i += 1) {
        const body = bodies[i];
        if (!body.alive || body.asleep || body.isStatic) {
            continue;
        }
        awake += 1;
        const linear = length(body.velocity);
        const angular = length(body.angular);
        maxSpeed = Math.max(maxSpeed, linear + angular * body.shape.radius);
        const candidate = linear < params.sleepLinear && angular < params.sleepAngular;
        if (candidate) {
            body.quietTime += params.frameDt;
        } else {
            body.quietTime = 0;
        }
        // The held body never sleeps, and keeps its island awake.
        if (body.quietTime < params.sleepTime || i === held) {
            islandQuiet[findRoot(parent, i)] = false;
        }
    }
    // The rest report: one record the game can read each frame at no cost.
    const rest = world.rest;
    rest.maxSpeed = maxSpeed;
    rest.awake = awake;
    rest.kineticEnergy = kineticEnergy(world);
    if (maxSpeed < params.sleepLinear) {
        rest.quietSeconds += params.frameDt;
    } else {
        rest.quietSeconds = 0;
    }
    rest.quiet = awake === 0 || rest.quietSeconds >= params.sleepTime;
    if (!params.sleeping) {
        return;
    }
    const islandId = new Array(bodies.length).fill(0);
    for (let i = 0; i < bodies.length; i += 1) {
        const body = bodies[i];
        if (!body.alive || body.asleep || body.isStatic) {
            continue;
        }
        const root = findRoot(parent, i);
        if (islandQuiet[root]) {
            if (islandId[root] === 0) {
                islandId[root] = world.nextIsland;
                world.nextIsland += 1;
            }
            body.asleep = true;
            body.hullsFresh = false;
            body.island = islandId[root];
            body.velocity = v3(0, 0, 0);
            body.angular = v3(0, 0, 0);
        }
    }
}

export function stepWorld(world) {
    collide(world);
    if (world.solverName === "soft_step") {
        world.lastSolve = solveSoftStep(world, world.contacts);
    } else {
        world.lastSolve = solveWrenchTransport(world, world.contacts, world.wrenchState, world.wrenchOptions);
    }
    persistContacts(world);
    updateSleep(world);
    world.frame += 1;
}

export function kineticEnergy(world) {
    let energy = 0;
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        if (!body.alive || body.asleep || body.isStatic) {
            continue;
        }
        energy += 0.5 * body.mass * dot(body.velocity, body.velocity);
        const local = quatRotateInverse(body.orientation, body.angular);
        const inertia = body.inertiaLocal;
        energy += 0.5 * (
            local.x * (inertia[0] * local.x + inertia[1] * local.y + inertia[2] * local.z) +
            local.y * (inertia[3] * local.x + inertia[4] * local.y + inertia[5] * local.z) +
            local.z * (inertia[6] * local.x + inertia[7] * local.y + inertia[8] * local.z));
    }
    return energy;
}

export function allAsleep(world) {
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        if (body.alive && !body.asleep && !body.isStatic) {
            return false;
        }
    }
    return true;
}

// The stable part of a pile that is still partly moving: every body that has
// been quiet for a quarter second (or is asleep) and is joined to the ground
// or a static body through a chain of loaded contacts between such bodies.
// One breadth-first walk over the retained manifolds; returns ascending ids.
export function stableSet(world) {
    const bodies = world.bodies;
    const settled = new Uint8Array(bodies.length);
    for (let i = 0; i < bodies.length; i += 1) {
        const body = bodies[i];
        if (body.alive && !body.isStatic && (body.asleep || body.quietTime >= 0.25)) {
            settled[i] = 1;
        }
    }
    const neighbours = [];
    for (let i = 0; i < bodies.length; i += 1) {
        neighbours.push([]);
    }
    const reached = new Uint8Array(bodies.length);
    const queue = [];
    const iterator = world.persisted.values();
    for (let item = iterator.next(); !item.done; item = iterator.next()) {
        const points = item.value;
        let loaded = false;
        for (let k = 0; k < points.length; k += 1) {
            if (points[k].warmNormal > 0) {
                loaded = true;
            }
        }
        if (!loaded || points.length === 0 || !settled[points[0].a]) {
            continue;
        }
        const a = points[0].a;
        const b = points[0].b;
        if (points[0].fixedB) {
            if (!reached[a]) {
                reached[a] = 1;
                queue.push(a);
            }
        } else if (settled[b]) {
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
    }
    for (let head = 0; head < queue.length; head += 1) {
        const list = neighbours[queue[head]];
        for (let k = 0; k < list.length; k += 1) {
            if (!reached[list[k]]) {
                reached[list[k]] = 1;
                queue.push(list[k]);
            }
        }
    }
    const stable = [];
    for (let i = 0; i < bodies.length; i += 1) {
        if (reached[i]) {
            stable.push(i);
        }
    }
    return stable;
}

// Deepest overlap among the current contacts, as a positive depth in metres.
export function maxPenetration(world) {
    let depth = 0;
    for (let k = 0; k < world.contacts.length; k += 1) {
        depth = Math.max(depth, -world.contacts[k].separation);
    }
    return depth;
}
