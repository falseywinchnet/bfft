// Benchmark scenes after the Zen Construction acceptance tests (specification
// section 11). Each scene builds a world for a named solver configuration,
// advances it, and returns measured numbers; verdicts are left to the caller.

import {
    v3, add, sub, scale, dot, cross, length, normalized, quatIdentity, quatFromAxisAngle, quatRotate,
} from "./math3.mjs";
import { boxShape, rockShape, makeRandom, randomUnit, randomRange } from "./shapes.mjs";
import {
    createWorld, addBody, stepWorld, kineticEnergy, maxPenetration, defaultWorldParams, allAsleep,
} from "./world.mjs";

export const FRAME_RATE = 60;

export const solverConfigs = {
    // The specification as written: 30 Hz soft contacts, 8 substeps, manifolds
    // reduced to four points, rolling resistance after each substep.
    soft_step_spec: { solver: "soft_step", contactHertz: 30, contactDamping: 10, substeps: 8, reduce: true, rolling: true },
    // The same solver in its strongest measured configuration at 8 substeps:
    // the stiffest contact the substep rate allows, every manifold point kept,
    // no post-hoc rolling resistance.
    soft_step_best: { solver: "soft_step", contactHertz: 120, contactDamping: 10, substeps: 8, reduce: false, rolling: false },
    // Twice the substeps, stiffer again.
    soft_step_16: { solver: "soft_step", contactHertz: 240, contactDamping: 10, substeps: 16, reduce: false, rolling: false },
    wrench: { solver: "wrench" },
};

export function makeWorld(configName, sleeping, overrides) {
    const config = solverConfigs[configName];
    const params = defaultWorldParams();
    params.sleeping = sleeping;
    if (config.solver === "soft_step") {
        params.contactHertz = config.contactHertz;
        params.contactDamping = config.contactDamping;
        params.substeps = config.substeps;
        params.reduceManifolds = config.reduce;
        params.rollingResistance = config.rolling;
    } else {
        params.reduceManifolds = false;
    }
    if (overrides !== undefined) {
        const names = Object.keys(overrides);
        for (let k = 0; k < names.length; k += 1) {
            params[names[k]] = overrides[names[k]];
        }
    }
    return createWorld(config.solver, params, config.wrenchOptions);
}

function lowestLocalZ(shape, orientation) {
    let lowest = Infinity;
    for (let h = 0; h < shape.hulls.length; h += 1) {
        const vertices = shape.hulls[h].vertices;
        for (let v = 0; v < vertices.length; v += 1) {
            lowest = Math.min(lowest, quatRotate(orientation, vertices[v]).z);
        }
    }
    return lowest;
}

function highestWorldZ(body) {
    let highest = -Infinity;
    for (let h = 0; h < body.shape.hulls.length; h += 1) {
        const vertices = body.shape.hulls[h].vertices;
        for (let v = 0; v < vertices.length; v += 1) {
            highest = Math.max(highest, body.position.z + quatRotate(body.orientation, vertices[v]).z);
        }
    }
    return highest;
}

function rotationBetween(a, b) {
    const overlap = Math.abs(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z);
    return 2 * Math.acos(Math.min(1, overlap));
}

function snapshot(world) {
    const poses = [];
    for (let i = 0; i < world.bodies.length; i += 1) {
        poses.push({ position: world.bodies[i].position, orientation: world.bodies[i].orientation });
    }
    return poses;
}

// Largest translation (metres) and rotation (radians) of any body since `poses`.
function driftSince(world, poses, into) {
    for (let i = 0; i < poses.length; i += 1) {
        const body = world.bodies[i];
        into.translation = Math.max(into.translation, length(sub(body.position, poses[i].position)));
        into.rotation = Math.max(into.rotation, rotationBetween(body.orientation, poses[i].orientation));
    }
}

function maxSpeed(world) {
    let speed = 0;
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        speed = Math.max(speed, length(body.velocity) + length(body.angular) * body.shape.radius);
    }
    return speed;
}

function newTimer() {
    return { total: 0, worst: 0, frames: 0 };
}

function timedStep(world, timer) {
    const started = performance.now();
    stepWorld(world);
    const elapsed = performance.now() - started;
    timer.total += elapsed;
    timer.worst = Math.max(timer.worst, elapsed);
    timer.frames += 1;
}

function timing(timer) {
    return { meanMs: timer.total / Math.max(1, timer.frames), worstMs: timer.worst, frames: timer.frames };
}

// Test 1: one rock dropped from 1 cm, then watched.
export function sceneRest(configName, sleeping, seed, kind, seconds) {
    const world = makeWorld(configName, sleeping);
    const shape = rockShape(seed, kind);
    addBody(world, shape, v3(0, 0, 0.01 - lowestLocalZ(shape, quatIdentity())));
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    let poses = null;
    const drift = { translation: 0, rotation: 0 };
    let lateSpeed = 0;
    let asleepAt = -1;
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, timer);
        if (asleepAt < 0 && world.bodies[0].asleep) {
            asleepAt = (f + 1) / FRAME_RATE;
        }
        if (f + 1 === 3 * FRAME_RATE) {
            poses = snapshot(world);
        }
        if (f + 1 > 3 * FRAME_RATE) {
            driftSince(world, poses, drift);
            lateSpeed = Math.max(lateSpeed, maxSpeed(world));
        }
    }
    return {
        driftMm: drift.translation * 1000,
        rotationDeg: drift.rotation * 180 / Math.PI,
        penetrationMm: maxPenetration(world) * 1000,
        lateSpeed: lateSpeed,
        asleepAt: asleepAt,
        timing: timing(timer),
    };
}

// A column of identical boxes spawned 1 mm apart.
export function sceneBoxTower(configName, sleeping, count, seconds) {
    const world = makeWorld(configName, sleeping);
    const shape = boxShape(0.1, 0.1, 0.05);
    for (let i = 0; i < count; i += 1) {
        addBody(world, shape, v3(0, 0, 0.025 + 0.051 * i + 0.001));
    }
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    const settleFrame = Math.min(5 * FRAME_RATE, Math.floor(frames / 2));
    let poses = null;
    const drift = { translation: 0, rotation: 0 };
    let lateEnergy = 0;
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, timer);
        if (f + 1 === settleFrame) {
            poses = snapshot(world);
        }
        if (f + 1 > settleFrame) {
            driftSince(world, poses, drift);
            lateEnergy = Math.max(lateEnergy, kineticEnergy(world));
        }
    }
    const top = world.bodies[count - 1];
    return {
        standing: top.position.z > 0.05 * (count - 1),
        topHeightErrorMm: (top.position.z - (0.025 + 0.05 * (count - 1))) * 1000,
        topLeanMm: Math.sqrt(top.position.x * top.position.x + top.position.y * top.position.y) * 1000,
        driftMm: drift.translation * 1000,
        lateEnergy: lateEnergy,
        penetrationMm: maxPenetration(world) * 1000,
        timing: timing(timer),
    };
}

// Builds a stack the way a careful player would: each rock is set on the one
// below, 2 mm above it, first over its centre of mass and, if that does not
// hold, at nearby offsets. A placement is kept when everything comes to rest,
// the new rock ends within 3 cm of where it was set, and no older rock has
// shifted more than 5 mm (stones resting on three points do re-seat a little
// under new load). Returns the world holding the rocks that were placed.
export function buildRockStack(configName, sleeping, seed, count, kind, timer) {
    const offsets = [v3(0, 0, 0)];
    for (let ring = 1; ring <= 3; ring += 1) {
        for (let k = 0; k < 6; k += 1) {
            const angle = (k + 0.5 * ring) * Math.PI / 3;
            offsets.push(v3(0.012 * ring * Math.cos(angle), 0.012 * ring * Math.sin(angle), 0));
        }
    }
    let placed = [];          // {shape, position, orientation}
    let persisted = null;
    let world = makeWorld(configName, sleeping);
    let attempts = 0;
    for (let i = 0; i < count; i += 1) {
        const shape = kind === "box" ? boxShape(0.12, 0.09, 0.03) : rockShape(seed * 100 + i, kind);
        const orientation = quatIdentity();
        let accepted = false;
        for (let c = 0; c < offsets.length && !accepted; c += 1) {
            attempts += 1;
            const trial = makeWorld(configName, sleeping);
            for (let b = 0; b < placed.length; b += 1) {
                addBody(trial, placed[b].shape, placed[b].position, placed[b].orientation);
            }
            if (persisted !== null) {
                trial.persisted = persisted;
            }
            let x = offsets[c].x;
            let y = offsets[c].y;
            let z = 0.002 - lowestLocalZ(shape, orientation);
            if (placed.length > 0) {
                const below = trial.bodies[placed.length - 1];
                x += below.position.x;
                y += below.position.y;
                z = highestWorldZ(below) + 0.002 - lowestLocalZ(shape, orientation);
            }
            const start = v3(x, y, z);
            addBody(trial, shape, start, orientation);
            const before = snapshot(trial);
            let quiet = 0;
            let settled = false;
            for (let f = 0; f < 4 * FRAME_RATE; f += 1) {
                timedStep(trial, timer);
                if (maxSpeed(trial) < 5.0e-4) {
                    quiet += 1;
                } else {
                    quiet = 0;
                }
                if (quiet >= 30 || allAsleep(trial)) {
                    settled = true;
                    break;
                }
            }
            const newest = trial.bodies[placed.length];
            const slid = Math.sqrt(
                (newest.position.x - start.x) * (newest.position.x - start.x) +
                (newest.position.y - start.y) * (newest.position.y - start.y));
            let disturbed = 0;
            for (let b = 0; b < placed.length; b += 1) {
                disturbed = Math.max(disturbed, length(sub(trial.bodies[b].position, before[b].position)));
            }
            if (process.env.STACK_VERBOSE) {
                console.log("rock " + i + " try " + c + " settled " + settled + " slid(mm) " + (slid * 1000).toFixed(2) +
                    " disturbed(mm) " + (disturbed * 1000).toFixed(3) + " speed " + maxSpeed(trial).toExponential(1));
            }
            if (settled && slid < 0.03 && disturbed < 0.005) {
                accepted = true;
                placed = [];
                for (let b = 0; b < trial.bodies.length; b += 1) {
                    placed.push({
                        shape: trial.bodies[b].shape,
                        position: trial.bodies[b].position,
                        orientation: trial.bodies[b].orientation,
                    });
                }
                persisted = trial.persisted;
                world = trial;
            }
        }
        if (!accepted) {
            break;
        }
    }
    // A fresh world at the accepted poses, with the requested sleeping mode.
    const final = makeWorld(configName, sleeping);
    for (let b = 0; b < placed.length; b += 1) {
        addBody(final, placed[b].shape, placed[b].position, placed[b].orientation);
    }
    if (persisted !== null) {
        final.persisted = persisted;
    }
    return { world: final, placed: placed.length, attempts: attempts };
}

// Test 2: a stack of flat rocks built by the placement helper, then watched.
export function sceneRockStack(configName, sleeping, seed, count, watchSeconds) {
    const timer = newTimer();
    const built = buildRockStack(configName, sleeping, seed, count, "flat", timer);
    const world = built.world;
    const watch = newTimer();
    const frames = Math.round(watchSeconds * FRAME_RATE);
    const settleFrames = Math.min(5 * FRAME_RATE, frames);
    let poses = snapshot(world);
    const drift = { translation: 0, rotation: 0 };
    let lateSpeed = 0;
    let asleepAt = -1;
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, watch);
        if (f + 1 === settleFrames) {
            poses = snapshot(world);
        }
        if (f + 1 > settleFrames) {
            driftSince(world, poses, drift);
            lateSpeed = Math.max(lateSpeed, maxSpeed(world));
        }
        if (asleepAt < 0 && allAsleep(world)) {
            asleepAt = (f + 1) / FRAME_RATE;
        }
    }
    let standing = built.placed === count;
    for (let i = 1; i < built.placed; i += 1) {
        if (world.bodies[i].position.z < world.bodies[i - 1].position.z) {
            standing = false;
        }
    }
    return {
        placed: built.placed,
        attempts: built.attempts,
        standing: standing,
        driftMm: drift.translation * 1000,
        rotationDeg: drift.rotation * 180 / Math.PI,
        lateSpeed: lateSpeed,
        asleepAt: asleepAt,
        penetrationMm: maxPenetration(world) * 1000,
        topZ: built.placed > 0 ? world.bodies[built.placed - 1].position.z : 0,
        buildTiming: timing(timer),
        timing: timing(watch),
    };
}

// Test 3a: a heavy slab on three light pebbles. `ratio` is slab mass over pebble mass.
export function sceneSlabOnPebbles(configName, sleeping, ratio, seconds) {
    const world = makeWorld(configName, sleeping);
    const pebble = boxShape(0.02, 0.02, 0.0192);           // about 0.02 kg
    const slabVolume = 0.2 * 0.2 * 0.04;
    const slab = boxShape(0.2, 0.2, 0.04, { density: pebble.mass * ratio / slabVolume });
    const spots = [v3(0.06, 0, 0), v3(-0.03, 0.052, 0), v3(-0.03, -0.052, 0)];
    for (let k = 0; k < 3; k += 1) {
        addBody(world, pebble, v3(spots[k].x, spots[k].y, 0.0096 + 0.0005));
    }
    addBody(world, slab, v3(0, 0, 0.0192 + 0.02 + 0.002));
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    let latePebbleSpeed = 0;
    let poses = null;
    const drift = { translation: 0, rotation: 0 };
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, timer);
        if (f + 1 === 2 * FRAME_RATE) {
            poses = snapshot(world);
        }
        if (f + 1 > 2 * FRAME_RATE) {
            driftSince(world, poses, drift);
            for (let k = 0; k < 3; k += 1) {
                latePebbleSpeed = Math.max(latePebbleSpeed, length(world.bodies[k].velocity));
            }
        }
    }
    return {
        slabMass: slab.mass,
        pebbleMass: pebble.mass,
        slabHeightErrorMm: (world.bodies[3].position.z - (0.0192 + 0.02)) * 1000,
        penetrationMm: maxPenetration(world) * 1000,
        latePebbleSpeed: latePebbleSpeed,
        driftMm: drift.translation * 1000,
        timing: timing(timer),
    };
}

// Test 3b: a light pebble on a heavy slab.
export function scenePebbleOnSlab(configName, sleeping, ratio, seconds) {
    const world = makeWorld(configName, sleeping);
    const pebble = boxShape(0.02, 0.02, 0.0192);
    const slabVolume = 0.2 * 0.2 * 0.04;
    const slab = boxShape(0.2, 0.2, 0.04, { density: pebble.mass * ratio / slabVolume });
    addBody(world, slab, v3(0, 0, 0.02 + 0.0005));
    addBody(world, pebble, v3(0.01, 0.02, 0.04 + 0.0096 + 0.002));
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    let latePebbleSpeed = 0;
    let poses = null;
    const drift = { translation: 0, rotation: 0 };
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, timer);
        if (f + 1 === 2 * FRAME_RATE) {
            poses = snapshot(world);
        }
        if (f + 1 > 2 * FRAME_RATE) {
            driftSince(world, poses, drift);
            latePebbleSpeed = Math.max(latePebbleSpeed, length(world.bodies[1].velocity));
        }
    }
    return {
        penetrationMm: maxPenetration(world) * 1000,
        latePebbleSpeed: latePebbleSpeed,
        driftMm: drift.translation * 1000,
        timing: timing(timer),
    };
}

// Test 4: a 20 x 5 x 2 cm slab on a 10 cm cube, its centre of mass
// `insideFraction` of the slab length inside (positive) or outside (negative)
// the cube's edge. `resistance` is the slab's rolling-resistance lever arm.
export function sceneOverhang(configName, sleeping, insideFraction, seconds) {
    const world = makeWorld(configName, sleeping);
    const cube = boxShape(0.1, 0.1, 0.1);
    const slab = boxShape(0.2, 0.05, 0.02);
    addBody(world, cube, v3(0, 0, 0.05));
    addBody(world, slab, v3(0.05 - insideFraction * 0.2, 0, 0.1 + 0.01 + 0.0002));
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    const start = world.bodies[1].orientation;
    let tippedAt = -1;
    let rotation = 0;
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, timer);
        rotation = rotationBetween(world.bodies[1].orientation, start) * 180 / Math.PI;
        if (tippedAt < 0 && rotation > 10) {
            tippedAt = (f + 1) / FRAME_RATE;
            break;
        }
    }
    return { rotationDeg: rotation, tippedAt: tippedAt, timing: timing(timer) };
}

// Test 5: a 10 cm box under gravity tilted so that tan(theta) = mu + offset.
// With `onSlab`, the box rests on a 100 kg slab and is measured against it.
export function sceneIncline(configName, sleeping, tangentOffset, onSlab, seconds) {
    const mu = 0.75;
    const tangent = mu + tangentOffset;
    const cosine = 1 / Math.sqrt(1 + tangent * tangent);
    const world = makeWorld(configName, sleeping, {
        gravity: v3(9.81 * tangent * cosine, 0, -9.81 * cosine),
        groundFriction: mu,
    });
    const box = boxShape(0.1, 0.1, 0.1, { friction: mu });
    let boxIndex = 0;
    if (onSlab) {
        // A wide slab keyed to the ground by friction 1.5 so only the box can slide.
        const slab = boxShape(0.6, 0.4, 0.05, { density: 100 / (0.6 * 0.4 * 0.05), friction: mu });
        world.params.groundFriction = 3.0 * 3.0 / mu;
        addBody(world, slab, v3(0, 0, 0.025));
        addBody(world, box, v3(-0.15, 0, 0.05 + 0.05 + 0.0002));
        boxIndex = 1;
    } else {
        addBody(world, box, v3(0, 0, 0.05 + 0.0002));
    }
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    const settle = Math.round(0.5 * FRAME_RATE);
    let startOffset = null;
    let travel = 0;
    for (let f = 0; f < frames; f += 1) {
        timedStep(world, timer);
        let offset = world.bodies[boxIndex].position;
        if (onSlab) {
            offset = sub(offset, world.bodies[0].position);
        }
        if (f + 1 === settle) {
            startOffset = offset;
        }
        if (f + 1 > settle) {
            travel = length(sub(offset, startOffset));
        }
    }
    return { travelMm: travel * 1000, timing: timing(timer) };
}

// Test 6: a round rock dropped from 0.3 m onto the ground.
export function sceneDrop(configName, sleeping, seed, seconds) {
    const world = makeWorld(configName, sleeping);
    const shape = rockShape(seed, "round");
    addBody(world, shape, v3(0, 0, 0.3 - lowestLocalZ(shape, quatIdentity())));
    const timer = newTimer();
    const frames = Math.round(seconds * FRAME_RATE);
    let touched = false;
    let impactSpeed = 0;
    let speedAfter = 0;
    let lowestAfter = Infinity;
    let bounce = 0;
    let restAt = -1;
    let quiet = 0;
    for (let f = 0; f < frames; f += 1) {
        const before = length(world.bodies[0].velocity);
        timedStep(world, timer);
        const body = world.bodies[0];
        if (!touched) {
            for (let k = 0; k < world.contacts.length; k += 1) {
                if (world.contacts[k].normalImpulse > 0) {
                    touched = true;
                    impactSpeed = before;
                }
            }
        }
        if (touched) {
            speedAfter = Math.max(speedAfter, length(body.velocity));
            const gap = body.position.z + lowestLocalZ(shape, body.orientation);
            lowestAfter = Math.min(lowestAfter, gap);
            bounce = Math.max(bounce, gap);
            if (maxSpeed(world) < 0.004) {
                quiet += 1;
                if (quiet === 30 && restAt < 0) {
                    restAt = (f + 1) / FRAME_RATE;
                }
            } else {
                quiet = 0;
            }
        }
    }
    return {
        impactSpeed: impactSpeed,
        maxSpeedAfterContact: speedAfter,
        bounceMm: bounce * 1000,
        deepestMm: -lowestAfter * 1000,
        restAt: restAt,
        timing: timing(timer),
    };
}

// Test 15: rocks dropped one after another into a heap, each from a random
// height of at most 0.5 m, then `seconds` of watching.
export function sceneHeap(configName, sleeping, seed, count, seconds) {
    const world = makeWorld(configName, sleeping);
    const kinds = ["round", "flat", "jagged"];
    const random = makeRandom(seed * 31 + 7);
    const spawnEvery = 18;
    const timer = newTimer();
    const frames = spawnEvery * count + Math.round(seconds * FRAME_RATE);
    let restAt = -1;
    let quiet = 0;
    let worstSpeed = 0;
    let worstPenetration = 0;
    let spawned = 0;
    for (let f = 0; f < frames; f += 1) {
        if (spawned < count && f === spawned * spawnEvery) {
            const shape = rockShape(seed * 1000 + spawned, kinds[spawned % 3]);
            const radius = 0.08 * Math.sqrt(randomUnit(random));
            const angle = 2 * Math.PI * randomUnit(random);
            const axis = normalized(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1)));
            addBody(world, shape,
                v3(radius * Math.cos(angle), radius * Math.sin(angle), randomRange(random, 0.3, 0.5)),
                quatFromAxisAngle(axis, randomRange(random, 0, 2 * Math.PI)));
            for (let b = 0; b < world.bodies.length; b += 1) {
                world.bodies[b].asleep = false;
            }
            spawned += 1;
        }
        timedStep(world, timer);
        worstSpeed = Math.max(worstSpeed, maxSpeed(world));
        worstPenetration = Math.max(worstPenetration, maxPenetration(world));
        if (spawned === count && maxSpeed(world) < 0.004) {
            quiet += 1;
            if (quiet === 30 && restAt < 0) {
                restAt = (f + 1 - spawnEvery * count) / FRAME_RATE;
            }
        } else {
            quiet = 0;
            restAt = -1;
        }
    }
    let finite = true;
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        if (!(Number.isFinite(body.position.x) && Number.isFinite(body.velocity.x) && Number.isFinite(body.orientation.w))) {
            finite = false;
        }
    }
    const result = {
        finite: finite,
        restAt: restAt,
        worstSpeed: worstSpeed,
        worstPenetrationMm: worstPenetration * 1000,
        finalPenetrationMm: maxPenetration(world) * 1000,
        finalEnergy: kineticEnergy(world),
        contacts: world.contacts.length,
        timing: timing(timer),
    };
    if (world.solverName === "wrench") {
        result.solverStats = world.wrenchState.stats;
    }
    return result;
}
