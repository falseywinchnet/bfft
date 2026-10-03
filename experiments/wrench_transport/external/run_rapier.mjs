// Runs the exported container scene in Rapier and writes the common result
// record. Usage:
//   node run_rapier.mjs scene.json out.json --steps-per-sample 1 --iterations 4 --modules DIR
// DIR is a directory whose node_modules holds @dimforge/rapier3d-compat.

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

let scenePath = "";
let outPath = "";
let stepsPerSample = 1;
let iterations = 4;
let modules = "";
let lengthUnit = 1;
for (let k = 2; k < process.argv.length; k += 1) {
    if (process.argv[k] === "--steps-per-sample") {
        stepsPerSample = Number(process.argv[k + 1]);
        k += 1;
    } else if (process.argv[k] === "--iterations") {
        iterations = Number(process.argv[k + 1]);
        k += 1;
    } else if (process.argv[k] === "--length-unit") {
        lengthUnit = Number(process.argv[k + 1]);
        k += 1;
    } else if (process.argv[k] === "--modules") {
        modules = process.argv[k + 1];
        k += 1;
    } else if (scenePath === "") {
        scenePath = process.argv[k];
    } else {
        outPath = process.argv[k];
    }
}

const requireFrom = createRequire(modules + "/package.json");
const RAPIER = requireFrom("@dimforge/rapier3d-compat");
await RAPIER.init();

const scene = JSON.parse(readFileSync(scenePath, "utf8"));
const dt = 1 / (scene.sampleRate * stepsPerSample);
const world = new RAPIER.World({ x: 0, y: 0, z: -9.81 });
world.timestep = dt;
world.numSolverIterations = iterations;
// Rapier scales its contact tolerances by the typical object size (default 1 m).
world.lengthUnit = lengthUnit;

const floorBody = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, 0, -0.5));
const floorCollider = world.createCollider(
    RAPIER.ColliderDesc.cuboid(5, 5, 0.5).setFriction(scene.friction).setRestitution(0), floorBody);
for (let k = 0; k < scene.walls.length; k += 1) {
    const wall = scene.walls[k];
    const wallBody = world.createRigidBody(
        RAPIER.RigidBodyDesc.fixed().setTranslation(wall.centre[0], wall.centre[1], wall.centre[2]));
    world.createCollider(
        RAPIER.ColliderDesc.cuboid(0.5 * wall.size[0], 0.5 * wall.size[1], 0.5 * wall.size[2])
            .setFriction(scene.friction).setRestitution(0), wallBody);
}
const bodies = [];
let weight = 0;
for (let i = 0; i < scene.bodies.length; i += 1) {
    const item = scene.bodies[i];
    const description = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(item.position[0], item.position[1], item.position[2])
        .setRotation({ w: item.quaternion[0], x: item.quaternion[1], y: item.quaternion[2], z: item.quaternion[3] })
        .setCanSleep(false);
    const body = world.createRigidBody(description);
    const collider = RAPIER.ColliderDesc.convexHull(new Float32Array(item.vertices));
    if (collider === null) {
        throw new Error("Rapier rejected hull " + i);
    }
    world.createCollider(collider.setDensity(scene.density).setFriction(scene.friction).setRestitution(0), body);
    bodies.push(body);
    weight += body.mass() * 9.81;
}

const sampleCount = Math.round(scene.seconds * scene.sampleRate);
const samples = [];
let wallMs = 0;
let worstMs = 0;
for (let sample = 0; sample < sampleCount; sample += 1) {
    for (let s = 0; s < stepsPerSample; s += 1) {
        const started = performance.now();
        world.step();
        const elapsed = performance.now() - started;
        wallMs += elapsed;
        worstMs = Math.max(worstMs, elapsed);
    }
    let maxSpeed = 0;
    let energy = 0;
    for (let i = 0; i < bodies.length; i += 1) {
        const v = bodies[i].linvel();
        const w = bodies[i].angvel();
        const linear = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        const angular = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
        maxSpeed = Math.max(maxSpeed, linear + angular * scene.bodies[i].radius);
        energy += 0.5 * bodies[i].mass() * linear * linear;
    }
    // Translational kinetic energy only; rotation is left out here.
    samples.push({ t: (sample + 1) / scene.sampleRate, maxSpeed: maxSpeed, kineticEnergy: energy });
}

// Normal force from the floor, from the last step's manifold impulses.
let floorForce = 0;
function addManifold(manifold) {
    for (let k = 0; k < manifold.numContacts(); k += 1) {
        floorForce += Math.abs(manifold.normal().z) * manifold.contactImpulse(k) / dt;
    }
}
function visitFloorPartner(other) {
    world.contactPair(floorCollider, other, addManifold);
}
world.contactPairsWith(floorCollider, visitFloorPartner);

const finalPoses = [];
for (let i = 0; i < bodies.length; i += 1) {
    const p = bodies[i].translation();
    const q = bodies[i].rotation();
    finalPoses.push({ p: [p.x, p.y, p.z], q: [q.w, q.x, q.y, q.z] });
}
const result = {
    engine: "rapier " + RAPIER.version(),
    config: iterations + " solver iterations, " + (scene.sampleRate * stepsPerSample) + " Hz, length unit " + lengthUnit + " m",
    dt: dt,
    samples: samples,
    finalPoses: finalPoses,
    supportForce: null,
    floorForce: floorForce,
    weight: weight,
    wallSeconds: wallMs / 1000,
    worstStepMs: worstMs,
    work: { stepsPerSecond: scene.sampleRate * stepsPerSample, solverIterations: iterations * sampleCount * stepsPerSample },
};
writeFileSync(outPath, JSON.stringify(result));
console.log(result.engine + " " + result.config + ": wall " + result.wallSeconds.toFixed(2) + " s, final max speed " +
    samples[samples.length - 1].maxSpeed.toExponential(2) + ", floor/weight " + (floorForce / weight).toFixed(4));
