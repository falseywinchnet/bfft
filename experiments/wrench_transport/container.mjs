// Container benchmark: 64 mixed convex bodies (cubes, spheres, cylinders,
// cones, sheets, prisms, arbitrary polyhedra, jagged rocks) released on a grid
// inside an open box with static walls, then left to pack.
//
//   node container.mjs --export scene.json            write the scene for other engines
//   node container.mjs --run wrench --out result.json  run one of this engine's solvers
//
// Every engine receives the same hull vertices, masses come from the same
// density, and every result file has the same layout (see runEngine below).

import { v3, length, normalized, quatIdentity, quatFromAxisAngle } from "./math3.mjs";
import {
    boxPoints, rockPoints, spherePoints, cylinderPoints, conePoints, prismPoints, polyhedronPoints,
    hullShape, makeRandom, randomRange,
} from "./shapes.mjs";
import { addBody, stepWorld, kineticEnergy } from "./world.mjs";
import { makeWorld } from "./scenes.mjs";

export const CONTAINER = {
    inner: 0.37,          // interior width and depth, metres
    wall: 0.04,           // wall thickness
    height: 0.5,
    friction: 0.5,
    density: 2600,
    seconds: 8,
    sampleRate: 60,
};

function shapePoints(kind, seed) {
    if (kind === 0) {
        return boxPoints(0.045, 0.045, 0.045);
    }
    if (kind === 1) {
        return spherePoints(0.036, 42);
    }
    if (kind === 2) {
        return cylinderPoints(0.028, 0.055, 16);
    }
    if (kind === 3) {
        return conePoints(0.032, 0.05, 16);
    }
    if (kind === 4) {
        return boxPoints(0.07, 0.045, 0.008);
    }
    if (kind === 5) {
        return prismPoints(seed, 0.034, 0.04, 5 + (seed % 3));
    }
    if (kind === 6) {
        return polyhedronPoints(seed, 0.04, 14);
    }
    return rockPoints(seed, "jagged");
}

export const KIND_NAMES = ["cube", "sphere", "cylinder", "cone", "sheet", "prism", "polyhedron", "rock"];

// The scene as plain data: static walls and bodies with cooked shapes.
export function buildContainerScene(seed) {
    const random = makeRandom(seed * 977 + 5);
    const options = { density: CONTAINER.density, friction: CONTAINER.friction, restitution: 0, rollingResistance: 0 };
    const half = 0.5 * CONTAINER.inner + 0.5 * CONTAINER.wall;
    const span = CONTAINER.inner + 2 * CONTAINER.wall;
    const walls = [
        { centre: v3(half, 0, 0.5 * CONTAINER.height), size: v3(CONTAINER.wall, span, CONTAINER.height) },
        { centre: v3(-half, 0, 0.5 * CONTAINER.height), size: v3(CONTAINER.wall, span, CONTAINER.height) },
        { centre: v3(0, half, 0.5 * CONTAINER.height), size: v3(span, CONTAINER.wall, CONTAINER.height) },
        { centre: v3(0, -half, 0.5 * CONTAINER.height), size: v3(span, CONTAINER.wall, CONTAINER.height) },
    ];
    const bodies = [];
    let index = 0;
    for (let layer = 0; layer < 4; layer += 1) {
        for (let row = 0; row < 4; row += 1) {
            for (let column = 0; column < 4; column += 1) {
                const kind = (index * 5 + layer) % 8;
                const shape = hullShape(shapePoints(kind, seed * 100 + index), options);
                const axis = normalized(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1)));
                bodies.push({
                    kind: kind,
                    shape: shape,
                    position: v3(0.088 * (column - 1.5), 0.088 * (row - 1.5), 0.06 + 0.09 * layer),
                    orientation: quatFromAxisAngle(axis, randomRange(random, 0, 2 * Math.PI)),
                });
                index += 1;
            }
        }
    }
    return { seed: seed, walls: walls, bodies: bodies };
}

export function sceneToJson(scene) {
    const bodies = [];
    for (let i = 0; i < scene.bodies.length; i += 1) {
        const body = scene.bodies[i];
        const vertices = [];
        const hull = body.shape.hulls[0];
        for (let v = 0; v < hull.vertices.length; v += 1) {
            vertices.push(hull.vertices[v].x, hull.vertices[v].y, hull.vertices[v].z);
        }
        bodies.push({
            kind: KIND_NAMES[body.kind],
            vertices: vertices,                 // body frame, origin at the centre of mass
            mass: body.shape.mass,
            inertia: body.shape.inertia,        // body frame, about the centre of mass, row-major
            radius: body.shape.radius,
            position: [body.position.x, body.position.y, body.position.z],
            quaternion: [body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z],
        });
    }
    const walls = [];
    for (let k = 0; k < scene.walls.length; k += 1) {
        const wall = scene.walls[k];
        walls.push({
            centre: [wall.centre.x, wall.centre.y, wall.centre.z],
            size: [wall.size.x, wall.size.y, wall.size.z],
        });
    }
    return {
        seed: scene.seed,
        gravity: [0, 0, -9.81],
        friction: CONTAINER.friction,
        density: CONTAINER.density,
        seconds: CONTAINER.seconds,
        sampleRate: CONTAINER.sampleRate,
        walls: walls,
        bodies: bodies,
    };
}

// Runs one of this engine's solver configurations at `hertz` frames per
// second and returns the common result record.
export function runEngine(scene, configName, hertz) {
    const world = makeWorld(configName, false, {
        frameDt: 1 / hertz,
        linearDamping: 0,
        angularDamping: 0,
        groundFriction: CONTAINER.friction,
    });
    const wallOptions = { friction: CONTAINER.friction, restitution: 0, rollingResistance: 0 };
    for (let k = 0; k < scene.walls.length; k += 1) {
        const wall = scene.walls[k];
        addBody(world, hullShape(boxPoints(wall.size.x, wall.size.y, wall.size.z), wallOptions), wall.centre,
            quatIdentity(), v3(0, 0, 0), v3(0, 0, 0), true);
    }
    const first = world.bodies.length;
    let weight = 0;
    for (let i = 0; i < scene.bodies.length; i += 1) {
        const body = scene.bodies[i];
        addBody(world, body.shape, body.position, body.orientation);
        weight += body.shape.mass * 9.81;
    }
    const frames = Math.round(CONTAINER.seconds * hertz);
    const every = Math.round(hertz / CONTAINER.sampleRate);
    const samples = [];
    let wallMs = 0;
    let worstMs = 0;
    let contactSum = 0;
    for (let f = 0; f < frames; f += 1) {
        const started = performance.now();
        stepWorld(world);
        const elapsed = performance.now() - started;
        wallMs += elapsed;
        worstMs = Math.max(worstMs, elapsed);
        contactSum += world.contacts.length;
        if ((f + 1) % every === 0) {
            let maxSpeed = 0;
            for (let i = first; i < world.bodies.length; i += 1) {
                const body = world.bodies[i];
                maxSpeed = Math.max(maxSpeed, length(body.velocity) + length(body.angular) * body.shape.radius);
            }
            samples.push({ t: (f + 1) / hertz, maxSpeed: maxSpeed, kineticEnergy: kineticEnergy(world) });
        }
    }
    // Vertical force the floor and walls return, from the last frame's impulses.
    let support = 0;
    let floorForce = 0;
    const substeps = world.solverName === "soft_step" ? world.params.substeps : 1;
    for (let k = 0; k < world.contacts.length; k += 1) {
        const contact = world.contacts[k];
        if (contact.fixedB) {
            support += (contact.warmNormal * contact.normal.z + contact.warmTangent.z) * substeps * hertz;
        }
        if (contact.b < 0) {
            floorForce += contact.warmNormal * substeps * hertz;
        }
    }
    const finalPoses = [];
    for (let i = first; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        finalPoses.push({
            p: [body.position.x, body.position.y, body.position.z],
            q: [body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z],
        });
    }
    const work = { stepsPerSecond: hertz, meanContacts: contactSum / frames };
    if (world.solverName === "wrench") {
        const stats = world.wrenchState.stats;
        work.newtonIterations = stats.newtonIterations;
        work.factorizations = stats.factorizations;
        work.passes = stats.passes;
        work.lineSearchEvaluations = stats.lineSearchEvaluations;
    } else {
        work.contactSweeps = frames * world.params.substeps * 2;
    }
    return {
        engine: "wrench_transport_js",
        config: configName + "@" + hertz + "Hz",
        dt: 1 / hertz,
        samples: samples,
        finalPoses: finalPoses,
        supportForce: support,
        floorForce: floorForce,
        weight: weight,
        wallSeconds: wallMs / 1000,
        worstStepMs: worstMs,
        work: work,
    };
}

async function main(argv) {
    const { writeFileSync } = await import("node:fs");
    let exportPath = "";
    let textPath = "";
    let runName = "";
    let outPath = "";
    let hertz = 60;
    let seed = 1;
    for (let k = 2; k < argv.length; k += 1) {
        if (argv[k] === "--export") {
            exportPath = argv[k + 1];
            k += 1;
        } else if (argv[k] === "--export-text") {
            textPath = argv[k + 1];
            k += 1;
        } else if (argv[k] === "--run") {
            runName = argv[k + 1];
            k += 1;
        } else if (argv[k] === "--out") {
            outPath = argv[k + 1];
            k += 1;
        } else if (argv[k] === "--hz") {
            hertz = Number(argv[k + 1]);
            k += 1;
        } else if (argv[k] === "--seed") {
            seed = Number(argv[k + 1]);
            k += 1;
        }
    }
    const scene = buildContainerScene(seed);
    if (exportPath !== "") {
        writeFileSync(exportPath, JSON.stringify(sceneToJson(scene)));
        console.log("wrote " + exportPath);
    }
    if (textPath !== "") {
        // The same scene as whitespace-separated numbers, for the C++ benchmark.
        const data = sceneToJson(scene);
        const lines = [String(data.walls.length)];
        for (let k = 0; k < data.walls.length; k += 1) {
            lines.push(data.walls[k].centre.join(" ") + " " + data.walls[k].size.join(" "));
        }
        lines.push(String(data.bodies.length));
        for (let i = 0; i < data.bodies.length; i += 1) {
            const body = data.bodies[i];
            lines.push(String(body.vertices.length / 3));
            lines.push(body.vertices.join(" "));
            lines.push(body.position.join(" ") + " " + body.quaternion.join(" "));
        }
        writeFileSync(textPath, lines.join("\n") + "\n");
        console.log("wrote " + textPath);
    }
    if (runName !== "") {
        const result = runEngine(scene, runName, hertz);
        const last = result.samples[result.samples.length - 1];
        console.log(result.config + ": wall " + result.wallSeconds.toFixed(2) + " s, final max speed " +
            last.maxSpeed.toExponential(2) + ", support/weight " + (result.supportForce / result.weight).toFixed(6));
        if (outPath !== "") {
            writeFileSync(outPath, JSON.stringify(result));
        }
    }
}

if (typeof process !== "undefined" && process.argv[1] !== undefined && process.argv[1].endsWith("container.mjs")) {
    await main(process.argv);
}
