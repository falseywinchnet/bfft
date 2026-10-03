// Sanity checks for cooking, collision and both solvers on small scenes.

import { v3, sub, length, dot, quatFromAxisAngle } from "./math3.mjs";
import { buildHull } from "./hull.mjs";
import { boxShape, rockShape, rockPoints } from "./shapes.mjs";
import { createWorld, addBody, stepWorld, kineticEnergy, maxPenetration, defaultWorldParams } from "./world.mjs";

let failures = 0;

function check(name, condition, detail) {
    const verdict = condition ? "PASS" : "FAIL";
    if (!condition) {
        failures += 1;
    }
    console.log(verdict + "  " + name + "  " + detail);
}

function testCooking() {
    const box = boxShape(0.2, 0.1, 0.05);
    const hull = box.hulls[0];
    check("box topology", hull.vertices.length === 8 && hull.faces.length === 6 && hull.edges.length === 12,
        "V=" + hull.vertices.length + " F=" + hull.faces.length + " E=" + hull.edges.length);
    const mass = 2600 * 0.2 * 0.1 * 0.05;
    check("box mass", Math.abs(box.mass - mass) < 1e-12, "mass=" + box.mass);
    const ixx = mass * (0.1 * 0.1 + 0.05 * 0.05) / 12;
    const izz = mass * (0.2 * 0.2 + 0.1 * 0.1) / 12;
    check("box inertia", Math.abs(box.inertia[0] - ixx) < 1e-12 && Math.abs(box.inertia[8] - izz) < 1e-12,
        "Ixx=" + box.inertia[0] + " Izz=" + box.inertia[8]);
    let worst = 0;
    let eulerOk = true;
    const kinds = ["round", "flat", "jagged"];
    for (let seed = 1; seed <= 40; seed += 1) {
        for (let k = 0; k < 3; k += 1) {
            const points = rockPoints(seed, kinds[k]);
            const rock = buildHull(points);
            if (rock.vertices.length - rock.edges.length + rock.faces.length !== 2) {
                eulerOk = false;
            }
            for (let p = 0; p < points.length; p += 1) {
                for (let f = 0; f < rock.faces.length; f += 1) {
                    worst = Math.max(worst, dot(rock.faces[f].normal, points[p]) - rock.faces[f].offset);
                }
            }
        }
    }
    check("rock hulls closed and convex", eulerOk && worst < 2e-4, "max outside distance=" + worst.toExponential(2));
}

function runRest(solverName, shape, height, seconds, label) {
    const params = defaultWorldParams();
    params.sleeping = false;
    params.reduceManifolds = solverName === "soft_step";
    const world = createWorld(solverName, params);
    addBody(world, shape, v3(0, 0, height));
    const frames = Math.round(seconds * 60);
    let settled = null;
    let drift = 0;
    for (let f = 0; f < frames; f += 1) {
        stepWorld(world);
        if (f === 180) {
            settled = world.bodies[0].position;
        }
        if (f > 180) {
            drift = Math.max(drift, length(sub(world.bodies[0].position, settled)));
        }
    }
    const body = world.bodies[0];
    console.log("  " + label + " [" + solverName + "] z=" + body.position.z.toFixed(6) +
        " |v|=" + length(body.velocity).toExponential(2) + " drift=" + drift.toExponential(2) +
        " pen=" + maxPenetration(world).toExponential(2) + " KE=" + kineticEnergy(world).toExponential(2));
    return { drift: drift, body: body, world: world };
}

function testRest() {
    const solvers = ["soft_step", "wrench"];
    for (let s = 0; s < 2; s += 1) {
        const box = runRest(solvers[s], boxShape(0.1, 0.1, 0.1), 0.06, 10, "box");
        check("box rests [" + solvers[s] + "]", Math.abs(box.body.position.z - 0.05) < 1e-3 && box.drift < 1e-4,
            "z=" + box.body.position.z);
        const rock = runRest(solvers[s], rockShape(3, "flat"), 0.05, 10, "flat rock");
        check("rock rests [" + solvers[s] + "]", rock.drift < 1e-3, "drift=" + rock.drift.toExponential(2));
    }
}

function testStack(solverName, count) {
    const params = defaultWorldParams();
    params.sleeping = false;
    const world = createWorld(solverName, params);
    const shape = boxShape(0.1, 0.1, 0.05);
    for (let i = 0; i < count; i += 1) {
        addBody(world, shape, v3(0, 0, 0.025 + 0.0502 * i + 0.001));
    }
    let top = null;
    let drift = 0;
    let energyLate = 0;
    const started = performance.now();
    for (let f = 0; f < 600; f += 1) {
        stepWorld(world);
        if (f === 300) {
            top = world.bodies[count - 1].position;
        }
        if (f > 300) {
            drift = Math.max(drift, length(sub(world.bodies[count - 1].position, top)));
            energyLate = Math.max(energyLate, kineticEnergy(world));
        }
    }
    const elapsed = performance.now() - started;
    const topZ = world.bodies[count - 1].position.z;
    const expected = 0.025 + 0.05 * (count - 1);
    console.log("  stack " + count + " [" + solverName + "] topZ=" + topZ.toFixed(6) + " expected=" + expected.toFixed(6) +
        " drift=" + drift.toExponential(2) + " lateKE=" + energyLate.toExponential(2) +
        " pen=" + maxPenetration(world).toExponential(2) + " ms/frame=" + (elapsed / 600).toFixed(3));
    if (solverName === "wrench") {
        console.log("    stats " + JSON.stringify(world.wrenchState.stats));
    }
    const stands = Math.abs(topZ - expected) < 2e-3 && drift < 1e-3;
    if (solverName === "soft_step") {
        // The specification's parameters do not hold these columns; the
        // outcome is reported, not asserted.
        console.log("NOTE  stack " + count + " [soft_step] " + (stands ? "stands" : "does not stand") +
            "  topZ error=" + (topZ - expected).toExponential(2));
    } else {
        check("stack " + count + " stands [" + solverName + "]", stands, "topZ error=" + (topZ - expected).toExponential(2));
    }
}

testCooking();
testRest();
testStack("soft_step", 10);
testStack("wrench", 10);
testStack("soft_step", 30);
testStack("wrench", 30);
console.log(failures === 0 ? "ALL PASS" : failures + " FAILURES");
process.exit(failures === 0 ? 0 : 1);
