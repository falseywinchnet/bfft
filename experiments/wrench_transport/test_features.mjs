// Checks for the features around the contact solve: static bodies,
// restitution, rolling resistance, the crane hold, the rest report and the
// stable-set query. All use the wrench solver.

import { v3, sub, length, quatIdentity, quatFromAxisAngle } from "./math3.mjs";
import { boxShape, rockShape } from "./shapes.mjs";
import {
    createWorld, addBody, stepWorld, defaultWorldParams, holdBody, releaseHold, stableSet,
} from "./world.mjs";

let failures = 0;

function check(name, condition, detail) {
    if (!condition) {
        failures += 1;
    }
    console.log((condition ? "PASS" : "FAIL") + "  " + name + "  " + detail);
}

function newWorld(sleeping) {
    const params = defaultWorldParams();
    params.sleeping = sleeping;
    params.reduceManifolds = false;
    return createWorld("wrench", params);
}

function run(world, frames) {
    for (let f = 0; f < frames; f += 1) {
        stepWorld(world);
    }
}

function testStatic() {
    const world = newWorld(false);
    const table = boxShape(0.4, 0.4, 0.05);
    addBody(world, table, v3(0, 0, 0.3), quatIdentity(), v3(0, 0, 0), v3(0, 0, 0), true);
    addBody(world, boxShape(0.1, 0.1, 0.1), v3(0.05, 0, 0.3 + 0.025 + 0.05 + 0.002));
    run(world, 300);
    const box = world.bodies[1];
    check("box rests on a static table", Math.abs(box.position.z - 0.375) < 1.0e-6 && length(box.velocity) < 1.0e-9,
        "z=" + box.position.z.toFixed(9) + " speed=" + length(box.velocity).toExponential(1));
    check("static table did not move", world.bodies[0].position.z === 0.3, "z=" + world.bodies[0].position.z);
}

function testRestitution() {
    const heights = [];
    const values = [0.0, 0.5];
    for (let k = 0; k < 2; k += 1) {
        const world = newWorld(false);
        const shape = boxShape(0.08, 0.08, 0.08, { restitution: values[k], rollingResistance: 0 });
        addBody(world, shape, v3(0, 0, 0.04 + 0.3));
        let touched = false;
        let peak = 0;
        for (let f = 0; f < 120; f += 1) {
            stepWorld(world);
            const gap = world.bodies[0].position.z - 0.04;
            if (gap < 0.002) {
                touched = true;
            }
            if (touched) {
                peak = Math.max(peak, gap);
            }
        }
        heights.push(peak);
    }
    // e = 0.5 from 0.3 m: ideal rebound 0.075 m; damping and the frame's
    // discrete impact take a little off.
    check("no restitution, no bounce", heights[0] < 0.002, "peak=" + (heights[0] * 1000).toFixed(3) + " mm");
    check("restitution 0.5 rebounds near e^2 h", heights[1] > 0.055 && heights[1] < 0.08,
        "peak=" + (heights[1] * 1000).toFixed(1) + " mm (ideal 75)");
}

function overhangTips(outside, resistance) {
    const world = newWorld(false);
    addBody(world, boxShape(0.1, 0.1, 0.1, { rollingResistance: resistance }), v3(0, 0, 0.05));
    addBody(world, boxShape(0.2, 0.05, 0.02, { rollingResistance: resistance }), v3(0.05 + outside, 0, 0.1102));
    run(world, 180);
    const q = world.bodies[1].orientation;
    return 2 * Math.acos(Math.min(1, Math.abs(q.w))) * 180 / Math.PI;
}

function testRolling() {
    // A lever arm of 2 mm holds a slab whose centre of mass is 1 mm past the
    // edge, and not one 4 mm past it. With no lever arm 1 mm is enough to tip.
    const held = overhangTips(0.001, 0.002);
    const tipped = overhangTips(0.004, 0.002);
    const bare = overhangTips(0.001, 0);
    check("rolling resistance holds 1 mm past the edge", held < 0.01, "rotation=" + held.toExponential(2) + " deg");
    check("rolling resistance yields 4 mm past the edge", tipped > 10, "rotation=" + tipped.toFixed(1) + " deg");
    check("without it 1 mm past the edge tips", bare > 10, "rotation=" + bare.toFixed(1) + " deg");
}

function testHold() {
    const world = newWorld(false);
    const shape = rockShape(5, "round");
    const id = addBody(world, shape, v3(0, 0, 0.3));
    const target = { position: v3(0, 0, 0.3), orientation: quatIdentity() };
    holdBody(world, id, target);
    run(world, 120);
    const body = world.bodies[id];
    check("held rock hangs on target", length(sub(body.position, target.position)) < 2.0e-4 &&
        Math.abs(world.holdState.tension - 1) < 0.01,
        "error=" + (length(sub(body.position, target.position)) * 1000).toExponential(2) + " mm tension=" +
        world.holdState.tension.toFixed(4));
    // Move 10 cm sideways over 2 s, then watch for overshoot.
    for (let f = 0; f < 120; f += 1) {
        world.hold.target = { position: v3(0.1 * (f + 1) / 120, 0, 0.3), orientation: quatIdentity() };
        stepWorld(world);
    }
    let overshoot = 0;
    for (let f = 0; f < 120; f += 1) {
        stepWorld(world);
        overshoot = Math.max(overshoot, body.position.x - 0.1);
    }
    check("move 10 cm: no overshoot beyond 2 mm", overshoot < 0.002 && Math.abs(body.position.x - 0.1) < 2.0e-4,
        "overshoot=" + (overshoot * 1000).toFixed(3) + " mm final x=" + body.position.x.toFixed(5));
    // Rotate 90 degrees about z over 2 s.
    for (let f = 0; f < 120; f += 1) {
        const angle = 0.5 * Math.PI * (f + 1) / 120;
        world.hold.target = { position: v3(0.1, 0, 0.3), orientation: quatFromAxisAngle(v3(0, 0, 1), angle) };
        stepWorld(world);
    }
    run(world, 120);
    const q = body.orientation;
    const goal = quatFromAxisAngle(v3(0, 0, 1), 0.5 * Math.PI);
    const overlap = Math.abs(q.w * goal.w + q.x * goal.x + q.y * goal.y + q.z * goal.z);
    check("rotate 90 degrees about z", 2 * Math.acos(Math.min(1, overlap)) < 0.002,
        "error=" + (2 * Math.acos(Math.min(1, overlap)) * 180 / Math.PI).toExponential(2) + " deg");
}

function buildBoxStack(world, count) {
    const shape = boxShape(0.12, 0.1, 0.04);
    for (let i = 0; i < count; i += 1) {
        addBody(world, shape, v3(0, 0, 0.02 + 0.04 * i + 0.0002 * (i + 1)));
    }
    run(world, 120);
}

function testSetDown() {
    const world = newWorld(false);
    buildBoxStack(world, 5);
    const before = [];
    for (let i = 0; i < 5; i += 1) {
        before.push(world.bodies[i].position);
    }
    const shape = boxShape(0.1, 0.1, 0.0769);       // about 2 kg
    const top = 0.2 + 0.0385;
    const id = addBody(world, shape, v3(0, 0, top + 0.03));
    let targetZ = top + 0.03;
    holdBody(world, id, { position: v3(0, 0, targetZ), orientation: quatIdentity() });
    run(world, 60);
    let previous = world.holdState.tension;
    let monotone = true;
    let lowered = 0;
    for (let f = 0; f < 600 && world.holdState.tension >= 0.05; f += 1) {
        targetZ -= 0.02 / 60;
        world.hold.target = { position: v3(0, 0, targetZ), orientation: quatIdentity() };
        stepWorld(world);
        if (world.holdState.tension > previous + 0.02) {
            monotone = false;
        }
        previous = world.holdState.tension;
        lowered += 1;
    }
    run(world, 300);
    let moved = 0;
    for (let i = 0; i < 5; i += 1) {
        moved = Math.max(moved, length(sub(world.bodies[i].position, before[i])));
    }
    check("set-down: tension falls monotonically to slack", monotone && world.holdState.tension < 0.05,
        "tension=" + world.holdState.tension.toFixed(3) + " after " + (lowered / 60).toFixed(2) + " s of lowering");
    check("set-down: stack undisturbed before release", moved < 3.0e-4, "moved=" + (moved * 1000).toExponential(2) + " mm");
    releaseHold(world);
    run(world, 300);
    moved = 0;
    for (let i = 0; i < 5; i += 1) {
        moved = Math.max(moved, length(sub(world.bodies[i].position, before[i])));
    }
    check("set-down: stack undisturbed after release", moved < 5.0e-4 && world.rest.quiet,
        "moved=" + (moved * 1000).toExponential(2) + " mm quiet=" + world.rest.quiet);
}

function testShove() {
    const world = newWorld(false);
    buildBoxStack(world, 5);
    const shape = boxShape(0.1, 0.1, 0.0769);
    const id = addBody(world, shape, v3(-0.2, 0, 0.18));
    holdBody(world, id, { position: v3(-0.2, 0, 0.18), orientation: quatIdentity() });
    run(world, 60);
    const body = world.bodies[id];
    const weightImpulse = body.mass * 9.81 / 60;
    let worst = 0;
    // Drive the target 15 cm into the top stack box over 3 s.
    for (let f = 0; f < 240; f += 1) {
        const x = -0.2 + 0.15 * Math.min(1, (f + 1) / 180);
        world.hold.target = { position: v3(x, 0, 0.18), orientation: quatIdentity() };
        stepWorld(world);
        worst = Math.max(worst, world.holdState.lateralImpulse / weightImpulse);
    }
    check("shove: side force capped at 0.35 of the held weight", worst <= 0.35 * 1.05 && worst > 0.3,
        "largest side force=" + worst.toFixed(4) + " m g");
}

function testRestAndStable() {
    const world = newWorld(true);
    buildBoxStack(world, 6);
    run(world, 60);
    check("rest report: settled stack is quiet", world.rest.quiet && world.rest.awake === 0,
        "quiet=" + world.rest.quiet + " awake=" + world.rest.awake + " KE=" + world.rest.kineticEnergy);
    // A rock dropped well beside the stack: the stack stays the stable set
    // while the newcomer is still moving.
    const loose = addBody(world, rockShape(9, "round"), v3(0.4, 0, 0.3));
    run(world, 10);
    const during = stableSet(world);
    check("stable set while a rock falls nearby", during.length === 6 && during.indexOf(loose) < 0 && !world.rest.quiet,
        "stable=" + during.join(",") + " quiet=" + world.rest.quiet);
    run(world, 300);
    const after = stableSet(world);
    check("stable set once it lands", after.length === 7 && world.rest.quiet, "stable=" + after.join(",") + " quiet=" + world.rest.quiet);
}

testStatic();
testRestitution();
testRolling();
testHold();
testSetDown();
testShove();
testRestAndStable();
console.log(failures === 0 ? "ALL PASS" : failures + " FAILURES");
process.exit(failures === 0 ? 0 : 1);
