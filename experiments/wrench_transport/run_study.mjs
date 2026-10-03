// Runs the benchmark scenes for every solver configuration and writes one
// JSON record. Usage: node run_study.mjs --out PATH [--quick] [--configs a,b]

import { writeFileSync } from "node:fs";
import {
    solverConfigs, sceneRest, sceneBoxTower, sceneRockStack, sceneSlabOnPebbles, scenePebbleOnSlab,
    sceneOverhang, sceneIncline, sceneDrop, sceneHeap,
} from "./scenes.mjs";

function parseArguments(argv) {
    const options = { out: "", quick: false, configs: Object.keys(solverConfigs) };
    for (let k = 2; k < argv.length; k += 1) {
        if (argv[k] === "--out") {
            options.out = argv[k + 1];
            k += 1;
        } else if (argv[k] === "--quick") {
            options.quick = true;
        } else if (argv[k] === "--configs") {
            options.configs = argv[k + 1].split(",");
            k += 1;
        }
    }
    return options;
}

function fixed(value, digits) {
    if (!Number.isFinite(value)) {
        return String(value);
    }
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 1.0e-3 || magnitude >= 1.0e5)) {
        return value.toExponential(2);
    }
    return value.toFixed(digits);
}

function report(scene, config, text) {
    console.log(scene.padEnd(26) + config.padEnd(18) + text);
}

const options = parseArguments(process.argv);
const record = { quick: options.quick, node: process.version, results: [] };
const seeds = options.quick ? 2 : 5;
const kinds = ["round", "flat", "jagged"];

for (let c = 0; c < options.configs.length; c += 1) {
    const config = options.configs[c];

    for (let k = 0; k < kinds.length; k += 1) {
        let worstDrift = 0;
        let worstRotation = 0;
        let worstPenetration = 0;
        let worstSpeed = 0;
        let meanMs = 0;
        for (let seed = 1; seed <= seeds; seed += 1) {
            const result = sceneRest(config, false, seed, kinds[k], options.quick ? 10 : 30);
            worstDrift = Math.max(worstDrift, result.driftMm);
            worstRotation = Math.max(worstRotation, result.rotationDeg);
            worstPenetration = Math.max(worstPenetration, result.penetrationMm);
            worstSpeed = Math.max(worstSpeed, result.lateSpeed);
            meanMs += result.timing.meanMs / seeds;
        }
        record.results.push({
            scene: "rest", kind: kinds[k], config: config, sleeping: false,
            driftMm: worstDrift, rotationDeg: worstRotation, penetrationMm: worstPenetration,
            lateSpeed: worstSpeed, meanMs: meanMs,
        });
        report("rest " + kinds[k], config, "drift " + fixed(worstDrift, 4) + " mm, rot " + fixed(worstRotation, 4) +
            " deg, pen " + fixed(worstPenetration, 4) + " mm, late speed " + fixed(worstSpeed, 5) + " m/s, " + fixed(meanMs, 3) + " ms");
    }

    const heights = options.quick ? [10, 20] : [10, 20, 40];
    for (let k = 0; k < heights.length; k += 1) {
        const result = sceneBoxTower(config, false, heights[k], options.quick ? 10 : 30);
        record.results.push({ scene: "box_tower", count: heights[k], config: config, sleeping: false, result: result });
        report("box tower " + heights[k], config, (result.standing ? "stands" : "FELL") + ", top error " +
            fixed(result.topHeightErrorMm, 3) + " mm, lean " + fixed(result.topLeanMm, 3) + " mm, drift " +
            fixed(result.driftMm, 4) + " mm, late KE " + fixed(result.lateEnergy, 3) + ", pen " +
            fixed(result.penetrationMm, 3) + " mm, " + fixed(result.timing.meanMs, 3) + " ms (worst " + fixed(result.timing.worstMs, 2) + ")");
    }

    for (let seed = 1; seed <= (options.quick ? 1 : 3); seed += 1) {
        for (let sleeping = 0; sleeping < 2; sleeping += 1) {
            const result = sceneRockStack(config, sleeping === 1, seed, 15, options.quick ? 10 : 60);
            record.results.push({ scene: "rock_stack", seed: seed, config: config, sleeping: sleeping === 1, result: result });
            report("rock stack 15 s" + seed + (sleeping ? " sleep" : ""), config, (result.standing ? "stands" : "FELL") +
                ", placed " + result.placed + " in " + result.attempts + " tries, drift " + fixed(result.driftMm, 4) + " mm, rot " + fixed(result.rotationDeg, 4) + " deg, late speed " +
                fixed(result.lateSpeed, 5) + ", asleep at " + fixed(result.asleepAt, 2) + " s, pen " +
                fixed(result.penetrationMm, 3) + " mm, " + fixed(result.timing.meanMs, 3) + " ms (worst " + fixed(result.timing.worstMs, 2) + ")");
        }
    }

    const ratios = [500, 5000];
    for (let k = 0; k < ratios.length; k += 1) {
        const under = sceneSlabOnPebbles(config, false, ratios[k], options.quick ? 10 : 30);
        record.results.push({ scene: "slab_on_pebbles", ratio: ratios[k], config: config, sleeping: false, result: under });
        report("slab on pebbles " + ratios[k], config, "slab height error " + fixed(under.slabHeightErrorMm, 3) +
            " mm, pen " + fixed(under.penetrationMm, 3) + " mm, pebble speed " + fixed(under.latePebbleSpeed, 5) +
            " m/s, drift " + fixed(under.driftMm, 4) + " mm, " + fixed(under.timing.meanMs, 3) + " ms");
        const over = scenePebbleOnSlab(config, false, ratios[k], options.quick ? 10 : 30);
        record.results.push({ scene: "pebble_on_slab", ratio: ratios[k], config: config, sleeping: false, result: over });
        report("pebble on slab " + ratios[k], config, "pen " + fixed(over.penetrationMm, 3) + " mm, pebble speed " +
            fixed(over.latePebbleSpeed, 5) + " m/s, drift " + fixed(over.driftMm, 4) + " mm");
    }

    const fractions = [0.05, 0.02, 0.01, -0.01, -0.02, -0.05];
    for (let k = 0; k < fractions.length; k += 1) {
        const result = sceneOverhang(config, false, fractions[k], fractions[k] > 0 ? (options.quick ? 10 : 60) : 3);
        record.results.push({ scene: "overhang", insideFraction: fractions[k], config: config, sleeping: false, result: result });
        report("overhang " + (fractions[k] * 100).toFixed(0) + "%", config, "rotation " + fixed(result.rotationDeg, 4) +
            " deg, tipped at " + fixed(result.tippedAt, 2) + " s");
    }

    for (let slab = 0; slab < 2; slab += 1) {
        const below = sceneIncline(config, false, -0.05, slab === 1, options.quick ? 10 : 30);
        const above = sceneIncline(config, false, 0.05, slab === 1, 3.5);
        record.results.push({ scene: "incline", onSlab: slab === 1, offset: -0.05, config: config, sleeping: false, result: below });
        record.results.push({ scene: "incline", onSlab: slab === 1, offset: 0.05, config: config, sleeping: false, result: above });
        report("incline" + (slab ? " on slab" : ""), config, "below: creep " + fixed(below.travelMm, 4) +
            " mm; above: slide " + fixed(above.travelMm, 1) + " mm in 3 s");
    }

    for (let seed = 1; seed <= (options.quick ? 1 : 3); seed += 1) {
        const result = sceneDrop(config, false, seed, 5);
        record.results.push({ scene: "drop", seed: seed, config: config, sleeping: false, result: result });
        report("drop s" + seed, config, "impact " + fixed(result.impactSpeed, 3) + " m/s, max after " +
            fixed(result.maxSpeedAfterContact, 3) + ", bounce " + fixed(result.bounceMm, 3) + " mm, deepest " +
            fixed(result.deepestMm, 3) + " mm, rest at " + fixed(result.restAt, 2) + " s");
    }

    const heap = sceneHeap(config, false, 1, 30, options.quick ? 6 : 12);
    record.results.push({ scene: "heap", count: 30, config: config, sleeping: false, result: heap });
    report("heap 30", config, (heap.finite ? "finite" : "NON-FINITE") + ", rest " + fixed(heap.restAt, 2) +
        " s, worst speed " + fixed(heap.worstSpeed, 2) + ", worst pen " + fixed(heap.worstPenetrationMm, 2) +
        " mm, final pen " + fixed(heap.finalPenetrationMm, 3) + " mm, final KE " + fixed(heap.finalEnergy, 3) + ", " +
        fixed(heap.timing.meanMs, 3) + " ms (worst " + fixed(heap.timing.worstMs, 2) + ")" +
        (heap.solverStats ? ", newton " + heap.solverStats.newtonIterations + " max " + heap.solverStats.maxNewtonIterations +
            " passes " + heap.solverStats.passes + " unconverged " + heap.solverStats.notConverged + " pass-limited " + heap.solverStats.passLimit : ""));
}

if (options.out !== "") {
    writeFileSync(options.out, JSON.stringify(record, null, 1));
    console.log("wrote " + options.out);
}
