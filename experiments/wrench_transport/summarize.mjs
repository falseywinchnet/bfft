// Prints the retained study record as Markdown tables.
// Usage: node summarize.mjs results/full_study.json

import { readFileSync } from "node:fs";

function number(value, digits) {
    if (value === undefined || value === null || !Number.isFinite(value)) {
        return String(value);
    }
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 1.0e-3 || magnitude >= 1.0e5)) {
        return value.toExponential(1);
    }
    return value.toFixed(digits);
}

function find(results, scene, config, extra) {
    const matches = [];
    for (let k = 0; k < results.length; k += 1) {
        const item = results[k];
        if (item.scene !== scene || item.config !== config) {
            continue;
        }
        let ok = true;
        const names = Object.keys(extra);
        for (let n = 0; n < names.length; n += 1) {
            if (item[names[n]] !== extra[names[n]]) {
                ok = false;
            }
        }
        if (ok) {
            matches.push(item);
        }
    }
    return matches;
}

function row(cells) {
    console.log("| " + cells.join(" | ") + " |");
}

const record = JSON.parse(readFileSync(process.argv[2], "utf8"));
const results = record.results;
const configs = [];
for (let k = 0; k < results.length; k += 1) {
    if (configs.indexOf(results[k].config) < 0) {
        configs.push(results[k].config);
    }
}

console.log("Sleeping is off unless a row says otherwise. Worst case over seeds.\n");
row(["Scene and measure"].concat(configs));
row(["---"].concat(configs.map(alignRight)));

function alignRight() {
    return "---:";
}

function line(label, pick) {
    const cells = [label];
    for (let c = 0; c < configs.length; c += 1) {
        cells.push(pick(configs[c]));
    }
    row(cells);
}

const kinds = ["round", "flat", "jagged"];
for (let k = 0; k < kinds.length; k += 1) {
    line("Rest, " + kinds[k] + " rock: drift after 3 s (mm)", restDrift.bind(null, kinds[k]));
    line("Rest, " + kinds[k] + " rock: overlap at rest (mm)", restPenetration.bind(null, kinds[k]));
}

function restDrift(kind, config) {
    const found = find(results, "rest", config, { kind: kind });
    return found.length ? number(found[0].driftMm, 3) : "-";
}

function restPenetration(kind, config) {
    const found = find(results, "rest", config, { kind: kind });
    return found.length ? number(found[0].penetrationMm, 3) : "-";
}

const towers = [10, 20, 40];
for (let k = 0; k < towers.length; k += 1) {
    line("Column of " + towers[k] + " boxes", towerVerdict.bind(null, towers[k]));
    line("Column of " + towers[k] + ": top height error (mm)", towerError.bind(null, towers[k]));
}

function towerVerdict(count, config) {
    const found = find(results, "box_tower", config, { count: count });
    if (!found.length) {
        return "-";
    }
    return found[0].result.standing ? "stands" : "falls";
}

function towerError(count, config) {
    const found = find(results, "box_tower", config, { count: count });
    if (!found.length || !found[0].result.standing) {
        return "-";
    }
    return number(found[0].result.topHeightErrorMm, 3);
}

line("15 flat stones: stacks built of 3 seeds", stackBuilt.bind(null, false));
line("15 flat stones: worst drift over 60 s (mm)", stackDrift.bind(null, false));
line("15 flat stones, sleeping on: stacks built", stackBuilt.bind(null, true));
line("15 flat stones, sleeping on: asleep after (s)", stackAsleep);

function stackBuilt(sleeping, config) {
    const found = find(results, "rock_stack", config, { sleeping: sleeping });
    let built = 0;
    let fewest = 99;
    for (let k = 0; k < found.length; k += 1) {
        if (found[k].result.standing) {
            built += 1;
        }
        fewest = Math.min(fewest, found[k].result.placed);
    }
    return built + " of " + found.length + " (fewest placed " + fewest + ")";
}

function stackDrift(sleeping, config) {
    const found = find(results, "rock_stack", config, { sleeping: sleeping });
    let worst = -1;
    for (let k = 0; k < found.length; k += 1) {
        if (found[k].result.standing) {
            worst = Math.max(worst, found[k].result.driftMm);
        }
    }
    return worst < 0 ? "-" : number(worst, 3);
}

function stackAsleep(config) {
    const found = find(results, "rock_stack", config, { sleeping: true });
    let worst = -1;
    for (let k = 0; k < found.length; k += 1) {
        if (found[k].result.standing) {
            worst = Math.max(worst, found[k].result.asleepAt);
        }
    }
    return worst < 0 ? "-" : number(worst, 2);
}

const ratios = [500, 5000];
for (let k = 0; k < ratios.length; k += 1) {
    line("Slab on three pebbles, " + ratios[k] + ":1: slab sinks (mm)", slabSink.bind(null, ratios[k]));
    line("Pebble on slab, " + ratios[k] + ":1: overlap (mm)", pebbleOverlap.bind(null, ratios[k]));
}

function slabSink(ratio, config) {
    const found = find(results, "slab_on_pebbles", config, { ratio: ratio });
    return found.length ? number(-found[0].result.slabHeightErrorMm, 3) : "-";
}

function pebbleOverlap(ratio, config) {
    const found = find(results, "pebble_on_slab", config, { ratio: ratio });
    return found.length ? number(found[0].result.penetrationMm, 3) : "-";
}

const fractions = [0.05, 0.02, 0.01, -0.01, -0.02, -0.05];
for (let k = 0; k < fractions.length; k += 1) {
    const label = fractions[k] > 0
        ? "Overhang, centre of mass " + (fractions[k] * 100) + "% inside: rotation (deg)"
        : "Overhang, " + (-fractions[k] * 100) + "% outside: tips after (s)";
    line(label, overhang.bind(null, fractions[k]));
}

function overhang(fraction, config) {
    const found = find(results, "overhang", config, { insideFraction: fraction });
    if (!found.length) {
        return "-";
    }
    if (fraction > 0) {
        return found[0].result.tippedAt > 0 ? "tips" : number(found[0].result.rotationDeg, 4);
    }
    return found[0].result.tippedAt > 0 ? number(found[0].result.tippedAt, 2) : "stays";
}

line("Incline below friction limit: creep (mm)", incline.bind(null, false, -0.05));
line("Incline above friction limit: slide in 3 s (mm)", incline.bind(null, false, 0.05));
line("Box on slab, below limit: creep (mm)", incline.bind(null, true, -0.05));

function incline(onSlab, offset, config) {
    const found = find(results, "incline", config, { onSlab: onSlab, offset: offset });
    return found.length ? number(found[0].result.travelMm, 3) : "-";
}

line("Drop from 0.3 m: worst bounce (mm)", dropWorst.bind(null, "bounceMm"));
line("Drop from 0.3 m: deepest overlap (mm)", dropWorst.bind(null, "deepestMm"));
line("Drop from 0.3 m: at rest after (s)", dropWorst.bind(null, "restAt"));

function dropWorst(field, config) {
    const found = find(results, "drop", config, {});
    let worst = -Infinity;
    for (let k = 0; k < found.length; k += 1) {
        worst = Math.max(worst, found[k].result[field]);
    }
    return found.length ? number(worst, 3) : "-";
}

line("Heap of 30: at rest after last drop (s)", heap.bind(null, "restAt"));
line("Heap of 30: final overlap (mm)", heap.bind(null, "finalPenetrationMm"));
line("Heap of 30: deepest overlap while pouring (mm)", heap.bind(null, "worstPenetrationMm"));
line("Heap of 30: mean step (ms, JavaScript)", heapTiming.bind(null, "meanMs"));
line("Heap of 30: worst step (ms, JavaScript)", heapTiming.bind(null, "worstMs"));

function heap(field, config) {
    const found = find(results, "heap", config, {});
    if (!found.length) {
        return "-";
    }
    const value = found[0].result[field];
    if (field === "restAt" && value < 0) {
        return "not within 12 s";
    }
    return number(value, 3);
}

function heapTiming(field, config) {
    const found = find(results, "heap", config, {});
    return found.length ? number(found[0].result.timing[field], 2) : "-";
}
