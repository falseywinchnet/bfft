// Scores container-benchmark results from any engine with one set of
// measures, computed here from each result's samples and final poses.
//
//   node container_report.mjs --seed 1 --out report.json result1.json result2.json ...
//
// Overlap uses exhaustive SAT on exported hulls, independently of each
// engine's contact manifold construction.

import { readFileSync, writeFileSync } from "node:fs";
import { v3, quatIdentity, quatRotate } from "./math3.mjs";
import { boxPoints, hullShape } from "./shapes.mjs";
import { transformHull, collideHulls, boundsOverlap } from "./collide.mjs";
import { buildContainerScene, CONTAINER } from "./container.mjs";

const QUIET_SPEED = 0.004;   // metres per second, the specification's sleep threshold

function scoreResult(scene, wallHulls, result) {
    const count = scene.bodies.length;
    const hulls = [];
    let top = 0;
    let centreSum = 0;
    let deepest = 0;
    let finite = true;
    for (let i = 0; i < count; i += 1) {
        const pose = result.finalPoses[i];
        const position = v3(pose.p[0], pose.p[1], pose.p[2]);
        const orientation = { w: pose.q[0], x: pose.q[1], y: pose.q[2], z: pose.q[3] };
        if (!(Number.isFinite(position.x) && Number.isFinite(position.y) && Number.isFinite(position.z))) {
            finite = false;
        }
        const hull = transformHull(scene.bodies[i].shape.hulls[0], position, orientation);
        hulls.push(hull);
        centreSum += position.z;
        for (let v = 0; v < hull.vertices.length; v += 1) {
            top = Math.max(top, hull.vertices[v].z);
            deepest = Math.max(deepest, -hull.vertices[v].z);
        }
    }
    let escaped = 0;
    const reach = 0.5 * CONTAINER.inner + 0.001;
    for (let i = 0; i < count; i += 1) {
        const p = result.finalPoses[i].p;
        if (Math.abs(p[0]) > reach || Math.abs(p[1]) > reach || p[2] < 0) {
            escaped += 1;
        }
    }
    let overlapSum = 0;
    let overlapPairs = 0;
    for (let i = 0; i < count; i += 1) {
        for (let w = 0; w < wallHulls.length; w += 1) {
            deepest = Math.max(deepest, overlapDepth(hulls[i], wallHulls[w]));
        }
        for (let j = i + 1; j < count; j += 1) {
            const depth = overlapDepth(hulls[i], hulls[j]);
            if (depth > 1.0e-5) {
                overlapSum += depth;
                overlapPairs += 1;
            }
            deepest = Math.max(deepest, depth);
        }
    }
    // Settling: the time after which the fastest body stays below the quiet
    // speed until the end of the run.
    const samples = result.samples;
    let settledAt = -1;
    for (let k = samples.length - 1; k >= 0; k -= 1) {
        if (samples[k].maxSpeed >= QUIET_SPEED) {
            break;
        }
        settledAt = samples[k].t;
    }
    if (settledAt >= samples[samples.length - 1].t) {
        settledAt = -1;
    }
    let lateSpeed = 0;
    let lateEnergy = 0;
    let lateCount = 0;
    const lateStart = samples[samples.length - 1].t - 1;
    for (let k = 0; k < samples.length; k += 1) {
        if (samples[k].t > lateStart) {
            lateSpeed = Math.max(lateSpeed, samples[k].maxSpeed);
            lateEnergy += samples[k].kineticEnergy;
            lateCount += 1;
        }
    }
    const simulated = samples[samples.length - 1].t;
    return {
        engine: result.engine,
        config: result.config,
        finite: finite,
        escaped: escaped,
        settledAt: settledAt,
        lastSecondMaxSpeed: lateSpeed,
        lastSecondMeanEnergy: lateEnergy / Math.max(1, lateCount),
        deepestOverlapMm: deepest * 1000,
        overlappingPairs: overlapPairs,
        meanOverlapMm: overlapPairs > 0 ? 1000 * overlapSum / overlapPairs : 0,
        pileTopMm: top * 1000,
        meanCentreHeightMm: 1000 * centreSum / count,
        supportOverWeight: result.supportForce === null ? null : result.supportForce / result.weight,
        floorOverWeight: result.floorForce / result.weight,
        totalMassKg: result.weight / 9.81,
        wallSeconds: result.wallSeconds,
        realTimeFactor: simulated / result.wallSeconds,
        worstStepMs: result.worstStepMs,
        work: result.work,
    };
}

// Independent exhaustive SAT depth. No manifold-generation or Gauss-map pruning.
function overlapDepth(a, b) {
    if (!boundsOverlap(a, b, 0)) return 0;
    let best = -Infinity;
    function testAxis(x, y, z) {
        const norm = Math.hypot(x, y, z);
        if (norm < 1e-12) return false;
        let amin=Infinity, amax=-Infinity, bmin=Infinity, bmax=-Infinity;
        for (const p of a.vertices) { const d=x*p.x+y*p.y+z*p.z; amin=Math.min(amin,d); amax=Math.max(amax,d); }
        for (const p of b.vertices) { const d=x*p.x+y*p.y+z*p.z; bmin=Math.min(bmin,d); bmax=Math.max(bmax,d); }
        best=Math.max(best, Math.max(bmin-amax,amin-bmax)/norm);
        return best >= 0;
    }
    for(const n of a.normals) if(testAxis(n.x,n.y,n.z)) return 0;
    for(const n of b.normals) if(testAxis(n.x,n.y,n.z)) return 0;
    for(const ea of a.hull.edges) {
        const a0=a.vertices[ea.v0], a1=a.vertices[ea.v1];
        const x=a1.x-a0.x,y=a1.y-a0.y,z=a1.z-a0.z;
        for(const eb of b.hull.edges) {
            const b0=b.vertices[eb.v0], b1=b.vertices[eb.v1];
            const u=b1.x-b0.x,v=b1.y-b0.y,w=b1.z-b0.z;
            if(testAxis(y*w-z*v,z*u-x*w,x*v-y*u)) return 0;
        }
    }
    return Math.max(0,-best);
}

function text(value, digits) {
    if (value === null || value === undefined) {
        return "n/a";
    }
    if (!Number.isFinite(value)) {
        return String(value);
    }
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 1.0e-3 || magnitude >= 1.0e5)) {
        return value.toExponential(1);
    }
    return value.toFixed(digits);
}

let seed = 1;
let outPath = "";
const files = [];
for (let k = 2; k < process.argv.length; k += 1) {
    if (process.argv[k] === "--seed") {
        seed = Number(process.argv[k + 1]);
        k += 1;
    } else if (process.argv[k] === "--out") {
        outPath = process.argv[k + 1];
        k += 1;
    } else {
        files.push(process.argv[k]);
    }
}
const scene = buildContainerScene(seed);
const wallHulls = [];
for (let k = 0; k < scene.walls.length; k += 1) {
    const wall = scene.walls[k];
    const shape = hullShape(boxPoints(wall.size.x, wall.size.y, wall.size.z));
    wallHulls.push(transformHull(shape.hulls[0], wall.centre, quatIdentity()));
}
const scores = [];
for (let k = 0; k < files.length; k += 1) {
    scores.push(scoreResult(scene, wallHulls, JSON.parse(readFileSync(files[k], "utf8"))));
}

const rows = [
    ["Settled (max speed stays below 4 mm/s) at (s)", "settledAt", 2],
    ["Fastest body in the last second (m/s)", "lastSecondMaxSpeed", 4],
    ["Mean kinetic energy in the last second (J)", "lastSecondMeanEnergy", 4],
    ["Deepest overlap at the end (mm)", "deepestOverlapMm", 3],
    ["Overlapping pairs (deeper than 0.01 mm)", "overlappingPairs", 0],
    ["Mean overlap of those pairs (mm)", "meanOverlapMm", 3],
    ["Bodies outside the container", "escaped", 0],
    ["Pile top (mm)", "pileTopMm", 1],
    ["Mean centre height (mm)", "meanCentreHeightMm", 1],
    ["Floor force / weight", "floorOverWeight", 4],
    ["Floor and wall vertical force / weight", "supportOverWeight", 6],
    ["Wall-clock for 8 s simulated (s)", "wallSeconds", 2],
    ["Simulated time / wall-clock", "realTimeFactor", 1],
    ["Worst step (ms)", "worstStepMs", 2],
];
const header = ["Measure"];
const divider = ["---"];
for (let k = 0; k < scores.length; k += 1) {
    header.push(scores[k].engine + "<br>" + scores[k].config);
    divider.push("---:");
}
console.log("| " + header.join(" | ") + " |");
console.log("| " + divider.join(" | ") + " |");
for (let r = 0; r < rows.length; r += 1) {
    const cells = [rows[r][0]];
    for (let k = 0; k < scores.length; k += 1) {
        let value = scores[k][rows[r][1]];
        if (rows[r][1] === "settledAt" && value < 0) {
            cells.push("not within 8 s");
        } else {
            cells.push(text(value, rows[r][2]));
        }
    }
    console.log("| " + cells.join(" | ") + " |");
}
const workCells = ["Work"];
for (let k = 0; k < scores.length; k += 1) {
    const work = scores[k].work;
    const names = Object.keys(work);
    const parts = [];
    for (let n = 0; n < names.length; n += 1) {
        parts.push(names[n] + " " + text(work[names[n]], 0));
    }
    workCells.push(parts.join("; "));
}
console.log("| " + workCells.join(" | ") + " |");
if (outPath !== "") {
    writeFileSync(outPath, JSON.stringify({ seed: seed, scores: scores }, null, 1));
}
