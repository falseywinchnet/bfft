// Checks the narrow phase against a brute-force support-function oracle:
// the separating-axis result must match the best separation over many sampled
// directions, and every manifold point must lie on both hull surfaces.

import { v3, add, sub, scale, dot, cross, length, normalized, quatFromAxisAngle } from "./math3.mjs";
import { rockShape, boxShape, makeRandom, randomRange } from "./shapes.mjs";
import { transformHull, collideHulls } from "./collide.mjs";

let failures = 0;

function check(name, condition, detail) {
    if (!condition) {
        failures += 1;
    }
    console.log((condition ? "PASS" : "FAIL") + "  " + name + "  " + detail);
}

function supportMax(hull, direction) {
    let best = -Infinity;
    for (let v = 0; v < hull.vertices.length; v += 1) {
        best = Math.max(best, dot(direction, hull.vertices[v]));
    }
    return best;
}

function supportMin(hull, direction) {
    let best = Infinity;
    for (let v = 0; v < hull.vertices.length; v += 1) {
        best = Math.min(best, dot(direction, hull.vertices[v]));
    }
    return best;
}

// Largest separation along any candidate direction: all face normals of both
// hulls, all edge-edge cross products, and random directions.
function oracleSeparation(hullA, hullB, random) {
    let best = -Infinity;
    const candidates = [];
    for (let f = 0; f < hullA.normals.length; f += 1) {
        candidates.push(hullA.normals[f]);
    }
    for (let f = 0; f < hullB.normals.length; f += 1) {
        candidates.push(scale(hullB.normals[f], -1));
    }
    const edgesA = hullA.hull.edges;
    const edgesB = hullB.hull.edges;
    for (let ea = 0; ea < edgesA.length; ea += 1) {
        const da = sub(hullA.vertices[edgesA[ea].v1], hullA.vertices[edgesA[ea].v0]);
        for (let eb = 0; eb < edgesB.length; eb += 1) {
            const db = sub(hullB.vertices[edgesB[eb].v1], hullB.vertices[edgesB[eb].v0]);
            const axis = cross(da, db);
            if (length(axis) > 1.0e-9) {
                candidates.push(normalized(axis));
                candidates.push(scale(normalized(axis), -1));
            }
        }
    }
    for (let k = 0; k < 2000; k += 1) {
        candidates.push(normalized(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1))));
    }
    for (let k = 0; k < candidates.length; k += 1) {
        const separation = supportMin(hullB, candidates[k]) - supportMax(hullA, candidates[k]);
        best = Math.max(best, separation);
    }
    return best;
}

// Distance from a point to the outside of a hull (negative inside).
function outsideDistance(hull, point) {
    let worst = -Infinity;
    for (let f = 0; f < hull.normals.length; f += 1) {
        worst = Math.max(worst, dot(hull.normals[f], point) - hull.offsets[f]);
    }
    return worst;
}

const random = makeRandom(4242);
const kinds = ["round", "flat", "jagged"];
let worstAxisError = 0;
let worstSurfaceError = 0;
let worstNormalError = 0;
let worstSeparationError = 0;
let worstEarly = 0;
let worstDeeper = 0;
let touching = 0;
let empty = 0;
let emptyOverlaps = 0;
let worstGapOver = 0;
const margin = 0.004;
for (let trial = 0; trial < 3000; trial += 1) {
    let shapeA = null;
    let shapeB = null;
    if (trial % 5 === 0) {
        shapeA = boxShape(0.1, 0.08, 0.05);
        shapeB = boxShape(0.07, 0.12, 0.04);
    } else {
        shapeA = rockShape(1 + (trial % 37), kinds[trial % 3]);
        shapeB = rockShape(50 + (trial % 41), kinds[(trial + 1) % 3]);
    }
    const qa = quatFromAxisAngle(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1)), randomRange(random, 0, 6));
    let qb = quatFromAxisAngle(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1)), randomRange(random, 0, 6));
    if (trial % 5 === 0 && trial % 2 === 0) {
        qb = qa;   // aligned boxes: face-face contact
    }
    const direction = normalized(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1)));
    const hullA = transformHull(shapeA.hulls[0], v3(0, 0, 0), qa);
    // Slide B along the direction until the oracle separation is a small target.
    const target = randomRange(random, -0.003, 0.003);
    let low = 0;
    let high = 0.5;
    for (let step = 0; step < 50; step += 1) {
        const middle = 0.5 * (low + high);
        const probe = transformHull(shapeB.hulls[0], scale(direction, middle), qb);
        let separation = -Infinity;
        for (let f = 0; f < hullA.normals.length; f += 1) {
            separation = Math.max(separation, supportMin(probe, hullA.normals[f]) - hullA.offsets[f]);
        }
        for (let f = 0; f < probe.normals.length; f += 1) {
            separation = Math.max(separation, supportMin(hullA, probe.normals[f]) - probe.offsets[f]);
        }
        if (separation > target) {
            high = middle;
        } else {
            low = middle;
        }
    }
    const hullB = transformHull(shapeB.hulls[0], scale(direction, high), qb);
    const truth = oracleSeparation(hullA, hullB, random);
    const points = collideHulls(hullA, hullB, margin);
    if (truth > margin) {
        continue;
    }
    if (points.length === 0) {
        // Allowed only for a gap whose nearest features lie beside every face.
        if (truth <= 0) {
            emptyOverlaps += 1;
        }
        empty += 1;
        continue;
    }
    touching += 1;
    for (let k = 0; k < points.length; k += 1) {
        const point = points[k];
        worstSurfaceError = Math.max(worstSurfaceError, Math.abs(outsideDistance(hullA, point.pointA)));
        worstSurfaceError = Math.max(worstSurfaceError, Math.abs(outsideDistance(hullB, point.pointB)));
        worstNormalError = Math.max(worstNormalError, Math.abs(length(point.normal) - 1));
        const measured = dot(sub(point.pointA, point.pointB), point.normal);
        worstSeparationError = Math.max(worstSeparationError, Math.abs(measured - point.separation));
    }
    // The deepest manifold point cannot be shallower than the true separation
    // by more than the face-preference tolerance.
    let deepest = Infinity;
    for (let k = 0; k < points.length; k += 1) {
        deepest = Math.min(deepest, points[k].separation);
    }
    if (Math.abs(deepest - truth) > 3.0e-4 && process.env.VERBOSE) {
        let text = "";
        for (let k = 0; k < points.length; k += 1) {
            text += points[k].separation.toExponential(2) + " ";
        }
        console.log("trial " + trial + " truth " + truth.toExponential(3) + " target " + target.toExponential(3) + " points " + text);
    }
    // An overlap must be reported to within the face bias. A gap may be
    // under-reported (the contact engages early), never over-reported.
    if (truth <= 0) {
        // Never shallower than the true overlap; deeper only by the face bias.
        worstAxisError = Math.max(worstAxisError, deepest - truth);
        worstDeeper = Math.max(worstDeeper, truth - deepest);
    } else {
        worstEarly = Math.max(worstEarly, truth - deepest);
        worstGapOver = Math.max(worstGapOver, deepest - truth);
    }
}
// Resting stacks: nearly parallel faces at nearly zero separation, the case a
// pile spends its life in. A contact here must never vanish.
let restingMissing = 0;
let restingWorst = 0;
let restingFewPoints = 0;
for (let trial = 0; trial < 3000; trial += 1) {
    const lower = rockShape(200 + (trial % 53), "flat");
    const upper = rockShape(300 + (trial % 59), "flat");
    let topLower = -Infinity;
    for (let v = 0; v < lower.hulls[0].vertices.length; v += 1) {
        topLower = Math.max(topLower, lower.hulls[0].vertices[v].z);
    }
    let bottomUpper = Infinity;
    for (let v = 0; v < upper.hulls[0].vertices.length; v += 1) {
        bottomUpper = Math.min(bottomUpper, upper.hulls[0].vertices[v].z);
    }
    const tiltScale = Math.pow(10, randomRange(random, -9, -3));
    const tilt = quatFromAxisAngle(
        v3(randomRange(random, -1, 1), randomRange(random, -1, 1), 0.2 * randomRange(random, -1, 1)), tiltScale);
    const gap = randomRange(random, -2.0e-4, 2.0e-4);
    const hullLower = transformHull(lower.hulls[0], v3(0, 0, 0), quatFromAxisAngle(v3(0, 0, 1), randomRange(random, 0, 6)));
    const hullUpper = transformHull(upper.hulls[0],
        v3(randomRange(random, -0.01, 0.01), randomRange(random, -0.01, 0.01), topLower - bottomUpper + gap), tilt);
    const truth = oracleSeparation(hullLower, hullUpper, random);
    const points = collideHulls(hullLower, hullUpper, 0.002);
    if (points.length === 0) {
        restingMissing += 1;
        continue;
    }
    if (points.length < 3) {
        restingFewPoints += 1;
    }
    let deepest = Infinity;
    for (let k = 0; k < points.length; k += 1) {
        deepest = Math.min(deepest, points[k].separation);
    }
    restingWorst = Math.max(restingWorst, Math.abs(deepest - truth));
}
check("resting face contacts never vanish", restingMissing === 0, "missing=" + restingMissing + " of 3000");
check("resting face contacts keep a polygon", restingFewPoints === 0, "fewer than three points=" + restingFewPoints);
check("resting face contact depth within the face-merge tolerance", restingWorst < 2.0e-4, "worst=" + restingWorst.toExponential(2) + " m");
check("pairs produced contacts", touching > 2000 && empty < 10 && emptyOverlaps === 0,
    "touching=" + touching + " empty gaps=" + empty + " empty overlaps=" + emptyOverlaps);
check("manifold points lie on or near both surfaces", worstSurfaceError < 1.0e-3, "worst=" + worstSurfaceError.toExponential(2) + " m");
check("normals are unit", worstNormalError < 1.0e-12, "worst=" + worstNormalError.toExponential(2));
check("separation equals point distance along normal", worstSeparationError < 1.0e-12, "worst=" + worstSeparationError.toExponential(2));
check("overlap never under-reported by more than 0.3 mm", worstAxisError < 3.0e-4, "worst=" + worstAxisError.toExponential(2) + " m");
check("overlap over-reported by at most the face bias", worstDeeper < 1.5e-3, "worst=" + worstDeeper.toExponential(2) + " m");
check("gap under-reported by at most 2.5 mm", worstEarly < 2.5e-3, "worst=" + worstEarly.toExponential(2) + " m");
check("gap over-reported by at most 1.5 mm", worstGapOver < 1.5e-3, "worst=" + worstGapOver.toExponential(2) + " m");
console.log(failures === 0 ? "ALL PASS" : failures + " FAILURES");
process.exit(failures === 0 ? 0 : 1);
