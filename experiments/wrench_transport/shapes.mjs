// Deterministic test shapes from the Zen Construction specification, section 11.

import { v3 } from "./math3.mjs";
import { cook } from "./hull.mjs";

// Small deterministic generator (mulberry32); state is the caller's record.
export function makeRandom(seed) {
    return { state: seed >>> 0 };
}

export function randomUnit(random) {
    random.state = (random.state + 0x6d2b79f5) >>> 0;
    let t = random.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function randomRange(random, low, high) {
    return low + (high - low) * randomUnit(random);
}

export function boxPoints(sx, sy, sz) {
    const points = [];
    for (let i = 0; i < 8; i += 1) {
        points.push(v3(
            (i & 1 ? 0.5 : -0.5) * sx,
            (i & 2 ? 0.5 : -0.5) * sy,
            (i & 4 ? 0.5 : -0.5) * sz));
    }
    return points;
}

export function boxShape(sx, sy, sz, options) {
    const desc = { hulls: [boxPoints(sx, sy, sz)] };
    copyOptions(desc, options);
    return cook(desc);
}

function copyOptions(desc, options) {
    if (options === undefined) {
        return;
    }
    if (options.density !== undefined) {
        desc.density = options.density;
    }
    if (options.friction !== undefined) {
        desc.friction = options.friction;
    }
    if (options.restitution !== undefined) {
        desc.restitution = options.restitution;
    }
    if (options.rollingResistance !== undefined) {
        desc.rollingResistance = options.rollingResistance;
    }
}

// 32 points on a deformed ellipsoid. kind: "round", "flat" or "jagged".
export function rockPoints(seed, kind) {
    const random = makeRandom(seed * 7919 + 17);
    let ax = 0;
    let ay = 0;
    let az = 0;
    let noise = 0.15;
    if (kind === "round") {
        ax = randomRange(random, 0.06, 0.09);
        ay = randomRange(random, 0.06, 0.09);
        az = randomRange(random, 0.06, 0.09);
    } else if (kind === "flat") {
        ax = randomRange(random, 0.10, 0.20);
        ay = randomRange(random, 0.08, 0.15);
        az = randomRange(random, 0.02, 0.04);
    } else {
        ax = randomRange(random, 0.03, 0.06);
        ay = randomRange(random, 0.03, 0.06);
        az = randomRange(random, 0.03, 0.06);
        noise = 0.3;
    }
    // Directions on a golden-angle lattice, so the 32 points cover the
    // ellipsoid evenly; the seed turns the lattice and sets each point's noise.
    const points = [];
    const turn = randomRange(random, 0, 2 * Math.PI);
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < 32; i += 1) {
        const zUnit = 1 - (2 * i + 1) / 32;
        const angle = turn + golden * i;
        const ring = Math.sqrt(Math.max(0, 1 - zUnit * zUnit));
        const stretch = 1 + noise * randomRange(random, -1, 1);
        // Axes are full extents; the ellipsoid semi-axes are half of them.
        let z = 0.5 * az * zUnit * stretch;
        if (kind === "flat") {
            // A flat stone has a bedding plane: its top and bottom are cut
            // level at 60% of the half thickness, like a flagstone.
            const cut = 0.3 * az;
            z = Math.min(cut, Math.max(-cut, z));
        }
        points.push(v3(
            0.5 * ax * ring * Math.cos(angle) * stretch,
            0.5 * ay * ring * Math.sin(angle) * stretch,
            z));
    }
    return points;
}

export function rockShape(seed, kind, options) {
    const desc = { hulls: [rockPoints(seed, kind)] };
    copyOptions(desc, options);
    return cook(desc);
}

export function concaveRockShape(seed, options) {
    const random = makeRandom(seed * 104729 + 5);
    const hulls = [];
    for (let part = 0; part < 3; part += 1) {
        const points = rockPoints(seed * 3 + part, "round");
        const shift = v3(
            randomRange(random, -0.03, 0.03),
            randomRange(random, -0.03, 0.03),
            randomRange(random, -0.015, 0.015));
        for (let i = 0; i < points.length; i += 1) {
            points[i] = v3(points[i].x + shift.x, points[i].y + shift.y, points[i].z + shift.z);
        }
        hulls.push(points);
    }
    const desc = { hulls: hulls };
    copyOptions(desc, options);
    return cook(desc);
}

// Primitive shapes for general scenes. Curved shapes are convex hulls of
// evenly spread surface points, which is also how the external engines used
// for comparison receive them.

export function spherePoints(radius, count) {
    const points = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < count; i += 1) {
        const zUnit = 1 - (2 * i + 1) / count;
        const ring = Math.sqrt(Math.max(0, 1 - zUnit * zUnit));
        points.push(v3(radius * ring * Math.cos(golden * i), radius * ring * Math.sin(golden * i), radius * zUnit));
    }
    return points;
}

export function cylinderPoints(radius, height, segments) {
    const points = [];
    for (let i = 0; i < segments; i += 1) {
        const angle = 2 * Math.PI * i / segments;
        points.push(v3(radius * Math.cos(angle), radius * Math.sin(angle), 0.5 * height));
        points.push(v3(radius * Math.cos(angle), radius * Math.sin(angle), -0.5 * height));
    }
    return points;
}

export function conePoints(radius, height, segments) {
    const points = [v3(0, 0, 0.75 * height)];
    for (let i = 0; i < segments; i += 1) {
        const angle = 2 * Math.PI * i / segments;
        points.push(v3(radius * Math.cos(angle), radius * Math.sin(angle), -0.25 * height));
    }
    return points;
}

// A right prism over a random convex polygon with `sides` corners.
export function prismPoints(seed, radius, height, sides) {
    const random = makeRandom(seed * 2654435761 + 11);
    const points = [];
    for (let i = 0; i < sides; i += 1) {
        const angle = 2 * Math.PI * (i + randomRange(random, -0.3, 0.3)) / sides;
        const reach = radius * randomRange(random, 0.75, 1);
        points.push(v3(reach * Math.cos(angle), reach * Math.sin(angle), 0.5 * height));
        points.push(v3(reach * Math.cos(angle), reach * Math.sin(angle), -0.5 * height));
    }
    return points;
}

// The hull of `count` random points in a ball: an arbitrary convex polyhedron.
export function polyhedronPoints(seed, radius, count) {
    const random = makeRandom(seed * 40503 + 3);
    const points = [];
    while (points.length < count) {
        const p = v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1));
        const size = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (size > 1 || size < 0.5) {
            continue;
        }
        points.push(v3(radius * p.x, radius * p.y, radius * p.z));
    }
    return points;
}

export function hullShape(points, options) {
    const desc = { hulls: [points] };
    copyOptions(desc, options);
    return cook(desc);
}
