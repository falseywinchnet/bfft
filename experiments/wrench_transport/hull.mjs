// Shape cooking: convex hull, coplanar-face merging, exact polyhedral mass
// properties, and the adjacency the collision stage needs.

import { v3, add, sub, scale, dot, cross, length, normalized } from "./math3.mjs";

const WELD_DISTANCE = 1.0e-4;          // metres
const HULL_PLANE_EPSILON = 1.0e-9;     // metres; closer points count as on the hull
const MERGE_COSINE = 0.9999619230641713;  // cos(0.5 degrees)
const MIN_PRINCIPAL_MOMENT = 1.0e-9;

function weldPoints(points) {
    const kept = [];
    for (let i = 0; i < points.length; i += 1) {
        let duplicate = false;
        for (let k = 0; k < kept.length; k += 1) {
            if (length(sub(points[i], kept[k])) < WELD_DISTANCE) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            kept.push(v3(points[i].x, points[i].y, points[i].z));
        }
    }
    return kept;
}

function makeTriangle(points, a, b, c) {
    const normal = normalized(cross(sub(points[b], points[a]), sub(points[c], points[a])));
    return { a: a, b: b, c: c, normal: normal, offset: dot(normal, points[a]), alive: true };
}

function initialTetrahedron(points) {
    let lowest = 0;
    let highest = 0;
    for (let i = 1; i < points.length; i += 1) {
        if (points[i].x < points[lowest].x) {
            lowest = i;
        }
        if (points[i].x > points[highest].x) {
            highest = i;
        }
    }
    if (lowest === highest) {
        return null;
    }
    const axis = normalized(sub(points[highest], points[lowest]));
    let third = -1;
    let thirdDistance = 0;
    for (let i = 0; i < points.length; i += 1) {
        const rel = sub(points[i], points[lowest]);
        const off = sub(rel, scale(axis, dot(rel, axis)));
        const distance = length(off);
        if (distance > thirdDistance) {
            thirdDistance = distance;
            third = i;
        }
    }
    if (third < 0 || thirdDistance < WELD_DISTANCE) {
        return null;
    }
    const planeNormal = normalized(
        cross(sub(points[highest], points[lowest]), sub(points[third], points[lowest])));
    let fourth = -1;
    let fourthDistance = 0;
    for (let i = 0; i < points.length; i += 1) {
        const distance = Math.abs(dot(planeNormal, sub(points[i], points[lowest])));
        if (distance > fourthDistance) {
            fourthDistance = distance;
            fourth = i;
        }
    }
    if (fourth < 0 || fourthDistance < WELD_DISTANCE) {
        return null;
    }
    return [lowest, highest, third, fourth];
}

// Incremental hull with farthest-point insertion. Returns outward triangles
// over the welded point list.
function hullTriangles(points) {
    const seed = initialTetrahedron(points);
    if (seed === null) {
        return null;
    }
    const inside = scale(
        add(add(points[seed[0]], points[seed[1]]), add(points[seed[2]], points[seed[3]])), 0.25);
    const triangles = [];
    const seedFaces = [
        [seed[0], seed[1], seed[2]],
        [seed[0], seed[1], seed[3]],
        [seed[0], seed[2], seed[3]],
        [seed[1], seed[2], seed[3]],
    ];
    for (let f = 0; f < 4; f += 1) {
        let triangle = makeTriangle(points, seedFaces[f][0], seedFaces[f][1], seedFaces[f][2]);
        if (dot(triangle.normal, inside) - triangle.offset > 0) {
            triangle = makeTriangle(points, seedFaces[f][0], seedFaces[f][2], seedFaces[f][1]);
        }
        triangles.push(triangle);
    }
    const used = new Array(points.length).fill(false);
    for (let k = 0; k < 4; k += 1) {
        used[seed[k]] = true;
    }
    for (;;) {
        // The unused point farthest outside any live triangle.
        let best = -1;
        let bestDistance = HULL_PLANE_EPSILON;
        for (let i = 0; i < points.length; i += 1) {
            if (used[i]) {
                continue;
            }
            let outside = 0;
            for (let t = 0; t < triangles.length; t += 1) {
                if (!triangles[t].alive) {
                    continue;
                }
                const distance = dot(triangles[t].normal, points[i]) - triangles[t].offset;
                if (distance > outside) {
                    outside = distance;
                }
            }
            if (outside > bestDistance) {
                bestDistance = outside;
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        used[best] = true;
        const apex = points[best];
        const visibleEdges = new Map();
        const visible = [];
        for (let t = 0; t < triangles.length; t += 1) {
            const triangle = triangles[t];
            if (!triangle.alive) {
                continue;
            }
            if (dot(triangle.normal, apex) - triangle.offset > HULL_PLANE_EPSILON) {
                visible.push(t);
                visibleEdges.set(triangle.a * points.length + triangle.b, true);
                visibleEdges.set(triangle.b * points.length + triangle.c, true);
                visibleEdges.set(triangle.c * points.length + triangle.a, true);
            }
        }
        const horizon = [];
        for (let k = 0; k < visible.length; k += 1) {
            const triangle = triangles[visible[k]];
            const corners = [triangle.a, triangle.b, triangle.c];
            for (let e = 0; e < 3; e += 1) {
                const from = corners[e];
                const to = corners[(e + 1) % 3];
                if (!visibleEdges.has(to * points.length + from)) {
                    horizon.push([from, to]);
                }
            }
            triangle.alive = false;
        }
        for (let k = 0; k < horizon.length; k += 1) {
            triangles.push(makeTriangle(points, horizon[k][0], horizon[k][1], best));
        }
    }
    const alive = [];
    for (let t = 0; t < triangles.length; t += 1) {
        if (triangles[t].alive) {
            alive.push(triangles[t]);
        }
    }
    return alive;
}

function findRoot(parent, index) {
    let root = index;
    while (parent[root] !== root) {
        root = parent[root];
    }
    while (parent[index] !== root) {
        const next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

// Groups adjacent triangles with matching normals and extracts each group's
// boundary loop, counter-clockwise seen from outside.
function mergeCoplanar(points, triangles) {
    const pointCount = points.length;
    const edgeOwner = new Map();
    for (let t = 0; t < triangles.length; t += 1) {
        const triangle = triangles[t];
        edgeOwner.set(triangle.a * pointCount + triangle.b, t);
        edgeOwner.set(triangle.b * pointCount + triangle.c, t);
        edgeOwner.set(triangle.c * pointCount + triangle.a, t);
    }
    const parent = [];
    for (let t = 0; t < triangles.length; t += 1) {
        parent.push(t);
    }
    for (let t = 0; t < triangles.length; t += 1) {
        const triangle = triangles[t];
        const corners = [triangle.a, triangle.b, triangle.c];
        for (let e = 0; e < 3; e += 1) {
            const twinKey = corners[(e + 1) % 3] * pointCount + corners[e];
            if (!edgeOwner.has(twinKey)) {
                continue;
            }
            const other = edgeOwner.get(twinKey);
            if (dot(triangle.normal, triangles[other].normal) > MERGE_COSINE) {
                const rootA = findRoot(parent, t);
                const rootB = findRoot(parent, other);
                if (rootA !== rootB) {
                    parent[Math.max(rootA, rootB)] = Math.min(rootA, rootB);
                }
            }
        }
    }
    const faces = [];
    const loops = [];
    for (let root = 0; root < triangles.length; root += 1) {
        if (findRoot(parent, root) !== root) {
            continue;
        }
        const next = new Map();
        let start = -1;
        for (let t = 0; t < triangles.length; t += 1) {
            if (findRoot(parent, t) !== root) {
                continue;
            }
            const triangle = triangles[t];
            const corners = [triangle.a, triangle.b, triangle.c];
            for (let e = 0; e < 3; e += 1) {
                const from = corners[e];
                const to = corners[(e + 1) % 3];
                const twinKey = to * pointCount + from;
                let interior = false;
                if (edgeOwner.has(twinKey)) {
                    interior = findRoot(parent, edgeOwner.get(twinKey)) === root;
                }
                if (!interior) {
                    next.set(from, to);
                    if (start < 0 || from < start) {
                        start = from;
                    }
                }
            }
        }
        const loop = [];
        let cursor = start;
        for (let guard = 0; guard <= next.size; guard += 1) {
            loop.push(cursor);
            cursor = next.get(cursor);
            if (cursor === start) {
                break;
            }
        }
        loops.push(loop);
    }
    // A vertex shared by only two merged faces lies inside a straight edge
    // between them; removing it from both keeps the edge lists consistent.
    const incidence = new Array(pointCount).fill(0);
    for (let f = 0; f < loops.length; f += 1) {
        for (let k = 0; k < loops[f].length; k += 1) {
            incidence[loops[f][k]] += 1;
        }
    }
    for (let f = 0; f < loops.length; f += 1) {
        const corners = [];
        for (let k = 0; k < loops[f].length; k += 1) {
            if (incidence[loops[f][k]] > 2) {
                corners.push(loops[f][k]);
            }
        }
        if (corners.length < 3) {
            continue;
        }
        // Newell normal of the loop; robust for slightly non-planar groups.
        let normal = v3(0, 0, 0);
        for (let k = 0; k < corners.length; k += 1) {
            const p = points[corners[k]];
            const q = points[corners[(k + 1) % corners.length]];
            normal = add(normal, cross(p, q));
        }
        normal = normalized(normal);
        let offset = 0;
        for (let k = 0; k < corners.length; k += 1) {
            offset += dot(normal, points[corners[k]]);
        }
        offset /= corners.length;
        faces.push({ loop: corners, normal: normal, offset: offset });
    }
    return faces;
}

// Convex hull of a point cloud as polygonal faces over a compact vertex list.
export function buildHull(inputPoints) {
    const points = weldPoints(inputPoints);
    if (points.length < 4) {
        throw new Error("hull needs at least four distinct points");
    }
    const triangles = hullTriangles(points);
    if (triangles === null) {
        throw new Error("hull points are degenerate");
    }
    const rawFaces = mergeCoplanar(points, triangles);
    const remap = new Array(points.length).fill(-1);
    const vertices = [];
    const faces = [];
    for (let f = 0; f < rawFaces.length; f += 1) {
        const loop = [];
        for (let k = 0; k < rawFaces[f].loop.length; k += 1) {
            const original = rawFaces[f].loop[k];
            if (remap[original] < 0) {
                remap[original] = vertices.length;
                vertices.push(points[original]);
            }
            loop.push(remap[original]);
        }
        faces.push({ loop: loop, normal: rawFaces[f].normal, offset: rawFaces[f].offset });
    }
    const hull = { vertices: vertices, faces: faces, edges: [], vertexFaces: [] };
    finishAdjacency(hull);
    return hull;
}

function finishAdjacency(hull) {
    const count = hull.vertices.length;
    const edgeIndex = new Map();
    hull.edges = [];
    hull.vertexFaces = [];
    for (let v = 0; v < count; v += 1) {
        hull.vertexFaces.push([]);
    }
    for (let f = 0; f < hull.faces.length; f += 1) {
        const loop = hull.faces[f].loop;
        for (let k = 0; k < loop.length; k += 1) {
            const from = loop[k];
            const to = loop[(k + 1) % loop.length];
            hull.vertexFaces[from].push(f);
            const low = Math.min(from, to);
            const high = Math.max(from, to);
            const key = low * count + high;
            if (edgeIndex.has(key)) {
                hull.edges[edgeIndex.get(key)].faceB = f;
            } else {
                edgeIndex.set(key, hull.edges.length);
                hull.edges.push({ v0: from, v1: to, faceA: f, faceB: -1 });
            }
        }
    }
    for (let e = 0; e < hull.edges.length; e += 1) {
        if (hull.edges[e].faceB < 0) {
            throw new Error("hull is not closed");
        }
    }
}

// Volume, first and second moments about the origin, by signed tetrahedra
// (origin, a, b, c) over each face fan.
function accumulateMoments(hull, totals) {
    for (let f = 0; f < hull.faces.length; f += 1) {
        const loop = hull.faces[f].loop;
        const a = hull.vertices[loop[0]];
        for (let k = 1; k + 1 < loop.length; k += 1) {
            const b = hull.vertices[loop[k]];
            const c = hull.vertices[loop[k + 1]];
            const det = dot(a, cross(b, c));
            totals.volume += det / 6;
            const sum = add(add(a, b), c);
            totals.first = add(totals.first, scale(sum, det / 24));
            const corners = [a, b, c, sum];
            const weight = det / 120;
            for (let n = 0; n < 4; n += 1) {
                const p = corners[n];
                totals.second[0] += weight * p.x * p.x;
                totals.second[1] += weight * p.x * p.y;
                totals.second[2] += weight * p.x * p.z;
                totals.second[4] += weight * p.y * p.y;
                totals.second[5] += weight * p.y * p.z;
                totals.second[8] += weight * p.z * p.z;
            }
        }
    }
}

// Cooks a compound of convex hulls into a shareable shape. desc.hulls is an
// array of point arrays in one shared frame; the result is recentred so the
// centre of mass is the body-frame origin.
export function cook(desc) {
    const hulls = [];
    for (let h = 0; h < desc.hulls.length; h += 1) {
        hulls.push(buildHull(desc.hulls[h]));
    }
    const totals = { volume: 0, first: v3(0, 0, 0), second: [0, 0, 0, 0, 0, 0, 0, 0, 0] };
    for (let h = 0; h < hulls.length; h += 1) {
        accumulateMoments(hulls[h], totals);
    }
    const density = desc.density === undefined ? 2600 : desc.density;
    const com = scale(totals.first, 1 / totals.volume);
    const mass = density * totals.volume;
    // Second moments about the centre of mass, then I = tr(C) 1 - C.
    const cxx = totals.second[0] - totals.volume * com.x * com.x;
    const cxy = totals.second[1] - totals.volume * com.x * com.y;
    const cxz = totals.second[2] - totals.volume * com.x * com.z;
    const cyy = totals.second[4] - totals.volume * com.y * com.y;
    const cyz = totals.second[5] - totals.volume * com.y * com.z;
    const czz = totals.second[8] - totals.volume * com.z * com.z;
    const inertia = [
        density * (cyy + czz) + 0, -density * cxy, -density * cxz,
        -density * cxy, density * (cxx + czz), -density * cyz,
        -density * cxz, -density * cyz, density * (cxx + cyy),
    ];
    inertia[0] = Math.max(inertia[0], MIN_PRINCIPAL_MOMENT);
    inertia[4] = Math.max(inertia[4], MIN_PRINCIPAL_MOMENT);
    inertia[8] = Math.max(inertia[8], MIN_PRINCIPAL_MOMENT);
    let radius = 0;
    for (let h = 0; h < hulls.length; h += 1) {
        const hull = hulls[h];
        for (let v = 0; v < hull.vertices.length; v += 1) {
            hull.vertices[v] = sub(hull.vertices[v], com);
            radius = Math.max(radius, length(hull.vertices[v]));
        }
        for (let f = 0; f < hull.faces.length; f += 1) {
            hull.faces[f].offset -= dot(hull.faces[f].normal, com);
        }
        let centroid = v3(0, 0, 0);
        for (let v = 0; v < hull.vertices.length; v += 1) {
            centroid = add(centroid, hull.vertices[v]);
        }
        hull.centroid = scale(centroid, 1 / hull.vertices.length);
    }
    return {
        hulls: hulls,
        comOffset: com,
        mass: mass,
        volume: totals.volume,
        radius: radius,
        inertia: inertia,
        friction: desc.friction === undefined ? 0.75 : desc.friction,
        restitution: desc.restitution === undefined ? 0.05 : desc.restitution,
        rollingResistance: desc.rollingResistance === undefined ? 0.002 : desc.rollingResistance,
    };
}
