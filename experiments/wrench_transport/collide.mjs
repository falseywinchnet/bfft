// Narrow phase: hull-hull separating-axis test with Gauss-map edge pruning,
// face clipping, hull-ground vertices, and four-point manifold reduction.
//
// A manifold point is {pointA, pointB, normal, separation}. The normal is the
// direction body B pushes body A. pointA and pointB are the world positions
// of the contact on each surface.

import { v3, add, sub, scale, addScaled, dot, cross, length, normalized, quatRotate } from "./math3.mjs";

const EDGE_PARALLEL_TOLERANCE = 1.0e-6;
// An edge pair counts as a separating-axis candidate only if its edge-to-edge
// distance matches the hulls' true separation along that axis, in metres.
const EDGE_SUPPORT_TOLERANCE = 1.0e-7;
// Preference for face axes (and for A's face), in metres. It keeps resting
// face contacts from flickering to single-point edge contacts. A biased face
// manifold is accepted only if it holds the depth its axis measured, so the
// bias cannot hide an overlap.
const FACE_BIAS = 1.0e-3;
// A face manifold's deepest point may be this much shallower than the
// separating-axis depth before the manifold is rejected, in metres.
const WITNESS_TOLERANCE = 2.0e-4;
// How far beside the reference face a lone witness vertex is still accepted.
const WITNESS_REACH = 5.0e-3;

// World-space copy of one cooked hull at a pose.
export function transformHull(hull, position, orientation) {
    const vertices = [];
    for (let v = 0; v < hull.vertices.length; v += 1) {
        vertices.push(add(position, quatRotate(orientation, hull.vertices[v])));
    }
    const normals = [];
    const offsets = [];
    for (let f = 0; f < hull.faces.length; f += 1) {
        const normal = quatRotate(orientation, hull.faces[f].normal);
        normals.push(normal);
        offsets.push(hull.faces[f].offset + dot(normal, position));
    }
    let low = v3(vertices[0].x, vertices[0].y, vertices[0].z);
    let high = v3(vertices[0].x, vertices[0].y, vertices[0].z);
    for (let v = 1; v < vertices.length; v += 1) {
        low = v3(Math.min(low.x, vertices[v].x), Math.min(low.y, vertices[v].y), Math.min(low.z, vertices[v].z));
        high = v3(Math.max(high.x, vertices[v].x), Math.max(high.y, vertices[v].y), Math.max(high.z, vertices[v].z));
    }
    return {
        hull: hull,
        vertices: vertices,
        normals: normals,
        offsets: offsets,
        centroid: add(position, quatRotate(orientation, hull.centroid)),
        low: low,
        high: high,
    };
}

export function boundsOverlap(a, b, margin) {
    if (a.low.x - margin > b.high.x || b.low.x - margin > a.high.x) {
        return false;
    }
    if (a.low.y - margin > b.high.y || b.low.y - margin > a.high.y) {
        return false;
    }
    if (a.low.z - margin > b.high.z || b.low.z - margin > a.high.z) {
        return false;
    }
    return true;
}

// Largest separation of `other` along the face normals of `reference`.
function queryFaces(reference, other) {
    let bestSeparation = -Infinity;
    let bestFace = -1;
    for (let f = 0; f < reference.normals.length; f += 1) {
        const normal = reference.normals[f];
        let lowest = Infinity;
        for (let v = 0; v < other.vertices.length; v += 1) {
            const distance = dot(normal, other.vertices[v]);
            if (distance < lowest) {
                lowest = distance;
            }
        }
        const separation = lowest - reference.offsets[f];
        if (separation > bestSeparation) {
            bestSeparation = separation;
            bestFace = f;
        }
    }
    return { separation: bestSeparation, face: bestFace };
}

// Largest separation over edge pairs that form a face of the Minkowski
// difference (Gregorius, GDC 2013).
function queryEdges(hullA, hullB) {
    let bestSeparation = -Infinity;
    let bestEdgeA = -1;
    let bestEdgeB = -1;
    let bestAxis = v3(0, 0, 0);
    const edgesA = hullA.hull.edges;
    const edgesB = hullB.hull.edges;
    for (let ea = 0; ea < edgesA.length; ea += 1) {
        const a = hullA.normals[edgesA[ea].faceA];
        const b = hullA.normals[edgesA[ea].faceB];
        const bxa = cross(b, a);
        const headA = hullA.vertices[edgesA[ea].v0];
        const directionA = sub(hullA.vertices[edgesA[ea].v1], headA);
        for (let eb = 0; eb < edgesB.length; eb += 1) {
            // c and d are the negated normals of B's two faces; d x c equals
            // the cross of the un-negated normals in the same order.
            const nc = hullB.normals[edgesB[eb].faceA];
            const nd = hullB.normals[edgesB[eb].faceB];
            const dxc = cross(nd, nc);
            const cba = -dot(nc, bxa);
            const dba = -dot(nd, bxa);
            const adc = dot(a, dxc);
            const bdc = dot(b, dxc);
            if (!(cba * dba < 0 && adc * bdc < 0 && cba * bdc > 0)) {
                continue;
            }
            const headB = hullB.vertices[edgesB[eb].v0];
            const directionB = sub(hullB.vertices[edgesB[eb].v1], headB);
            let axis = cross(directionA, directionB);
            const axisLength = length(axis);
            if (axisLength < EDGE_PARALLEL_TOLERANCE * length(directionA) * length(directionB)) {
                continue;
            }
            axis = scale(axis, 1 / axisLength);
            if (dot(axis, sub(headA, hullA.centroid)) < 0) {
                axis = scale(axis, -1);
            }
            const separation = dot(axis, sub(headB, headA));
            if (separation > bestSeparation) {
                // For a true Minkowski-difference face the two edges are the
                // supports along the axis, so the edge-to-edge distance is the
                // separation along it. Nearly parallel faces make the arc
                // test admit pairs that are not; measure the axis to be sure.
                let lowestB = Infinity;
                for (let v = 0; v < hullB.vertices.length; v += 1) {
                    lowestB = Math.min(lowestB, dot(axis, hullB.vertices[v]));
                }
                let highestA = -Infinity;
                for (let v = 0; v < hullA.vertices.length; v += 1) {
                    highestA = Math.max(highestA, dot(axis, hullA.vertices[v]));
                }
                if (Math.abs((lowestB - highestA) - separation) > EDGE_SUPPORT_TOLERANCE) {
                    continue;
                }
                bestSeparation = separation;
                bestEdgeA = ea;
                bestEdgeB = eb;
                bestAxis = axis;
            }
        }
    }
    return { separation: bestSeparation, edgeA: bestEdgeA, edgeB: bestEdgeB, axis: bestAxis };
}

// Sutherland-Hodgman: keeps the part of `polygon` with dot(normal, p) <= offset.
function clipPolygon(polygon, normal, offset) {
    const out = [];
    for (let k = 0; k < polygon.length; k += 1) {
        const current = polygon[k];
        const following = polygon[(k + 1) % polygon.length];
        const currentDistance = dot(normal, current) - offset;
        const followingDistance = dot(normal, following) - offset;
        if (currentDistance <= 0) {
            out.push(current);
        }
        if ((currentDistance < 0 && followingDistance > 0) || (currentDistance > 0 && followingDistance < 0)) {
            const t = currentDistance / (currentDistance - followingDistance);
            out.push(addScaled(current, sub(following, current), t));
        }
    }
    return out;
}

// Face contact with `reference` face index on hull `ref`, clipped incident
// face of hull `inc`. `referenceIsA` selects which side of the pair owns it.
// Returns the points, the deepest kept separation, and the witness: the
// incident hull's deepest vertex along the reference normal, its separation,
// and how far its projection lies outside the reference polygon.
function faceManifold(ref, inc, face, margin, referenceIsA) {
    const normal = ref.normals[face];
    let support = 0;
    let supportDistance = Infinity;
    for (let v = 0; v < inc.vertices.length; v += 1) {
        const distance = dot(normal, inc.vertices[v]);
        if (distance < supportDistance) {
            supportDistance = distance;
            support = v;
        }
    }
    // The most anti-parallel incident face among those sharing the witness.
    const candidates = inc.hull.vertexFaces[support];
    let incident = candidates[0];
    let lowestDot = Infinity;
    for (let k = 0; k < candidates.length; k += 1) {
        const alignment = dot(inc.normals[candidates[k]], normal);
        if (alignment < lowestDot) {
            lowestDot = alignment;
            incident = candidates[k];
        }
    }
    let polygon = [];
    const incidentLoop = inc.hull.faces[incident].loop;
    for (let k = 0; k < incidentLoop.length; k += 1) {
        polygon.push(inc.vertices[incidentLoop[k]]);
    }
    const loop = ref.hull.faces[face].loop;
    let witnessOutside = -Infinity;
    for (let k = 0; k < loop.length; k += 1) {
        const from = ref.vertices[loop[k]];
        const to = ref.vertices[loop[(k + 1) % loop.length]];
        const side = normalized(cross(sub(to, from), normal));
        const sideOffset = dot(side, from);
        witnessOutside = Math.max(witnessOutside, dot(side, inc.vertices[support]) - sideOffset);
        if (polygon.length > 0) {
            polygon = clipPolygon(polygon, side, sideOffset);
        }
    }
    const result = {
        points: [],
        deepest: Infinity,
        witnessSeparation: supportDistance - ref.offsets[face],
        witnessOutside: witnessOutside,
        witnessPoint: null,
    };
    for (let k = 0; k < polygon.length; k += 1) {
        const separation = dot(normal, polygon[k]) - ref.offsets[face];
        if (separation > margin) {
            continue;
        }
        result.deepest = Math.min(result.deepest, separation);
        result.points.push(facePoint(polygon[k], normal, separation, referenceIsA));
    }
    result.witnessPoint = facePoint(inc.vertices[support], normal, result.witnessSeparation, referenceIsA);
    return result;
}

function facePoint(onIncident, normal, separation, referenceIsA) {
    const onReference = addScaled(onIncident, normal, -separation);
    if (referenceIsA) {
        return { pointA: onReference, pointB: onIncident, normal: scale(normal, -1), separation: separation };
    }
    return { pointA: onIncident, pointB: onReference, normal: normal, separation: separation };
}

function edgeManifold(hullA, hullB, query) {
    const edgeA = hullA.hull.edges[query.edgeA];
    const edgeB = hullB.hull.edges[query.edgeB];
    const p1 = hullA.vertices[edgeA.v0];
    const d1 = sub(hullA.vertices[edgeA.v1], p1);
    const p2 = hullB.vertices[edgeB.v0];
    const d2 = sub(hullB.vertices[edgeB.v1], p2);
    const r = sub(p1, p2);
    const a = dot(d1, d1);
    const e = dot(d2, d2);
    const f = dot(d2, r);
    const c = dot(d1, r);
    const b = dot(d1, d2);
    const denominator = a * e - b * b;
    let s = 0;
    if (denominator > 0) {
        s = Math.min(1, Math.max(0, (b * f - c * e) / denominator));
    }
    let t = (b * s + f) / e;
    if (t < 0) {
        t = 0;
        s = Math.min(1, Math.max(0, -c / a));
    } else if (t > 1) {
        t = 1;
        s = Math.min(1, Math.max(0, (b - c) / a));
    }
    const onA = addScaled(p1, d1, s);
    const onB = addScaled(p2, d2, t);
    return [{ pointA: onA, pointB: onB, normal: scale(query.axis, -1), separation: query.separation }];
}

// Deepest point, the point farthest from it, then the points that maximise
// triangle and quadrilateral area. Ties resolve to the lowest index.
export function reduceManifold(points) {
    if (points.length <= 4) {
        return points;
    }
    const taken = new Array(points.length).fill(false);
    let first = 0;
    for (let k = 1; k < points.length; k += 1) {
        if (points[k].separation < points[first].separation) {
            first = k;
        }
    }
    taken[first] = true;
    let second = -1;
    let farthest = -1;
    for (let k = 0; k < points.length; k += 1) {
        if (taken[k]) {
            continue;
        }
        const distance = length(sub(points[k].pointA, points[first].pointA));
        if (distance > farthest) {
            farthest = distance;
            second = k;
        }
    }
    taken[second] = true;
    const normal = points[first].normal;
    const base = sub(points[second].pointA, points[first].pointA);
    let third = -1;
    let largest = -1;
    let thirdSigned = 0;
    for (let k = 0; k < points.length; k += 1) {
        if (taken[k]) {
            continue;
        }
        const signed = dot(cross(base, sub(points[k].pointA, points[first].pointA)), normal);
        if (Math.abs(signed) > largest) {
            largest = Math.abs(signed);
            third = k;
            thirdSigned = signed;
        }
    }
    taken[third] = true;
    // The fourth point adds the most area outside the triangle: the largest
    // signed area against any triangle edge, on the side away from the triangle.
    const corners = [first, second, third];
    const orientation = thirdSigned >= 0 ? 1 : -1;
    let fourth = -1;
    let gain = 0;
    for (let k = 0; k < points.length; k += 1) {
        if (taken[k]) {
            continue;
        }
        for (let e = 0; e < 3; e += 1) {
            const from = points[corners[e]].pointA;
            const to = points[corners[(e + 1) % 3]].pointA;
            const signed = orientation * dot(cross(sub(to, from), sub(points[k].pointA, from)), normal);
            if (-signed > gain) {
                gain = -signed;
                fourth = k;
            }
        }
    }
    const reduced = [points[first], points[second], points[third]];
    if (fourth >= 0) {
        reduced.push(points[fourth]);
    }
    return reduced;
}

// Contact points between two world hulls, or an empty array beyond `margin`.
export function collideHulls(hullA, hullB, margin) {
    const faceA = queryFaces(hullA, hullB);
    if (faceA.separation > margin) {
        return [];
    }
    const faceB = queryFaces(hullB, hullA);
    if (faceB.separation > margin) {
        return [];
    }
    const edge = queryEdges(hullA, hullB);
    if (edge.separation > margin) {
        return [];
    }
    // Bias toward faces, and toward A's face, for temporal coherence.
    let useA = true;
    let faceSeparation = faceA.separation;
    let otherSeparation = faceB.separation;
    if (faceB.separation > faceA.separation + FACE_BIAS + 0.05 * Math.abs(faceA.separation)) {
        useA = false;
        faceSeparation = faceB.separation;
        otherSeparation = faceA.separation;
    }
    const bias = FACE_BIAS + 0.05 * Math.abs(faceSeparation);

    if (edge.edgeA >= 0 && edge.separation > faceSeparation + bias) {
        return edgeManifold(hullA, hullB, edge);
    }
    let manifold = null;
    if (useA) {
        manifold = faceManifold(hullA, hullB, faceA.face, margin, true);
    } else {
        manifold = faceManifold(hullB, hullA, faceB.face, margin, false);
    }
    // A face manifold stands only if it kept the depth the axis test
    // measured. Clipping loses it when the nearest features lie beside the
    // reference face, which the face bias can hide.
    if (manifold.deepest <= manifold.witnessSeparation + WITNESS_TOLERANCE) {
        return manifold.points;
    }
    if (otherSeparation >= faceSeparation - bias) {
        let other = null;
        if (useA) {
            other = faceManifold(hullB, hullA, faceB.face, margin, false);
        } else {
            other = faceManifold(hullA, hullB, faceA.face, margin, true);
        }
        if (other.deepest <= other.witnessSeparation + WITNESS_TOLERANCE) {
            return other.points;
        }
    }
    if (edge.edgeA >= 0 && edge.separation >= faceSeparation - bias) {
        return edgeManifold(hullA, hullB, edge);
    }
    // Otherwise the witness vertex itself carries the measured depth, as long
    // as it sits close beside the reference face.
    if (manifold.witnessOutside <= WITNESS_REACH + Math.abs(manifold.witnessSeparation)) {
        manifold.points.push(manifold.witnessPoint);
    }
    return manifold.points;
}

// Contact points of a world hull against the ground plane z = groundZ.
// The body is side A; the ground is side B with normal (0, 0, 1).
export function collideGround(hull, groundZ, margin) {
    const points = [];
    for (let v = 0; v < hull.vertices.length; v += 1) {
        const vertex = hull.vertices[v];
        const separation = vertex.z - groundZ;
        if (separation < margin) {
            points.push({
                pointA: vertex,
                pointB: v3(vertex.x, vertex.y, groundZ),
                normal: v3(0, 0, 1),
                separation: separation,
            });
        }
    }
    return points;
}
