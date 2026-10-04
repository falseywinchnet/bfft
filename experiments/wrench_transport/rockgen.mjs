// Rock generator.
//
// A rock is a radius field over directions: an ellipsoid, cut by a few
// fracture planes, times one plus a sum of noise octaves whose amplitude
// falls with frequency. The octaves are split at a cutoff wavelength set by
// how many hull vertices the physics may spend:
//
//   - the bands below the cutoff are geometry. They are sampled at the vertex
//     budget and become the convex collision hull;
//   - the bands above the cutoff never reach the physics as shape. They are
//     kept as a height map over directions (the deformation map, for the
//     renderer) and summarised as a roughness angle that raises the friction
//     coefficient: mu = tan(basic friction angle + roughness angle), Patton's
//     law for rough rock joints.
//
// The render mesh carries every band, drawn midway between the collision hull
// and the designed surface so that what is seen touching is, to within half
// the hull's facet error plus the texture height, what the solver has touching.

import { v3, add, sub, scale, dot, cross, length, normalized } from "./math3.mjs";
import { buildHull, cook } from "./hull.mjs";

export const defaultRockOptions = {
    size: 0.09,                 // mean diameter, metres
    aspect: [1.25, 1.0, 0.6],   // relative extents of the three axes
    amplitude: 0.12,            // first octave's relative amplitude
    hurst: 0.9,                 // amplitude falls by 2^-hurst per octave
    octaves: 7,
    fractures: 4,               // flat faces cut into the rock
    fractureDepth: 0.22,        // how far inside the surface a cut may sit, as a fraction of the radius
    bedding: 0,                 // 0, or the fraction of the thin axis kept between two parallel bedding planes
    vertexBudget: 48,           // collision hull vertices
    basicFrictionAngle: 31,     // degrees, smooth rock on rock
    roughnessGain: 0.5,         // share of the texture's rms slope angle that acts as roughness angle
    roughnessAngleLimit: 12,    // degrees, cap on what texture may add
    renderSubdivisions: 4,      // icosphere levels: 4 gives 2562 vertices
    mapSize: 32,                // deformation map: six faces of mapSize x mapSize heights
    density: 2600,
    restitution: 0.05,
    rollingResistance: 0.002,
};

function hash3(ix, iy, iz, seed) {
    let h = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(iz, 2147483647) ^ Math.imul(seed, 1274126177);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return h;
}

function lattice(ix, iy, iz, seed, fx, fy, fz) {
    // A gradient from twelve edge directions, dotted with the offset.
    const h = hash3(ix, iy, iz, seed) & 15;
    const u = h < 8 ? fx : fy;
    let w = fz;
    if (h < 4) {
        w = fy;
    } else if (h === 12 || h === 14) {
        w = fx;
    }
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? w : -w);
}

function fade(t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// Gradient noise in about [-1, 1], one octave, lattice period 1.
function noise3(x, y, z, seed) {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const iz = Math.floor(z);
    const fx = x - ix;
    const fy = y - iy;
    const fz = z - iz;
    const u = fade(fx);
    const v = fade(fy);
    const w = fade(fz);
    const n000 = lattice(ix, iy, iz, seed, fx, fy, fz);
    const n100 = lattice(ix + 1, iy, iz, seed, fx - 1, fy, fz);
    const n010 = lattice(ix, iy + 1, iz, seed, fx, fy - 1, fz);
    const n110 = lattice(ix + 1, iy + 1, iz, seed, fx - 1, fy - 1, fz);
    const n001 = lattice(ix, iy, iz + 1, seed, fx, fy, fz - 1);
    const n101 = lattice(ix + 1, iy, iz + 1, seed, fx - 1, fy, fz - 1);
    const n011 = lattice(ix, iy + 1, iz + 1, seed, fx, fy - 1, fz - 1);
    const n111 = lattice(ix + 1, iy + 1, iz + 1, seed, fx - 1, fy - 1, fz - 1);
    const x00 = n000 + u * (n100 - n000);
    const x10 = n010 + u * (n110 - n010);
    const x01 = n001 + u * (n101 - n001);
    const x11 = n011 + u * (n111 - n011);
    const y0 = x00 + v * (x10 - x00);
    const y1 = x01 + v * (x11 - x01);
    return y0 + w * (y1 - y0);
}

function randomUnit(state) {
    state.value = (state.value + 0x6d2b79f5) >>> 0;
    let t = state.value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

// The rock's recipe: everything the radius field needs, derived from the seed.
function makeRecipe(seed, options) {
    const state = { value: (seed * 2654435761) >>> 0 };
    const mean = (options.aspect[0] + options.aspect[1] + options.aspect[2]) / 3;
    const axes = [];
    for (let k = 0; k < 3; k += 1) {
        const jitter = 0.85 + 0.3 * randomUnit(state);
        axes.push(0.5 * options.size * options.aspect[k] / mean * jitter);
    }
    const radius = Math.cbrt(axes[0] * axes[1] * axes[2]);
    const fractures = [];
    for (let k = 0; k < options.fractures; k += 1) {
        const z = 2 * randomUnit(state) - 1;
        const angle = 2 * Math.PI * randomUnit(state);
        const ring = Math.sqrt(Math.max(0, 1 - z * z));
        const normal = v3(ring * Math.cos(angle), ring * Math.sin(angle), z);
        const reach = 1 / Math.sqrt(
            (normal.x / axes[0]) * (normal.x / axes[0]) +
            (normal.y / axes[1]) * (normal.y / axes[1]) +
            (normal.z / axes[2]) * (normal.z / axes[2]));
        fractures.push({ normal: normal, offset: reach * (1 - options.fractureDepth * randomUnit(state)) });
    }
    if (options.bedding > 0) {
        // A slab splits along two nearly parallel planes across its thin axis.
        const tilt = 0.06;
        fractures.push({
            normal: normalized(v3(tilt * (2 * randomUnit(state) - 1), tilt * (2 * randomUnit(state) - 1), 1)),
            offset: axes[2] * options.bedding,
        });
        fractures.push({
            normal: normalized(v3(tilt * (2 * randomUnit(state) - 1), tilt * (2 * randomUnit(state) - 1), -1)),
            offset: axes[2] * options.bedding,
        });
    }
    // Octave k has lattice frequency 2^k per rock radius; a wave spans two
    // lattice cells, so its surface wavelength is about 2 radius / 2^k. The
    // cutoff is twice the mean spacing of the hull's vertices: shorter waves
    // cannot be carried by the hull.
    const spacing = Math.sqrt(4 * Math.PI * radius * radius / options.vertexBudget);
    const cutoffWavelength = 2 * spacing;
    let cutoffOctave = options.octaves;
    for (let k = 0; k < options.octaves; k += 1) {
        if (2 * radius / Math.pow(2, k) < cutoffWavelength) {
            cutoffOctave = k;
            break;
        }
    }
    return {
        seed: seed,
        axes: axes,
        radius: radius,
        fractures: fractures,
        cutoffOctave: cutoffOctave,
        cutoffWavelength: cutoffWavelength,
        offsets: [randomUnit(state) * 64, randomUnit(state) * 64, randomUnit(state) * 64],
    };
}

// Sum of octaves [first, last) at unit direction d, relative to the radius.
function octaveSum(recipe, options, d, first, last) {
    let sum = 0;
    for (let k = first; k < last; k += 1) {
        const frequency = Math.pow(2, k);
        const amplitude = options.amplitude * Math.pow(2, -options.hurst * k);
        sum += amplitude * noise3(
            d.x * frequency + recipe.offsets[0],
            d.y * frequency + recipe.offsets[1],
            d.z * frequency + recipe.offsets[2], recipe.seed + k * 7919);
    }
    return sum;
}

// Radius of the geometric (below-cutoff) surface along unit direction d.
function lowRadius(recipe, options, d) {
    const axes = recipe.axes;
    let radius = 1 / Math.sqrt(
        (d.x / axes[0]) * (d.x / axes[0]) + (d.y / axes[1]) * (d.y / axes[1]) + (d.z / axes[2]) * (d.z / axes[2]));
    radius *= 1 + octaveSum(recipe, options, d, 0, recipe.cutoffOctave);
    for (let k = 0; k < recipe.fractures.length; k += 1) {
        const facing = dot(d, recipe.fractures[k].normal);
        if (facing > 1.0e-6) {
            radius = Math.min(radius, recipe.fractures[k].offset / facing);
        }
    }
    return radius;
}

// Height of the texture (above-cutoff bands) along d, in metres.
function textureHeight(recipe, options, d) {
    return recipe.radius * octaveSum(recipe, options, d, recipe.cutoffOctave, options.octaves);
}

function hullRadius(hull, d) {
    let radius = Infinity;
    for (let f = 0; f < hull.faces.length; f += 1) {
        const facing = dot(hull.faces[f].normal, d);
        if (facing > 1.0e-9) {
            radius = Math.min(radius, hull.faces[f].offset / facing);
        }
    }
    return radius;
}

function goldenDirections(count) {
    const directions = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < count; i += 1) {
        const z = 1 - (2 * i + 1) / count;
        const ring = Math.sqrt(Math.max(0, 1 - z * z));
        directions.push(v3(ring * Math.cos(golden * i), ring * Math.sin(golden * i), z));
    }
    return directions;
}

// Icosphere directions and triangles after `levels` subdivisions.
function icosphere(levels) {
    const t = (1 + Math.sqrt(5)) / 2;
    const raw = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
        [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ];
    const directions = [];
    for (let k = 0; k < raw.length; k += 1) {
        directions.push(normalized(v3(raw[k][0], raw[k][1], raw[k][2])));
    }
    let triangles = [
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
    ];
    for (let level = 0; level < levels; level += 1) {
        const midpoints = new Map();
        const next = [];
        for (let k = 0; k < triangles.length; k += 3) {
            const a = triangles[k];
            const b = triangles[k + 1];
            const c = triangles[k + 2];
            const ab = midpoint(directions, midpoints, a, b);
            const bc = midpoint(directions, midpoints, b, c);
            const ca = midpoint(directions, midpoints, c, a);
            next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
        }
        triangles = next;
    }
    return { directions: directions, triangles: triangles };
}

function midpoint(directions, midpoints, a, b) {
    const key = a < b ? a * 1048576 + b : b * 1048576 + a;
    if (midpoints.has(key)) {
        return midpoints.get(key);
    }
    directions.push(normalized(add(directions[a], directions[b])));
    midpoints.set(key, directions.length - 1);
    return directions.length - 1;
}

// Direction of texel (i, j) on cube face `face`; faces are +x, -x, +y, -y, +z, -z.
export function cubeDirection(face, i, j, size) {
    const a = 2 * (i + 0.5) / size - 1;
    const b = 2 * (j + 0.5) / size - 1;
    if (face === 0) {
        return normalized(v3(1, a, b));
    }
    if (face === 1) {
        return normalized(v3(-1, a, b));
    }
    if (face === 2) {
        return normalized(v3(a, 1, b));
    }
    if (face === 3) {
        return normalized(v3(a, -1, b));
    }
    if (face === 4) {
        return normalized(v3(a, b, 1));
    }
    return normalized(v3(a, b, -1));
}

// Generates one rock. Returns the cooked physics shape, the render mesh in
// the shape's body frame, the deformation map and the numbers that describe
// the split.
export function generateRock(seed, overrides) {
    const options = {};
    const names = Object.keys(defaultRockOptions);
    for (let k = 0; k < names.length; k += 1) {
        options[names[k]] = defaultRockOptions[names[k]];
    }
    if (overrides !== undefined) {
        const given = Object.keys(overrides);
        for (let k = 0; k < given.length; k += 1) {
            options[given[k]] = overrides[given[k]];
        }
    }
    const recipe = makeRecipe(seed, options);

    // Collision hull: the geometric surface sampled at the vertex budget.
    // A polyhedron with its corners on a convex surface lies inside it, so
    // the corners are pushed out until the hull and the surface agree on
    // average.
    const samples = goldenDirections(options.vertexBudget);
    let points = [];
    for (let k = 0; k < samples.length; k += 1) {
        points.push(scale(samples[k], lowRadius(recipe, options, samples[k])));
    }
    let hull = buildHull(points);
    const check = goldenDirections(800);
    let ratioSum = 0;
    for (let k = 0; k < check.length; k += 1) {
        ratioSum += lowRadius(recipe, options, check[k]) / hullRadius(hull, check[k]);
    }
    const inflate = ratioSum / check.length;
    const inflated = [];
    for (let k = 0; k < points.length; k += 1) {
        inflated.push(scale(points[k], inflate));
    }
    points = inflated;
    hull = buildHull(points);

    // Texture statistics on the deformation map: root-mean-square height and
    // slope of the bands above the cutoff.
    const size = options.mapSize;
    const map = new Float32Array(6 * size * size);
    let heightSquares = 0;
    let slopeSquares = 0;
    let slopeCount = 0;
    for (let face = 0; face < 6; face += 1) {
        for (let j = 0; j < size; j += 1) {
            for (let i = 0; i < size; i += 1) {
                const d = cubeDirection(face, i, j, size);
                const height = textureHeight(recipe, options, d);
                map[(face * size + j) * size + i] = height;
                heightSquares += height * height;
            }
        }
    }
    const probe = goldenDirections(600);
    const step = 0.25 * recipe.radius / Math.pow(2, options.octaves - 1);
    for (let k = 0; k < probe.length; k += 1) {
        const d = probe[k];
        const side = normalized(cross(d, Math.abs(d.z) < 0.9 ? v3(0, 0, 1) : v3(1, 0, 0)));
        const other = cross(d, side);
        const here = textureHeight(recipe, options, d);
        const alongSide = textureHeight(recipe, options, normalized(add(d, scale(side, step / recipe.radius))));
        const alongOther = textureHeight(recipe, options, normalized(add(d, scale(other, step / recipe.radius))));
        const slopeSide = (alongSide - here) / step;
        const slopeOther = (alongOther - here) / step;
        slopeSquares += slopeSide * slopeSide + slopeOther * slopeOther;
        slopeCount += 1;
    }
    const rmsHeight = Math.sqrt(heightSquares / map.length);
    const rmsSlope = Math.sqrt(slopeSquares / Math.max(1, slopeCount));
    const roughnessAngle = Math.min(options.roughnessAngleLimit,
        options.roughnessGain * Math.atan(rmsSlope) * 180 / Math.PI);
    const friction = Math.tan((options.basicFrictionAngle + roughnessAngle) * Math.PI / 180);

    const shape = cook({
        hulls: [points],
        density: options.density,
        friction: friction,
        restitution: options.restitution,
        rollingResistance: options.rollingResistance,
    });

    // Render mesh: the hull's own radius plus the texture, so the drawn rock
    // sits on the collision surface, with normals from every band.
    const sphere = icosphere(options.renderSubdivisions);
    const count = sphere.directions.length;
    const positions = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    const shade = new Float32Array(count);
    let deviationMax = 0;
    let deviationSum = 0;
    for (let k = 0; k < count; k += 1) {
        const d = sphere.directions[k];
        const onHull = hullRadius(hull, d);
        const low = lowRadius(recipe, options, d);
        const height = textureHeight(recipe, options, d);
        deviationMax = Math.max(deviationMax, Math.abs(onHull - low));
        deviationSum += Math.abs(onHull - low);
        // Halfway between the hull's flat facets and the designed surface: the
        // outline loses its corners and stays within half the facet error of
        // what the solver collides.
        const radius = 0.5 * (onHull + low) + height;
        positions[k * 3] = d.x * radius - shape.comOffset.x;
        positions[k * 3 + 1] = d.y * radius - shape.comOffset.y;
        positions[k * 3 + 2] = d.z * radius - shape.comOffset.z;
        shade[k] = height / Math.max(1.0e-9, 3 * rmsHeight);
    }
    // Normals: area-weighted face normals of the hull-based mesh, bent by the
    // gradient of the geometric and texture bands so creases read as stone.
    const triangles = sphere.triangles;
    for (let k = 0; k < triangles.length; k += 3) {
        const a = triangles[k];
        const b = triangles[k + 1];
        const c = triangles[k + 2];
        const pa = v3(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]);
        const pb = v3(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]);
        const pc = v3(positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]);
        const normal = cross(sub(pb, pa), sub(pc, pa));
        const corners = [a, b, c];
        for (let n = 0; n < 3; n += 1) {
            normals[corners[n] * 3] += normal.x;
            normals[corners[n] * 3 + 1] += normal.y;
            normals[corners[n] * 3 + 2] += normal.z;
        }
    }
    for (let k = 0; k < count; k += 1) {
        const n = normalized(v3(normals[k * 3], normals[k * 3 + 1], normals[k * 3 + 2]));
        normals[k * 3] = n.x;
        normals[k * 3 + 1] = n.y;
        normals[k * 3 + 2] = n.z;
    }
    return {
        shape: shape,
        render: { positions: positions, normals: normals, shade: shade, indices: new Uint32Array(triangles) },
        deformationMap: { size: size, heights: map },
        summary: {
            seed: seed,
            diameter: 2 * recipe.radius,
            hullVertices: hull.vertices.length,
            hullFaces: hull.faces.length,
            cutoffWavelength: recipe.cutoffWavelength,
            geometryOctaves: recipe.cutoffOctave,
            textureOctaves: options.octaves - recipe.cutoffOctave,
            textureRmsHeight: rmsHeight,
            textureRmsSlope: rmsSlope,
            roughnessAngleDegrees: roughnessAngle,
            friction: friction,
            hullFromGeometryMax: deviationMax,
            hullFromGeometryMean: deviationSum / count,
            mass: shape.mass,
        },
    };
}

// The mix of rocks a player would pick from, after photographs of balanced
// stones and beach cobble fields: mostly water-rounded discs and cobbles,
// some blocky fractured stones, some flat slabs and blades, a few small
// jagged shards. Each class sets shape ratios, how rough and how fractured
// the stone is, a size range and a colour family.
export const rockClasses = [
    {
        name: "river disc", weight: 0.32, size: [0.07, 0.2], aspect: [1.0, 0.82, 0.4],
        amplitude: 0.045, hurst: 1.1, fractures: 0, fractureDepth: 0, bedding: 0, basicFrictionAngle: 29,
        colours: [[0.66, 0.6, 0.52], [0.72, 0.68, 0.62], [0.6, 0.47, 0.4], [0.5, 0.5, 0.52], [0.7, 0.58, 0.5]],
    },
    {
        name: "cobble", weight: 0.16, size: [0.06, 0.16], aspect: [1.0, 0.86, 0.72],
        amplitude: 0.06, hurst: 1.0, fractures: 1, fractureDepth: 0.12, bedding: 0, basicFrictionAngle: 29,
        colours: [[0.68, 0.63, 0.56], [0.55, 0.42, 0.38], [0.58, 0.58, 0.6], [0.74, 0.7, 0.64]],
    },
    {
        name: "block", weight: 0.2, size: [0.06, 0.18], aspect: [1.0, 0.85, 0.68],
        amplitude: 0.09, hurst: 0.9, fractures: 7, fractureDepth: 0.3, bedding: 0, basicFrictionAngle: 31,
        colours: [[0.6, 0.59, 0.56], [0.52, 0.52, 0.53], [0.64, 0.6, 0.54]],
    },
    {
        name: "slab", weight: 0.2, size: [0.08, 0.22], aspect: [1.0, 0.72, 0.3],
        amplitude: 0.08, hurst: 0.9, fractures: 4, fractureDepth: 0.3, bedding: 0.8, basicFrictionAngle: 31,
        colours: [[0.42, 0.44, 0.47], [0.5, 0.5, 0.5], [0.46, 0.42, 0.4]],
    },
    {
        name: "shard", weight: 0.12, size: [0.03, 0.07], aspect: [1.0, 0.62, 0.42],
        amplitude: 0.15, hurst: 0.7, fractures: 8, fractureDepth: 0.45, bedding: 0, basicFrictionAngle: 31,
        colours: [[0.4, 0.4, 0.43], [0.5, 0.46, 0.42], [0.36, 0.38, 0.4]],
    },
];

// One rock drawn from the mix. Returns generateRock's record with the class
// name and a colour added.
export function generateMixedRock(seed, overrides) {
    const state = { value: (seed * 40503 + 977) >>> 0 };
    let pick = randomUnit(state);
    let chosen = rockClasses[rockClasses.length - 1];
    for (let k = 0; k < rockClasses.length; k += 1) {
        if (pick < rockClasses[k].weight) {
            chosen = rockClasses[k];
            break;
        }
        pick -= rockClasses[k].weight;
    }
    // Sizes are skewed toward the small end of the class range.
    const draw = randomUnit(state);
    const size = chosen.size[0] + (chosen.size[1] - chosen.size[0]) * draw * draw;
    const options = {
        size: size,
        aspect: [
            chosen.aspect[0],
            chosen.aspect[1] * (0.85 + 0.3 * randomUnit(state)),
            chosen.aspect[2] * (0.8 + 0.4 * randomUnit(state)),
        ],
        amplitude: chosen.amplitude * (0.75 + 0.5 * randomUnit(state)),
        hurst: chosen.hurst,
        fractures: chosen.fractures,
        fractureDepth: chosen.fractureDepth,
        bedding: chosen.bedding,
        basicFrictionAngle: chosen.basicFrictionAngle,
    };
    if (overrides !== undefined) {
        const given = Object.keys(overrides);
        for (let k = 0; k < given.length; k += 1) {
            options[given[k]] = overrides[given[k]];
        }
    }
    const rock = generateRock(seed, options);
    const colour = chosen.colours[Math.floor(randomUnit(state) * chosen.colours.length) % chosen.colours.length];
    const tone = 0.9 + 0.2 * randomUnit(state);
    rock.className = chosen.name;
    rock.colour = [colour[0] * tone, colour[1] * tone, colour[2] * tone];
    rock.summary.className = chosen.name;
    return rock;
}
