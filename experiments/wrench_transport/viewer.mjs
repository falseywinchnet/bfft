// WebGL viewer for the prototype: steps a chosen scene with a chosen solver
// and draws each body's render mesh (generated rocks carry a detailed mesh;
// other shapes are drawn from their hull faces). No dependencies.

import {
    v3, sub, scale, dot, cross, normalized, quatIdentity, quatFromAxisAngle, quatRotate, quatToMat3,
} from "./math3.mjs";
import { boxShape, boxPoints, hullShape, makeRandom, randomUnit, randomRange } from "./shapes.mjs";
import { addBody, stepWorld, maxPenetration, stableSet } from "./world.mjs";
import { makeWorld } from "./scenes.mjs";
import { generateRock, generateMixedRock } from "./rockgen.mjs";
import { buildContainerScene } from "./container.mjs";

const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl", { antialias: true });
const statusLine = document.getElementById("status");
const sceneSelect = document.getElementById("scene");
const solverSelect = document.getElementById("solver");
const pauseButton = document.getElementById("pause");
const restartButton = document.getElementById("restart");
const dropButton = document.getElementById("drop");

const view = { yaw: 0.7, pitch: 0.3, distance: 1.0, target: v3(0, 0, 0.14), paused: false, dragging: false, lastX: 0, lastY: 0 };
const session = { world: null, frame: 0, pending: [], random: makeRandom(1), stepMs: 0, sceneName: "", nextRock: 1, stableCount: 0 };
const LIGHT = normalized(v3(0.45, 0.3, 0.84));

const VERTEX_SOURCE = `
attribute vec3 position;
attribute vec3 normal;
attribute float shade;
uniform mat4 viewProjection;
uniform mat3 rotation;
uniform vec3 translation;
uniform mediump float flatten;
uniform mediump vec3 light;
varying vec3 worldNormal;
varying vec3 localPosition;
varying float shadeOut;
varying float height;
void main() {
    vec3 world = rotation * position + translation;
    if (flatten > 0.5) {
        // Project along the light onto the ground for a planar shadow.
        world = world - light * (world.z / light.z);
        world.z = 0.0004;
    }
    worldNormal = rotation * normal;
    localPosition = position;
    shadeOut = shade;
    height = world.z;
    gl_Position = viewProjection * vec4(world, 1.0);
}`;

const FRAGMENT_SOURCE = `
precision mediump float;
uniform vec3 baseColor;
uniform mediump vec3 light;
uniform mediump float flatten;
uniform float alpha;
uniform float grain;
varying vec3 worldNormal;
varying vec3 localPosition;
varying float shadeOut;
varying float height;
float hash(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}
void main() {
    if (flatten > 0.5) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.42);
        return;
    }
    vec3 n = normalize(worldNormal);
    float direct = max(dot(n, light), 0.0);
    float sky = 0.5 + 0.5 * n.z;
    // Crevices darker, crests lighter; fine speckle for mineral grain.
    float relief = clamp(0.5 + 0.5 * shadeOut, 0.0, 1.0);
    float speckle = hash(floor(localPosition * 900.0));
    vec3 albedo = baseColor * (0.78 + 0.3 * relief) * (1.0 - grain * 0.16 * speckle);
    float ground = clamp(0.55 + height * 6.0, 0.55, 1.0);
    vec3 color = albedo * (0.2 + 0.32 * sky * ground + 0.75 * direct);
    gl_FragColor = vec4(color, alpha);
}`;

const GROUND_VERTEX = `
attribute vec3 position;
uniform mat4 viewProjection;
varying vec2 plane;
void main() {
    plane = position.xy;
    gl_Position = viewProjection * vec4(position, 1.0);
}`;

const GROUND_FRAGMENT = `
precision mediump float;
varying vec2 plane;
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
void main() {
    // Sand: a warm base with fine grain, fading into the background far away.
    float grainValue = hash(floor(plane * 700.0));
    vec3 sand = vec3(0.56, 0.5, 0.4) * (0.9 + 0.14 * grainValue);
    float fade = clamp(1.0 - length(plane) / 1.6, 0.0, 1.0);
    gl_FragColor = vec4(mix(vec3(0.11, 0.125, 0.14), sand, fade), 1.0);
}`;

function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
}

function link(vertexSource, fragmentSource) {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
}

const bodyProgram = link(VERTEX_SOURCE, FRAGMENT_SOURCE);
const groundProgram = link(GROUND_VERTEX, GROUND_FRAGMENT);
const bodyUniforms = {
    viewProjection: gl.getUniformLocation(bodyProgram, "viewProjection"),
    rotation: gl.getUniformLocation(bodyProgram, "rotation"),
    translation: gl.getUniformLocation(bodyProgram, "translation"),
    flatten: gl.getUniformLocation(bodyProgram, "flatten"),
    light: gl.getUniformLocation(bodyProgram, "light"),
    baseColor: gl.getUniformLocation(bodyProgram, "baseColor"),
    alpha: gl.getUniformLocation(bodyProgram, "alpha"),
    grain: gl.getUniformLocation(bodyProgram, "grain"),
};
const bodyAttributes = {
    position: gl.getAttribLocation(bodyProgram, "position"),
    normal: gl.getAttribLocation(bodyProgram, "normal"),
    shade: gl.getAttribLocation(bodyProgram, "shade"),
};
const groundBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, groundBuffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-3, -3, 0, 3, -3, 0, 3, 3, 0, -3, -3, 0, 3, 3, 0, -3, 3, 0]), gl.STATIC_DRAW);

// GPU buffers for a shape, built once and kept on the shape record.
function meshFor(shape) {
    if (shape.gpu !== undefined) {
        return shape.gpu;
    }
    let positions = null;
    let normals = null;
    let shade = null;
    let indices = null;
    if (shape.render !== undefined) {
        positions = shape.render.positions;
        normals = shape.render.normals;
        shade = shape.render.shade;
        indices = new Uint16Array(shape.render.indices);
    } else {
        // Flat-shaded faces straight from the hulls.
        const p = [];
        const n = [];
        const idx = [];
        for (let h = 0; h < shape.hulls.length; h += 1) {
            const hull = shape.hulls[h];
            for (let f = 0; f < hull.faces.length; f += 1) {
                const loop = hull.faces[f].loop;
                const base = p.length / 3;
                for (let k = 0; k < loop.length; k += 1) {
                    const vertex = hull.vertices[loop[k]];
                    p.push(vertex.x, vertex.y, vertex.z);
                    n.push(hull.faces[f].normal.x, hull.faces[f].normal.y, hull.faces[f].normal.z);
                }
                for (let k = 1; k + 1 < loop.length; k += 1) {
                    idx.push(base, base + k, base + k + 1);
                }
            }
        }
        positions = new Float32Array(p);
        normals = new Float32Array(n);
        shade = new Float32Array(p.length / 3);
        indices = new Uint16Array(idx);
    }
    const gpu = {
        position: gl.createBuffer(),
        normal: gl.createBuffer(),
        shade: gl.createBuffer(),
        index: gl.createBuffer(),
        count: indices.length,
        grain: shape.render !== undefined ? 1 : 0.35,
    };
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.position);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.normal);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.shade);
    gl.bufferData(gl.ARRAY_BUFFER, shade, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpu.index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    shape.gpu = gpu;
    return gpu;
}

function rockShapeFor(seed, overrides) {
    const rock = generateRock(seed, overrides);
    rock.shape.render = rock.render;
    rock.shape.summary = rock.summary;
    return rock.shape;
}

function mixedRockShape(seed) {
    const rock = generateMixedRock(seed);
    rock.shape.render = rock.render;
    rock.shape.summary = rock.summary;
    rock.shape.colour = rock.colour;
    return rock.shape;
}

function lowestLocalZ(shape, orientation) {
    let lowest = Infinity;
    for (let h = 0; h < shape.hulls.length; h += 1) {
        const vertices = shape.hulls[h].vertices;
        for (let v = 0; v < vertices.length; v += 1) {
            lowest = Math.min(lowest, quatRotate(orientation, vertices[v]).z);
        }
    }
    return lowest;
}

function highestWorldZ(world) {
    let highest = 0;
    for (let i = 0; i < world.bodies.length; i += 1) {
        const body = world.bodies[i];
        if (body.isStatic) {
            continue;
        }
        const vertices = body.shape.hulls[0].vertices;
        for (let v = 0; v < vertices.length; v += 1) {
            highest = Math.max(highest, body.position.z + quatRotate(body.orientation, vertices[v]).z);
        }
    }
    return highest;
}

// Scene builders either add bodies now or queue {frame, shape, stack}.
function buildScene(name) {
    const world = makeWorld(solverSelect.value, true);
    session.world = world;
    session.frame = 0;
    session.pending = [];
    session.random = makeRandom(7);
    session.sceneName = name;
    view.target = v3(0, 0, 0.12);
    view.distance = 0.9;
    if (name === "rocks set one on another") {
        for (let i = 0; i < 12; i += 1) {
            session.pending.push({
                frame: i * 80,
                shape: mixedRockShape(200 + i),
                stack: true,
            });
        }
    } else if (name === "rock pile") {
        for (let i = 0; i < 30; i += 1) {
            session.pending.push({ frame: i * 18, shape: mixedRockShape(400 + i), stack: false });
        }
    } else if (name === "container, mixed shapes") {
        const scene = buildContainerScene(1);
        for (let k = 0; k < scene.walls.length; k += 1) {
            const wall = scene.walls[k];
            const shape = hullShape(boxPoints(wall.size.x, wall.size.y, wall.size.z), { friction: 0.5, rollingResistance: 0 });
            shape.transparent = true;
            addBody(world, shape, wall.centre, quatIdentity(), v3(0, 0, 0), v3(0, 0, 0), true);
        }
        for (let i = 0; i < scene.bodies.length; i += 1) {
            addBody(world, scene.bodies[i].shape, scene.bodies[i].position, scene.bodies[i].orientation);
        }
        world.params.groundFriction = 0.5;
        view.target = v3(0, 0, 0.1);
        view.distance = 1.1;
        view.pitch = 0.55;
    } else if (name === "box tower 20") {
        const shape = boxShape(0.1, 0.1, 0.05);
        for (let i = 0; i < 20; i += 1) {
            addBody(world, shape, v3(0, 0, 0.025 + 0.051 * i + 0.001));
        }
        view.target = v3(0, 0, 0.45);
        view.distance = 2.0;
    } else if (name === "slab on pebbles") {
        const pebble = boxShape(0.02, 0.02, 0.0192);
        const slab = boxShape(0.2, 0.2, 0.04, { density: pebble.mass * 500 / (0.2 * 0.2 * 0.04) });
        addBody(world, pebble, v3(0.06, 0, 0.0101));
        addBody(world, pebble, v3(-0.03, 0.052, 0.0101));
        addBody(world, pebble, v3(-0.03, -0.052, 0.0101));
        addBody(world, slab, v3(0, 0, 0.0192 + 0.02 + 0.002));
        view.distance = 0.6;
    }
}

function spawnPending() {
    const world = session.world;
    while (session.pending.length > 0 && session.pending[0].frame <= session.frame) {
        const item = session.pending.shift();
        const random = session.random;
        if (item.stack) {
            let x = 0;
            let y = 0;
            if (world.bodies.length > 0) {
                const below = world.bodies[world.bodies.length - 1];
                x = below.position.x;
                y = below.position.y;
            }
            const z = highestWorldZ(world) + 0.002 - lowestLocalZ(item.shape, quatIdentity());
            addBody(world, item.shape, v3(x, y, z));
        } else {
            const radius = 0.08 * Math.sqrt(randomUnit(random));
            const angle = 2 * Math.PI * randomUnit(random);
            const axis = normalized(v3(randomRange(random, -1, 1), randomRange(random, -1, 1), randomRange(random, -1, 1)));
            addBody(world, item.shape,
                v3(radius * Math.cos(angle), radius * Math.sin(angle), randomRange(random, 0.3, 0.5)),
                quatFromAxisAngle(axis, randomRange(random, 0, 2 * Math.PI)));
        }
    }
}

function dropRock() {
    const world = session.world;
    session.nextRock += 1;
    const shape = mixedRockShape(9000 + session.nextRock);
    addBody(world, shape, v3(0, 0, highestWorldZ(world) + 0.15));
}

function viewProjection() {
    const cp = Math.cos(view.pitch);
    const forward = v3(-Math.cos(view.yaw) * cp, -Math.sin(view.yaw) * cp, -Math.sin(view.pitch));
    const right = normalized(cross(forward, v3(0, 0, 1)));
    const up = cross(right, forward);
    const eye = sub(view.target, scale(forward, view.distance));
    const near = 0.02;
    const far = 20;
    const f = 1 / Math.tan(0.5 * 0.62);
    const aspect = canvas.width / canvas.height;
    const tx = -dot(right, eye);
    const ty = -dot(up, eye);
    const tz = dot(forward, eye);
    const a = (far + near) / (near - far);
    const b = 2 * far * near / (near - far);
    // Column-major P * V; view rows are right, up and -forward.
    return new Float32Array([
        f / aspect * right.x, f * up.x, -a * forward.x, forward.x,
        f / aspect * right.y, f * up.y, -a * forward.y, forward.y,
        f / aspect * right.z, f * up.z, -a * forward.z, forward.z,
        f / aspect * tx, f * ty, a * tz + b, -tz,
    ]);
}

const palette = [
    [0.62, 0.58, 0.52], [0.5, 0.5, 0.53], [0.6, 0.52, 0.45], [0.47, 0.5, 0.46], [0.66, 0.62, 0.55], [0.52, 0.47, 0.45],
];

function drawBody(body, index, flatten, stable) {
    const gpu = meshFor(body.shape);
    const m = quatToMat3(body.orientation);
    gl.uniformMatrix3fv(bodyUniforms.rotation, false,
        new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]));
    gl.uniform3f(bodyUniforms.translation, body.position.x, body.position.y, body.position.z);
    gl.uniform1f(bodyUniforms.flatten, flatten ? 1 : 0);
    const color = body.shape.colour !== undefined ? body.shape.colour : palette[index % palette.length];
    // Bodies in the stable set are drawn at full tone; moving ones warmer.
    const warm = stable ? 0 : 0.12;
    gl.uniform3f(bodyUniforms.baseColor, color[0] + warm, color[1], color[2] - 0.5 * warm);
    gl.uniform1f(bodyUniforms.alpha, body.shape.transparent ? 0.18 : 1);
    gl.uniform1f(bodyUniforms.grain, gpu.grain);
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.position);
    gl.vertexAttribPointer(bodyAttributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.normal);
    gl.vertexAttribPointer(bodyAttributes.normal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.shade);
    gl.vertexAttribPointer(bodyAttributes.shade, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpu.index);
    gl.drawElements(gl.TRIANGLES, gpu.count, gl.UNSIGNED_SHORT, 0);
}

function draw() {
    const world = session.world;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.11, 0.125, 0.14, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const matrix = viewProjection();

    gl.useProgram(groundProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(groundProgram, "viewProjection"), false, matrix);
    gl.bindBuffer(gl.ARRAY_BUFFER, groundBuffer);
    const groundPosition = gl.getAttribLocation(groundProgram, "position");
    gl.enableVertexAttribArray(groundPosition);
    gl.vertexAttribPointer(groundPosition, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.useProgram(bodyProgram);
    gl.uniformMatrix4fv(bodyUniforms.viewProjection, false, matrix);
    gl.uniform3f(bodyUniforms.light, LIGHT.x, LIGHT.y, LIGHT.z);
    gl.enableVertexAttribArray(bodyAttributes.position);
    gl.enableVertexAttribArray(bodyAttributes.normal);
    gl.enableVertexAttribArray(bodyAttributes.shade);
    const stable = new Uint8Array(world.bodies.length);
    const stableIds = stableSet(world);
    for (let k = 0; k < stableIds.length; k += 1) {
        stable[stableIds[k]] = 1;
    }
    // Planar shadows first, blended over the sand.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    for (let i = 0; i < world.bodies.length; i += 1) {
        if (!world.bodies[i].shape.transparent) {
            drawBody(world.bodies[i], i, true, true);
        }
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    for (let i = 0; i < world.bodies.length; i += 1) {
        if (!world.bodies[i].shape.transparent) {
            drawBody(world.bodies[i], i, false, stable[i] === 1);
        }
    }
    gl.enable(gl.BLEND);
    gl.depthMask(false);
    for (let i = 0; i < world.bodies.length; i += 1) {
        if (world.bodies[i].shape.transparent) {
            drawBody(world.bodies[i], i, false, true);
        }
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    session.stableCount = stableIds.length;
}

function updateStatus() {
    const world = session.world;
    const rest = world.rest;
    let text = session.sceneName + " | " + solverSelect.value + " | bodies " + world.bodies.length +
        " | contacts " + world.contacts.length + " | awake " + rest.awake +
        " | stable set " + session.stableCount + " | " + (rest.quiet ? "QUIET" : "moving, max " + rest.maxSpeed.toExponential(1) + " m/s") +
        " | deepest overlap " + (maxPenetration(world) * 1000).toFixed(4) + " mm" +
        " | step " + session.stepMs.toFixed(2) + " ms";
    if (world.solverName === "wrench") {
        text += " | passes " + world.lastSolve.passes + ", Newton " + world.lastSolve.iterations;
    }
    statusLine.textContent = text;
}

function tick() {
    if (!view.paused) {
        spawnPending();
        const started = performance.now();
        stepWorld(session.world);
        session.stepMs = 0.9 * session.stepMs + 0.1 * (performance.now() - started);
        session.frame += 1;
    }
    draw();
    updateStatus();
    window.requestAnimationFrame(tick);
}

function onPointerDown(event) {
    view.dragging = true;
    view.lastX = event.clientX;
    view.lastY = event.clientY;
}

function onPointerUp() {
    view.dragging = false;
}

function onPointerMove(event) {
    if (!view.dragging) {
        return;
    }
    view.yaw -= 0.008 * (event.clientX - view.lastX);
    view.pitch = Math.min(1.45, Math.max(0.02, view.pitch + 0.008 * (event.clientY - view.lastY)));
    view.lastX = event.clientX;
    view.lastY = event.clientY;
}

function onWheel(event) {
    event.preventDefault();
    view.distance = Math.min(6, Math.max(0.2, view.distance * (event.deltaY > 0 ? 1.1 : 0.9)));
}

function onPause() {
    view.paused = !view.paused;
    pauseButton.textContent = view.paused ? "Resume" : "Pause";
}

function onRestart() {
    buildScene(sceneSelect.value);
}

function populate(select, names, labels) {
    for (let k = 0; k < names.length; k += 1) {
        const option = document.createElement("option");
        option.value = names[k];
        option.textContent = labels === undefined ? names[k] : labels[k];
        select.appendChild(option);
    }
}

// Advances the simulation without waiting for animation frames (for tests).
function advance(frames) {
    for (let f = 0; f < frames; f += 1) {
        spawnPending();
        stepWorld(session.world);
        session.frame += 1;
    }
    draw();
    updateStatus();
}

populate(sceneSelect, ["rocks set one on another", "rock pile", "container, mixed shapes", "box tower 20", "slab on pebbles"]);
// "wrench" is this engine's solver. The soft-step entries are the solver the
// specification prescribed, kept only to compare against.
populate(solverSelect, ["wrench", "soft_step_spec", "soft_step_best", "soft_step_16"], [
    "wrench (our solver)",
    "comparison: spec soft-step, as written",
    "comparison: soft-step, stiffest 8 substeps",
    "comparison: soft-step, 16 substeps",
]);
canvas.addEventListener("pointerdown", onPointerDown);
window.addEventListener("pointerup", onPointerUp);
window.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("wheel", onWheel, { passive: false });
pauseButton.addEventListener("click", onPause);
restartButton.addEventListener("click", onRestart);
dropButton.addEventListener("click", dropRock);
sceneSelect.addEventListener("change", onRestart);
solverSelect.addEventListener("change", onRestart);
window.wrenchViewer = { session: session, view: view, advance: advance };
buildScene(sceneSelect.value);
window.requestAnimationFrame(tick);
