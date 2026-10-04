// Small fixed-size vector, quaternion and matrix arithmetic for the prototype.
// Vectors are {x, y, z} records, quaternions {w, x, y, z}, 3x3 matrices are
// row-major arrays of nine doubles.

export function v3(x, y, z) {
    return { x: x, y: y, z: z };
}

export function add(a, b) {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function sub(a, b) {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function scale(a, s) {
    return { x: a.x * s, y: a.y * s, z: a.z * s };
}

export function addScaled(a, b, s) {
    return { x: a.x + b.x * s, y: a.y + b.y * s, z: a.z + b.z * s };
}

export function dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function cross(a, b) {
    return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    };
}

export function length(a) {
    return Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

export function normalized(a) {
    const len = length(a);
    if (len === 0) {
        return { x: 0, y: 0, z: 0 };
    }
    return { x: a.x / len, y: a.y / len, z: a.z / len };
}

export function quatIdentity() {
    return { w: 1, x: 0, y: 0, z: 0 };
}

export function quatMul(a, b) {
    return {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

export function quatConj(q) {
    return { w: q.w, x: -q.x, y: -q.y, z: -q.z };
}

export function quatNormalized(q) {
    const len = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return { w: q.w / len, x: q.x / len, y: q.y / len, z: q.z / len };
}

export function quatFromAxisAngle(axis, angle) {
    const unit = normalized(axis);
    const half = 0.5 * angle;
    const s = Math.sin(half);
    return { w: Math.cos(half), x: unit.x * s, y: unit.y * s, z: unit.z * s };
}

export function quatRotate(q, v) {
    // v + 2 w (u x v) + 2 u x (u x v), u = vector part
    const ux = q.x;
    const uy = q.y;
    const uz = q.z;
    const cx = uy * v.z - uz * v.y;
    const cy = uz * v.x - ux * v.z;
    const cz = ux * v.y - uy * v.x;
    const dx = uy * cz - uz * cy;
    const dy = uz * cx - ux * cz;
    const dz = ux * cy - uy * cx;
    return {
        x: v.x + 2 * (q.w * cx + dx),
        y: v.y + 2 * (q.w * cy + dy),
        z: v.z + 2 * (q.w * cz + dz),
    };
}

export function quatRotateInverse(q, v) {
    return quatRotate(quatConj(q), v);
}

// q + (h/2) (0, w) q, renormalized: first-order orientation update.
export function quatIntegrate(q, w, h) {
    const half = 0.5 * h;
    const out = {
        w: q.w + half * (-w.x * q.x - w.y * q.y - w.z * q.z),
        x: q.x + half * (w.x * q.w + w.y * q.z - w.z * q.y),
        y: q.y + half * (w.y * q.w + w.z * q.x - w.x * q.z),
        z: q.z + half * (w.z * q.w + w.x * q.y - w.y * q.x),
    };
    return quatNormalized(out);
}

export function quatToMat3(q) {
    const xx = q.x * q.x;
    const yy = q.y * q.y;
    const zz = q.z * q.z;
    const xy = q.x * q.y;
    const xz = q.x * q.z;
    const yz = q.y * q.z;
    const wx = q.w * q.x;
    const wy = q.w * q.y;
    const wz = q.w * q.z;
    return [
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ];
}

export function mat3MulVec(m, v) {
    return {
        x: m[0] * v.x + m[1] * v.y + m[2] * v.z,
        y: m[3] * v.x + m[4] * v.y + m[5] * v.z,
        z: m[6] * v.x + m[7] * v.y + m[8] * v.z,
    };
}

export function mat3Mul(a, b) {
    const out = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let row = 0; row < 3; row += 1) {
        for (let col = 0; col < 3; col += 1) {
            out[row * 3 + col] =
                a[row * 3] * b[col] +
                a[row * 3 + 1] * b[3 + col] +
                a[row * 3 + 2] * b[6 + col];
        }
    }
    return out;
}

export function mat3Transpose(m) {
    return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

export function mat3Inverse(m) {
    const c00 = m[4] * m[8] - m[5] * m[7];
    const c01 = m[5] * m[6] - m[3] * m[8];
    const c02 = m[3] * m[7] - m[4] * m[6];
    const det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    const inv = 1 / det;
    return [
        c00 * inv, (m[2] * m[7] - m[1] * m[8]) * inv, (m[1] * m[5] - m[2] * m[4]) * inv,
        c01 * inv, (m[0] * m[8] - m[2] * m[6]) * inv, (m[2] * m[3] - m[0] * m[5]) * inv,
        c02 * inv, (m[1] * m[6] - m[0] * m[7]) * inv, (m[0] * m[4] - m[1] * m[3]) * inv,
    ];
}

// R * m * R^T for a rotation matrix R: body-frame tensor to world frame.
export function mat3Rotated(rotation, m) {
    return mat3Mul(mat3Mul(rotation, m), mat3Transpose(rotation));
}
