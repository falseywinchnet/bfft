// Macro-frame elimination: a sparse block Cholesky factorization over the
// contact graph, six degrees of freedom per body.
//
// Eliminating a body folds its inertia and its load into the bodies it still
// touches (the Schur complement is the "macro-frame" those bodies see), and the
// reverse sweep hands each eliminated body its response. One up sweep and one
// down sweep solve the linear system exactly, so a load change reaches every
// body in the connected group within one solve.
//
// Blocks are 6x6, row-major Float64Array(36). Positions are elimination order.

const B = 6;
const BB = 36;

export function createFactor() {
    return {
        nodeCount: 0,
        order: [],           // order[position] = node
        position: [],        // position[node] = elimination position
        structure: [],       // structure[p] = later positions coupled to p, ascending
        diagonal: [],        // diagonal[p]: block, overwritten by its Cholesky factor
        upper: [],           // upper[p][k]: block (p, structure[p][k])
        transfer: [],        // transfer[p][k] = inverse(diagonal[p]) * upper[p][k]
        lookup: new Map(),   // p * nodeCount + q -> block for p < q
        work: new Float64Array(0),
        fillBlocks: 0,
    };
}

// Chooses a minimum-degree elimination order for the graph given by `pairs`
// (flat node index pairs) and allocates every block the factor will touch.
export function analyze(factor, nodeCount, pairs) {
    const adjacency = new Uint8Array(nodeCount * nodeCount);
    for (let k = 0; k + 1 < pairs.length; k += 2) {
        const i = pairs[k];
        const j = pairs[k + 1];
        if (i !== j) {
            adjacency[i * nodeCount + j] = 1;
            adjacency[j * nodeCount + i] = 1;
        }
    }
    const eliminated = new Uint8Array(nodeCount);
    const order = [];
    const position = new Array(nodeCount).fill(-1);
    const neighbourNodes = [];
    let originalLinks = 0;
    for (let k = 0; k < adjacency.length; k += 1) {
        originalLinks += adjacency[k];
    }
    for (let step = 0; step < nodeCount; step += 1) {
        let best = -1;
        let bestDegree = nodeCount + 1;
        for (let i = 0; i < nodeCount; i += 1) {
            if (eliminated[i]) {
                continue;
            }
            let degree = 0;
            for (let j = 0; j < nodeCount; j += 1) {
                if (!eliminated[j] && adjacency[i * nodeCount + j]) {
                    degree += 1;
                }
            }
            if (degree < bestDegree) {
                bestDegree = degree;
                best = i;
            }
        }
        eliminated[best] = 1;
        order.push(best);
        position[best] = step;
        const neighbours = [];
        for (let j = 0; j < nodeCount; j += 1) {
            if (!eliminated[j] && adjacency[best * nodeCount + j]) {
                neighbours.push(j);
            }
        }
        for (let a = 0; a < neighbours.length; a += 1) {
            for (let b = a + 1; b < neighbours.length; b += 1) {
                adjacency[neighbours[a] * nodeCount + neighbours[b]] = 1;
                adjacency[neighbours[b] * nodeCount + neighbours[a]] = 1;
            }
        }
        neighbourNodes.push(neighbours);
    }
    factor.nodeCount = nodeCount;
    factor.order = order;
    factor.position = position;
    factor.structure = [];
    factor.diagonal = [];
    factor.upper = [];
    factor.transfer = [];
    factor.lookup = new Map();
    factor.work = new Float64Array(nodeCount * B);
    let storedLinks = 0;
    for (let p = 0; p < nodeCount; p += 1) {
        const later = [];
        for (let k = 0; k < neighbourNodes[p].length; k += 1) {
            later.push(position[neighbourNodes[p][k]]);
        }
        later.sort(ascending);
        factor.structure.push(later);
        factor.diagonal.push(new Float64Array(BB));
        const upperRow = [];
        const transferRow = [];
        for (let k = 0; k < later.length; k += 1) {
            const block = new Float64Array(BB);
            upperRow.push(block);
            transferRow.push(new Float64Array(BB));
            factor.lookup.set(p * nodeCount + later[k], block);
            storedLinks += 1;
        }
        factor.upper.push(upperRow);
        factor.transfer.push(transferRow);
    }
    factor.fillBlocks = storedLinks - originalLinks / 2;
}

function ascending(a, b) {
    return a - b;
}

export function clearBlocks(factor) {
    for (let p = 0; p < factor.nodeCount; p += 1) {
        factor.diagonal[p].fill(0);
        const row = factor.upper[p];
        for (let k = 0; k < row.length; k += 1) {
            row[k].fill(0);
        }
    }
}

export function diagonalBlock(factor, node) {
    return factor.diagonal[factor.position[node]];
}

// The stored block for an off-diagonal node pair and whether it is stored
// transposed (when nodeI is eliminated after nodeJ).
export function couplingBlock(factor, nodeI, nodeJ) {
    const p = factor.position[nodeI];
    const q = factor.position[nodeJ];
    if (p < q) {
        return { block: factor.lookup.get(p * factor.nodeCount + q), transposed: false };
    }
    return { block: factor.lookup.get(q * factor.nodeCount + p), transposed: true };
}

// In-place Cholesky of a 6x6 block: lower triangle receives L, L L^T = block.
function choleskyBlock(block) {
    for (let j = 0; j < B; j += 1) {
        let pivot = block[j * B + j];
        for (let k = 0; k < j; k += 1) {
            pivot -= block[j * B + k] * block[j * B + k];
        }
        if (!(pivot > 0)) {
            return false;
        }
        pivot = Math.sqrt(pivot);
        block[j * B + j] = pivot;
        for (let i = j + 1; i < B; i += 1) {
            let value = block[i * B + j];
            for (let k = 0; k < j; k += 1) {
                value -= block[i * B + k] * block[j * B + k];
            }
            block[i * B + j] = value / pivot;
        }
    }
    return true;
}

// Solves (L L^T) x = b for six values at `offset` with stride `stride`.
function choleskySolve(lower, values, offset, stride) {
    for (let i = 0; i < B; i += 1) {
        let value = values[offset + i * stride];
        for (let k = 0; k < i; k += 1) {
            value -= lower[i * B + k] * values[offset + k * stride];
        }
        values[offset + i * stride] = value / lower[i * B + i];
    }
    for (let i = B - 1; i >= 0; i -= 1) {
        let value = values[offset + i * stride];
        for (let k = i + 1; k < B; k += 1) {
            value -= lower[k * B + i] * values[offset + k * stride];
        }
        values[offset + i * stride] = value / lower[i * B + i];
    }
}

// target -= left^T * right for 6x6 blocks.
function subtractTransposeProduct(target, left, right) {
    for (let i = 0; i < B; i += 1) {
        for (let j = 0; j < B; j += 1) {
            let sum = 0;
            for (let k = 0; k < B; k += 1) {
                sum += left[k * B + i] * right[k * B + j];
            }
            target[i * B + j] -= sum;
        }
    }
}

// Numeric factorization of the assembled blocks. Returns false if a pivot is
// not positive (the matrix was not positive definite to working precision).
export function factorize(factor) {
    const count = factor.nodeCount;
    for (let p = 0; p < count; p += 1) {
        const lower = factor.diagonal[p];
        if (!choleskyBlock(lower)) {
            return false;
        }
        const later = factor.structure[p];
        const upperRow = factor.upper[p];
        const transferRow = factor.transfer[p];
        for (let k = 0; k < later.length; k += 1) {
            const transfer = transferRow[k];
            transfer.set(upperRow[k]);
            for (let column = 0; column < B; column += 1) {
                choleskySolve(lower, transfer, column, B);
            }
        }
        for (let k1 = 0; k1 < later.length; k1 += 1) {
            const q = later[k1];
            subtractTransposeProduct(factor.diagonal[q], upperRow[k1], transferRow[k1]);
            for (let k2 = k1 + 1; k2 < later.length; k2 += 1) {
                const target = factor.lookup.get(q * count + later[k2]);
                subtractTransposeProduct(target, upperRow[k1], transferRow[k2]);
            }
        }
    }
    return true;
}

// Solves H x = rhs. Both vectors are indexed by node (six values per node).
export function solve(factor, rhs, x) {
    const count = factor.nodeCount;
    const work = factor.work;
    for (let p = 0; p < count; p += 1) {
        const node = factor.order[p];
        for (let i = 0; i < B; i += 1) {
            work[p * B + i] = rhs[node * B + i];
        }
    }
    // Up sweep: each eliminated body hands its load to the bodies it touches.
    for (let p = 0; p < count; p += 1) {
        const later = factor.structure[p];
        const transferRow = factor.transfer[p];
        for (let k = 0; k < later.length; k += 1) {
            const transfer = transferRow[k];
            const q = later[k];
            for (let j = 0; j < B; j += 1) {
                let sum = 0;
                for (let i = 0; i < B; i += 1) {
                    sum += transfer[i * B + j] * work[p * B + i];
                }
                work[q * B + j] -= sum;
            }
        }
    }
    for (let p = 0; p < count; p += 1) {
        choleskySolve(factor.diagonal[p], work, p * B, 1);
    }
    // Down sweep: each body receives the response of the bodies it handed to.
    for (let p = count - 1; p >= 0; p -= 1) {
        const later = factor.structure[p];
        const transferRow = factor.transfer[p];
        for (let k = 0; k < later.length; k += 1) {
            const transfer = transferRow[k];
            const q = later[k];
            for (let i = 0; i < B; i += 1) {
                let sum = 0;
                for (let j = 0; j < B; j += 1) {
                    sum += transfer[i * B + j] * work[q * B + j];
                }
                work[p * B + i] -= sum;
            }
        }
    }
    for (let p = 0; p < count; p += 1) {
        const node = factor.order[p];
        for (let i = 0; i < B; i += 1) {
            x[node * B + i] = work[p * B + i];
        }
    }
}
