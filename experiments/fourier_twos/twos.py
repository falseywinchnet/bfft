"""Exact Fourier encodings with shifts, additions and sign-changing wraps.

Only diagnostic_complex uses complex arithmetic. Integer multiplication below
is confined to indices, bounds and size accounting, never transform samples.
The expanded transform is a Fermat-ring FFT plus bounded Kronecker lifting.
The compact transform retains cyclotomic orbits, not numeric complex bins.
"""
from dataclasses import dataclass
import cmath
import math


def power_two(n):
    return isinstance(n, int) and n >= 2 and n & (n - 1) == 0


@dataclass(frozen=True)
class ShiftPlan:
    n: int
    bound: int
    lane_bits: int = 0

    def __post_init__(self):
        if not power_two(self.n) or not isinstance(self.bound, int) or self.bound < 0:
            raise ValueError("power-of-two N >= 2 and nonnegative integer bound required")
        # Strictly B > 2*N*A+1 gives an unambiguous balanced polynomial lift.
        minimum = (2 * self.n * self.bound + 1).bit_length()
        if self.lane_bits and self.lane_bits < minimum:
            raise ValueError("insufficient guard bits for the declared input range")
        object.__setattr__(self, 'lane_bits', self.lane_bits or minimum)

    @property
    def degree(self):
        return self.n >> 1

    @property
    def width(self):
        return self.degree * self.lane_bits

    @property
    def modulus(self):
        return (1 << self.width) + 1

    def reduce(self, value):
        """Canonical mod (2**M+1), with no remainder/division instruction."""
        mask = (1 << self.width) - 1
        q = self.modulus
        while value < 0 or value >= q:
            value = (value & mask) - (value >> self.width)
        return value

    def shift(self, value, bits):
        """Multiply by 2**bits in the ring; negative bits mean modular inverse."""
        # Remainder is on a public shift count, not on signal data.
        bits %= self.width << 1
        if bits >= self.width:
            return self.reduce(-(value << (bits - self.width)))
        return self.reduce(value << bits)

    def fft(self, values, inverse=False):
        """Natural-order input/output, including an explicitly paid bit reversal."""
        if len(values) != self.n or any(type(x) is not int for x in values):
            raise ValueError("exactly N Python integers required")
        v = [self.reduce(x) for x in values]
        j = 0
        for i in range(1, self.n):
            bit = self.n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                v[i], v[j] = v[j], v[i]
        length = 2
        while length <= self.n:
            half = length >> 1
            step = self.lane_bits * (self.n // length)
            if inverse:
                step = -step
            for start in range(0, self.n, length):
                shift = 0
                for j in range(half):
                    a = v[start + j]
                    b = self.shift(v[start + half + j], shift)
                    v[start + j] = self.reduce(a + b)
                    v[start + half + j] = self.reduce(a - b)
                    shift += step
            length <<= 1
        if inverse:
            v = [self.shift(x, -(self.n.bit_length() - 1)) for x in v]
        return v

    def forward(self, values):
        if any(type(x) is not int or abs(x) > self.bound for x in values):
            raise ValueError("input exceeds declared integer range")
        return self.fft(values)

    def coefficients(self, value):
        """Decode the unique small coefficients; valid after a bounded forward."""
        value = self.reduce(value)
        if value > self.modulus >> 1:
            value -= self.modulus
        base = 1 << self.lane_bits
        half = base >> 1
        result = []
        for _ in range(self.degree):
            digit = ((value + half) & (base - 1)) - half
            result.append(digit)
            value = (value - digit) >> self.lane_bits
        if value:
            raise ValueError("residue has no balanced lift of this degree")
        return result


def polynomial_bin(values, k):
    """Independent exact oracle: reduce sum x[j]*z**(j*k) modulo z**(N/2)+1."""
    n = len(values)
    if not power_two(n):
        raise ValueError("power-of-two input required")
    d = n >> 1
    result = [0] * d
    for j, x in enumerate(values):
        e = (j * k) & (n - 1)
        if e < d:
            result[e] += x
        else:
            result[e - d] -= x
    return result


def compact_forward(values):
    """N integers encoding every bin through its cyclotomic/Galois orbit.

    Packet key m means roots of exact order m. Its coefficients evaluate at
    exp(-2*pi*i*u/m), for odd u. Key 1 holds DC. No twiddle coefficients.
    """
    if not power_two(len(values)) or any(type(x) is not int for x in values):
        raise ValueError("power-of-two integer input required")
    v = list(values)
    packets = {}
    while len(v) > 1:
        h = len(v) >> 1
        packets[len(v)] = [v[j] - v[j + h] for j in range(h)]
        v = [v[j] + v[j + h] for j in range(h)]
    packets[1] = v
    return packets


def compact_inverse(packets):
    """Exact inverse of an admitted integer packet set; halves must be exact."""
    n = max(packets)
    if not power_two(n) or set(packets) != {1 << j for j in range(n.bit_length())}:
        raise ValueError("incomplete packet set")
    if len(packets[1]) != 1:
        raise ValueError("one DC coefficient required")
    v = list(packets[1])
    m = 2
    while m <= n:
        minus = packets[m]
        if len(minus) != len(v):
            raise ValueError("incorrect packet degree")
        sums = [a + b for a, b in zip(v, minus)]
        differences = [a - b for a, b in zip(v, minus)]
        if any(x & 1 for x in sums + differences):
            raise ValueError("packets do not represent an integer input")
        v = [x >> 1 for x in sums + differences]
        m <<= 1
    return v


def compact_bin(packets, n, k):
    """Materialize one bin in the common exact basis, using signed placement."""
    if not 0 <= k < n or max(packets) != n:
        raise ValueError("bin/packet size mismatch")
    if k == 0:
        return packets[1] + [0] * ((n >> 1) - 1)
    g = k & -k
    m = n // g
    odd = k // g
    result = [0] * (n >> 1)
    for j, x in enumerate(packets[m]):
        e = (g * odd * j) & (n - 1)
        if e < n >> 1:
            result[e] += x
        else:
            result[e - (n >> 1)] -= x
    return result


def orbit_lane_bits(n, bound):
    """Tighter guards: order-m orbit coefficients have magnitude <= 2*N*A/m."""
    if not power_two(n) or type(bound) is not int or bound < 0:
        raise ValueError("power-of-two N and nonnegative integer bound required")
    return {m: (4 * (n // m) * bound + 1).bit_length()
            for m in (1 << j for j in range(1, n.bit_length()))}


def pack_samples(values, plan):
    """Pack bounded samples as one signed binary-segmented integer.

    Reference Horner packing allocates growing integers; its cost is separate
    from the packed folding kernel and is not claimed to be linear in Python.
    """
    if len(values) != plan.n or any(type(x) is not int or abs(x) > plan.bound for x in values):
        raise ValueError("input exceeds plan length/type/range")
    word = 0
    for x in reversed(values):
        word = (word << plan.lane_bits) + x
    return word


def fold_packed(word, plan):
    """All compact orbit residues from a guarded packed input, without unpacking.

    Precondition: word came from pack_samples(values, plan). Global guard width
    is retained, unlike the tighter variable-width pack_orbits representation.
    This uses only log2(N) packed splitting/folding stages, of decreasing width.
    """
    packets = {}
    m = plan.n
    while m > 1:
        width = (m >> 1) * plan.lane_bits
        mask = (1 << width) - 1
        low, high = word & mask, word >> width
        packets[m] = ShiftPlan(m, 0, plan.lane_bits).reduce(low - high)
        # Remainder modulo B^(m/2)-1, then center it. The global guard
        # ensures the centered value is the exact folded coefficient polynomial.
        word = low + high
        if word >= mask:
            word -= mask
        elif word < 0:
            word += mask
        if word > mask >> 1:
            word -= mask
        m >>= 1
    packets[1] = word
    return packets


def _bits_for(lane_bits, m):
    return lane_bits[m] if isinstance(lane_bits, dict) else lane_bits


def pack_orbits(packets, lane_bits):
    """Guarded binary segmentation of each orbit, retaining N coefficient lanes."""
    packed = {1: packets[1][0]}
    for m, coefficients in packets.items():
        if m == 1:
            continue
        bits = _bits_for(lane_bits, m)
        ring = ShiftPlan(m, 0, bits)
        value = 0
        for c in reversed(coefficients):
            if abs(c) >= (1 << (bits - 1)):
                raise ValueError("orbit coefficient needs more guard bits")
            value = (value << bits) + c
        packed[m] = ring.reduce(value)
    return packed


def unpack_orbits(packed, lane_bits):
    return {m: [value] if m == 1 else ShiftPlan(m, 0, _bits_for(lane_bits, m)).coefficients(value)
            for m, value in packed.items()}


def phase_shift_orbits(packed, lane_bits, samples):
    """Exact circular time delay = one modular bit shift per non-DC orbit.

    No numeric twiddle table, per-bin expansion, or signal multiplication.
    Input must have been packed with enough guard bits for its coefficients.
    """
    return {m: value if m == 1 else
            ShiftPlan(m, 0, _bits_for(lane_bits, m)).shift(value, _bits_for(lane_bits, m) * samples)
            for m, value in packed.items()}


def diagnostic_complex(coefficients, n):
    """PAID numeric projection, deliberately separate from shift-only execution.

    This direct O(N) per-bin oracle uses trigonometry and general products.
    It is not part of any claim of twiddle-free ordinary complex output.
    """
    terms = [c * cmath.exp(-2j * math.pi * r / n)
             for r, c in enumerate(coefficients) if c]
    return complex(math.fsum(t.real for t in terms), math.fsum(t.imag for t in terms))
