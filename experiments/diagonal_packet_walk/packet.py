"""Exact finite diagonal coset coordinates; dense matrices are diagnostic only."""
import numpy as np


class Packet:
    def __init__(self, exponent, slope=1):
        if exponent < 1:
            raise ValueError("exponent must be positive")
        self.r = exponent
        self.m = m = 2 ** exponent
        self.n = m * m
        self.a = a = 2 ** (exponent // 2)
        self.b = b = m // a
        self.slope = slope
        if (2 * slope * a * a) % m:
            raise ValueError("subgroup is not self-annihilating")
        q, p, j, l = np.indices((b, a, b, a))
        self.pack = (p + a*j + m*((q + slope*a*j + b*l) % m)).ravel()
        self.unpack = np.argsort(self.pack)

    def forward(self, x):
        """Natural sample order -> (q,p,alpha,beta) packet coordinates."""
        v = np.asarray(x)[self.pack].reshape(self.b, self.a, self.b, self.a)
        return np.fft.fftn(v, axes=(2, 3), norm="ortho").ravel()

    def inverse(self, z):
        v = np.fft.ifftn(np.asarray(z).reshape(self.b, self.a, self.b, self.a),
                         axes=(2, 3), norm="ortho").ravel()
        return v[self.unpack]

    def monomial(self):
        """P=QGQ*: output row reads one column with a unit phase."""
        b, a, m, s = self.b, self.a, self.m, self.slope
        q, p, alpha, beta = np.indices((b, a, b, a))
        pp = (-beta) % a
        qp = (-alpha - s*pp) % b
        ap = (s*p + q) % b
        bp = p
        col = np.ravel_multi_index((qp, pp, ap, bp), (b,a,b,a)).ravel()
        phase = np.exp(-2j*np.pi*(p*qp + q*pp)/m).ravel()
        return col, phase

    def carry_natural(self, x):
        """C=FG*: controlled fractional circular shift; exact, not cheap by fiat."""
        m = self.m
        v = np.asarray(x).reshape(m, m)  # rows v, columns u
        modes = np.fft.ifft(v, axis=0, norm="ortho")
        t, u = np.indices((m,m))
        modes *= np.exp(-2j*np.pi*t*u/(m*m))
        return np.fft.fft(modes, axis=0, norm="ortho").ravel()

    def carry(self, z):
        return self.forward(self.carry_natural(self.inverse(z)))

    def carry_blocks(self, z):
        """Carry without conversion to natural coordinates; independent b² blocks.

        Input/output layout remains (q,p,alpha,beta). NumPy's axis transforms
        are a reference implementation, not an assertion of free movement.
        """
        b, a, m = self.b, self.a, self.m
        q, p, alpha, beta = np.indices((b,a,b,a))
        tau = (-beta) % a
        edge = np.exp(-2j*np.pi*tau*q/m)
        v = np.asarray(z).reshape(b,a,b,a) * edge.conj()
        v = np.fft.ifftn(v, axes=(0,2), norm="ortho")
        # q and alpha now denote dual indices h and j, respectively.
        v *= np.exp(-2j*np.pi*(tau+a*q)*(p+a*alpha)/(m*m))
        return (edge*np.fft.fftn(v, axes=(0,2), norm="ortho")).ravel()

    def transform_blocks(self, x):
        col, phase = self.monomial()
        return self.inverse(self.carry_blocks(phase*self.forward(x)[col]))

    def transform_cancelled(self, x):
        """Both adjacent b-FFT pairs cancelled algebraically; no N-DFT call.

        Unnormalized transforms, ordinary (unnormalized) forward DFT output.
        This exposes conventional row/column Fourier work in the packet route.
        """
        a,b,m=self.a,self.b,self.m
        work=np.asarray(x)[self.pack].reshape(b,a,b,a)
        work=np.fft.fft(work,axis=3)
        p,beta,h,alpha=np.indices((a,a,b,b))
        tau=(-beta)%a
        qp=(-alpha-self.slope*tau)%b
        v=work[qp,tau,h,p]*np.exp(-2j*np.pi*p*(qp/m+self.slope*h/b))
        v=np.fft.ifft(v,axis=3)*b
        v*=np.exp(-2j*np.pi*(tau+a*h)*(p+a*alpha)/(m*m))
        v=np.fft.fft(v,axis=2)
        v*=np.exp(-2j*np.pi*tau*h/m)
        v=v.transpose(2,0,3,1)
        v=np.fft.ifft(v,axis=3)*a
        return v.ravel()[self.unpack]

    def transform(self, x):
        col, phase = self.monomial()
        return self.inverse(self.carry(phase*self.forward(x)[col]))

    def matrix(self):
        """Q; only for small-size independent matrix verification."""
        return np.column_stack([self.forward(x) for x in np.eye(self.n)])


def direct_matrices(m):
    n = m*m
    t = np.arange(n) % m
    f = np.arange(n) // m
    F = np.exp(-2j*np.pi*((np.outer(np.arange(n),np.arange(n))) % n)/n)/m
    G = np.exp(-2j*np.pi*((np.outer(t,f)+np.outer(f,t)) % m)/m)/m
    return F, G


def components(mask):
    remaining = set(range(len(mask)))
    result = []
    while remaining:
        seed = min(remaining)
        seen, frontier = {seed}, [seed]
        while frontier:
            i = frontier.pop()
            new = set(np.flatnonzero(mask[i] | mask[:, i])) - seen
            seen.update(new)
            frontier.extend(new)
        remaining.difference_update(seen)
        result.append(sorted(seen))
    return result
