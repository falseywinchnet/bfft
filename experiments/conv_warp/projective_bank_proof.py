"""High-precision reference for the addendum's finite projective edge bank.

Coordinates and density coefficients may be decimal strings. The caller sets
mpmath precision; this is an algebraic reference for numerical verification.
"""
from math import comb
import mpmath as mp


def power_integral(exponent, left, right):
    h = (right-left)/left
    if exponent == -1:
        return mp.log1p(h)
    n = exponent+1
    return left**n * mp.expm1(n*mp.log1p(h))/n


def oriented(vertices):
    area2 = sum(x*v-y*u for (x,y),(u,v) in zip(vertices, vertices[1:]+vertices[:1]))
    return vertices if area2 >= 0 else list(reversed(vertices))


def polynomial_moment(vertices, ax, ay):
    result = mp.mpf(0)
    for (x,y),(u,v) in zip(vertices, vertices[1:]+vertices[:1]):
        dx,dy=u-x,v-y
        for i in range(ax+2):
            for j in range(ay+1):
                result += (mp.mpf(comb(ax+1,i)*comb(ay,j))*x**(ax+1-i)*dx**i
                           * y**(ay-j)*dy**j *dy/((ax+1)*(i+j+1)))
    return result


def monomial_bank(vertices, gamma, alpha, beta, degree=10):
    """Return integrals x^a y^b/(gamma+alpha*x+beta*y)^3, a+b<=degree."""
    vertices=oriented([(mp.mpf(x),mp.mpf(y)) for x,y in vertices])
    gamma,alpha,beta=map(mp.mpf,(gamma,alpha,beta))
    if min(gamma+alpha*x+beta*y for x,y in vertices) <= 0:
        raise ValueError('The normalized denominator must be positive.')
    if alpha == beta == 0:
        return {(a,b):polynomial_moment(vertices,a,b)/gamma**3
                for a in range(degree+1) for b in range(degree+1-a)}
    swapped=abs(beta)>abs(alpha)
    if swapped:
        vertices=[(y,x) for x,y in vertices]
        alpha,beta=beta,alpha
    transformed=oriented([(gamma+alpha*x+beta*y,y) for x,y in vertices])
    edge_moments={}
    for k in range(degree+1):
        for l in range(degree+1-k):
            value=mp.mpf(0)
            for (t0,y0),(t1,y1) in zip(transformed,transformed[1:]+transformed[:1]):
                if t0 == t1:
                    continue
                slope=(y1-y0)/(t1-t0)
                intercept=y0-slope*t0
                value -= mp.fsum(comb(l+1,j)*slope**j*intercept**(l+1-j)
                                 *power_integral(k-3+j,t0,t1)
                                 for j in range(l+2))/(l+1)
            edge_moments[k,l]=value
    result={}
    for a in range(degree+1):
        for b in range(degree+1-a):
            value=mp.fsum(comb(a,k)*comb(a-k,j)*(-gamma)**(a-k-j)*(-beta)**j
                          *edge_moments[k,b+j] for k in range(a+1) for j in range(a-k+1))
            result[(b,a) if swapped else (a,b)]=value/(alpha**a*abs(alpha))
    return result


def quintic_bank(vertices, gamma, alpha, beta):
    moments=monomial_bank(vertices,gamma,alpha,beta)
    return [[mp.fsum(comb(5,i)*comb(5-i,a-i)*(-1)**(a-i)
                    *comb(5,j)*comb(5-j,b-j)*(-1)**(b-j)*moments[a,b]
                    for a in range(i,6) for b in range(j,6))
             for j in range(6)] for i in range(6)]
