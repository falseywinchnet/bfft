"""Report the initial normalized Frobenius phase chamber."""

from __future__ import annotations

from math import sqrt

from mellin_current import prime_sieve, trace_at_prime
from run_probe import CONTROL, RECORD


def phase_chamber(curve, limit: int = 500) -> dict[str, object]:
    primes, _ = prime_sieve(limit)
    rows = [
        (prime, trace_at_prime(curve, prime))
        for prime in primes
        if curve.conductor % prime != 0
    ]
    first_positive = next((prime for prime, trace in rows if trace > 0), None)
    negative_prefix = []
    for prime, trace in rows:
        if trace >= 0:
            break
        negative_prefix.append((prime, trace))
    phase_values = [
        trace / (2.0 * sqrt(prime)) for prime, trace in negative_prefix
    ]
    return {
        "curve": curve.name,
        "negative_prefix_last_prime": negative_prefix[-1][0] if negative_prefix else None,
        "first_positive_prime": first_positive,
        "prefix_count": len(negative_prefix),
        "prefix_mean_phase": sum(phase_values) / len(phase_values),
        "prefix_weighted_phase": (
            sum(trace for prime, trace in negative_prefix)
            / sum(2.0 * sqrt(prime) for prime, trace in negative_prefix)
        ),
    }


def main() -> None:
    for curve in (RECORD, CONTROL):
        print(phase_chamber(curve))


if __name__ == "__main__":
    main()
