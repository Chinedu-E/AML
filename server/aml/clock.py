"""Seedable random source and simulated clock."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class SimClock:
    """Monotonic simulated time in seconds since epoch start."""

    now: float = 0.0

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("cannot move the simulated clock backwards")
        self.now += seconds
        return self.now

    def set(self, timestamp: float) -> float:
        if timestamp < self.now:
            raise ValueError("cannot set the simulated clock to an earlier time")
        self.now = timestamp
        return self.now


@dataclass
class SimRng:
    """Wrapper around random.Random for deterministic simulation draws."""

    seed: int = 42
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def choice[T](self, seq: list[T] | tuple[T, ...]) -> T:
        return self._rng.choice(seq)

    def sample[T](self, population: list[T] | tuple[T, ...], k: int) -> list[T]:
        return self._rng.sample(list(population), k)

    def shuffle[T](self, xs: list[T]) -> None:
        self._rng.shuffle(xs)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._rng.gauss(mu, sigma)

    def lognormal(self, mu: float, sigma: float) -> float:
        return self._rng.lognormvariate(mu, sigma)

    def expovariate(self, lambd: float) -> float:
        if lambd <= 0:
            raise ValueError("lambd must be positive")
        return self._rng.expovariate(lambd)

    def zipf_index(self, n: int, alpha: float = 1.2) -> int:
        """Return an index in [0, n) with Zipf-like preference for low ranks."""
        if n <= 0:
            raise ValueError("n must be positive")
        weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return n - 1

    def poisson_interarrival(self, rate_per_second: float) -> float:
        """Seconds until the next event of a Poisson process with the given rate."""
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        return self.expovariate(rate_per_second)

    def round_amount(self, amount: float, places: int = 2) -> float:
        factor = 10**places
        return math.floor(amount * factor + 0.5) / factor
