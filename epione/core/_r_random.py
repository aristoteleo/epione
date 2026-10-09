"""Bit-exact emulation of R's default RNG (Mersenne-Twister + Rejection sampling).

Reproduces ``set.seed(seed); sample(x, size)`` / ``runif`` from R >= 3.6 with the
default ``RNGkind("Mersenne-Twister", "Inversion", "Rejection")`` so ports of R
code that sample (ArchR ``addModuleScore``, ``plotGroups(maxCells=)``) draw the
very same elements as the R reference. Algorithm follows R's ``src/main/RNG.c``.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["RRandom"]

_MASK32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)


class RRandom:
    """R-compatible random stream.

    ``RRandom(1).sample(x, k)`` returns the same elements, in the same order, as
    ``set.seed(1); sample(x, k)`` in R. Successive calls continue the stream,
    exactly like successive R calls after a single ``set.seed``.
    """

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        s = int(seed) & _MASK32
        for _ in range(50):
            s = (69069 * s + 1) & _MASK32
        key = np.empty(625, dtype=np.uint64)
        for j in range(625):
            s = (69069 * s + 1) & _MASK32
            key[j] = s
        # key[0] is R's mti slot (FixupSeeds sets it to 624 -> regenerate on first draw)
        self._bitgen = np.random.MT19937()
        self._bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": key[1:].astype(np.uint32), "pos": 624},
        }

    def unif_rand(self) -> float:
        value = float(self._bitgen.random_raw()) * 2.3283064365386963e-10
        if value <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - value <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return value

    def runif(self, n: int) -> np.ndarray:
        return np.array([self.unif_rand() for _ in range(int(n))])

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """R's ``R_unif_index`` (rejection sampling, R >= 3.6)."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int | None = None, replace: bool = False) -> np.ndarray:
        """0-based equivalent of R ``sample.int(n, size, replace)``."""
        n = int(n)
        size = n if size is None else int(size)
        if replace or size < 2:
            return np.array([self.unif_index(n) for _ in range(size)], dtype=np.int64)
        if size > n:
            raise ValueError("cannot take a sample larger than the population when 'replace = FALSE'")
        pool = list(range(n))
        out = np.empty(size, dtype=np.int64)
        m = n
        for i in range(size):
            j = self.unif_index(m)
            out[i] = pool[j]
            m -= 1
            pool[j] = pool[m]
        return out

    def sample(self, x: Sequence, size: int | None = None, replace: bool = False):
        """R ``sample(x, size)`` for a vector ``x`` of length >= 2."""
        arr = np.asarray(x)
        return arr[self.sample_int(len(arr), size, replace)]
