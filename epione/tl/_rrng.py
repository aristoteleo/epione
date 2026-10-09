"""Bit-exact emulation of R's default random number generator.

Reproduces ``set.seed()`` + ``unif_rand()`` (Mersenne-Twister, R's seed
scrambling and ``fixup``) and ``sample.int()`` without replacement
(``sample.kind = "Rejection"``, R >= 3.6), so ports of R code that draw
random numbers can follow the exact same stream as the R reference.
Ported from R 4.4.3 ``src/main/RNG.c`` and ``src/main/random.c`` (GPL-2+).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

_I2_32M1 = 2.328306437080797e-10
_MT_N = 624


class RRandom:
    """R's Mersenne-Twister stream.

    ``RRandom(seed)`` is equivalent to ``set.seed(seed)`` in R;
    ``RRandom.from_state(.Random.seed)`` resumes an exported R state.
    """

    def __init__(self, seed: Optional[int] = 1):
        self._bg = np.random.MT19937(0)
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        i_seed = np.empty(_MT_N + 1, dtype=np.uint64)
        for j in range(_MT_N + 1):
            s = (69069 * s + 1) & 0xFFFFFFFF
            i_seed[j] = s
        # FixupSeeds(initial=1): i_seed[0] holds mti and is forced to N.
        self._set_mt(i_seed[1:].astype(np.uint32), _MT_N)

    @classmethod
    def from_state(cls, random_seed: Sequence[int]) -> "RRandom":
        """Resume from R's ``.Random.seed`` (626 ints, kind code first)."""
        st = np.asarray(random_seed, dtype=np.int64)
        if st.size != _MT_N + 2 or st[0] % 100 != 3:
            raise ValueError("only R's default Mersenne-Twister .Random.seed is supported")
        obj = cls(None)
        mti = int(st[1])
        if mti <= 0:
            mti = _MT_N
        obj._set_mt((st[2:] & 0xFFFFFFFF).astype(np.uint32), mti)
        return obj

    def _set_mt(self, key: np.ndarray, pos: int) -> None:
        self._bg.state = {"bit_generator": "MT19937",
                          "state": {"key": key, "pos": int(pos)}}

    def state(self) -> np.ndarray:
        """Current state in R's ``.Random.seed`` layout."""
        st = self._bg.state["state"]
        key = st["key"].astype(np.int64)
        key = np.where(key > 0x7FFFFFFF, key - 0x100000000, key)
        return np.concatenate([[10403, st["pos"]], key]).astype(np.int64)

    def unif_rand(self, n: int) -> np.ndarray:
        """``n`` draws of R's ``unif_rand()`` in (0, 1)."""
        y = self._bg.random_raw(int(n)).astype(np.float64)
        u = y * 2.3283064365386963e-10
        u[u <= 0.0] = 0.5 * _I2_32M1
        u[(1.0 - u) <= 0.0] = 1.0 - 0.5 * _I2_32M1
        return u

    def runif(self, n: int, a: float = 0.0, b: float = 1.0) -> np.ndarray:
        return a + (b - a) * self.unif_rand(n)

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand(1)[0] * 65536))
            v = 65536 * v + v1
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: float) -> int:
        """R's ``R_unif_index(dn)`` (rejection sampling)."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int) -> np.ndarray:
        """``sample.int(n, size)`` without replacement, 1-based like R."""
        x = list(range(n))
        out = np.empty(size, dtype=np.int64)
        for i in range(size):
            j = self.unif_index(n)
            out[i] = x[j] + 1
            n -= 1
            x[j] = x[n]
        return out
