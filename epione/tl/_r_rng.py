"""Bit-exact emulation of R's default RNG (Mersenne-Twister + Rejection
sampling, R >= 3.6) for ``set.seed(seed); sample.int(n, size)``.

Lets ArchR ports reproduce R's random draws (e.g. the seed cells of
``correlateMatrices``) without calling R.
"""
from __future__ import annotations

import math

import numpy as np

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)


_M1 = 4294967087
_M2 = 4294944443


def _ctrunc_div(a: int, b: int) -> int:
    q = abs(a) // b
    return q if a >= 0 else -q


class RRandom:
    """R's RNG seeded like ``set.seed(seed)``.

    ``kind`` is ``"Mersenne-Twister"`` (R's default) or ``"L'Ecuyer-CMRG"``
    (what ``ArchR::addArchRThreads(threads > 1)`` switches the session to).
    """

    def __init__(self, seed: int, kind: str = "Mersenne-Twister"):
        if kind not in ("Mersenne-Twister", "L'Ecuyer-CMRG"):
            raise ValueError(f"unsupported RNG kind {kind!r}")
        self.kind = kind
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):  # RNG_Init(): initial scrambling
            s = (69069 * s + 1) & 0xFFFFFFFF
        if kind == "L'Ecuyer-CMRG":
            words = []
            for _ in range(6):
                s = (69069 * s + 1) & 0xFFFFFFFF
                while s >= _M2:
                    s = (69069 * s + 1) & 0xFFFFFFFF
                words.append(s)
            self._ii = words
            return
        words = []
        for _ in range(_N + 1):  # fill dummy[0..624]
            s = (69069 * s + 1) & 0xFFFFFFFF
            words.append(s)
        self._mt = np.array(words[1:], dtype=np.uint64)
        self._mti = _N  # FixupSeeds(): dummy[0] = mti = N

    def _lecuyer(self) -> float:
        ii = self._ii
        p1 = 1403580 * ii[1] - 810728 * ii[0]
        p1 -= _ctrunc_div(p1, _M1) * _M1
        if p1 < 0:
            p1 += _M1
        ii[0], ii[1], ii[2] = ii[1], ii[2], p1
        p2 = 527612 * ii[5] - 1370589 * ii[3]
        p2 -= _ctrunc_div(p2, _M2) * _M2
        if p2 < 0:
            p2 += _M2
        ii[3], ii[4], ii[5] = ii[4], ii[5], p2
        return ((p1 - p2) if p1 > p2 else (p1 - p2 + _M1)) * 2.328306549295727688e-10

    def _genrand(self) -> float:
        if self.kind == "L'Ecuyer-CMRG":
            return self._lecuyer()
        mt = self._mt
        if self._mti >= _N:
            for kk in range(_N):
                y = (int(mt[kk]) & _UPPER) | (int(mt[(kk + 1) % _N]) & _LOWER)
                v = int(mt[(kk + _M) % _N]) ^ (y >> 1)
                if y & 1:
                    v ^= _MATRIX_A
                mt[kk] = v
            self._mti = 0
        y = int(mt[self._mti])
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return (y & 0xFFFFFFFF) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        v = self._genrand()
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int, replace: bool = False) -> np.ndarray:
        """``sample.int(n, size, replace)``; returns 1-based indices like R."""
        if replace:
            return np.array([self.unif_index(n) + 1 for _ in range(size)], dtype=np.int64)
        if size > n:
            raise ValueError("cannot take a sample larger than the population when replace=False")
        x = list(range(n))
        out = np.empty(size, dtype=np.int64)
        m = n
        for i in range(size):
            j = self.unif_index(m)
            out[i] = x[j] + 1
            m -= 1
            x[j] = x[m]
        return out
