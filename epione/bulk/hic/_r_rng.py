"""R's default uniform random number generator, bit for bit.

R's default ``RNGkind`` is ``"Mersenne-Twister"``. ``set.seed(s)`` scrambles
the seed with the linear congruential map ``s -> 69069 * s + 1`` (mod 2**32):
50 times, then 625 more times to fill the state (the first word is the
position, then forced to 624; R-4.1.3 ``src/main/RNG.c``, ``RNG_Init`` /
``FixupSeeds``). ``unif_rand()`` returns ``y * 2.3283064365386963e-10`` for the
next 32-bit Mersenne-Twister output ``y``, nudged into ``(0, 1)`` by ``fixup``.

The Mersenne-Twister recursion and tempering are the standard MT19937 ones,
so the state is handed to :class:`numpy.random.MT19937`. Used by the exact
ports of R code that draws random numbers (e.g. ``rrcov::CovSde``).
"""
from __future__ import annotations

import numpy as np

_MASK = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2**32 - 1)


class RRandom:
    """Uniform draws identical to R after ``set.seed(seed)``.

    Arguments:
        seed: integer passed to R's ``set.seed`` (default kind
            ``"Mersenne-Twister"``).

    Example:
        ``RRandom(21).unif_rand(3)`` equals ``set.seed(21); runif(3)`` in R.
    """

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        s = int(seed) & _MASK
        for _ in range(50):
            s = (69069 * s + 1) & _MASK
        words = np.empty(625, dtype=np.uint64)
        for j in range(625):
            s = (69069 * s + 1) & _MASK
            words[j] = s
        # FixupSeeds(initial=1): dummy[0] (the position) = 624, i.e. the next
        # draw regenerates the whole state from the 624 words that follow.
        self._bitgen = np.random.MT19937()
        self._bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": words[1:].astype(np.uint32), "pos": 624},
        }

    def random_seed(self) -> np.ndarray:
        """R's ``.Random.seed`` for this state (``10403``, position, 624 words)."""
        st = self._bitgen.state["state"]
        words = np.asarray(st["key"], dtype=np.uint32).view(np.int32)
        return np.r_[np.int32(10403), np.int32(st["pos"]), words]

    def get_state(self) -> dict:
        """Snapshot of the generator state (restore with :meth:`set_state`)."""
        return self._bitgen.state

    def set_state(self, state: dict) -> None:
        self._bitgen.state = state

    def unif_rand(self, size: int | None = None):
        """Next uniform(s) in ``(0, 1)``; one draw each, as R's ``unif_rand()``."""
        n = 1 if size is None else int(size)
        y = self._bitgen.random_raw(n).astype(np.float64)
        v = y * 2.3283064365386963e-10
        # fixup(): keep strictly inside (0, 1)
        v = np.where(v <= 0.0, 0.5 * _I2_32M1, v)
        v = np.where(1.0 - v <= 0.0, 1.0 - 0.5 * _I2_32M1, v)
        return float(v[0]) if size is None else v
