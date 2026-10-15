"""MOODS-exact PWM scanning on the GPU (torch).

:func:`moods_any_hit` returns, for every sequence and motif, whether MOODS
(``MOODS.scan.Scanner(7).scan``, as called by :func:`pychromvar.match_motif`)
reports at least one hit on either strand. It reproduces MOODS 1.9 exactly
rather than approximately:

* sequences are split into maximal runs of ``ACGTacgt``; every other
  character is a separator, and no hit overlaps one;
* scores are float64 sums taken in MOODS's order: matrix columns ``0..m-1``
  for matrices of length ``m <= 7`` (the window size); for longer matrices
  the 7-column window at ``window_position`` first, then the remaining
  columns in MOODS's look-ahead order, with MOODS's early-termination test
  ``score + lookahead[i] < threshold`` applied before every look-ahead step;
* a hit is ``score >= threshold``;
* MOODS never reports a matrix shorter than the window whose last base is
  the last base of an ACGT run (the tail loop of ``Scanner::process_matches``
  uses ``size < end - i``). This is reproduced.

Runs shorter than the window go through a separate branch of
``process_matches`` whose result depends on the whole motif set. Sequences
that contain such a run are not emulated: they are scanned by MOODS itself on
the CPU, so the combined result stays exact.

The log-odds matrices, their reverse complements and the thresholds are
computed by ``MOODS.tools`` exactly as pychromvar does, so MOODS remains a
dependency of this path (it is cheap: it runs once per motif, not per peak).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

WINDOW = 7  # pychromvar: MOODS.scan.Scanner(7)
_DELTA_SCALE = 1.0  # tests raise it to route every window through the exact stage

_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


# ---------------------------------------------------------------------------
# MOODS Motif0 preprocessing, transcribed from motif_0.cpp (double precision)
# ---------------------------------------------------------------------------

def _expected_differences(mat: np.ndarray, bg: Sequence[float]) -> List[float]:
    a, m = mat.shape
    ret = []
    for i in range(m):
        mx = -np.inf
        for j in range(a):
            mx = max(mx, float(mat[j, i]))
        r = mx
        for j in range(a):
            r -= float(bg[j]) * float(mat[j, i])
        ret.append(r)
    return ret


def _window_position(ed: List[float], l: int, m: int) -> int:
    if l >= m:
        return 0
    current = 0.0
    for i in range(l):
        current += ed[i]
    mx = current
    wp = 0
    for i in range(m - l):
        current -= ed[i]
        current += ed[i + l]
        if current > mx:
            mx = current
            wp = i + 1
    return wp


def _lookahead_order(ed: List[float], l: int, wp: int, m: int) -> List[int]:
    if l >= m:
        return []
    order = list(range(wp)) + list(range(wp + l, m))
    # std::sort with comp(i, j) = ed[i] > ed[j]. For the <= 16 elements of
    # all but the longest motifs libstdc++ sorts by insertion sort, which is
    # stable; Python's sort is stable too.
    return sorted(order, key=lambda i: -ed[i])


def _lookahead_scores(mat: np.ndarray, order: List[int], l: int, m: int) -> List[float]:
    if l >= m:
        return []
    a = mat.shape[0]
    scores = [0.0] * (m - l)
    total = 0.0
    for i in range(m - l - 1, -1, -1):
        mx = -np.inf
        for j in range(a):
            mx = max(mx, float(mat[j, order[i]]))
        total += mx
        scores[i] = total
    return scores


class _Matrix:
    __slots__ = ("mat", "m", "T", "steps", "la", "need")

    def __init__(self, mat, bg, threshold):
        self.mat = np.asarray(mat, dtype=np.float64)       # (4, m)
        self.m = self.mat.shape[1]
        self.T = float(threshold)
        if self.m <= WINDOW:
            self.steps = list(range(self.m))
            self.la = []
            # middle loop covers starts <= end - 7; the tail loop starts with
            # size < end - i. Together: start <= end - (m + 1) for m < 7.
            self.need = self.m if self.m == WINDOW else self.m + 1
        else:
            ed = _expected_differences(self.mat, bg)
            wp = _window_position(ed, WINDOW, self.m)
            order = _lookahead_order(ed, WINDOW, wp, self.m)
            self.steps = list(range(wp, wp + WINDOW)) + order
            self.la = _lookahead_scores(self.mat, order, WINDOW, self.m)
            self.need = self.m


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------

def _encode(seqs: Sequence[str]):
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    L = int(lens.max()) if len(seqs) else 0
    codes = np.full((len(seqs), L), 4, dtype=np.uint8)
    for i, s in enumerate(seqs):
        if lens[i]:
            codes[i, : lens[i]] = _LUT[np.frombuffer(s.encode("latin-1"), dtype=np.uint8)]
    return codes, lens


def _run_remaining(codes: np.ndarray) -> np.ndarray:
    """``rem[b, p]`` = length of the ACGT run starting at ``p`` (0 if
    ``p`` is a separator)."""
    B, L = codes.shape
    bad = codes >= 4
    pos = np.where(bad, np.arange(L)[None, :], L)
    next_bad = np.minimum.accumulate(pos[:, ::-1], axis=1)[:, ::-1]
    return (next_bad - np.arange(L)[None, :]).astype(np.int32)


def _has_short_run(codes: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """True for sequences with an ACGT run shorter than the window."""
    good = codes < 4
    B, L = codes.shape
    # run starts: good and (p == 0 or previous bad)
    prev_bad = np.ones_like(good)
    prev_bad[:, 1:] = ~good[:, :-1]
    starts = good & prev_bad
    rem = _run_remaining(codes)
    short = starts & (rem < WINDOW)
    return short.any(axis=1)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def moods_any_hit(
    seqs: Sequence[str],
    matrices: Sequence,
    thresholds: Sequence[float],
    bg: Sequence[float],
    n_motifs: int,
    *,
    dev,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """``(n_seqs, n_motifs)`` uint8: 1 where MOODS reports >= 1 hit of matrix
    ``j`` or of its reverse complement ``j + n_motifs``.

    ``matrices`` / ``thresholds`` are the ``2 * n_motifs`` MOODS log-odds
    matrices (forward then reverse complement) and thresholds, in the layout
    pychromvar passes to ``Scanner.set_motifs``.
    """
    from ..core._gpu import get_torch, memory_budget

    torch = get_torch()
    n_seqs = len(seqs)
    out = np.zeros((n_seqs, n_motifs), dtype=np.uint8)
    if n_seqs == 0:
        return out

    mats = [_Matrix(m, bg, t) for m, t in zip(matrices, thresholds)]
    codes, lens = _encode(seqs)
    fallback = _has_short_run(codes, lens) | (lens < WINDOW)
    fb_idx = np.where(fallback)[0]
    gpu_idx = np.where(~fallback)[0]

    if fb_idx.size:
        import MOODS.scan
        scanner = MOODS.scan.Scanner(WINDOW)
        scanner.set_motifs(matrices=list(matrices), bg=list(bg),
                           thresholds=list(thresholds))
        for i in fb_idx:
            res = scanner.scan(seqs[i])
            for j in range(n_motifs):
                if len(res[j]) > 0 or len(res[j + n_motifs]) > 0:
                    out[i, j] = 1

    if gpu_idx.size == 0:
        return out

    # Group matrices by length; each group is scanned in one batched pass.
    groups = {}
    for k, mt in enumerate(mats):
        groups.setdefault(mt.m, []).append(k)

    codes_g = codes[gpu_idx]
    L = codes_g.shape[1]
    rem = _run_remaining(codes_g)
    codes_t = torch.as_tensor(codes_g.astype(np.int64), device=dev)
    rem_t = torch.as_tensor(rem, device=dev)

    if batch_size is None:
        budget = memory_budget(dev)
        kmax = max(len(v) for v in groups.values())
        # float32 scores + masks of (B, K, P), the one-hot input and scratch
        batch_size = max(1, int(budget // (16 * kmax * max(L, 1) + 64 * L)))
    hit = torch.zeros((gpu_idx.size, 2 * n_motifs), dtype=torch.bool, device=dev)
    n_amb = 0

    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False      # exact float32 products
    try:
        for m, ks in sorted(groups.items()):
            if m > L:
                continue
            g = _Group([mats[k] for k in ks], dev, torch)
            ks_t = torch.as_tensor(ks, device=dev)
            for b0 in range(0, gpu_idx.size, batch_size):
                b1 = min(b0 + batch_size, gpu_idx.size)
                C = codes_t[b0:b1]
                h, na = g.scan(C, rem_t[b0:b1], torch)
                n_amb += na
                hit[b0:b1, ks_t] |= h
            if verbose:
                print(f"  └─ [motif_scan] length {m}: {len(ks)} matrices", flush=True)
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32
    if verbose:
        print(f"  └─ [motif_scan] {n_amb:,} near-threshold windows rescored "
              f"exactly", flush=True)

    hit = hit.cpu().numpy()
    out[gpu_idx] = (hit[:, :n_motifs] | hit[:, n_motifs:]).astype(np.uint8)
    return out


class _Group:
    """Matrices of one length ``m``, scanned together.

    Two stages keep the result exact while doing almost all work in fast
    float32 convolutions:

    1. ``conv1d`` of the one-hot sequence with the float32 matrices gives a
       score ``s32`` whose error is bounded by ``delta`` (a generous multiple
       of the float32 rounding bound of an ``m``-term sum). Windows with
       ``s32 >= T + delta`` are hits and windows with ``s32 < T - delta`` are
       not, whatever the order of the float64 sum.
    2. The few windows with ``|s32 - T| < delta`` are rescored exactly as
       MOODS does (float64, MOODS's column order and early-termination test).
    """

    def __init__(self, mats, dev, torch):
        self.m = m = mats[0].m
        self.K = len(mats)
        self.long = m > WINDOW
        W = np.stack([mt.mat for mt in mats])                       # (K, 4, m)
        self.W32 = torch.as_tensor(W.astype(np.float32), device=dev)
        W5 = np.zeros((self.K, m, 5), dtype=np.float64)
        W5[:, :, :4] = np.transpose(W, (0, 2, 1))
        self.W5 = torch.as_tensor(W5, device=dev)
        self.T = torch.as_tensor([mt.T for mt in mats], dtype=torch.float64, device=dev)
        self.T32 = self.T.to(torch.float32)
        amax = np.abs(W).max(axis=1).sum(axis=1)                    # sum_j max_c |w|
        delta = (m * 2.0 ** -20 * amax + 1e-6) * _DELTA_SCALE
        self.delta = torch.as_tensor(delta.astype(np.float32), device=dev)
        self.need = torch.as_tensor([mt.need for mt in mats], device=dev)
        self.steps = torch.as_tensor([mt.steps for mt in mats], device=dev)   # (K, m)
        if self.long:
            self.LA = torch.as_tensor([mt.la for mt in mats], dtype=torch.float64, device=dev)

    def scan(self, C, rem, torch):
        """``C`` (B, L) int64 codes, ``rem`` (B, L) run lengths ->
        ((B, K) bool any-hit, number of rescored windows)."""
        import torch.nn.functional as F
        B, L = C.shape
        P = L - self.m + 1
        onehot = F.one_hot(C, 5)[:, :, :4].permute(0, 2, 1).to(torch.float32)
        s32 = F.conv1d(onehot, self.W32)                             # (B, K, P)
        del onehot
        valid = rem[:, None, :P] >= self.need[None, :, None]
        T = self.T32[None, :, None]
        d = self.delta[None, :, None]
        hit = ((s32 >= T + d) & valid).any(dim=2)                    # (B, K)
        amb = valid & ((s32 - T).abs() < d) & ~hit[:, :, None]
        del s32, valid
        idx = amb.nonzero()
        del amb
        if idx.shape[0]:
            b, k, p = idx[:, 0], idx[:, 1], idx[:, 2]
            ok = self._exact(C, b, k, p, torch)
            hit[b[ok], k[ok]] = True
        return hit, int(idx.shape[0])

    def _exact(self, C, b, k, p, torch):
        """MOODS's float64 score of windows (b, k, p) and its hit decision."""
        score = torch.zeros(b.shape[0], dtype=torch.float64, device=C.device)
        alive = torch.ones(b.shape[0], dtype=torch.bool, device=C.device)
        T = self.T[k]
        for t in range(self.m):
            if self.long and t >= WINDOW:
                # check_hit: stop when score + lookahead[i] < T
                alive &= (score + self.LA[k, t - WINDOW]) >= T
            col = self.steps[k, t]
            code = C[b, p + col]
            score += self.W5[k, col, code]
            if self.long and t == WINDOW - 1:
                # window_match: score_w + lookahead_scores[0] >= T
                alive &= (score + self.LA[k, 0]) >= T
        return alive & (score >= T)
