"""Minimal linear algebra over plain Python lists — no external dependencies.

Used by the CBOW trainer and the PCA projection.  Vectors are ``list[float]``;
matrices are ``list[list[float]]`` (row-major).  Operations are written for
correctness and readability, not speed.
"""

from __future__ import annotations

import math

__all__ = [
    "dot", "add_", "scale_", "norm", "cosine_similarity",
    "mat_row", "mat_add_outer_",
    "mean_vector", "pca2",
]


# ── Vector operations ─────────────────────────────────────────────────────────

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add_(target: list[float], delta: list[float], scale: float = 1.0) -> None:
    """In-place: target += delta * scale."""
    for i in range(len(target)):
        target[i] += delta[i] * scale


def scale_(v: list[float], s: float) -> None:
    """In-place: v *= s."""
    for i in range(len(v)):
        v[i] *= s


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


# ── Matrix operations ─────────────────────────────────────────────────────────

def mat_row(matrix: list[list[float]], i: int) -> list[float]:
    """Return a reference to row i (not a copy — callers may mutate it)."""
    return matrix[i]


def mat_add_outer_(
    matrix: list[list[float]],
    row_idx: int,
    delta: list[float],
    scale: float = 1.0,
) -> None:
    """In-place: matrix[row_idx] += delta * scale."""
    add_(matrix[row_idx], delta, scale)


# ── Aggregation ───────────────────────────────────────────────────────────────

def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return the element-wise mean of a non-empty list of vectors."""
    d = len(vectors[0])
    result = [0.0] * d
    for v in vectors:
        add_(result, v)
    n = len(vectors)
    for i in range(d):
        result[i] /= n
    return result


# ── PCA (2 components via power iteration) ────────────────────────────────────

def pca2(matrix: list[list[float]]) -> list[tuple[float, float]]:
    """Project each row of ``matrix`` onto its top-2 principal components.

    Uses power iteration to find the two dominant eigenvectors of the
    covariance matrix.  Sufficient for visualization; not a general PCA.

    Returns a list of (x, y) pairs, one per row.
    """
    n = len(matrix)
    if n == 0:
        return []
    d = len(matrix[0])

    # Centre the data
    mean = [sum(matrix[r][c] for r in range(n)) / n for c in range(d)]
    centred = [[matrix[r][c] - mean[c] for c in range(d)] for r in range(n)]

    def _power_iter(data: list[list[float]], iters: int = 50) -> list[float]:
        """Return the dominant eigenvector of data^T data via power iteration."""
        # Start from the row with the largest norm
        v = list(max(data, key=lambda row: sum(x * x for x in row)))
        nv = norm(v)
        if nv == 0.0:
            v = [1.0] + [0.0] * (d - 1)
        else:
            scale_(v, 1.0 / nv)
        for _ in range(iters):
            # w = (data^T data) v  computed as data^T (data v)
            # data v: project each row onto v
            projections = [dot(row, v) for row in data]
            # data^T (data v): weighted sum of rows
            w = [0.0] * d
            for proj, row in zip(projections, data):
                add_(w, row, proj)
            nw = norm(w)
            if nw == 0.0:
                break
            scale_(w, 1.0 / nw)
            v = w
        return v

    # First principal component
    pc1 = _power_iter(centred)

    # Deflate: remove the pc1 component from each row
    deflated = [
        [row[c] - dot(row, pc1) * pc1[c] for c in range(d)]
        for row in centred
    ]

    # Second principal component
    pc2 = _power_iter(deflated)

    return [(dot(row, pc1), dot(row, pc2)) for row in centred]
