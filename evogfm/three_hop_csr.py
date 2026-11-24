"""Exact three-hop walk counts at requested entries of an unweighted CSR graph."""

import numpy as np
from numba import njit


@njit(cache=True)
def overlap(indices, lo, hi, other_lo, other_hi):
    if hi - lo > other_hi - other_lo:
        lo, hi, other_lo, other_hi = other_lo, other_hi, lo, hi
    count = 0
    if 8 * (hi - lo) < other_hi - other_lo:
        for i in range(lo, hi):
            value = indices[i]
            position = other_lo + np.searchsorted(indices[other_lo:other_hi], value)
            if position < other_hi and indices[position] == value:
                count += 1
        return count
    while lo < hi and other_lo < other_hi:
        left, right = indices[lo], indices[other_lo]
        if left == right:
            count += 1
        if left <= right:
            lo += 1
        if left >= right:
            other_lo += 1
    return count


@njit(cache=True)
def three_hop_walk_counts(indptr, indices, left, right):
    result = np.zeros(len(left), dtype=np.float32)
    for i in range(len(left)):
        u, v = left[i], right[i]
        ulo, uhi = indptr[u], indptr[u + 1]
        vlo, vhi = indptr[v], indptr[v + 1]
        if uhi - ulo > vhi - vlo:
            ulo, uhi, vlo, vhi = vlo, vhi, ulo, uhi
        count = 0
        for j in range(ulo, uhi):
            mid = indices[j]
            count += overlap(indices, indptr[mid], indptr[mid + 1], vlo, vhi)
        result[i] = count
    return result
