"""Deterministic similarity graphs and community-comparison primitives."""

from __future__ import annotations

import igraph as ig
import leidenalg as la
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score


def similarity_matrix(profiles: np.ndarray, geometry: str) -> np.ndarray:
    """Construct similarities without learning across months."""
    p = np.asarray(profiles, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("profiles must be a finite nonnegative matrix")
    if not np.allclose(p.sum(axis=1), 1):
        raise ValueError("profile rows must sum to one")
    if geometry == "hellinger":
        x = np.sqrt(p)
    elif geometry == "cosine":
        x = p / np.linalg.norm(p, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    similarities = np.clip(x @ x.T, 0, 1)
    np.fill_diagonal(similarities, 0)
    return similarities


def knn_graph(similarities: np.ndarray, k: int) -> sparse.csr_matrix:
    """Union kNN, max weights, deterministic ties by sorted input row index."""
    n = len(similarities)
    if similarities.shape != (n, n) or not 1 <= k < n:
        raise ValueError("Need a square matrix and 1 <= k < n")
    ranked = similarities.copy()
    np.fill_diagonal(ranked, -np.inf)
    neighbors = np.argsort(-ranked, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k)
    columns = neighbors.ravel()
    weights = similarities[rows, columns]
    directed = sparse.csr_matrix((weights, (rows, columns)), shape=(n, n))
    result = directed.maximum(directed.T).tocsr()
    result.setdiag(0)
    result.eliminate_zeros()
    return result


def equal_edge_budget_graph(graph: sparse.csr_matrix, budget: int) -> sparse.csr_matrix:
    """Retain the strongest undirected edges, with deterministic index ties.

    This control changes the kNN topology; isolate counts are therefore reported.
    """
    upper = sparse.triu(graph, k=1).tocoo()
    if budget < 1 or budget > len(upper.data):
        raise ValueError("Invalid undirected edge budget")
    order = np.lexsort((upper.col, upper.row, -upper.data))[:budget]
    one_way = sparse.csr_matrix(
        (upper.data[order], (upper.row[order], upper.col[order])), shape=graph.shape
    )
    return (one_way + one_way.T).tocsr()


def leiden_partition(graph: sparse.csr_matrix, resolution: float, seed: int) -> np.ndarray:
    upper = sparse.triu(graph, k=1).tocoo()
    native = ig.Graph(n=graph.shape[0], edges=list(zip(upper.row, upper.col, strict=True)))
    partition = la.find_partition(
        native,
        la.RBConfigurationVertexPartition,
        weights=upper.data.tolist(),
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )
    return np.asarray(partition.membership, dtype=int)


def graph_statistics(graph: sparse.csr_matrix) -> dict[str, float | int]:
    degrees = np.diff(graph.indptr)
    n = graph.shape[0]
    components, labels = connected_components(graph, directed=False)
    return {
        "n_edges": graph.nnz // 2,
        "density": graph.nnz / (n * (n - 1)),
        "degree_min": int(degrees.min()),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(degrees.max()),
        "n_components": components,
        "largest_component_share": float(np.bincount(labels).max() / n),
        "n_isolates": int(np.sum(degrees == 0)),
    }


def align_labels(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Maximum overlap matching. Unmatched candidate groups get new negative IDs."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) == 0:
        raise ValueError("Label arrays must be nonempty, one-dimensional and same length")
    # Canonical group order by first member, independent of arbitrary label values.
    ref_values = reference[np.sort(np.unique(reference, return_index=True)[1])]
    cand_values = candidate[np.sort(np.unique(candidate, return_index=True)[1])]
    ref_lookup = {value: i for i, value in enumerate(ref_values)}
    cand_lookup = {value: i for i, value in enumerate(cand_values)}
    ref_codes = np.array([ref_lookup[v] for v in reference])
    cand_codes = np.array([cand_lookup[v] for v in candidate])
    overlap = np.zeros((len(ref_values), len(cand_values)), dtype=np.int64)
    np.add.at(overlap, (ref_codes, cand_codes), 1)
    ref_index, cand_index = linear_sum_assignment(-overlap)
    used = set(int(v) for v in ref_values)
    missing = min(-1, min(used) - 1)
    mapping = np.empty(len(cand_values), dtype=np.int64)
    for i in range(len(cand_values)):
        mapping[i] = missing
        missing -= 1
    for r, c in zip(ref_index, cand_index, strict=True):
        if overlap[r, c] > 0:
            mapping[c] = int(ref_values[r])
    return mapping[cand_codes]


def compare_partitions(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    aligned = align_labels(reference, candidate)
    return {
        "ari_to_primary": float(adjusted_rand_score(reference, candidate)),
        "aligned_agreement": float(np.mean(reference == aligned)),
        "candidate_n_clusters": int(len(np.unique(candidate))),
    }


def neighbor_jaccard(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    a, b = left.astype(bool).astype(np.int8), right.astype(bool).astype(np.int8)
    intersection = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    union = np.asarray((a + b).astype(bool).sum(axis=1)).ravel()
    return np.divide(intersection, union, out=np.full(len(union), np.nan), where=union > 0)


def co_membership_jaccard(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Jaccard of each node's co-cluster peers, excluding itself; label invariant.

    Two empty peer sets (singleton in both partitions) have similarity1 by convention.
    Singleton guards are assessed separately, so this does not make trivial partitions good.
    """
    if np.asarray(reference).shape != np.asarray(candidate).shape:
        raise ValueError("Partition lengths differ")
    _, a, counts_a = np.unique(reference, return_inverse=True, return_counts=True)
    _, b, counts_b = np.unique(candidate, return_inverse=True, return_counts=True)
    overlap = np.zeros((len(counts_a), len(counts_b)), dtype=int)
    np.add.at(overlap, (a, b), 1)
    intersection = overlap[a, b] - 1
    union = counts_a[a] + counts_b[b] - overlap[a, b] - 1
    return np.divide(intersection, union, out=np.ones(len(a), dtype=float), where=union > 0)
