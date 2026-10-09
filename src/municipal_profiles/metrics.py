"""Partition diagnostics with explicit metric variants and undefined-value guards.

See ``docs/metrics.md`` for equations, sources and domain conventions. No metric
here supplies independent evidence for the existence of economic communities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, silhouette_score

_METRICS = (
    "silhouette",
    "calinski_harabasz",
    "s_dbw",
    "avi",
    "avu",
    "modularity",
)


def _validate_features_and_labels(
    X: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    raw = np.asarray(X)
    if np.iscomplexobj(raw):
        raise ValueError("X must contain real numbers")
    features = np.asarray(raw, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("X must have shape (n_samples, n_features), n_features > 0")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values")

    assignments = np.asarray(labels)
    if assignments.ndim != 1 or len(assignments) != len(features):
        raise ValueError("labels must be a one-dimensional array aligned with X")
    if np.iscomplexobj(assignments):
        raise ValueError("labels must be real numbers or strings")
    for value in assignments:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            raise ValueError("labels must not contain missing or infinite values")
    try:
        _, inverse, counts = np.unique(assignments, return_inverse=True, return_counts=True)
    except TypeError as error:
        raise ValueError("labels must contain mutually comparable scalar values") from error
    return features, inverse.astype(np.int64), counts.astype(np.int64)


def _validate_adjacency(adjacency: Any, n_samples: int) -> sparse.csr_matrix:
    if sparse.issparse(adjacency):
        if np.iscomplexobj(adjacency):
            raise ValueError("adjacency must contain real weights")
    elif np.iscomplexobj(np.asarray(adjacency)):
        raise ValueError("adjacency must contain real weights")
    matrix = sparse.csr_matrix(adjacency, dtype=np.float64, copy=True)
    if matrix.shape != (n_samples, n_samples):
        raise ValueError("adjacency must have shape (n_samples, n_samples)")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("adjacency weights must be finite and nonnegative")
    if np.any(matrix.diagonal() != 0):
        raise ValueError("adjacency must have a zero diagonal (no self-loops)")
    difference = matrix - matrix.T
    if difference.nnz:
        magnitude = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
        if np.max(np.abs(difference.data)) > 1e-12 + 1e-10 * magnitude:
            raise ValueError("adjacency must be symmetric")
        # Project only tolerated numerical asymmetry; never alter the caller's CSR.
        matrix = ((matrix + matrix.T) * 0.5).tocsr()
    return matrix


def _validate_distances(distances: ArrayLike | None, n_samples: int) -> NDArray[np.float64] | None:
    if distances is None:
        return None
    raw = np.asarray(distances)
    if np.iscomplexobj(raw):
        raise ValueError("distances must contain real values")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (n_samples, n_samples):
        raise ValueError("distances must have shape (n_samples, n_samples)")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("distances must be finite and nonnegative")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("distances must have a zero diagonal")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("distances must be symmetric")
    return matrix


def _s_dbw(
    groups: list[NDArray[np.float64]],
    centroids: NDArray[np.float64],
    global_variance_norm: float,
) -> tuple[float, str]:
    """Halkidi/Vazirgiannis 2002, PDF pp. 69-72: pair-union density variant."""
    if global_variance_norm == 0:
        return float("nan"), "undefined_zero_global_variance"
    variance_norms = np.array([np.linalg.norm(np.var(group, axis=0, ddof=0)) for group in groups])
    n_clusters = len(groups)
    radius = float(np.sqrt(variance_norms.sum()) / n_clusters)
    if not np.isfinite(radius):
        return float("nan"), "undefined_nonfinite_radius"
    if radius == 0:
        return float("nan"), "undefined_zero_density_radius"

    scattering = float(variance_norms.mean() / global_variance_norm)
    pair_sum = 0.0
    for first in range(n_clusters):
        for second in range(first + 1, n_clusters):
            # The source defines density(u) over C_first union C_second.
            # Its support stays the same for both centroids and their midpoint.
            union = np.concatenate((groups[first], groups[second]), axis=0)
            first_density = np.count_nonzero(
                np.linalg.norm(union - centroids[first], axis=1) <= radius
            )
            second_density = np.count_nonzero(
                np.linalg.norm(union - centroids[second], axis=1) <= radius
            )
            denominator = max(first_density, second_density)
            if denominator == 0:
                return float("nan"), "undefined_zero_density_denominator"
            midpoint = (centroids[first] + centroids[second]) * 0.5
            middle_density = np.count_nonzero(np.linalg.norm(union - midpoint, axis=1) <= radius)
            pair_sum += middle_density / denominator

    # Symmetric unordered pairs replace the source's ordered-pair sum.
    between_density = 2.0 * pair_sum / (n_clusters * (n_clusters - 1))
    result = scattering + between_density
    if not np.isfinite(result):
        return float("nan"), "undefined_nonfinite_result"
    return float(result), "ok"


def _graph_metrics(
    adjacency: sparse.csr_matrix,
    labels: NDArray[np.int64],
    n_clusters: int,
    nontrivial_partition: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    values = {name: float("nan") for name in ("avi", "avu", "modularity")}
    status = dict.fromkeys(values, "undefined_trivial_partition")
    if adjacency.nnz == 0:
        return values, dict.fromkeys(values, "undefined_edgeless_graph")

    # Q is invariant to a common positive weight scale. Normalizing avoids
    # overflow in degree sums without changing the graph or metric.
    weighted = adjacency.copy()
    weighted.data /= np.max(weighted.data)
    membership = sparse.csr_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), labels)),
        shape=(len(labels), n_clusters),
    )
    cluster_weights = (membership.T @ weighted @ membership).toarray()
    cluster_degrees = cluster_weights.sum(axis=1)
    twice_total_weight = float(cluster_degrees.sum())
    values["modularity"] = float(
        np.sum(
            np.diag(cluster_weights) / twice_total_weight
            - (cluster_degrees / twice_total_weight) ** 2
        )
    )
    status["modularity"] = "ok" if nontrivial_partition else "defined_trivial_partition"
    if not nontrivial_partition:
        return values, status

    topology = adjacency.copy()
    topology.data = np.ones_like(topology.data)
    block = (membership.T @ topology @ membership).toarray()
    internal = np.diag(block)
    external = block.sum(axis=1) - internal
    volume = internal + external
    if np.any(volume == 0):
        status["avi"] = "undefined_zero_cluster_volume"
    else:
        values["avi"] = float(np.mean(internal / volume))
        status["avi"] = "ok_binary_topology"

    denominators = external[:, None] + external[None, :] - block
    off_diagonal = ~np.eye(n_clusters, dtype=bool)
    if np.any(denominators[off_diagonal] == 0):
        status["avu"] = "undefined_zero_pair_boundary"
    else:
        # Shalileh et al. (2026), eq. 21 divides by K, not K*(K-1).
        values["avu"] = float(np.sum(block[off_diagonal] / denominators[off_diagonal]) / n_clusters)
        status["avu"] = "ok_binary_topology"
    return values, status


def evaluate_partition(
    X: ArrayLike,
    labels: ArrayLike,
    adjacency: Any,
    distances: ArrayLike | None = None,
) -> dict[str, Any]:
    """Evaluate an aligned feature table, partition and undirected weighted graph.

    ``X`` is the Euclidean feature matrix, without additional scaling here.
    ``distances``, if supplied, must be Euclidean distances between those rows;
    it is validated structurally but not recomputed to prove that equivalence.
    All labels, including -1, denote ordinary groups: no rows are discarded.

    Returns the six numeric metric keys, three size diagnostics,
    ``partition_status`` and a per-metric ``metric_status`` mapping. A valid
    but undefined metric is NaN. Invalid or misaligned inputs raise ValueError.
    Inputs are never modified. Empty data retain diagnostics and NaN metrics.
    """
    features, assignments, sizes = _validate_features_and_labels(X, labels)
    n_samples = len(features)
    n_clusters = len(sizes)
    graph = _validate_adjacency(adjacency, n_samples)
    distance_matrix = _validate_distances(distances, n_samples)

    if n_samples == 0:
        partition_status = "empty"
    elif n_clusters == 1:
        partition_status = "one_cluster"
    elif n_clusters == n_samples:
        partition_status = "all_singletons"
    else:
        partition_status = "nontrivial"
    nontrivial = partition_status == "nontrivial"
    status = dict.fromkeys(_METRICS, "undefined_trivial_partition")
    result: dict[str, Any] = dict.fromkeys(_METRICS, float("nan"))
    result.update(
        n_clusters=n_clusters,
        smallest_cluster_size=int(sizes.min()) if n_clusters else 0,
        largest_cluster_share=float(sizes.max() / n_samples) if n_samples else float("nan"),
        partition_status=partition_status,
        metric_status=status,
    )

    graph_values, graph_status = _graph_metrics(graph, assignments, n_clusters, nontrivial)
    result.update(graph_values)
    status.update(graph_status)
    if not nontrivial:
        return result

    groups = [features[assignments == group] for group in range(n_clusters)]
    centroids = np.array([group.mean(axis=0) for group in groups])
    global_variance_norm = float(np.linalg.norm(np.var(features, axis=0, ddof=0)))
    if global_variance_norm == 0 or not np.isfinite(global_variance_norm):
        reason = (
            "undefined_zero_global_variance"
            if global_variance_norm == 0
            else "undefined_nonfinite_variance"
        )
        for name in ("silhouette", "calinski_harabasz", "s_dbw"):
            status[name] = reason
        return result

    if distance_matrix is None:
        result["silhouette"] = float(silhouette_score(features, assignments, metric="euclidean"))
    else:
        result["silhouette"] = float(
            silhouette_score(distance_matrix, assignments, metric="precomputed")
        )
    status["silhouette"] = "ok"

    within_ss = float(
        sum(np.sum((group - center) ** 2) for group, center in zip(groups, centroids, strict=True))
    )
    if within_ss == 0 or not np.isfinite(within_ss):
        status["calinski_harabasz"] = "undefined_zero_or_nonfinite_within_dispersion"
    else:
        result["calinski_harabasz"] = float(calinski_harabasz_score(features, assignments))
        status["calinski_harabasz"] = "ok"
    result["s_dbw"], status["s_dbw"] = _s_dbw(groups, centroids, global_variance_norm)
    for name in ("silhouette", "calinski_harabasz"):
        if not np.isfinite(result[name]) and status[name] == "ok":
            result[name] = float("nan")
            status[name] = "undefined_nonfinite_result"
    return result
