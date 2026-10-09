import numpy as np
import pytest
from scipy import sparse

from municipal_profiles.networks import (
    align_labels,
    co_membership_jaccard,
    compare_partitions,
    equal_edge_budget_graph,
    graph_statistics,
    knn_graph,
    leiden_partition,
    neighbor_jaccard,
    similarity_matrix,
)


def test_similarity_and_tied_neighbors_are_deterministic():
    p = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.9, 0.1]])
    a = knn_graph(similarity_matrix(p, "hellinger"), 1)
    assert a[0, 1] > 0 and a[1, 0] > 0
    assert a[0, 2] > 0  # Row2 chooses row0, so union degree can exceed k.
    assert np.all(a.diagonal() == 0)
    assert (a - a.T).nnz == 0
    np.testing.assert_allclose(neighbor_jaccard(a, a), 1)


def test_geometries_and_invalid_inputs():
    p = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    h = similarity_matrix(p, "hellinger")
    c = similarity_matrix(p, "cosine")
    assert h[0, 1] == 0 and c[0, 1] == 0
    assert h[0, 2] == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError):
        similarity_matrix(p * 2, "hellinger")
    with pytest.raises(ValueError):
        knn_graph(h, 3)


def test_label_permutation_and_splits_preserve_meaning():
    ref = np.array([1, 1, 2, 2, 3, 3])
    shuffled = np.array([80, 80, 7, 7, 4, 4])
    np.testing.assert_array_equal(align_labels(ref, shuffled), ref)
    assert compare_partitions(ref, shuffled)["ari_to_primary"] == 1
    split = np.array([10, 10, 20, 20, 30, 40])
    aligned = align_labels(ref, split)
    assert np.sum(aligned == ref) == 5
    assert len(np.unique(aligned)) == 4


def test_equal_edge_control_and_separated_communities():
    a = np.zeros((8, 8))
    a[:4, :4] = 1
    a[4:, 4:] = 1
    np.fill_diagonal(a, 0)
    graph = sparse.csr_matrix(a)
    labels = leiden_partition(graph, 1, 13)
    assert len(np.unique(labels[:4])) == 1
    assert len(np.unique(labels[4:])) == 1
    assert labels[0] != labels[4]
    reduced = equal_edge_budget_graph(graph, 5)
    assert reduced.nnz // 2 == 5
    assert graph_statistics(graph)["n_components"] == 2
    assert graph_statistics(graph)["n_isolates"] == 0


def test_zero_overlap_match_does_not_invent_lineage():
    reference = np.array([0] * 5 + [1] + [2] * 9)
    candidate = np.array([10] * 6 + [20] * 5 + [30] * 4)
    aligned = align_labels(reference, candidate)
    assert np.all(aligned[-4:] < 0)


def test_tied_splits_are_label_invariant_and_not_confident():
    ref = np.array([0, 0, 0, 0])
    left = np.array([10, 10, 20, 20])
    right = np.array([20, 20, 10, 10])
    np.testing.assert_array_equal(align_labels(ref, left), align_labels(ref, right))
    np.testing.assert_allclose(co_membership_jaccard(ref, left), 1 / 3)
    np.testing.assert_array_equal(
        co_membership_jaccard(ref, left), co_membership_jaccard(ref, right)
    )
