"""Hand-calculated cases and independent references for partition diagnostics."""

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from municipal_profiles.metrics import evaluate_partition

METRIC_NAMES = (
    "silhouette",
    "calinski_harabasz",
    "s_dbw",
    "avi",
    "avu",
    "modularity",
)


def two_triangles(bridge_weight=0.0):
    adjacency = np.zeros((6, 6))
    for start in (0, 3):
        adjacency[start : start + 3, start : start + 3] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    adjacency[2, 3] = adjacency[3, 2] = bridge_weight
    return sparse.csr_matrix(adjacency)


@pytest.fixture
def separated_features():
    return np.array([[-1.0], [0.0], [1.0], [9.0], [10.0], [11.0]])


def test_disjoint_triangles_have_isolation_but_undefined_unifiability(separated_features):
    result = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], two_triangles())
    assert result["avi"] == pytest.approx(1.0)
    assert result["modularity"] == pytest.approx(0.5)
    assert np.isnan(result["avu"])
    assert result["metric_status"]["avu"] == "undefined_zero_pair_boundary"
    assert result["n_clusters"] == 2
    assert result["smallest_cluster_size"] == 3
    assert result["largest_cluster_share"] == pytest.approx(0.5)


def test_bridge_has_hand_calculated_scores(separated_features):
    result = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], two_triangles(1))
    assert result["avi"] == pytest.approx(6 / 7)
    assert result["avu"] == pytest.approx(1)
    assert result["modularity"] == pytest.approx(5 / 14)


def test_weighted_modularity_keeps_weights_avi_avu_use_topology(separated_features):
    result = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], two_triangles(2))
    assert result["modularity"] == pytest.approx(1 / 4)
    assert result["avi"] == pytest.approx(6 / 7)
    assert result["avu"] == pytest.approx(1)
    assert result["metric_status"]["avi"] == "ok_binary_topology"


def test_three_groups_check_avu_normalization():
    # Three internally connected pairs; one edge between every pair of groups.
    adjacency = np.zeros((6, 6))
    for first, second in [(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (0, 4)]:
        adjacency[first, second] = adjacency[second, first] = 1
    result = evaluate_partition(np.arange(6)[:, None], [0, 0, 1, 1, 2, 2], adjacency)
    # Each group: I=2, E=2. Each ordered pair: B/(E+E-B)=1/3.
    assert result["avi"] == pytest.approx(1 / 2)
    assert result["avu"] == pytest.approx(2 / 3)
    assert result["modularity"] == pytest.approx(1 / 6)


def test_s_dbw_separated_centers_manual_result(separated_features):
    result = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], two_triangles())
    # Within variances 2/3, global variance 77/3, no points near midpoint 5.
    assert result["s_dbw"] == pytest.approx(2 / 77)
    assert result["metric_status"]["s_dbw"] == "ok"


def test_s_dbw_pair_union_density_variant():
    features = np.array([-2, 0, 2, 0, 2, 4], dtype=float)[:, None]
    result = evaluate_partition(features, [0, 0, 0, 1, 1, 1], two_triangles())
    # Scat=8/11; radius=2/sqrt(3). The union has two points near each
    # centroid and four near midpoint 1, so between-density=4/2=2.
    assert result["s_dbw"] == pytest.approx(30 / 11)


def test_zero_density_denominator_is_not_perfect_score():
    features = np.array([-1, 1, 9, 11], dtype=float)[:, None]
    result = evaluate_partition(features, [0, 0, 1, 1], np.zeros((4, 4)))
    # Radius=sqrt(2)/2; every observation is distance 1 from its own center.
    assert np.isnan(result["s_dbw"])
    assert result["metric_status"]["s_dbw"] == "undefined_zero_density_denominator"


def test_zero_within_variance_has_explicit_undefined_status():
    result = evaluate_partition([[0], [0], [10], [10]], [0, 0, 1, 1], np.zeros((4, 4)))
    assert result["silhouette"] == pytest.approx(1)
    assert np.isnan(result["calinski_harabasz"])
    assert np.isnan(result["s_dbw"])
    assert result["metric_status"]["s_dbw"] == "undefined_zero_density_radius"


def test_zero_global_variance_does_not_report_geometry_quality():
    result = evaluate_partition(np.ones((6, 2)), [0, 0, 0, 1, 1, 1], two_triangles(1))
    for name in ("silhouette", "calinski_harabasz", "s_dbw"):
        assert np.isnan(result[name])
        assert result["metric_status"][name] == "undefined_zero_global_variance"
    assert result["modularity"] == pytest.approx(5 / 14)


@pytest.mark.parametrize("labels", [[0] * 6, list(range(6))])
def test_trivial_partitions_keep_q_but_flag_quality(separated_features, labels):
    result = evaluate_partition(separated_features, labels, two_triangles(1))
    for name in METRIC_NAMES[:-1]:
        assert np.isnan(result[name])
        assert result["metric_status"][name] == "undefined_trivial_partition"
    assert result["metric_status"]["modularity"] == "defined_trivial_partition"
    if len(set(labels)) == 1:
        assert result["modularity"] == pytest.approx(0)
        assert result["partition_status"] == "one_cluster"
    else:
        # Degrees are 2,2,3,3,2,2: sum(degree**2)/(2m)**2=34/196.
        assert result["modularity"] == pytest.approx(-17 / 98)
        assert result["partition_status"] == "all_singletons"


def test_edgeless_graph_leaves_geometry_metrics_available(separated_features):
    result = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], sparse.csr_matrix((6, 6)))
    for name in ("avi", "avu", "modularity"):
        assert np.isnan(result[name])
        assert result["metric_status"][name] == "undefined_edgeless_graph"
    assert result["s_dbw"] == pytest.approx(2 / 77)


def test_isolated_singleton_does_not_disappear():
    adjacency = sparse.block_diag((two_triangles(), sparse.csr_matrix((1, 1))))
    result = evaluate_partition(np.arange(7)[:, None], [0, 0, 0, 1, 1, 1, 2], adjacency)
    assert result["n_clusters"] == 3
    assert result["smallest_cluster_size"] == 1
    assert result["largest_cluster_share"] == pytest.approx(3 / 7)
    assert np.isnan(result["avi"])
    assert result["metric_status"]["avi"] == "undefined_zero_cluster_volume"
    assert result["modularity"] == pytest.approx(0.5)


def test_sklearn_geometry_reference_and_precomputed_distances(separated_features):
    labels = np.array([0, 0, 0, 1, 1, 1])
    distances = squareform(pdist(separated_features))
    result = evaluate_partition(separated_features, labels, two_triangles(1), distances)
    assert result["silhouette"] == pytest.approx(silhouette_score(separated_features, labels))
    assert result["calinski_harabasz"] == pytest.approx(
        calinski_harabasz_score(separated_features, labels)
    )
    direct = evaluate_partition(separated_features, labels, two_triangles(1))
    assert result["silhouette"] == pytest.approx(direct["silhouette"])


def test_label_and_row_permutation_preserve_metrics(separated_features):
    labels = np.array(["east"] * 3 + ["west"] * 3)
    graph = two_triangles(2)
    baseline = evaluate_partition(separated_features, labels, graph)
    order = np.array([4, 0, 5, 2, 1, 3])
    renamed_labels = np.where(labels == "east", 71, -1)
    permuted = evaluate_partition(
        separated_features[order], renamed_labels[order], graph[order][:, order]
    )
    for name in METRIC_NAMES:
        assert permuted[name] == pytest.approx(baseline[name])
    assert baseline["metric_status"] == permuted["metric_status"]


def test_common_weight_scaling_and_feature_translation(separated_features):
    baseline = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], two_triangles(2))
    changed = evaluate_partition(
        separated_features * 3 + 100,
        [0, 0, 0, 1, 1, 1],
        two_triangles(2) * 1e100,
    )
    for name in METRIC_NAMES:
        assert changed[name] == pytest.approx(baseline[name])


def test_inputs_are_not_modified(separated_features):
    features = separated_features.copy()
    graph = two_triangles(2)
    graph_before = graph.copy()
    distances = squareform(pdist(features))
    distances_before = distances.copy()
    evaluate_partition(features, [0, 0, 0, 1, 1, 1], graph, distances)
    np.testing.assert_array_equal(features, separated_features)
    assert (graph != graph_before).nnz == 0
    np.testing.assert_array_equal(distances, distances_before)


def test_empty_data_has_explicit_diagnostics():
    result = evaluate_partition(np.empty((0, 2)), [], sparse.csr_matrix((0, 0)))
    assert result["partition_status"] == "empty"
    assert result["n_clusters"] == 0
    assert result["smallest_cluster_size"] == 0
    assert np.isnan(result["largest_cluster_share"])
    assert all(np.isnan(result[name]) for name in METRIC_NAMES)


@pytest.mark.parametrize(
    "adjacency, message",
    [
        (np.eye(4), "self-loops"),
        (np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), "symmetric"),
        (-np.ones((4, 4)) + np.eye(4), "nonnegative"),
        (np.full((4, 4), np.nan), "finite"),
        (np.zeros((3, 3)), "shape"),
    ],
)
def test_invalid_adjacency_is_rejected(adjacency, message):
    with pytest.raises(ValueError, match=message):
        evaluate_partition(np.arange(4)[:, None], [0, 0, 1, 1], adjacency)


@pytest.mark.parametrize("labels", [[0, 1], [[0], [0], [1], [1]], [0, 0, None, 1]])
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        evaluate_partition(np.arange(4)[:, None], labels, np.zeros((4, 4)))


def test_invalid_precomputed_distances_are_rejected():
    distances = np.ones((4, 4))
    with pytest.raises(ValueError, match="zero diagonal"):
        evaluate_partition(np.arange(4)[:, None], [0, 0, 1, 1], np.zeros((4, 4)), distances)


def test_modularity_against_networkx_if_available(separated_features):
    nx = pytest.importorskip("networkx")
    graph = two_triangles(2)
    result = evaluate_partition(separated_features, [0, 0, 0, 1, 1, 1], graph)
    nx_graph = nx.from_scipy_sparse_array(graph)
    expected = nx.community.modularity(nx_graph, [{0, 1, 2}, {3, 4, 5}], weight="weight")
    assert result["modularity"] == pytest.approx(expected)
