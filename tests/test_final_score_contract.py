import numpy as np

from src.obs_hypertension.recommender.pipeline.master_recommender_pipeline import (
    MasterRecommenderConfig,
    _compute_final_score_matrix,
    _minmax_scale_1d,
)


def test_minmax_scale_maps_risk_to_unit_interval():
    values = np.array([0.0, 2.0, 4.0], dtype="float32")
    result = _minmax_scale_1d(values)

    expected = np.array([0.0, 0.5, 1.0], dtype="float32")
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_minmax_scale_constant_risk_becomes_zero():
    values = np.array([3.0, 3.0, 3.0], dtype="float32")
    result = _minmax_scale_1d(values)

    np.testing.assert_array_equal(result, np.zeros(3, dtype="float32"))


def test_default_final_score_weights_match_manuscript_configuration():
    config = MasterRecommenderConfig()

    assert np.isclose(config.similarity_weight, 0.30)
    assert np.isclose(config.efficiency_weight, 0.50)
    assert np.isclose(config.safety_weight, 0.20)
    assert np.isclose(
        config.similarity_weight + config.efficiency_weight + config.safety_weight,
        1.0,
    )


def test_final_score_uses_inverse_normalized_risk_as_safety():
    similarity = np.array([[0.2, 0.8]], dtype="float32")
    efficiency = np.array([0.4, 0.6], dtype="float32")
    normalized_risk = np.array([0.0, 1.0], dtype="float32")
    config = MasterRecommenderConfig()

    result = _compute_final_score_matrix(
        similarity_matrix=similarity,
        efficiency_scores=efficiency,
        risk_scores_normalized=normalized_risk,
        config=config,
    )

    safety = 1.0 - normalized_risk
    expected = (
        0.30 * similarity
        + 0.50 * efficiency[None, :]
        + 0.20 * safety[None, :]
    )

    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_final_score_remains_within_unit_interval_for_unit_inputs():
    similarity = np.array([[0.0, 0.5, 1.0]], dtype="float32")
    efficiency = np.array([0.0, 0.5, 1.0], dtype="float32")
    normalized_risk = np.array([1.0, 0.5, 0.0], dtype="float32")
    config = MasterRecommenderConfig()

    result = _compute_final_score_matrix(
        similarity_matrix=similarity,
        efficiency_scores=efficiency,
        risk_scores_normalized=normalized_risk,
        config=config,
    )

    assert np.isfinite(result).all()
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()
