import numpy as np

from bearing_diagnosis.anomaly_detection import fit_reference, score_features


def test_normal_reference_scores_shifted_points_higher() -> None:
    rng = np.random.default_rng(2026)
    normal = rng.normal(size=(200, 8))
    reference = fit_reference(normal)
    _, normal_probability = score_features(normal[:20], reference)
    _, shifted_probability = score_features(np.full((20, 8), 8.0), reference)
    assert float(np.median(shifted_probability)) > float(np.median(normal_probability))
    assert np.all((shifted_probability >= 0) & (shifted_probability <= 1))
