import numpy as np
from models import FeatureVector, Emitter, MatchResult, Observation, Anomaly, Baseline, MessageType


def test_feature_vector_from_array():
    fv = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert len(fv.values) == 7
    assert fv.spectral_centroid == 1.0
    assert fv.bandwidth_3db == 2.0
    assert fv.bandwidth_10db == 3.0
    assert fv.snr == 4.0
    assert fv.papr == 5.0
    assert fv.spectral_flatness == 6.0
    assert fv.kurtosis == 7.0


def test_feature_vector_to_numpy():
    fv = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    arr = fv.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (7,)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_feature_vector_validation_rejects_wrong_length():
    import pytest
    with pytest.raises(Exception):
        FeatureVector(values=[1.0, 2.0])


def test_emitter_defaults():
    e = Emitter(name="RADAR-A", tags=["hostile"], freq_range_start=99.0, freq_range_end=100.0)
    assert e.id  # auto-generated
    assert e.name == "RADAR-A"
    assert e.tags == ["hostile"]
    assert e.notes == ""


def test_match_result_ordering():
    m1 = MatchResult(emitter_id="a", emitter_name="A", confidence=0.94)
    m2 = MatchResult(emitter_id="b", emitter_name="B", confidence=0.61)
    results = sorted([m2, m1], key=lambda r: r.confidence, reverse=True)
    assert results[0].confidence == 0.94


def test_anomaly_model():
    a = Anomaly(
        type="power_anomaly",
        emitter_id="abc",
        emitter_name="RADAR-A",
        severity=2.5,
        baseline_value=-50.0,
        current_value=-42.0,
        message="Power +8dB above baseline",
    )
    assert a.type == "power_anomaly"
    assert a.timestamp  # auto-set


def test_message_type_match():
    assert MessageType.MATCH == 0x03
