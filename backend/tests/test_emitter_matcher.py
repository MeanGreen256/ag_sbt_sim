import pytest
import os
import tempfile
import numpy as np
from emitter_db import EmitterDB
from emitter_matcher import EmitterMatcher
from models import FeatureVector


@pytest.fixture
def matcher():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EmitterDB(path)
    m = EmitterMatcher(db)
    yield m
    db.close()
    os.unlink(path)


def test_enroll_creates_emitter(matcher):
    vectors = [FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])]
    emitter = matcher.enroll("RADAR-A", ["hostile"], vectors, freq_start=99.0, freq_end=100.0)
    assert emitter.name == "RADAR-A"
    assert emitter.tags == ["hostile"]


def test_enroll_stores_fingerprint(matcher):
    vectors = [
        FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        FeatureVector(values=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]),
    ]
    emitter = matcher.enroll("RADAR-B", [], vectors, freq_start=99.0, freq_end=100.0)
    fps = matcher.db.get_fingerprints(emitter.id)
    assert len(fps) == 2


def test_match_identical_vector(matcher):
    vec = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    matcher.enroll("EXACT", [], [vec], freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    results = matcher.match(vec)
    assert len(results) >= 1
    assert results[0].confidence > 0.99
    assert results[0].emitter_name == "EXACT"


def test_match_similar_vector(matcher):
    vec = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    matcher.enroll("SIMILAR", [], [vec], freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    query = FeatureVector(values=[1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05])
    results = matcher.match(query)
    assert len(results) >= 1
    assert results[0].confidence > 0.99


def test_match_orthogonal_vector(matcher):
    vec = FeatureVector(values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    matcher.enroll("DIR-X", [], [vec], freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    query = FeatureVector(values=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = matcher.match(query)
    assert len(results) >= 1
    assert results[0].confidence < 0.1


def test_match_returns_sorted_by_confidence(matcher):
    matcher.enroll("A", [], [FeatureVector(values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])], freq_start=99.0, freq_end=100.0)
    matcher.enroll("B", [], [FeatureVector(values=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])], freq_start=100.0, freq_end=101.0)
    matcher.load_library()
    query = FeatureVector(values=[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = matcher.match(query)
    for i in range(len(results) - 1):
        assert results[i].confidence >= results[i + 1].confidence


def test_match_empty_library(matcher):
    matcher.load_library()
    query = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    results = matcher.match(query)
    assert results == []


def test_enrich_adds_fingerprints(matcher):
    vec1 = FeatureVector(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    emitter = matcher.enroll("ENRICH", [], [vec1], freq_start=99.0, freq_end=100.0)
    vec2 = FeatureVector(values=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1])
    matcher.enrich(emitter.id, [vec2])
    fps = matcher.db.get_fingerprints(emitter.id)
    assert len(fps) == 2


def test_match_multiple_fingerprints_takes_best(matcher):
    vecs = [
        FeatureVector(values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        FeatureVector(values=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    matcher.enroll("MULTI", [], vecs, freq_start=99.0, freq_end=100.0)
    matcher.load_library()
    query = FeatureVector(values=[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = matcher.match(query)
    assert len(results) == 1
    assert results[0].emitter_name == "MULTI"
    assert results[0].confidence > 0.9
