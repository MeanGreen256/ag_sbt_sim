import numpy as np
from models import FeatureVector, Emitter, MatchResult
from emitter_db import EmitterDB


class EmitterMatcher:
    """Cosine-similarity matching engine for emitter fingerprints."""

    def __init__(self, db: EmitterDB):
        self.db = db
        self._library: list[tuple[str, str, np.ndarray]] = []

    def load_library(self) -> None:
        fingerprints = self.db.get_all_fingerprints()
        emitters = {e.id: e for e in self.db.list_emitters()}
        self._library = []
        for fp in fingerprints:
            emitter = emitters.get(fp.emitter_id)
            if emitter:
                vec = np.array(fp.feature_vector, dtype=np.float64)
                self._library.append((emitter.id, emitter.name, vec))

    def match(self, vector: FeatureVector) -> list[MatchResult]:
        if not self._library:
            return []
        query = vector.to_numpy()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query_unit = query / query_norm

        best_scores: dict[str, tuple[str, float]] = {}
        for emitter_id, emitter_name, lib_vec in self._library:
            lib_norm = np.linalg.norm(lib_vec)
            if lib_norm == 0:
                continue
            sim = float(np.dot(query_unit, lib_vec / lib_norm))
            sim = max(0.0, sim)
            if emitter_id not in best_scores or sim > best_scores[emitter_id][1]:
                best_scores[emitter_id] = (emitter_name, sim)

        results = [
            MatchResult(emitter_id=eid, emitter_name=name, confidence=round(score, 4))
            for eid, (name, score) in best_scores.items()
        ]
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def enroll(self, name, tags, vectors, freq_start, freq_end) -> Emitter:
        emitter = self.db.create_emitter(name, tags, freq_start, freq_end)
        for vec in vectors:
            quality = min(1.0, max(0.1, vec.snr / 40.0))
            self.db.add_fingerprint(emitter.id, vec.values, quality_score=quality)
        self.load_library()
        return emitter

    def enrich(self, emitter_id, vectors) -> None:
        for vec in vectors:
            quality = min(1.0, max(0.1, vec.snr / 40.0))
            self.db.add_fingerprint(emitter_id, vec.values, quality_score=quality)
        self.load_library()
