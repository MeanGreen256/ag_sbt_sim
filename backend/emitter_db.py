import sqlite3
import json
from datetime import datetime, timedelta
from uuid import uuid4
from models import Emitter, Fingerprint, Observation, Baseline


class EmitterDB:
    """SQLite-backed storage for emitter library, fingerprints, observations, and baselines."""

    def __init__(self, db_path: str = "emitters.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS emitters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                freq_range_start REAL,
                freq_range_end REAL,
                notes TEXT DEFAULT '',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fingerprints (
                id TEXT PRIMARY KEY,
                emitter_id TEXT NOT NULL REFERENCES emitters(id) ON DELETE CASCADE,
                feature_vector TEXT NOT NULL,
                quality_score REAL DEFAULT 1.0,
                captured_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emitter_id TEXT NOT NULL REFERENCES emitters(id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                frequency REAL,
                power_db REAL,
                bandwidth REAL,
                is_active INTEGER DEFAULT 1,
                match_confidence REAL
            );

            CREATE TABLE IF NOT EXISTS baselines (
                emitter_id TEXT PRIMARY KEY REFERENCES emitters(id) ON DELETE CASCADE,
                power_mean REAL,
                power_std REAL,
                freq_mean REAL,
                freq_std REAL,
                typical_hours TEXT,
                duty_on_seconds REAL,
                duty_off_seconds REAL,
                updated_at TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- Emitter CRUD ---

    def create_emitter(self, name, tags, freq_start, freq_end) -> Emitter:
        now = datetime.now().isoformat()
        emitter = Emitter(name=name, tags=tags, freq_range_start=freq_start, freq_range_end=freq_end,
                          first_seen=datetime.now(), last_seen=datetime.now(), created_at=datetime.now())
        self.conn.execute(
            "INSERT INTO emitters (id, name, tags, freq_range_start, freq_range_end, notes, first_seen, last_seen, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (emitter.id, emitter.name, json.dumps(emitter.tags), emitter.freq_range_start, emitter.freq_range_end, emitter.notes, now, now, now))
        self.conn.commit()
        return emitter

    def get_emitter(self, emitter_id) -> Emitter | None:
        row = self.conn.execute(
            "SELECT id, name, tags, freq_range_start, freq_range_end, notes, first_seen, last_seen, created_at FROM emitters WHERE id = ?",
            (emitter_id,)).fetchone()
        if row is None:
            return None
        return Emitter(id=row[0], name=row[1], tags=json.loads(row[2]), freq_range_start=row[3], freq_range_end=row[4], notes=row[5],
                       first_seen=datetime.fromisoformat(row[6]), last_seen=datetime.fromisoformat(row[7]), created_at=datetime.fromisoformat(row[8]))

    def list_emitters(self) -> list[Emitter]:
        rows = self.conn.execute(
            "SELECT id, name, tags, freq_range_start, freq_range_end, notes, first_seen, last_seen, created_at FROM emitters ORDER BY name"
        ).fetchall()
        return [Emitter(id=r[0], name=r[1], tags=json.loads(r[2]), freq_range_start=r[3], freq_range_end=r[4], notes=r[5],
                        first_seen=datetime.fromisoformat(r[6]), last_seen=datetime.fromisoformat(r[7]), created_at=datetime.fromisoformat(r[8])) for r in rows]

    def update_emitter(self, emitter_id, name=None, tags=None, notes=None) -> Emitter:
        updates, params = [], []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if updates:
            params.append(emitter_id)
            self.conn.execute(f"UPDATE emitters SET {', '.join(updates)} WHERE id = ?", params)
            self.conn.commit()
        return self.get_emitter(emitter_id)

    def delete_emitter(self, emitter_id) -> None:
        self.conn.execute("DELETE FROM emitters WHERE id = ?", (emitter_id,))
        self.conn.commit()

    def update_last_seen(self, emitter_id) -> None:
        self.conn.execute("UPDATE emitters SET last_seen = ? WHERE id = ?", (datetime.now().isoformat(), emitter_id))
        self.conn.commit()

    # --- Fingerprints ---

    def add_fingerprint(self, emitter_id, feature_vector, quality_score=1.0) -> Fingerprint:
        fp = Fingerprint(emitter_id=emitter_id, feature_vector=feature_vector, quality_score=quality_score)
        self.conn.execute(
            "INSERT INTO fingerprints (id, emitter_id, feature_vector, quality_score, captured_at) VALUES (?, ?, ?, ?, ?)",
            (fp.id, fp.emitter_id, json.dumps(fp.feature_vector), fp.quality_score, fp.captured_at.isoformat()))
        self.conn.commit()
        return fp

    def get_fingerprints(self, emitter_id) -> list[Fingerprint]:
        rows = self.conn.execute(
            "SELECT id, emitter_id, feature_vector, quality_score, captured_at FROM fingerprints WHERE emitter_id = ?",
            (emitter_id,)).fetchall()
        return [Fingerprint(id=r[0], emitter_id=r[1], feature_vector=json.loads(r[2]), quality_score=r[3], captured_at=datetime.fromisoformat(r[4])) for r in rows]

    def get_all_fingerprints(self) -> list[Fingerprint]:
        rows = self.conn.execute(
            "SELECT id, emitter_id, feature_vector, quality_score, captured_at FROM fingerprints"
        ).fetchall()
        return [Fingerprint(id=r[0], emitter_id=r[1], feature_vector=json.loads(r[2]), quality_score=r[3], captured_at=datetime.fromisoformat(r[4])) for r in rows]

    # --- Observations ---

    def log_observation(self, emitter_id, frequency, power_db, bandwidth, is_active=True, match_confidence=0.0, timestamp=None) -> None:
        ts = (timestamp or datetime.now()).isoformat()
        self.conn.execute(
            "INSERT INTO observations (emitter_id, timestamp, frequency, power_db, bandwidth, is_active, match_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (emitter_id, ts, frequency, power_db, bandwidth, int(is_active), match_confidence))
        self.conn.commit()

    def get_observations(self, emitter_id, since=None, until=None) -> list[Observation]:
        query = "SELECT emitter_id, timestamp, frequency, power_db, bandwidth, is_active, match_confidence FROM observations WHERE emitter_id = ?"
        params = [emitter_id]
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())
        query += " ORDER BY timestamp"
        rows = self.conn.execute(query, params).fetchall()
        return [Observation(emitter_id=r[0], timestamp=datetime.fromisoformat(r[1]), frequency=r[2], power_db=r[3], bandwidth=r[4], is_active=bool(r[5]), match_confidence=r[6]) for r in rows]

    def prune_observations(self, max_age_hours=24) -> int:
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        cursor = self.conn.execute("DELETE FROM observations WHERE timestamp < ?", (cutoff,))
        self.conn.commit()
        return cursor.rowcount

    # --- Baselines ---

    def compute_baseline(self, emitter_id, window_hours=4) -> Baseline | None:
        since = (datetime.now() - timedelta(hours=window_hours)).isoformat()
        rows = self.conn.execute(
            "SELECT frequency, power_db, timestamp, is_active FROM observations WHERE emitter_id = ? AND timestamp >= ? ORDER BY timestamp",
            (emitter_id, since)).fetchall()
        if not rows:
            return None

        import numpy as np
        from collections import Counter

        active_powers = [r[1] for r in rows if r[3]]
        active_freqs = [r[0] for r in rows if r[3]]
        power_mean = float(np.mean(active_powers)) if active_powers else 0.0
        power_std = float(np.std(active_powers)) if active_powers else 0.0
        freq_mean = float(np.mean(active_freqs)) if active_freqs else 0.0
        freq_std = float(np.std(active_freqs)) if active_freqs else 0.0

        hour_counts, hour_active = Counter(), Counter()
        for r in rows:
            h = datetime.fromisoformat(r[2]).hour
            hour_counts[h] += 1
            if r[3]:
                hour_active[h] += 1
        typical_hours = [[h, h + 1] for h in sorted(hour_counts) if hour_counts[h] > 0 and hour_active[h] / hour_counts[h] > 0.3]

        baseline = Baseline(emitter_id=emitter_id, power_mean=power_mean, power_std=power_std,
                            freq_mean=freq_mean, freq_std=freq_std, typical_hours=typical_hours, updated_at=datetime.now())
        self.conn.execute(
            "INSERT INTO baselines (emitter_id, power_mean, power_std, freq_mean, freq_std, typical_hours, duty_on_seconds, duty_off_seconds, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(emitter_id) DO UPDATE SET "
            "power_mean=excluded.power_mean, power_std=excluded.power_std, freq_mean=excluded.freq_mean, freq_std=excluded.freq_std, "
            "typical_hours=excluded.typical_hours, updated_at=excluded.updated_at",
            (emitter_id, power_mean, power_std, freq_mean, freq_std, json.dumps(typical_hours), 0.0, 0.0, baseline.updated_at.isoformat()))
        self.conn.commit()
        return baseline

    def get_baseline(self, emitter_id) -> Baseline | None:
        row = self.conn.execute(
            "SELECT emitter_id, power_mean, power_std, freq_mean, freq_std, typical_hours, duty_on_seconds, duty_off_seconds, updated_at FROM baselines WHERE emitter_id = ?",
            (emitter_id,)).fetchone()
        if row is None:
            return None
        return Baseline(emitter_id=row[0], power_mean=row[1], power_std=row[2], freq_mean=row[3], freq_std=row[4],
                        typical_hours=json.loads(row[5]), duty_on_seconds=row[6], duty_off_seconds=row[7], updated_at=datetime.fromisoformat(row[8]))
