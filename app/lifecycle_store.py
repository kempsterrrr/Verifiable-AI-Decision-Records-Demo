"""Storage for lifecycle proof records (training, registration)."""

import json
import os
import tempfile
import threading


class LifecycleStore:
    """Thread-safe JSON file storage for lifecycle proof envelopes."""

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            self._write([])

    def _read(self) -> list[dict]:
        try:
            with open(self._filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, records: list[dict]) -> None:
        dir_name = os.path.dirname(self._filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp_path, self._filepath)
        except Exception:
            os.unlink(tmp_path)
            raise

    def append(self, envelope: dict) -> None:
        with self._lock:
            records = self._read()
            records.append(envelope)
            self._write(records)

    def list_all(self) -> list[dict]:
        with self._lock:
            return self._read()

    def get_by_event_id(self, event_id: str) -> dict | None:
        with self._lock:
            for rec in self._read():
                if rec.get("record", {}).get("event_id") == event_id:
                    return rec
        return None

    def get_by_type(self, event_type: str) -> list[dict]:
        with self._lock:
            return [
                r for r in self._read()
                if r.get("record", {}).get("event_type") == event_type
            ]

    def get_by_run_id(self, run_id: str) -> dict | None:
        with self._lock:
            for rec in self._read():
                if rec.get("record", {}).get("run_id") == run_id:
                    return rec
        return None

    def get_by_model_version(self, model_name: str, version: str) -> dict | None:
        with self._lock:
            for rec in self._read():
                r = rec.get("record", {})
                if (r.get("event_type") == "model_registered"
                        and r.get("model_name") == model_name
                        and str(r.get("model_version")) == str(version)):
                    return rec
        return None

    def update(self, event_id: str, envelope: dict) -> bool:
        with self._lock:
            records = self._read()
            for i, rec in enumerate(records):
                if rec.get("record", {}).get("event_id") == event_id:
                    records[i] = envelope
                    self._write(records)
                    return True
        return False
