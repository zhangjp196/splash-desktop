"""Process-local storage for Responses API results, under a byte budget."""

import threading
from dataclasses import dataclass

from . import json_codec
from .lru import LRUCache


@dataclass(frozen=True, slots=True)
class StoredResponse:
    response_json: bytes
    history_json: bytes

    @property
    def size(self):
        return len(self.response_json) + len(self.history_json)

    @property
    def response(self):
        return json_codec.loads(self.response_json)


class ResponseStore:
    """Process-local Responses state with one strict byte-budgeted LRU."""

    BUDGET_BYTES = 64 * 1024 * 1024

    def __init__(self):
        self.records = LRUCache(self.BUDGET_BYTES)
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, response_id):
        with self.lock:
            record = self.records.get(response_id)
            if record is None:
                self.misses += 1
            else:
                self.hits += 1
        return record

    def put(self, response, history_items):
        record = StoredResponse(
            json_codec.encode(response), json_codec.encode(history_items)
        )
        with self.lock:
            return self.records.put(response["id"], record, record.size)

    def delete(self, response_id):
        with self.lock:
            return self.records.pop(response_id) is not None

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.records),
                "bytes": self.records.bytes,
                "budget_bytes": self.records.budget_bytes,
                "evictions": self.records.evictions,
                "hits": self.hits,
                "misses": self.misses,
            }
