from typing import List, Optional
from . import computation_history_entry as comp
import os
import json

SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved_state.json")


class HistoryCache:
    def __init__(self):
        self._entries: List[comp.ComputationRecord] = []

    def add_entry(self, entry: comp.ComputationRecord):
        self._entries.append(entry)
        self.save_to_disk()

    def get_entry(self, index: int) -> Optional[comp.ComputationRecord]:
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def get_labels(self) -> List[str]:
        return [entry.display_name() for entry in self._entries]

    def get_all_entries(self) -> List[comp.ComputationRecord]:
        return self._entries[:]

    def save_to_disk(self):
        data = [entry.to_dict() for entry in self._entries]
        try:
            with open(SAVE_PATH, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[History Save Error] {e}")

    def load_from_disk(self):
        if not os.path.exists(SAVE_PATH):
            return
        try:
            with open(SAVE_PATH, 'r') as f:
                data = json.load(f)
                self._entries = [comp.ComputationRecord.from_dict(entry) for entry in data]
        except Exception as e:
            print(f"[History Load Error] {e}")

    def clear(self):
        """Clear the in-memory and saved history."""
        self._entries.clear()
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)


cache = HistoryCache()
cache.load_from_disk()
