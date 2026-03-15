from __future__ import annotations

import difflib
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SchemaHasher:
    @staticmethod
    def hash_schema(schema_profile: dict) -> str:
        hash_input = {
            "file_name": schema_profile.get("file_name", ""),
            "column_count": schema_profile.get("column_count", 0),
            "columns": []
        }

        for col in schema_profile.get("columns", []):
            col_info = {
                "name": col.get("name", ""),
                "type": col.get("inferred_type", "")
            }

            profile = col.get("profile", {})
            if col.get("inferred_type") == "categorical":
                top_values = profile.get("top_values", [])
                col_info["values"] = sorted([
                    str(v.get("value", "")) for v in top_values[:10]
                ])

            hash_input["columns"].append(col_info)

        hash_input["columns"].sort(key=lambda x: x["name"])

        hash_str = json.dumps(hash_input, sort_keys=True)
        full_hash = hashlib.sha256(hash_str.encode()).hexdigest()

        return full_hash[:16]


class PlanCache:
    SIMILARITY_THRESHOLD = 0.85
    MAX_ENTRIES_PER_SCHEMA = 100

    def __init__(self, cache_path: str = "plan_cache.json"):
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[dict]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded plan cache: {sum(len(v) for v in self._cache.values())} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache, starting fresh: {e}")
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, default=str)
            logger.debug(f"Saved plan cache to {self.cache_path}")
        except IOError as e:
            logger.error(f"Failed to save cache: {e}")

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        def normalize(q: str) -> str:
            return " ".join(q.lower().strip().split())

        q1_norm = normalize(query1)
        q2_norm = normalize(query2)

        return difflib.SequenceMatcher(None, q1_norm, q2_norm).ratio()

    def lookup(
        self,
        query: str,
        schema_hash: str,
        threshold: Optional[float] = None
    ) -> Optional[dict]:
        threshold = threshold or self.SIMILARITY_THRESHOLD

        entries = self._cache.get(schema_hash, [])

        if not entries:
            logger.debug(f"Cache miss: No entries for schema {schema_hash}")
            return None

        best_match = None
        best_similarity = 0.0

        for entry in entries:
            cached_query = entry.get("query", "")
            similarity = self._calculate_similarity(query, cached_query)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match and best_similarity >= threshold:
            logger.info(f"Cache HIT: similarity={best_similarity:.2f}, query='{best_match['query'][:50]}...'")

            best_match["hit_count"] = best_match.get("hit_count", 0) + 1
            best_match["last_accessed"] = datetime.now().isoformat()
            self._save_cache()

            return best_match.get("plan")

        logger.debug(f"Cache miss: Best similarity {best_similarity:.2f} < threshold {threshold}")
        return None

    def save(
        self,
        query: str,
        schema_hash: str,
        plan: dict,
        metadata: Optional[dict] = None
    ) -> None:
        if schema_hash not in self._cache:
            self._cache[schema_hash] = []

        entries = self._cache[schema_hash]

        for entry in entries:
            if self._calculate_similarity(query, entry["query"]) > 0.95:
                entry["plan"] = plan
                entry["updated_at"] = datetime.now().isoformat()
                entry["hit_count"] = entry.get("hit_count", 0)
                logger.debug(f"Updated existing cache entry for query")
                self._save_cache()
                return

        new_entry = {
            "query": query,
            "plan": plan,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "hit_count": 0,
            "metadata": metadata or {}
        }

        entries.append(new_entry)

        if len(entries) > self.MAX_ENTRIES_PER_SCHEMA:
            entries.sort(key=lambda x: x.get("last_accessed", ""))
            self._cache[schema_hash] = entries[-self.MAX_ENTRIES_PER_SCHEMA:]

        logger.info(f"Cached new plan for query: '{query[:50]}...'")
        self._save_cache()

    def invalidate_schema(self, schema_hash: str) -> int:
        count = len(self._cache.get(schema_hash, []))
        if schema_hash in self._cache:
            del self._cache[schema_hash]
            self._save_cache()
            logger.info(f"Invalidated {count} cache entries for schema {schema_hash}")
        return count

    def get_stats(self) -> dict:
        total_entries = sum(len(v) for v in self._cache.values())
        total_hits = sum(
            entry.get("hit_count", 0)
            for entries in self._cache.values()
            for entry in entries
        )

        return {
            "total_schemas": len(self._cache),
            "total_entries": total_entries,
            "total_hits": total_hits,
            "cache_file": str(self.cache_path)
        }

    def clear(self) -> None:
        self._cache = {}
        self._save_cache()
        logger.info("Cache cleared")


class ConversationMemory:
    def __init__(self, memory_dir: str = "conversations"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

    def _get_thread_path(self, thread_id: str) -> Path:
        return self.memory_dir / f"{thread_id}.json"

    def create_thread(self, thread_id: str, metadata: Optional[dict] = None) -> dict:
        thread_path = self._get_thread_path(thread_id)

        thread_data = {
            "thread_id": thread_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "turns": [],
            "pending_clarification": None
        }

        with open(thread_path, "w", encoding="utf-8") as f:
            json.dump(thread_data, f, indent=2, default=str)

        logger.info(f"Created new thread: {thread_id}")
        return thread_data

    def load_thread(self, thread_id: str) -> Optional[dict]:
        thread_path = self._get_thread_path(thread_id)

        if not thread_path.exists():
            return None

        try:
            with open(thread_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load thread {thread_id}: {e}")
            return None

    def save_thread(self, thread_id: str, thread_data: dict) -> None:
        thread_path = self._get_thread_path(thread_id)
        thread_data["updated_at"] = datetime.now().isoformat()

        with open(thread_path, "w", encoding="utf-8") as f:
            json.dump(thread_data, f, indent=2, default=str)

    def add_turn(
        self,
        thread_id: str,
        user_query: str,
        response: str,
        state_snapshot: Optional[dict] = None
    ) -> None:
        thread_data = self.load_thread(thread_id)
        if not thread_data:
            thread_data = self.create_thread(thread_id)

        turn = {
            "turn_number": len(thread_data["turns"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "response": response,
            "state_snapshot": state_snapshot
        }

        thread_data["turns"].append(turn)
        thread_data["pending_clarification"] = None

        self.save_thread(thread_id, thread_data)
        logger.debug(f"Added turn {turn['turn_number']} to thread {thread_id}")

    def set_pending_clarification(
        self,
        thread_id: str,
        clarification_question: str,
        partial_state: dict
    ) -> None:
        thread_data = self.load_thread(thread_id)
        if not thread_data:
            thread_data = self.create_thread(thread_id)

        thread_data["pending_clarification"] = {
            "question": clarification_question,
            "asked_at": datetime.now().isoformat(),
            "partial_state": partial_state
        }

        self.save_thread(thread_id, thread_data)
        logger.info(f"Thread {thread_id} awaiting clarification")

    def get_pending_clarification(self, thread_id: str) -> Optional[dict]:
        thread_data = self.load_thread(thread_id)
        if thread_data:
            return thread_data.get("pending_clarification")
        return None

    def get_conversation_context(
        self,
        thread_id: str,
        max_turns: int = 5
    ) -> list[dict]:
        thread_data = self.load_thread(thread_id)
        if not thread_data:
            return []

        turns = thread_data.get("turns", [])
        return turns[-max_turns:] if turns else []

    def list_threads(self) -> list[dict]:
        threads = []
        for path in self.memory_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    threads.append({
                        "thread_id": data.get("thread_id"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "turn_count": len(data.get("turns", [])),
                        "has_pending": data.get("pending_clarification") is not None
                    })
            except Exception:
                continue

        return sorted(threads, key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete_thread(self, thread_id: str) -> bool:
        thread_path = self._get_thread_path(thread_id)
        if thread_path.exists():
            thread_path.unlink()
            logger.info(f"Deleted thread: {thread_id}")
            return True
        return False


_plan_cache: Optional[PlanCache] = None
_conversation_memory: Optional[ConversationMemory] = None


def get_plan_cache() -> PlanCache:
    global _plan_cache
    if _plan_cache is None:
        _plan_cache = PlanCache()
    return _plan_cache


def get_conversation_memory() -> ConversationMemory:
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory
