# SPDX-License-Identifier: MIT
"""Tests for agent_memory.py — Agent Memory Layer."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_memory import (
    AgentMemory,
    AgentStats,
    ReferenceType,
    SelfReference,
    TfIdfStore,
    VideoRecord,
)


# ---------------------------------------------------------------------------
# TF-IDF Store
# ---------------------------------------------------------------------------

class TestTfIdfStore:
    def test_add_and_search(self):
        store = TfIdfStore()
        store.add("d1", "PowerPC G4 vintage hardware mining")
        store.add("d2", "Modern x86 benchmark comparison")
        store.add("d3", "PowerPC architecture deep dive")
        results = store.search("PowerPC hardware")
        assert len(results) >= 1
        # d1 or d3 should be top result
        top_id = results[0][0]
        assert top_id in ("d1", "d3")

    def test_empty_store(self):
        store = TfIdfStore()
        assert store.search("anything") == []

    def test_remove(self):
        store = TfIdfStore()
        store.add("d1", "test document")
        store.remove("d1")
        assert store.search("test") == []

    def test_short_words_filtered(self):
        store = TfIdfStore()
        store.add("d1", "a an the is")
        # All words <= 2 chars, should be empty after tokenization
        assert store.search("a an") == []


# ---------------------------------------------------------------------------
# AgentMemory basics
# ---------------------------------------------------------------------------

@pytest.fixture
def mem(tmp_path):
    return AgentMemory(
        agent="test_bot",
        db_path=tmp_path / "memory.db",
        now_fn=lambda: 1700000000.0,
    )


class TestIngest:
    def test_ingest_video(self, mem):
        mem.ingest_video("v1", "My First Video", "Description here", ["tag1"])
        stats = mem.get_stats()
        assert stats.total_videos == 1

    def test_ingest_multiple(self, mem):
        for i in range(5):
            mem.ingest_video(f"v{i}", f"Video {i}", f"Description {i}")
        assert mem.get_stats().total_videos == 5

    def test_ingest_with_opinions(self, mem):
        mem.ingest_video("v1", "Hot Take", opinions=["PowerPC is better"])
        results = mem.search("PowerPC")
        assert len(results) == 0  # opinions not in search text
        # But video is stored
        assert mem.get_stats().total_videos == 1


class TestSearch:
    def test_search_by_title(self, mem):
        mem.ingest_video("v1", "PowerPC G4 Mining Guide", tags=["hardware"])
        mem.ingest_video("v2", "Cooking Pasta Basics", tags=["cooking"])
        results = mem.search("PowerPC hardware")
        assert results[0][0].video_id == "v1"

    def test_search_by_tags(self, mem):
        mem.ingest_video("v1", "Video One", tags=["blockchain", "mining"])
        mem.ingest_video("v2", "Video Two", tags=["cooking", "recipe"])
        results = mem.search("blockchain mining")
        assert results[0][0].video_id == "v1"

    def test_has_covered_topic(self, mem):
        mem.ingest_video("v1", "Deep Dive into Rust Programming")
        assert mem.has_covered_topic("Rust programming") is True
        assert mem.has_covered_topic("quantum physics") is False


# ---------------------------------------------------------------------------
# Self-reference suggestions
# ---------------------------------------------------------------------------

class TestSuggestReference:
    def test_first_time_topic(self, mem):
        ref = mem.suggest_reference("Brand New Topic")
        assert ref is not None
        assert ref.type == ReferenceType.FIRST_TIME

    def test_followup_recent_video(self, tmp_path):
        """Video from 3 days ago → followup reference."""
        now = 1700000000.0
        mem = AgentMemory(
            agent="bot", db_path=tmp_path / "m.db",
            now_fn=lambda: now,
        )
        # Video from 3 days ago
        mem.ingest_video("v1", "PowerPC Architecture Overview",
                         "A deep dive into PowerPC",
                         tags=["hardware", "powerpc"])
        # Hack created_at to 3 days ago
        with sqlite3.connect(str(tmp_path / "m.db")) as conn:
            conn.execute(
                "UPDATE agent_videos SET created_at=? WHERE video_id='v1'",
                (now - 3 * 86400,),
            )
        mem._store = TfIdfStore()
        mem._load_into_store()

        ref = mem.suggest_reference("More PowerPC Thoughts", "thinking about powerpc")
        assert ref is not None
        assert ref.type in (ReferenceType.FOLLOWUP, ReferenceType.CALLBACK,
                            ReferenceType.CHANGED_MIND)
        assert "v1" == ref.related_video_id or ref.related_title is not None

    def test_callback_old_video(self, tmp_path):
        """Video from 30 days ago → callback reference."""
        now = 1700000000.0
        mem = AgentMemory(
            agent="bot", db_path=tmp_path / "m.db",
            now_fn=lambda: now,
        )
        mem.ingest_video("v1", "Vintage Hardware Mining Setup",
                         "How to mine with old computers",
                         tags=["mining", "vintage"])
        with sqlite3.connect(str(tmp_path / "m.db")) as conn:
            conn.execute(
                "UPDATE agent_videos SET created_at=? WHERE video_id='v1'",
                (now - 30 * 86400,),
            )
        mem._store = TfIdfStore()
        mem._load_into_store()

        ref = mem.suggest_reference("Mining with Vintage Computers", "vintage mining setup")
        assert ref is not None
        assert ref.type in (ReferenceType.CALLBACK, ReferenceType.CHANGED_MIND)

    def test_changed_mind_with_opinions(self, tmp_path):
        now = 1700000000.0
        mem = AgentMemory(
            agent="bot", db_path=tmp_path / "m.db",
            now_fn=lambda: now,
        )
        mem.ingest_video("v1", "Why Modern Hardware Wins",
                         "Modern beats vintage",
                         tags=["hardware"],
                         opinions=["Modern hardware is objectively better"])
        ref = mem.suggest_reference("Hardware Comparison", "modern versus vintage hardware")
        assert ref is not None
        assert ref.type == ReferenceType.CHANGED_MIND
        assert "Modern hardware" in ref.text

    def test_series_detection(self, mem):
        mem.ingest_video("v1", "Cooking Series Part 1")
        ref = mem.suggest_reference("Cooking Series Part 2")
        assert ref is not None
        assert ref.type == ReferenceType.SERIES
        assert "Part 2" in ref.text

    def test_milestone_detection(self, tmp_path):
        now = 1700000000.0
        mem = AgentMemory(
            agent="bot", db_path=tmp_path / "m.db",
            now_fn=lambda: now,
        )
        # Ingest 49 videos, then the 50th
        for i in range(50):
            mem.ingest_video(f"v{i}", f"Video number {i}", tags=["stuff"])
        ref = mem.suggest_reference("Milestone Video")
        assert ref is not None
        assert ref.type == ReferenceType.MILESTONE
        assert "50" in ref.text


# Need sqlite3 import for test
import sqlite3


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_stats(self, mem):
        stats = mem.get_stats()
        assert stats.total_videos == 0
        assert stats.first_upload is None

    def test_top_topics(self, mem):
        mem.ingest_video("v1", "A", tags=["hardware", "mining"])
        mem.ingest_video("v2", "B", tags=["hardware", "vintage"])
        mem.ingest_video("v3", "C", tags=["cooking"])
        stats = mem.get_stats()
        # hardware appears 2x, should be top
        assert stats.top_topics[0][0] == "hardware"
        assert stats.top_topics[0][1] == 2

    def test_series_detection_in_stats(self, mem):
        mem.ingest_video("v1", "Cooking Series Part 1")
        mem.ingest_video("v2", "Cooking Series Part 2")
        mem.ingest_video("v3", "Unrelated Video")
        stats = mem.get_stats()
        assert "Cooking Series" in stats.current_series


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_survives_reload(self, tmp_path):
        db = tmp_path / "mem.db"
        m1 = AgentMemory(agent="bot", db_path=db)
        m1.ingest_video("v1", "Test Video", "desc", ["tag"])

        m2 = AgentMemory(agent="bot", db_path=db)
        assert m2.get_stats().total_videos == 1
        results = m2.search("Test Video")
        assert len(results) == 1

    def test_agent_isolation(self, tmp_path):
        db = tmp_path / "mem.db"
        m1 = AgentMemory(agent="bot_a", db_path=db)
        m2 = AgentMemory(agent="bot_b", db_path=db)
        m1.ingest_video("v1", "Bot A Video")
        m2.ingest_video("v2", "Bot B Video")
        assert m1.get_stats().total_videos == 1
        assert m2.get_stats().total_videos == 1


# ---------------------------------------------------------------------------
# Integration: natural self-reference example
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_agent_references_two_week_old_video(self, tmp_path):
        """The bounty example: agent references own video from 2 weeks ago."""
        now = 1700000000.0
        mem = AgentMemory(
            agent="cosmo", db_path=tmp_path / "m.db",
            now_fn=lambda: now,
        )
        # Two weeks ago
        mem.ingest_video("old_vid", "Why SPARC Machines Are Underrated",
                         "A deep look at Sun Microsystems SPARC architecture",
                         tags=["hardware", "sparc", "vintage"])
        with sqlite3.connect(str(tmp_path / "m.db")) as conn:
            conn.execute(
                "UPDATE agent_videos SET created_at=? WHERE video_id='old_vid'",
                (now - 14 * 86400,),
            )
        mem._store = TfIdfStore()
        mem._load_into_store()

        # New video about similar topic
        ref = mem.suggest_reference(
            "SPARC vs PowerPC — Which Aged Better?",
            "Comparing two vintage architectures",
        )
        assert ref is not None
        assert ref.related_video_id == "old_vid"
        assert "SPARC" in ref.related_title
        assert ref.type in (ReferenceType.FOLLOWUP, ReferenceType.CALLBACK,
                            ReferenceType.CHANGED_MIND)
