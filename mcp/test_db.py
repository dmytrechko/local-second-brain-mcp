"""Tests for db.py — DuckDB layer."""

import os
import pytest
from embeddings import get_embedding
import db

DB = "./test_brain.duckdb"


@pytest.fixture(autouse=True)
def fresh_db():
    if os.path.exists(DB):
        os.unlink(DB)
    db.init_db(DB)
    yield
    if os.path.exists(DB):
        os.unlink(DB)


FAKE_EMB = [0.1] * 1536


def _insert(filepath="test/memory.md", title="Test", content="hello world content", tags=None):
    return db.upsert_memory(DB, filepath, title, content, tags or ["test"], FAKE_EMB, "2026-01-01")


def test_upsert_and_get():
    _insert()
    result = db.get_memory_by_filepath(DB, "test/memory.md")
    assert result is not None
    assert result["title"] == "Test"
    assert result["content"] == "hello world content"


def test_upsert_updates_existing():
    _insert()
    db.upsert_memory(DB, "test/memory.md", "Updated", "new content", ["updated"], [], FAKE_EMB, "2026-01-02")
    result = db.get_memory_by_filepath(DB, "test/memory.md")
    assert result["title"] == "Updated"
    assert result["content"] == "new content"


def test_delete():
    _insert()
    assert db.delete_memory(DB, "test/memory.md") is True
    assert db.get_memory_by_filepath(DB, "test/memory.md") is None


def test_delete_nonexistent():
    assert db.delete_memory(DB, "does/not/exist.md") is False


def test_get_all_filepaths():
    _insert("a.md")
    _insert("b.md")
    fps = db.get_all_filepaths(DB)
    assert "a.md" in fps
    assert "b.md" in fps


def test_get_all_memories():
    _insert("a.md")
    _insert("b.md")
    mems = db.get_all_memories(DB)
    assert len(mems) == 2


def test_get_recent_memories():
    _insert("a.md")
    db.upsert_memory(DB, "b.md", "B", "body", ["test"], FAKE_EMB, "2026-06-01")
    recent = db.get_recent_memories(DB, limit=1)
    assert recent[0]["filepath"] == "b.md"


def test_get_recent_memories_subfolder():
    _insert("projects/a.md")
    _insert("me/b.md")
    results = db.get_recent_memories(DB, subfolder="projects")
    assert all(r["filepath"].startswith("projects/") for r in results)


def test_tag_filter():
    _insert("a.md", tags=["python", "ai"])
    _insert("b.md", tags=["health"])
    results = db.get_memories_by_tags(DB, ["python"])
    assert len(results) == 1
    assert results[0]["filepath"] == "a.md"


def test_tag_filter_match_all():
    _insert("a.md", tags=["python", "ai"])
    _insert("b.md", tags=["python"])
    results = db.get_memories_by_tags(DB, ["python", "ai"], match_all=True)
    assert len(results) == 1
    assert results[0]["filepath"] == "a.md"


def test_get_tag_counts():
    _insert("a.md", tags=["python", "ai"])
    _insert("b.md", tags=["python"])
    counts = db.get_tag_counts(DB)
    assert counts["python"] == 2
    assert counts["ai"] == 1


def test_vector_search():
    emb_a = get_embedding("machine learning neural networks")
    emb_b = get_embedding("cooking recipes pasta")
    db.upsert_memory(DB, "a.md", "A", "machine learning neural networks", ["test"], emb_a, "2026-01-01")
    db.upsert_memory(DB, "b.md", "B", "cooking recipes pasta", ["test"], emb_b, "2026-01-01")
    query_emb = get_embedding("neural networks")
    results = db.search_by_vector(DB, query_emb, limit=2)
    assert len(results) > 0
    assert results[0]["filepath"] == "a.md"


def test_fts_search():
    _insert("a.md", content="Python is a programming language")
    _insert("b.md", content="Rust is a systems language")
    db.rebuild_fts_index(DB)
    results = db.search_fts(DB, "Python")
    assert len(results) == 1
    assert results[0]["filepath"] == "a.md"


def test_fts_search_no_results():
    _insert("a.md", content="something unrelated")
    db.rebuild_fts_index(DB)
    results = db.search_fts(DB, "xyznotfound")
    assert results == []
