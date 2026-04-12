"""Tests for entity/relation extraction and graph queries."""

import os
import pytest
import db
from extract import extract_entities_and_relations, _derive_relations

DB = "./test_extract.duckdb"
FAKE_EMB = [0.1] * 384


@pytest.fixture(autouse=True)
def fresh_db():
    if os.path.exists(DB):
        os.unlink(DB)
    db.init_db(DB)
    yield
    if os.path.exists(DB):
        os.unlink(DB)


def _insert(filepath, title="Test", content="hello", tags=None):
    return db.upsert_memory(DB, filepath, title, content, tags or [], FAKE_EMB, "2026-01-01")


# ── extract.py tests ──────────────────────────────────────────────────────────

def test_extraction_returns_entities():
    entities, relations = extract_entities_and_relations(
        "Acme Project",
        "Acme is a task management app built with Firebase. Alice is the founder.",
    )
    assert isinstance(entities, list)
    assert isinstance(relations, list)
    assert all("name" in e and "type" in e for e in entities)


def test_extraction_returns_relations():
    entities, relations = extract_entities_and_relations(
        "Acme Project",
        "Acme uses Firebase for its backend.",
    )
    assert isinstance(relations, list)
    assert all("subject" in r and "predicate" in r and "object" in r for r in relations)


def test_extraction_empty_content():
    entities, relations = extract_entities_and_relations("Untitled", "")
    assert entities == []
    assert relations == []


def test_derive_relations_co_occurrence():
    entities = [
        {"name": "Acme", "type": "project"},
        {"name": "Firebase", "type": "tool"},
    ]
    text = "Acme uses Firebase for storage."
    relations = _derive_relations(entities, text)
    assert len(relations) == 1
    assert relations[0]["subject"] == "Acme"
    assert relations[0]["object"] == "Firebase"
    assert relations[0]["predicate"] == "related_to"


def test_derive_relations_no_duplicate_pairs():
    entities = [{"name": "A", "type": "concept"}, {"name": "B", "type": "concept"}]
    text = "A and B are here. A and B appear again."
    relations = _derive_relations(entities, text)
    assert len(relations) == 1


# ── db graph query tests ──────────────────────────────────────────────────────

def test_upsert_and_get_entities():
    mid = _insert("a.md", title="Acme")
    db.upsert_entities_relations(
        DB, mid,
        [{"name": "Acme", "type": "project"}, {"name": "Firebase", "type": "tool"}],
        [{"subject": "Acme", "predicate": "uses", "object": "Firebase"}],
    )
    entities = db.get_entities_for_memory(DB, mid)
    assert len(entities) == 2
    names = {e["name"] for e in entities}
    assert "Acme" in names
    assert "Firebase" in names


def test_get_related_by_shared_entity():
    mid_a = _insert("a.md", title="Acme")
    mid_b = _insert("b.md", title="Globex")
    mid_c = _insert("c.md", title="Hobbies")

    db.upsert_entities_relations(DB, mid_a, [{"name": "OpenAI", "type": "tool"}], [])
    db.upsert_entities_relations(DB, mid_b, [{"name": "OpenAI", "type": "tool"}], [])
    db.upsert_entities_relations(DB, mid_c, [{"name": "Gardening", "type": "concept"}], [])

    related = db.get_related_by_entities(DB, mid_a)
    fps = [r["filepath"] for r in related]
    assert "b.md" in fps
    assert "c.md" not in fps


def test_related_includes_shared_entity_names():
    mid_a = _insert("a.md")
    mid_b = _insert("b.md")

    db.upsert_entities_relations(DB, mid_a, [{"name": "Firebase", "type": "tool"}], [])
    db.upsert_entities_relations(DB, mid_b, [{"name": "Firebase", "type": "tool"}], [])

    related = db.get_related_by_entities(DB, mid_a)
    assert len(related) == 1
    assert related[0]["shared_entities"][0]["name"] == "Firebase"


def test_upsert_entities_replaces_existing():
    mid = _insert("a.md")
    db.upsert_entities_relations(DB, mid, [{"name": "OldTool", "type": "tool"}], [])
    db.upsert_entities_relations(DB, mid, [{"name": "NewTool", "type": "tool"}], [])
    entities = db.get_entities_for_memory(DB, mid)
    names = {e["name"] for e in entities}
    assert "NewTool" in names
    assert "OldTool" not in names
