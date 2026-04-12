"""DuckDB database layer for the second brain.

Schema:
  memories    — core memory records + embeddings (FLOAT[] column)
  tags        — unique tag names
  memory_tags — many-to-many memory <-> tag
  entities    — extracted named entities (future use)
  relations   — directed entity relations (future use)

Vector search: cosine similarity via numpy (fast for personal vaults).
Full-text search: DuckDB FTS extension (BM25). Index rebuilt after sync.
"""

import json
import re

import numpy as np

import duckdb


def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_db(db_path: str) -> None:
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY,
            filepath   VARCHAR NOT NULL UNIQUE,
            title      VARCHAR NOT NULL,
            content    VARCHAR NOT NULL,
            tags       VARCHAR NOT NULL DEFAULT '[]',
            embedding  FLOAT[],
            updated_at VARCHAR NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id   INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id INTEGER NOT NULL,
            tag_id    INTEGER NOT NULL,
            PRIMARY KEY (memory_id, tag_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id        INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL,
            name      VARCHAR NOT NULL,
            type      VARCHAR NOT NULL DEFAULT 'unknown'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id        INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL,
            subject   VARCHAR NOT NULL,
            predicate VARCHAR NOT NULL,
            object    VARCHAR NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id         INTEGER PRIMARY KEY,
            memory_id  INTEGER NOT NULL,
            text       VARCHAR NOT NULL,
            due_date   VARCHAR NOT NULL,
            status     VARCHAR NOT NULL DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id         INTEGER PRIMARY KEY,
            memory_id  INTEGER NOT NULL,
            text       VARCHAR NOT NULL,
            status     VARCHAR NOT NULL DEFAULT 'active',
            project    VARCHAR NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id           INTEGER PRIMARY KEY,
            memory_id    INTEGER NOT NULL,
            filepath     VARCHAR NOT NULL,
            heading      VARCHAR NOT NULL DEFAULT '',
            content_hash VARCHAR NOT NULL,
            volatility   FLOAT   NOT NULL DEFAULT 0.5,
            updated_at   VARCHAR NOT NULL,
            accessed_at  VARCHAR NOT NULL DEFAULT ''
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS memories_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS tags_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS entities_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS relations_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS reminders_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS tasks_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS sections_seq START 1")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_filepath ON memories(filepath)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_memory ON entities(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_memory ON relations(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_memory ON reminders(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_memory ON tasks(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_filepath ON sections(filepath)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_memory ON sections(memory_id)")


def rebuild_fts_index(db_path: str) -> None:
    """Rebuild FTS index — call after bulk upserts (e.g. end of sync)."""
    conn = get_connection(db_path)
    conn.execute("INSTALL fts; LOAD fts;")
    conn.execute("PRAGMA create_fts_index('memories', 'id', 'title', 'content', overwrite=1)")
    conn.close()


def upsert_memory(
    db_path: str,
    filepath: str,
    title: str,
    content: str,
    tags: list[str],
    embedding: list[float],
    updated_at: str,
) -> int:
    conn = get_connection(db_path)

    existing = conn.execute("SELECT id FROM memories WHERE filepath = ?", [filepath]).fetchone()
    if existing:
        memory_id = existing[0]
        conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", [memory_id])
        conn.execute("""
            UPDATE memories
            SET title=?, content=?, tags=?, embedding=?, updated_at=?
            WHERE id=?
        """, [title, content, json.dumps(tags), embedding, updated_at, memory_id])
    else:
        memory_id = conn.execute("SELECT nextval('memories_seq')").fetchone()[0]
        conn.execute("""
            INSERT INTO memories (id, filepath, title, content, tags, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [memory_id, filepath, title, content, json.dumps(tags), embedding, updated_at])
    for tag in tags:
        row = conn.execute("SELECT id FROM tags WHERE name = ?", [tag]).fetchone()
        if row:
            tag_id = row[0]
        else:
            tag_id = conn.execute("SELECT nextval('tags_seq')").fetchone()[0]
            conn.execute("INSERT INTO tags (id, name) VALUES (?, ?)", [tag_id, tag])
        conn.execute(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            [memory_id, tag_id],
        )

    return memory_id


def delete_memory(db_path: str, filepath: str) -> bool:
    conn = get_connection(db_path)
    row = conn.execute("SELECT id FROM memories WHERE filepath = ?", [filepath]).fetchone()
    if not row:
        return False
    memory_id = row[0]
    conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM entities WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM relations WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM reminders WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM tasks WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM sections WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM memories WHERE id = ?", [memory_id])
    return True


def search_by_vector(
    db_path: str,
    embedding: list[float],
    limit: int = 10,
    tag_filter: list[str] | None = None,
    subfolder: str | None = None,
) -> list[dict]:
    """Cosine similarity search via numpy — no extension conflicts."""
    conn = get_connection(db_path)

    where = "WHERE embedding IS NOT NULL"
    params: list = []
    if tag_filter:
        placeholders = ",".join(["?"] * len(tag_filter))
        where += f" AND id IN (SELECT mt.memory_id FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id WHERE t.name IN ({placeholders}))"
        params.extend(tag_filter)
    if subfolder:
        where += " AND filepath LIKE ?"
        params.append(subfolder.rstrip("/") + "/%")

    rows = conn.execute(
        f"SELECT id, filepath, title, content, tags, updated_at, embedding FROM memories {where}",
        params,
    ).fetchall()

    if not rows:
        return []

    q = np.array(embedding, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-10

    scored = []
    for row in rows:
        emb = np.array(row[7], dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        score = float(np.dot(q, emb))
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "id": r[0], "filepath": r[1], "title": r[2], "content": r[3],
            "tags": r[4], "updated_at": r[5], "distance": 1 - score,
        }
        for score, r in scored[:limit]
    ]


def search_fts(
    db_path: str,
    query: str,
    tag_filter: list[str] | None = None,
    subfolder: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Full-text BM25 search. Requires rebuild_fts_index() to have been called."""
    conn = get_connection(db_path)
    conn.execute("INSTALL fts; LOAD fts;")

    query = re.sub(r'[*\\"\'()|&!<>~^]', ' ', query).strip()

    extra = ""
    params: list = [query]
    if tag_filter:
        placeholders = ",".join(["?"] * len(tag_filter))
        extra += f" AND id IN (SELECT mt.memory_id FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id WHERE t.name IN ({placeholders}))"
        params.extend(tag_filter)
    if subfolder:
        extra += " AND filepath LIKE ?"
        params.append(subfolder.rstrip("/") + "/%")
    params.append(limit)

    rows = conn.execute(f"""
        SELECT id, filepath, title, content, tags, updated_at, score
        FROM (
            SELECT *, fts_main_memories.match_bm25(id, ?) AS score
            FROM memories
        ) sq
        WHERE score IS NOT NULL{extra}
        ORDER BY score DESC
        LIMIT ?
    """, params).fetchall()

    return [
        {
            "id": r[0], "filepath": r[1], "title": r[2], "content": r[3],
            "tags": r[4], "updated_at": r[5],
        }
        for r in rows
    ]


def get_recent_memories(
    db_path: str, limit: int = 10, subfolder: str | None = None
) -> list[dict]:
    conn = get_connection(db_path)
    if subfolder:
        rows = conn.execute(
            "SELECT id, filepath, title, content, tags, updated_at FROM memories WHERE filepath LIKE ? ORDER BY updated_at DESC LIMIT ?",
            [subfolder.rstrip("/") + "/%", limit],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, filepath, title, content, tags, updated_at FROM memories ORDER BY updated_at DESC LIMIT ?",
            [limit],
        ).fetchall()
    return [_row(r) for r in rows]


def get_memory_by_filepath(db_path: str, filepath: str) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT id, filepath, title, content, tags, updated_at FROM memories WHERE filepath = ?",
        [filepath],
    ).fetchone()
    return _row(row) if row else None


def get_memories_by_tags(db_path: str, tags: list[str], match_all: bool = False) -> list[dict]:
    conn = get_connection(db_path)
    placeholders = ",".join(["?"] * len(tags))
    if match_all:
        rows = conn.execute(f"""
            SELECT m.id, m.filepath, m.title, m.content, m.tags, m.updated_at
            FROM memories m
            JOIN memory_tags mt ON mt.memory_id = m.id
            JOIN tags t ON t.id = mt.tag_id
            WHERE t.name IN ({placeholders})
            GROUP BY m.id, m.filepath, m.title, m.content, m.tags, m.updated_at
            HAVING COUNT(DISTINCT t.name) = ?
            ORDER BY m.updated_at DESC
        """, [*tags, len(tags)]).fetchall()
    else:
        rows = conn.execute(f"""
            SELECT DISTINCT m.id, m.filepath, m.title, m.content, m.tags, m.updated_at
            FROM memories m
            JOIN memory_tags mt ON mt.memory_id = m.id
            JOIN tags t ON t.id = mt.tag_id
            WHERE t.name IN ({placeholders})
            ORDER BY m.updated_at DESC
        """, tags).fetchall()
    return [_row(r) for r in rows]


def get_all_memories(db_path: str) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, filepath, title, content, tags, updated_at FROM memories ORDER BY updated_at DESC"
    ).fetchall()
    return [_row(r) for r in rows]


def get_all_filepaths(db_path: str) -> set[str]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT filepath FROM memories").fetchall()
    return {r[0] for r in rows}


def get_tag_counts(db_path: str) -> dict[str, int]:
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT t.name, COUNT(mt.memory_id) AS count
        FROM tags t
        JOIN memory_tags mt ON mt.tag_id = t.id
        GROUP BY t.name
        ORDER BY count DESC, t.name
    """).fetchall()
    return {r[0]: r[1] for r in rows}


def upsert_entities_relations(
    db_path: str,
    memory_id: int,
    entities: list[dict],
    relations: list[dict],
) -> None:
    """Replace all entities and relations for a memory."""
    conn = get_connection(db_path)
    conn.execute("DELETE FROM entities WHERE memory_id = ?", [memory_id])
    conn.execute("DELETE FROM relations WHERE memory_id = ?", [memory_id])
    for ent in entities:
        eid = conn.execute("SELECT nextval('entities_seq')").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (id, memory_id, name, type) VALUES (?, ?, ?, ?)",
            [eid, memory_id, ent.get("name", ""), ent.get("type", "other")],
        )
    for rel in relations:
        rid = conn.execute("SELECT nextval('relations_seq')").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (id, memory_id, subject, predicate, object) VALUES (?, ?, ?, ?, ?)",
            [rid, memory_id, rel.get("subject", ""), rel.get("predicate", ""), rel.get("object", "")],
        )


def get_related_by_entities(db_path: str, memory_id: int) -> list[dict]:
    """Find other memories that share entities with the given memory."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT DISTINCT m.id, m.filepath, m.title, m.content, m.tags, m.updated_at,
               e1.name AS shared_entity, e1.type AS entity_type
        FROM entities e1
        JOIN entities e2 ON lower(e2.name) = lower(e1.name) AND e2.memory_id != e1.memory_id
        JOIN memories m ON m.id = e2.memory_id
        WHERE e1.memory_id = ?
        ORDER BY m.updated_at DESC
    """, [memory_id]).fetchall()

    seen: dict[int, dict] = {}
    for r in rows:
        mid = r[0]
        if mid not in seen:
            seen[mid] = {**_row(r[:7]), "shared_entities": []}
        seen[mid]["shared_entities"].append({"name": r[7], "type": r[8]})
    return list(seen.values())


def get_entities_for_memory(db_path: str, memory_id: int) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT name, type FROM entities WHERE memory_id = ? ORDER BY name",
        [memory_id],
    ).fetchall()
    return [{"name": r[0], "type": r[1]} for r in rows]


def upsert_reminders(
    db_path: str,
    memory_id: int,
    reminders: list[dict],
) -> None:
    """Replace all reminders for a memory. Each reminder: {text, due_date, status}."""
    conn = get_connection(db_path)
    conn.execute("DELETE FROM reminders WHERE memory_id = ?", [memory_id])
    for r in reminders:
        rid = conn.execute("SELECT nextval('reminders_seq')").fetchone()[0]
        conn.execute(
            "INSERT INTO reminders (id, memory_id, text, due_date, status) VALUES (?, ?, ?, ?, ?)",
            [rid, memory_id, r.get("text", ""), r.get("due_date", ""), r.get("status", "pending")],
        )


def get_reminders(
    db_path: str,
    status: str | None = None,
    due_before: str | None = None,
    due_after: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Query reminders with optional filters. Returns list with memory filepath/title."""
    conn = get_connection(db_path)
    where = []
    params: list = []
    if status:
        where.append("r.status = ?")
        params.append(status)
    if due_before:
        where.append("r.due_date <= ?")
        params.append(due_before)
    if due_after:
        where.append("r.due_date >= ?")
        params.append(due_after)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)

    rows = conn.execute(f"""
        SELECT r.id, r.text, r.due_date, r.status, m.filepath, m.title
        FROM reminders r
        JOIN memories m ON m.id = r.memory_id
        {where_sql}
        ORDER BY r.due_date ASC
        LIMIT ?
    """, params).fetchall()

    return [{
        "id": r[0], "text": r[1], "due_date": r[2], "status": r[3],
        "filepath": r[4], "title": r[5],
    } for r in rows]


def update_reminder_status(db_path: str, reminder_id: int, status: str) -> None:
    """Mark reminder as done/snoozed/etc."""
    conn = get_connection(db_path)
    conn.execute("UPDATE reminders SET status = ? WHERE id = ?", [status, reminder_id])


def upsert_tasks(db_path: str, memory_id: int, tasks: list[dict]) -> None:
    """Replace all tasks for a memory. Each task: {text, status, project}."""
    conn = get_connection(db_path)
    conn.execute("DELETE FROM tasks WHERE memory_id = ?", [memory_id])
    for t in tasks:
        tid = conn.execute("SELECT nextval('tasks_seq')").fetchone()[0]
        conn.execute(
            "INSERT INTO tasks (id, memory_id, text, status, project) VALUES (?, ?, ?, ?, ?)",
            [tid, memory_id, t["text"], t.get("status", "active"), t.get("project", "")],
        )


def get_tasks(
    db_path: str,
    status: str | None = None,
    project: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query tasks with optional status/project filter."""
    conn = get_connection(db_path)
    where = []
    params: list = []
    if status:
        where.append("t.status = ?")
        params.append(status)
    if project:
        where.append("lower(t.project) LIKE lower(?)")
        params.append(f"%{project}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = conn.execute(f"""
        SELECT t.id, t.text, t.status, t.project, m.filepath
        FROM tasks t
        JOIN memories m ON m.id = t.memory_id
        {where_sql}
        ORDER BY t.status ASC, t.project ASC
        LIMIT ?
    """, params).fetchall()
    return [{"id": r[0], "text": r[1], "status": r[2], "project": r[3], "filepath": r[4]} for r in rows]


def update_task_status(db_path: str, task_id: int, status: str) -> None:
    conn = get_connection(db_path)
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", [status, task_id])


def upsert_sections(
    db_path: str,
    memory_id: int,
    filepath: str,
    sections: list[dict],
    now: str,
) -> None:
    """Upsert section rows for a file. Only updates updated_at when hash changes.
    Removes sections no longer present in the file (renamed/deleted headings).
    Does NOT overwrite volatility — preserves previously set values.
    """
    conn = get_connection(db_path)
    current_headings = {s["heading"] for s in sections}

    # Remove stale sections (renamed or deleted)
    existing = conn.execute(
        "SELECT heading FROM sections WHERE filepath = ?", [filepath]
    ).fetchall()
    for (heading,) in existing:
        if heading not in current_headings:
            conn.execute(
                "DELETE FROM sections WHERE filepath = ? AND heading = ?",
                [filepath, heading],
            )

    for s in sections:
        row = conn.execute(
            "SELECT id, content_hash, volatility FROM sections WHERE filepath = ? AND heading = ?",
            [filepath, s["heading"]],
        ).fetchone()
        if row:
            sid, old_hash, volatility = row
            if old_hash != s["content_hash"]:
                conn.execute(
                    "UPDATE sections SET content_hash = ?, updated_at = ? WHERE id = ?",
                    [s["content_hash"], now, sid],
                )
        else:
            sid = conn.execute("SELECT nextval('sections_seq')").fetchone()[0]
            conn.execute(
                "INSERT INTO sections (id, memory_id, filepath, heading, content_hash, volatility, updated_at, accessed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [sid, memory_id, filepath, s["heading"], s["content_hash"], 0.5, now, ""],
            )


def touch_sections_accessed(db_path: str, filepath: str, now: str) -> None:
    """Update accessed_at for all sections of a file."""
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE sections SET accessed_at = ? WHERE filepath = ?",
        [now, filepath],
    )


def set_section_volatility(db_path: str, filepath: str, heading: str, volatility: float) -> None:
    """Set volatility for a specific section (or all sections if heading is empty)."""
    conn = get_connection(db_path)
    if heading:
        conn.execute(
            "UPDATE sections SET volatility = ? WHERE filepath = ? AND heading = ?",
            [volatility, filepath, heading],
        )
    else:
        conn.execute(
            "UPDATE sections SET volatility = ? WHERE filepath = ?",
            [volatility, filepath],
        )


def get_section_headings(db_path: str, filepaths: list[str]) -> dict[str, list[str]]:
    """Return {filepath: [heading, ...]} for a list of filepaths. Excludes empty preamble heading."""
    if not filepaths:
        return {}
    conn = get_connection(db_path)
    placeholders = ", ".join("?" for _ in filepaths)
    rows = conn.execute(
        f"SELECT filepath, heading FROM sections WHERE filepath IN ({placeholders}) AND heading != '' ORDER BY id ASC",
        filepaths,
    ).fetchall()
    result: dict[str, list[str]] = {fp: [] for fp in filepaths}
    for fp, heading in rows:
        result[fp].append(heading)
    return result


def get_stale_sections(
    db_path: str,
    limit: int = 10,
) -> list[dict]:
    """Return sections ranked by staleness = days_since_updated * volatility."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT
            s.filepath,
            s.heading,
            s.volatility,
            s.updated_at,
            s.accessed_at,
            m.title,
            date_diff('day', CAST(s.updated_at AS DATE), current_date) AS days_stale
        FROM sections s
        JOIN memories m ON m.id = s.memory_id
        WHERE s.volatility > 0
        ORDER BY date_diff('day', CAST(s.updated_at AS DATE), current_date) * s.volatility DESC
        LIMIT ?
    """, [limit]).fetchall()
    return [{
        "filepath": r[0],
        "heading": r[1] or "(preamble)",
        "volatility": r[2],
        "updated_at": r[3],
        "accessed_at": r[4] or "never",
        "title": r[5],
        "days_stale": r[6],
        "staleness_score": round(r[6] * r[2], 1),
    } for r in rows]


def _row(r: tuple) -> dict:
    return {
        "id": r[0], "filepath": r[1], "title": r[2],
        "content": r[3], "tags": r[4], "updated_at": r[5],
    }
