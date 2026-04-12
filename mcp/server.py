"""
Second Brain MCP Server

Tools exposed:
  - remember        : create a new memory file + index it
  - update_memory   : append content, replace a section, or update tags in an existing memory
  - search          : unified search — semantic + full-text + tag filter + recent
  - get_related     : find memories related via shared entities
  - get_memory      : fetch full content of a specific memory by filepath
  - memory_overview : global overview — folders, tags, total count, unsynced files
  - list_memories   : list all memories optionally filtered by subfolder
  - forget          : delete a memory file + remove from index
  - list_tasks      : list open (or all) tasks, optionally filtered by context
  - complete_task   : mark a task as done
  - add_task        : add a new task to a memory file
  - set_volatility  : set how likely a note/section is to go stale, 0-1
  - self_reflect    : synthesize brain health — staleness, orphans, task/reminder load
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastmcp import FastMCP

import db
import embeddings as emb
import extract as ext
import parser as par
from tasks import extract_tasks


# ── Config ────────────────────────────────────────────────────────────────────

MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "../memory")).resolve()
DB_PATH = os.getenv("DB_PATH", "./brain.duckdb")

# Ensure DB is initialised on startup
db.init_db(DB_PATH)

mcp = FastMCP("second-brain")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filepath(relative: str) -> Path:
    """Resolve a relative path safely inside MEMORY_DIR."""
    path = (MEMORY_DIR / relative).resolve()
    if not str(path).startswith(str(MEMORY_DIR)):
        raise ValueError("Path escapes memory directory")
    return path


def _format_memory(m: dict, include_content: bool = False) -> dict:
    result = {
        "filepath": m["filepath"],
        "title": m["title"],
        "tags": json.loads(m["tags"]) if isinstance(m["tags"], str) else m["tags"],
        "updated_at": m["updated_at"],
    }
    if include_content:
        result["content"] = m["content"]
    if "distance" in m:
        result["relevance_score"] = round(1 - float(m["distance"]), 4)
    return result


def _index_file(filepath: Path) -> tuple[dict, int]:
    """Parse + embed + upsert + extract entities for a single file. Returns (parsed, memory_id)."""
    parsed = par.parse_memory_file(filepath)
    embedding = emb.get_embedding(parsed["embed_text"])
    rel = par.relative_filepath(MEMORY_DIR, filepath)
    memory_id = db.upsert_memory(
        db_path=DB_PATH,
        filepath=rel,
        title=parsed["title"],
        content=parsed["content"],
        tags=parsed["tags"],
        embedding=embedding,
        updated_at=parsed["updated_at"],
    )
    entities, relations = ext.extract_entities_and_relations(parsed["title"], parsed["content"])
    if entities:
        db.upsert_entities_relations(DB_PATH, memory_id, entities, relations)
    return parsed, memory_id


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def remember(
    title: str,
    content: str,
    tags: list[str],
    subfolder: str = "",
    volatility: float = 0.5,
) -> str:
    """
    Create a new memory and add it to the second brain.

    Args:
        title:      Title of the memory (also used as filename).
        content:    Markdown body of the memory.
        tags:       List of tags to categorise this memory.
        subfolder:  Subfolder inside memory/ (e.g. "me", "projects", "health", "personal", "people", "daily").
                    Can be a new folder name — it will be created automatically.
        volatility: How likely this content is to go stale (0.0–1.0).
                    0.1 = stable reference (concepts, setup), 0.5 = default,
                    0.9 = fast-changing (project status, tasks, decisions).

    Returns the filepath of the created memory.
    """
    # Build filename from title
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    filename = f"{slug}.md"

    target_dir = (MEMORY_DIR / subfolder).resolve() if subfolder else MEMORY_DIR
    if not str(target_dir).startswith(str(MEMORY_DIR)):
        raise ValueError("Invalid subfolder: path escapes memory directory")
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename

    # Build frontmatter + body
    tag_list = "\n".join(f"  - {t}" for t in tags)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    file_content = f"""---
tags:
{tag_list}
created: {now}
---

# {title}

{content}
"""

    filepath.write_text(file_content, encoding="utf-8")

    parsed, memory_id = _index_file(filepath)
    rel = par.relative_filepath(MEMORY_DIR, filepath)
    sections = par.split_sections(parsed["content"])
    db.upsert_sections(DB_PATH, memory_id, rel, sections, parsed["updated_at"])
    db.set_section_volatility(DB_PATH, rel, "", max(0.0, min(1.0, volatility)))

    return f"Memory created and indexed: {rel}"


@mcp.tool()
def update_memory(
    filepath: str,
    mode: str,
    content: str = "",
    section: str = "",
    tags: list[str] | None = None,
) -> str:
    """
    Update an existing memory file. Three modes:

    - "append"          : Add content to the end of the file.
    - "replace_section" : Replace a named markdown section (## Heading) with new content.
                          Requires `section` (the heading text, e.g. "Status") and `content`.
    - "set_tags"        : Replace all tags with a new list. Requires `tags`.

    Args:
        filepath: Relative path of the memory to update (e.g. "me/tasks.md").
        mode:     One of: "append", "replace_section", "set_tags".
        content:  New content to append or replace with (for append / replace_section).
        section:  Heading text of the section to replace (for replace_section), without ##.
        tags:     New tag list (for set_tags).

    Returns a confirmation or error message.
    """
    abs_path = _safe_filepath(filepath)
    if not abs_path.exists():
        return f"Error: file not found: {filepath}"

    text = abs_path.read_text(encoding="utf-8")

    if mode == "append":
        if not content:
            return "Error: content is required for append mode"
        separator = "\n" if text.endswith("\n") else "\n\n"
        text = text + separator + content.strip() + "\n"

    elif mode == "replace_section":
        if not section or not content:
            return "Error: section and content are required for replace_section mode"

        lines = text.splitlines(keepends=True)
        section_name = section.strip().lower()

        # Find all matching heading lines and their level (##, ###, etc.)
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
        matches = []
        for i, line in enumerate(lines):
            m = heading_re.match(line.rstrip("\n\r"))
            if m and m.group(2).lower() == section_name:
                matches.append((i, len(m.group(1))))  # (line_index, heading_level)

        if not matches:
            # Section not found — append it
            separator = "\n" if text.endswith("\n") else "\n\n"
            text = text + separator + f"## {section.strip()}\n\n" + content.strip() + "\n"
        else:
            if len(matches) > 1:
                # Multiple matches — update the last one (most likely the current/active section)
                target_idx, heading_level = matches[-1]
            else:
                target_idx, heading_level = matches[0]

            # Find where the section body ends: next heading of same or higher level, or EOF
            end_idx = len(lines)
            for i in range(target_idx + 1, len(lines)):
                m = heading_re.match(lines[i].rstrip("\n\r"))
                if m and len(m.group(1)) <= heading_level:
                    end_idx = i
                    break

            new_section_lines = [lines[target_idx], "\n"] + [
                l + ("\n" if not l.endswith("\n") else "")
                for l in (content.strip() + "\n").splitlines(keepends=True)
            ]
            # Preserve blank line before next heading if there was one
            if end_idx < len(lines):
                new_section_lines.append("\n")

            lines = lines[:target_idx] + new_section_lines + lines[end_idx:]
            text = "".join(lines)

            if len(matches) > 1:
                return (
                    f"Warning: {len(matches)} sections named '{section}' found — "
                    f"updated the last one. Consider using unique section names.\n"
                    f"Memory updated (replace_section): {filepath}"
                )

    elif mode == "set_tags":
        if tags is None:
            return "Error: tags list is required for set_tags mode"
        tag_lines = "\n".join(f"  - {t}" for t in tags)
        new_tags_block = f"tags:\n{tag_lines}"
        # Replace existing tags block in frontmatter
        pattern = re.compile(r"tags:\n(?:  - .+\n)+", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(new_tags_block + "\n", text)
        else:
            # No tags block yet — insert after opening ---
            text = re.sub(r"(---\n)", r"\1" + new_tags_block + "\n", text, count=1)

    else:
        return f"Error: unknown mode '{mode}'. Use: append, replace_section, set_tags"

    abs_path.write_text(text, encoding="utf-8")
    parsed, memory_id = _index_file(abs_path)
    sections = par.split_sections(parsed["content"])
    db.upsert_sections(DB_PATH, memory_id, filepath, sections, parsed["updated_at"])
    return f"Memory updated ({mode}): {filepath}"


@mcp.tool()
def search(
    query: str = "",
    tags: list[str] | None = None,
    subfolder: str = "",
    limit: int = 8,
) -> list[dict]:
    """
    Unified search across all memories.

    Behaviour:
      - query provided  : runs semantic (vector) AND full-text (FTS) search,
                          merges and deduplicates results.
      - tags provided   : filters to memories with those tags.
      - subfolder       : limits to a folder (e.g. "projects", "me").
      - nothing passed  : returns most recently updated memories.

    Args:
        query:     Natural language or exact keyword/ID query (optional).
        tags:      Tag filter (optional).
        subfolder: Folder filter (optional).
        limit:     Max results (default 8).
    """
    tag_filter = tags or None
    folder = subfolder or None

    if not query:
        if tag_filter:
            results = db.get_memories_by_tags(DB_PATH, tag_filter, match_all=False)
            if folder:
                results = [r for r in results if r["filepath"].startswith(folder.rstrip("/") + "/")]
            return [_format_memory(r, include_content=True) for r in results[:limit]]
        return [_format_memory(r, include_content=True) for r in db.get_recent_memories(DB_PATH, limit=limit, subfolder=folder)]

    # Semantic search
    embedding = emb.get_embedding(query)
    semantic = db.search_by_vector(DB_PATH, embedding, limit=limit, tag_filter=tag_filter, subfolder=folder)

    # Full-text search
    try:
        fts = db.search_fts(DB_PATH, query, tag_filter=tag_filter, subfolder=folder, limit=limit)
    except Exception:
        fts = []

    # Merge: semantic first, then FTS-only hits
    seen = {r["filepath"] for r in semantic}
    merged = list(semantic)
    for r in fts:
        if r["filepath"] not in seen:
            merged.append(r)
            seen.add(r["filepath"])

    now = datetime.now(tz=timezone.utc).isoformat()
    results = merged[:limit]
    for r in results:
        db.touch_sections_accessed(DB_PATH, r["filepath"], now)
    headings_map = db.get_section_headings(DB_PATH, [r["filepath"] for r in results])
    out = [_format_memory(r, include_content=True) for r in results]
    for item in out:
        item["sections"] = headings_map.get(item["filepath"], [])
    return out


@mcp.tool()
def get_related(filepath: str) -> dict:
    """
    Find memories related to a given memory via shared extracted entities.

    Args:
        filepath: Relative path of the memory (e.g. "projects/my-app.md").

    Returns the source memory, its entities, and other memories that share those entities.
    """
    memory = db.get_memory_by_filepath(DB_PATH, filepath)
    if not memory:
        return {"error": f"Memory not found: {filepath}"}

    entities = db.get_entities_for_memory(DB_PATH, memory["id"])
    related = db.get_related_by_entities(DB_PATH, memory["id"])
    related = [r for r in related if r["filepath"] != filepath]

    now = datetime.now(tz=timezone.utc).isoformat()
    db.touch_sections_accessed(DB_PATH, filepath, now)

    return {
        "source": _format_memory(memory, include_content=True),
        "entities": entities,
        "related": [
            {**_format_memory(r), "shared_entities": r["shared_entities"]}
            for r in related
        ],
    }


@mcp.tool()
def get_memory(filepath: str, section: str = "") -> dict:
    """
    Retrieve the full content of a specific memory by its filepath.

    Args:
        filepath: Relative path of the memory (e.g. "concepts/event-sourcing.md").
        section:  Optional H2 heading to return only that section's content.
                  If omitted, the full file is returned.
                  Call without section first to see available sections, then
                  call again with a specific section to read just that part.

    Returns the memory with content and a list of available section headings.
    """
    memory = db.get_memory_by_filepath(DB_PATH, filepath)
    if not memory:
        return {"error": f"Memory not found: {filepath}"}

    now = datetime.now(tz=timezone.utc).isoformat()
    db.touch_sections_accessed(DB_PATH, filepath, now)

    all_sections = par.split_sections(memory["content"])
    section_headings = [s["heading"] for s in all_sections if s["heading"]]

    if section:
        match = next((s for s in all_sections if s["heading"].lower() == section.lower()), None)
        if not match:
            return {
                "error": f"Section '{section}' not found in {filepath}",
                "available_sections": section_headings,
            }
        result = _format_memory(memory)
        result["section"] = section
        result["content"] = match["body"]
        result["available_sections"] = section_headings
        return result

    result = _format_memory(memory, include_content=True)
    result["available_sections"] = section_headings
    return result


@mcp.tool()
def forget(filepath: str) -> str:
    """
    Delete a memory from the index and remove its markdown file.

    Args:
        filepath: Relative path of the memory to delete (e.g. "people/john-doe.md").

    Returns a confirmation or error message.
    """
    abs_path = _safe_filepath(filepath)

    removed_from_db = db.delete_memory(DB_PATH, filepath)

    if abs_path.exists():
        abs_path.unlink()
        file_status = "file deleted"
    else:
        file_status = "file not found on disk (already removed?)"

    if removed_from_db:
        return f"Memory forgotten: {filepath} ({file_status})"
    return f"Memory not in index (may not have been synced): {filepath} ({file_status})"


@mcp.tool()
def memory_overview() -> dict:
    """
    Get a high-level overview of the second brain.
    Returns total memory count, folder breakdown, tag counts, active tasks, and upcoming reminders.
    Use this at the start of a session to orient yourself before searching.
    """
    all_mems = db.get_all_memories(DB_PATH)

    # Folder breakdown
    folders: dict[str, int] = {}
    for m in all_mems:
        folder = str(Path(m["filepath"]).parent)
        folders[folder] = folders.get(folder, 0) + 1

    # Tags with counts
    tags = db.get_tag_counts(DB_PATH)

    # Active tasks — top 10, grouped by project
    active_tasks = db.get_tasks(DB_PATH, status="active", limit=10)
    waiting_tasks = db.get_tasks(DB_PATH, status="waiting", limit=5)

    # Upcoming + overdue reminders
    today = datetime.now().strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    upcoming_reminders = db.get_reminders(DB_PATH, status="pending", due_after=today, due_before=future)
    overdue_reminders = db.get_reminders(DB_PATH, status="pending", due_before=today)

    return {
        "total_memories": len(all_mems),
        "folders": dict(sorted(folders.items())),
        "tags": tags,
        "tasks": {
            "active": active_tasks,
            "waiting": waiting_tasks,
        },
        "reminders": {
            "overdue": overdue_reminders,
            "upcoming_14d": upcoming_reminders,
        },
    }


@mcp.tool()
def list_memories(subfolder: str = "") -> list[dict]:
    """
    List all indexed memories, optionally filtered to a subfolder.

    Args:
        subfolder: Optional subfolder prefix to filter by (e.g. "projects").

    Returns a list of memory summaries (no content body).
    """
    all_mems = db.get_all_memories(DB_PATH)
    if subfolder:
        prefix = subfolder.rstrip("/") + "/"
        all_mems = [m for m in all_mems if m["filepath"].startswith(prefix)]
    headings_map = db.get_section_headings(DB_PATH, [m["filepath"] for m in all_mems])
    out = [_format_memory(m) for m in all_mems]
    for item in out:
        item["sections"] = headings_map.get(item["filepath"], [])
    return out


@mcp.tool()
def get_reminders(overdue: bool = False, upcoming_days: int = 7, status: str = "pending") -> list[dict]:
    """
    Get reminders extracted from memories.

    Args:
        overdue: If true, only return reminders past their due date.
        upcoming_days: Number of days ahead to look for upcoming reminders (default 7).
        status: Filter by status (pending, done, snoozed). Default "pending".

    Returns reminders with text, due_date, status, and source memory info.
    """
    from datetime import datetime, timedelta
    from reminders import get_overdue_reminders, get_upcoming_reminders

    if overdue:
        return get_overdue_reminders(DB_PATH)

    today = datetime.now().strftime('%Y-%m-%d')
    future = (datetime.now() + timedelta(days=upcoming_days)).strftime('%Y-%m-%d')
    return db.get_reminders(DB_PATH, status=status, due_after=today, due_before=future)


@mcp.tool()
def get_all_reminders() -> list[dict]:
    """
    Get all reminders regardless of date, sorted by due date.
    Useful for seeing the full reminder list.
    """
    return db.get_reminders(DB_PATH, limit=100)


@mcp.tool()
def create_reminder(filepath: str, text: str, due_date: str) -> str:
    """
    Create a reminder linked to an existing memory.

    Args:
        filepath: Relative path of the memory to attach the reminder to (e.g. "projects/acme.md").
        text:     Reminder description.
        due_date: Due date in YYYY-MM-DD format.

    Returns a confirmation or error message.
    """
    memory = db.get_memory_by_filepath(DB_PATH, filepath)
    if not memory:
        return f"Error: memory not found: {filepath}"

    try:
        datetime.strptime(due_date, "%Y-%m-%d")
    except ValueError:
        return f"Error: due_date must be in YYYY-MM-DD format, got: {due_date}"

    db.upsert_reminders(DB_PATH, memory["id"], [{"text": text, "due_date": due_date, "status": "pending"}])
    return f"Reminder created: '{text}' due {due_date} → {filepath}"


@mcp.tool()
def update_reminder(reminder_id: int, status: str) -> str:
    """
    Update a reminder's status (pending, done, snoozed).

    Args:
        reminder_id: The ID of the reminder to update.
        status: New status value.

    Returns confirmation message.
    """
    db.update_reminder_status(DB_PATH, reminder_id, status)
    return f"Reminder {reminder_id} marked as {status}"


# ── Volatility & reflection ───────────────────────────────────────────────

@mcp.tool()
def set_volatility(filepath: str, volatility: float, section: str = "") -> str:
    """
    Set how likely a note (or specific section) is to go stale.

    Args:
        filepath:   Relative path of the memory.
        volatility: 0.0–1.0. 0.1 = stable (concepts, setup), 0.5 = default,
                    0.9 = fast-changing (project status, decisions, task lists).
        section:    Optional H2 heading to scope to one section. Empty = all sections.

    Returns confirmation.
    """
    v = max(0.0, min(1.0, volatility))
    db.set_section_volatility(DB_PATH, filepath, section, v)
    scope = f"section '{section}'" if section else "all sections"
    return f"Volatility set to {v} for {scope} of {filepath}"


@mcp.tool()
def self_reflect(focus: str = "") -> dict:
    """
    Synthesize the health of the second brain.

    Surfaces stale sections, never-accessed notes, active task/reminder load,
    and orphan notes (no entities, never read).

    Args:
        focus: Optional keyword to narrow reflection (e.g. "acme", "health").
               Empty = whole brain.

    Returns a structured reflection report for the AI to reason over.
    """
    now = datetime.now(tz=timezone.utc).isoformat()

    # Stale sections
    stale = db.get_stale_sections(DB_PATH, limit=10)
    if focus:
        stale = [s for s in stale if focus.lower() in s["filepath"].lower()
                 or focus.lower() in s["heading"].lower()]

    # Never-accessed memories
    all_mems = db.get_all_memories(DB_PATH)
    never_accessed = []
    for m in all_mems:
        fp = m["filepath"]
        if focus and focus.lower() not in fp.lower():
            continue
        conn_check = db.get_connection(DB_PATH)
        row = conn_check.execute(
            "SELECT COUNT(*) FROM sections WHERE filepath = ? AND (accessed_at = '' OR accessed_at IS NULL)",
            [fp],
        ).fetchone()
        total = conn_check.execute(
            "SELECT COUNT(*) FROM sections WHERE filepath = ?", [fp]
        ).fetchone()
        conn_check.close()
        if total and total[0] > 0 and row and row[0] == total[0]:
            never_accessed.append({"filepath": fp, "title": m["title"], "updated_at": m["updated_at"]})

    # Task + reminder load
    active_tasks = db.get_tasks(DB_PATH, status="active", limit=50)
    today = datetime.now().strftime("%Y-%m-%d")
    overdue = db.get_reminders(DB_PATH, status="pending", due_before=today)

    return {
        "stale_sections": stale,
        "never_accessed": never_accessed[:10],
        "task_load": {
            "active_count": len(active_tasks),
            "by_project": _group_tasks_by_project(active_tasks),
        },
        "overdue_reminders": overdue,
        "focus": focus or "whole brain",
    }


def _group_tasks_by_project(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tasks:
        p = t["project"] or "General"
        counts[p] = counts.get(p, 0) + 1
    return counts


# ── Tasks ────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_tasks(status: str = "active", project: str = "") -> list[dict]:
    """
    List tasks from task notes (notes tagged 'tasks').

    Args:
        status:  Filter by status: "active" (default), "waiting", "backlog", "done", or "" for all.
        project: Optional project/area filter (e.g. "Project", "Personal"). Partial match.

    Returns a list of tasks with id, text, status, project, and source filepath.
    """
    return db.get_tasks(
        DB_PATH,
        status=status or None,
        project=project or None,
    )


@mcp.tool()
def complete_task(task_id: int) -> str:
    """
    Mark a task as done by its ID.

    Args:
        task_id: The ID of the task (from list_tasks).

    Returns confirmation message.
    """
    db.update_task_status(DB_PATH, task_id, "done")
    return f"Task {task_id} marked as done"


@mcp.tool()
def add_task(text: str, project: str = "", status: str = "active", filepath: str = "tasks.md") -> str:
    """
    Add a new task to the task note (tasks.md by default).

    Appends a checkbox under the matching ## Status / ### Project section.
    Creates the section if it doesn't exist.

    Args:
        text:     Task description.
        project:  Project or area name (e.g. "Project", "Personal"). Maps to ### heading.
        status:   Status section to add under: "active" (default), "waiting", "backlog".
        filepath: Override target file (default: tasks.md).

    Returns confirmation message.
    """
    path = _safe_filepath(filepath)
    if not path.exists():
        return f"Error: memory not found: {filepath}"

    content = path.read_text(encoding="utf-8")
    status_heading = status.capitalize()
    new_line = f"- [ ] {text}\n"

    if project:
        # Look for ### Project under the right ## Status section
        h2_pattern = re.compile(rf"^## {re.escape(status_heading)}\b.*$", re.MULTILINE | re.IGNORECASE)
        h2_match = h2_pattern.search(content)
        if h2_match:
            # Find ### project heading within this status section
            section_start = h2_match.end()
            next_h2 = re.search(r"^## ", content[section_start:], re.MULTILINE)
            section_end = section_start + next_h2.start() if next_h2 else len(content)
            section = content[section_start:section_end]

            h3_match = re.search(rf"^### {re.escape(project)}\b.*$", section, re.MULTILINE | re.IGNORECASE)
            if h3_match:
                insert_pos = section_start + h3_match.end() + 1
                content = content[:insert_pos] + new_line + content[insert_pos:]
            else:
                # Append new ### project at end of status section
                content = content[:section_end].rstrip() + f"\n\n### {project}\n\n{new_line}" + content[section_end:]
        else:
            content = content.rstrip() + f"\n\n## {status_heading}\n\n### {project}\n\n{new_line}"
    else:
        h2_pattern = re.compile(rf"^## {re.escape(status_heading)}\b.*$", re.MULTILINE | re.IGNORECASE)
        h2_match = h2_pattern.search(content)
        if h2_match:
            insert_pos = content.find("\n", h2_match.end()) + 1
            content = content[:insert_pos] + new_line + content[insert_pos:]
        else:
            content = content.rstrip() + f"\n\n## {status_heading}\n\n{new_line}"

    path.write_text(content, encoding="utf-8")

    parsed, memory_id = _index_file(path)
    sections = par.split_sections(parsed["content"])
    db.upsert_sections(DB_PATH, memory_id, filepath, sections, parsed["updated_at"])
    if "tasks" in parsed["tags"]:
        db.upsert_tasks(DB_PATH, memory_id, extract_tasks(parsed["content"]))

    return f"Task added ({status} / {project or 'no project'}): '{text}'"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
