# Second Brain MCP

Most AI assistants forget everything between sessions. Second Brain MCP fixes that — it gives your AI a persistent, searchable memory that lives in plain markdown files on your machine. No cloud, no API keys, no vendor lock-in.

You write notes in `memory/` (or let the AI write them). The MCP server handles indexing, retrieval, and reflection. Works with any MCP-compatible client: Claude Desktop, Windsurf, Cursor, Claude CLI.

**Key features:**
- **Fully local** — embeddings, entity extraction, and storage all run on your machine
- **Human-readable** — notes are plain markdown, editable in Obsidian or any editor
- **Staleness-aware** — tracks when sections were last updated and accessed; surfaces what's going stale via `self_reflect`
- **Structured tasks** — explicit task management with status/project bucketing, not extracted from arbitrary prose
- **Explicit reminders** — created intentionally via tool, linked to a specific note
- **Section-level reads** — AI can fetch just one `## Section` of a large note instead of loading the whole file

**Stack:** DuckDB · sentence-transformers (all-MiniLM-L6-v2) · GLiNER · FastMCP

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`brew install uv` or `pip install uv`)

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd <repo>/mcp
uv sync
```

### 2. Add your notes

Create markdown files in `memory/` (any subfolder). Recommended structure:

```
memory/
  projects/    — per-project context and decisions
  people/      — contacts and notes on people
  daily/       — daily notes and scratch space
```

Add any subfolders you need — everything is indexed recursively.

### 4. Index for the first time

```bash
uv run python sync.py
```

Re-run after editing notes. To rebuild from scratch:

```bash
uv run python sync.py --clean
```

---

## MCP Tools

**Memory**

| Tool | Description |
|------|-------------|
| `remember` | Create a new memory note and index it immediately |
| `update_memory` | Append content, replace a section, or update tags |
| `search` | Unified semantic + full-text search with optional tag/folder filters |
| `get_memory` | Fetch a note by filepath — optionally just one `## Section` |
| `get_related` | Find related notes via shared entities |
| `list_memories` | List all notes with tags and section headings, optionally filtered by subfolder |
| `memory_overview` | Global overview — folders, tags, active tasks, upcoming reminders |
| `forget` | Delete a note from disk and index |

**Tasks**

| Tool | Description |
|------|-------------|
| `list_tasks` | List tasks filtered by status and/or project |
| `add_task` | Add a task to `tasks.md` under the right status/project heading |
| `complete_task` | Mark a task as done |

**Reminders**

| Tool | Description |
|------|-------------|
| `create_reminder` | Create a reminder linked to a specific note |
| `get_reminders` | Get upcoming or overdue reminders |
| `get_all_reminders` | Get all reminders sorted by due date |
| `update_reminder` | Mark a reminder as done, snoozed, etc. |

**Staleness & Reflection**

| Tool | Description |
|------|-------------|
| `set_volatility` | Mark a note or section as fast-changing (0.9) or stable (0.1) |
| `self_reflect` | Synthesize brain health — stale sections, never-accessed notes, task load |

---

## Client Configuration

### Windsurf / VS Code

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/repo/mcp", "python", "/path/to/repo/mcp/server.py"],
      "env": {
        "MEMORY_DIR": "/path/to/repo/memory",
        "DB_PATH": "/path/to/repo/mcp/brain.duckdb"
      }
    }
  }
}
```

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/repo/mcp", "python", "/path/to/repo/mcp/server.py"],
      "env": {
        "MEMORY_DIR": "/path/to/repo/memory",
        "DB_PATH": "/path/to/repo/mcp/brain.duckdb"
      }
    }
  }
}
```

### Claude CLI

```bash
claude mcp add second-brain \
  uv run --project /path/to/repo/mcp python /path/to/repo/mcp/server.py \
  --env MEMORY_DIR=/path/to/repo/memory \
  --env DB_PATH=/path/to/repo/mcp/brain.duckdb
```

---

## Note Format

```markdown
---
tags:
  - python
  - architecture
created: 2026-04-07
---

# My Note Title

Content goes here.
```

Reminders are created explicitly via the `create_reminder` tool and linked to a specific note filepath. Tasks live in `memory/tasks.md` with `## Status` and `### Project` headings — `add_task` and `complete_task` write directly to that file.

---

## Security

- All DB queries use parameterized statements — no SQL injection risk
- `MEMORY_DIR` is enforced as a sandbox — file operations cannot escape it
- `brain.duckdb` is stored **unencrypted** — keep it on an encrypted volume (macOS FileVault, LUKS, etc.) if your notes are sensitive

## Architecture

```
mcp/
  server.py      — FastMCP server, all tool definitions
  db.py          — DuckDB layer (vector search, FTS, entities, reminders, tasks, sections)
  embeddings.py  — Local sentence-transformers embeddings
  extract.py     — Local GLiNER entity extraction
  parser.py      — Markdown + frontmatter parser, section splitter
  tasks.py       — Structured task parser (status/project headings)
  reminders.py   — Reminder helpers
  sync.py        — CLI re-index script (run after editing notes)
  canvas.py      — Generates reminders.md and entity hubs for Obsidian
```
