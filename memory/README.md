# Memory Vault

This folder is your second brain — open it in Obsidian or any markdown editor.

## Structure

```
memory/
  projects/    — per-project context, decisions, status
  people/      — notes on people and contacts
  daily/       — daily notes, scratch space
```

Add any subfolders you need — the MCP server indexes everything recursively.

## Tagging

Add tags in the YAML frontmatter of any note:

```yaml
---
tags: [python, architecture, backend]
---
```

## Syncing to the index

After editing notes, run:

```bash
cd mcp && uv run python sync.py
```

This re-reads all markdown files, extracts tags, generates embeddings, and updates the search index.
