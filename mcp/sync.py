"""
Manual sync script: re-reads all markdown files, re-embeds, updates DB.

Usage:
    cd mcp
    python sync.py          # incremental sync
    python sync.py --clean  # drop DB and rebuild from scratch
"""

import argparse
import os
import sys
from pathlib import Path

from canvas import generate_entity_hubs, generate_reminders_md
from db import init_db, upsert_memory, get_all_filepaths, delete_memory, rebuild_fts_index, upsert_entities_relations, upsert_tasks, upsert_sections
from embeddings import get_embedding
from extract import extract_entities_and_relations
from parser import find_all_markdown_files, parse_memory_file, relative_filepath, split_sections
from tasks import extract_tasks


def get_config() -> tuple[Path, str]:
    memory_dir = Path(os.getenv("MEMORY_DIR", "../memory")).resolve()
    db_path = os.getenv("DB_PATH", "./brain.duckdb")
    return memory_dir, db_path


def sync(clean: bool = False) -> None:
    memory_dir, db_path = get_config()

    if not memory_dir.exists():
        print(f"ERROR: memory dir not found: {memory_dir}")
        sys.exit(1)

    if clean and Path(db_path).exists():
        Path(db_path).unlink()
        wal = Path(db_path + ".wal")
        if wal.exists():
            wal.unlink()
        print(f"Dropped existing DB: {db_path}")

    init_db(db_path)

    md_files = find_all_markdown_files(memory_dir)
    print(f"Found {len(md_files)} markdown files in {memory_dir}")

    existing_fps = get_all_filepaths(db_path)
    current_fps: set[str] = set()

    indexed = 0
    errors = 0

    for filepath in md_files:
        rel = relative_filepath(memory_dir, filepath)
        current_fps.add(rel)
        try:
            parsed = parse_memory_file(filepath)
            embedding = get_embedding(parsed["embed_text"])
            memory_id = upsert_memory(
                db_path=db_path,
                filepath=rel,
                title=parsed["title"],
                content=parsed["content"],
                tags=parsed["tags"],
                embedding=embedding,
                updated_at=parsed["updated_at"],
            )
            entities, relations = extract_entities_and_relations(parsed["title"], parsed["content"])
            if entities:
                upsert_entities_relations(db_path, memory_id, entities, relations)

            upsert_sections(db_path, memory_id, rel, split_sections(parsed["content"]), parsed["updated_at"])

            if "tasks" in parsed["tags"]:
                tasks = extract_tasks(parsed["content"])
                if tasks:
                    upsert_tasks(db_path, memory_id, tasks)
            else:
                tasks = []

            notes = []
            if entities:
                notes.append(f"{len(entities)} entities")
            if tasks:  # only non-empty if tags contained 'tasks'
                notes.append(f"{len(tasks)} tasks")
            note_str = f"  ({', '.join(notes)})" if notes else ""
            print(f"  ✓ {rel}  [{', '.join(parsed['tags']) or 'no tags'}]{note_str}")
            indexed += 1
        except Exception as e:
            print(f"  ✗ {rel}: {e}")
            errors += 1

    stale = existing_fps - current_fps
    for fp in stale:
        delete_memory(db_path, fp)
        print(f"  - removed stale: {fp}")

    print("Rebuilding FTS index...")
    rebuild_fts_index(db_path)

    print(f"\nSync complete: {indexed} indexed, {len(stale)} removed, {errors} errors")

    generate_entity_hubs(db_path, str(memory_dir))

    reminders_output = str(memory_dir / "reminders.md")
    generate_reminders_md(db_path, reminders_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync memory markdown files to the brain DB.")
    parser.add_argument("--clean", action="store_true", help="Drop DB and rebuild from scratch")
    args = parser.parse_args()
    sync(clean=args.clean)
