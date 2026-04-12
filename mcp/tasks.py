"""Task extraction from structured task notes.

Expected format:
    ## Active          ← H2 = status section
    ### Project      ← H3 = project/area (optional)
    - [ ] Do thing     ← open task
    - [x] Done thing   ← completed task

    ## Waiting On
    ## Backlog
    ## Done

Status is derived from the H2 heading:
    "active"   → active
    "waiting"  → waiting
    "backlog" / "someday" → backlog
    "done"     → done
    anything else → active
"""

import re

_TASK_RE = re.compile(r"^- \[( |x|X)\] (.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_H3_RE = re.compile(r"^### (.+)$", re.MULTILINE)

_STATUS_MAP = {
    "active": "active",
    "waiting on": "waiting",
    "waiting": "waiting",
    "backlog": "backlog",
    "someday": "backlog",
    "someday / backlog": "backlog",
    "done": "done",
}


def _normalize_status(heading: str) -> str:
    return _STATUS_MAP.get(heading.strip().lower(), "active")


def extract_tasks(content: str) -> list[dict]:
    """Parse structured task note into list of {text, status, project} dicts."""
    if not content:
        return []

    # Collect all section boundaries: (pos, level, text)
    sections: list[tuple[int, int, str]] = []
    for m in _H2_RE.finditer(content):
        sections.append((m.start(), 2, m.group(1).strip()))
    for m in _H3_RE.finditer(content):
        sections.append((m.start(), 3, m.group(1).strip()))
    sections.sort()

    tasks = []
    for m in _TASK_RE.finditer(content):
        checked = m.group(1).lower() == "x"
        text = m.group(2).strip()
        pos = m.start()

        # Find nearest H2 and H3 before this task
        status_heading = ""
        project = ""
        for s_pos, level, heading in reversed(sections):
            if s_pos >= pos:
                continue
            if level == 3 and not project:
                project = heading
            if level == 2 and not status_heading:
                status_heading = heading
            if status_heading and project:
                break

        tasks.append({
            "text": text,
            "status": "done" if checked else _normalize_status(status_heading),
            "project": project,
        })

    return tasks
