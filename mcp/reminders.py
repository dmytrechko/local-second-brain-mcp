"""Reminder query helpers used by server.py tools."""

from datetime import datetime, timedelta


def get_overdue_reminders(db_path: str) -> list[dict]:
    """Get all pending reminders that are past due."""
    from db import get_reminders
    today = datetime.now().strftime('%Y-%m-%d')
    return get_reminders(db_path, status="pending", due_before=today)


def get_upcoming_reminders(db_path: str, days: int = 7) -> list[dict]:
    """Get pending reminders due within specified days."""
    from db import get_reminders
    today = datetime.now().strftime('%Y-%m-%d')
    future = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    return get_reminders(db_path, status="pending", due_after=today, due_before=future)
