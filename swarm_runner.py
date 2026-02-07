#!/usr/bin/env python3
"""Lightweight multi-agent swarm runner for Codex and Claude CLIs.

This script manages a SQLite task queue and dispatches queued tasks to provider
commands defined in `config.json`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SWARM_DIR = Path(__file__).resolve().parent
DB_PATH = SWARM_DIR / "state" / "swarm.db"
RUNS_DIR = SWARM_DIR / "runs"
CONFIG_PATH = SWARM_DIR / "config.json"
SESSIONS_DIR = SWARM_DIR / "sessions"
CURRENT_SESSION_POINTER = SESSIONS_DIR / ".current"

DEFAULT_CONFIG: dict[str, Any] = {
    "runner": {
        "poll_interval_sec": 2,
        "task_timeout_sec": 1800,
    },
    "providers": {
        "codex": {
            "command": [
                "codex",
                "exec",
                "--full-auto",
                "--cd",
                "{repo_path}",
                "{prompt}",
            ]
        },
        "claude": {
            "command": [
                "claude",
                "-p",
                "--permission-mode",
                "bypassPermissions",
                "{prompt}",
            ]
        },
    },
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  role TEXT NOT NULL,
  provider TEXT NOT NULL,
  repo_path TEXT NOT NULL,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  priority INTEGER NOT NULL DEFAULT 100,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 2,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  worker_id TEXT,
  output_path TEXT,
  last_error TEXT,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status_priority
  ON tasks(status, priority, id);
"""

ALLOWED_ROLES = {"planner", "builder", "reviewer", "integrator"}
ALLOWED_STATUS = {"queued", "running", "succeeded", "failed"}
PRINT_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        config = deep_merge(DEFAULT_CONFIG, user_config)
    return config


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_storage(db_path: Path) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect_db(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def make_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def detect_git_branch(repo_path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            branch = proc.stdout.strip()
            if branch:
                return branch
    except FileNotFoundError:
        pass
    return "unknown"


def set_current_session(session_path: Path) -> None:
    relative_name = session_path.name
    CURRENT_SESSION_POINTER.write_text(f"{relative_name}\n", encoding="utf-8")


def resolve_session_path(session_file: str | None) -> Path:
    if session_file:
        explicit = Path(session_file).resolve()
        if explicit.exists():
            return explicit
        raise ValueError(f"Session file not found: {explicit}")

    if CURRENT_SESSION_POINTER.exists():
        pointer_value = CURRENT_SESSION_POINTER.read_text(encoding="utf-8").strip()
        if pointer_value:
            candidate = (SESSIONS_DIR / pointer_value).resolve()
            if candidate.exists():
                return candidate

    candidates = sorted(SESSIONS_DIR.glob("session-*.md"))
    if candidates:
        return candidates[-1].resolve()

    raise ValueError(
        "No session log found. Start one with "
        "'python3 swarm_runner.py session start --goal \"...\"'."
    )


def update_session_metadata(session_path: Path, key: str, value: str) -> None:
    lines = session_path.read_text(encoding="utf-8").splitlines()
    prefix = f"- {key}:"

    for idx, line in enumerate(lines):
        if line.startswith(prefix):
            lines[idx] = f"{prefix} {value}"
            break
    else:
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].startswith("- "):
            insert_at += 1
        lines.insert(insert_at, f"{prefix} {value}")

    session_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_session_lines(session_path: Path, lines: list[str]) -> None:
    prefix = "" if session_path.read_text(encoding="utf-8").endswith("\n") else "\n"
    with session_path.open("a", encoding="utf-8") as handle:
        handle.write(prefix)
        handle.write("\n".join(lines).rstrip())
        handle.write("\n")


def start_session(args: argparse.Namespace) -> Path:
    now = utc_now()
    session_id = make_session_id()
    repo_path = Path(args.repo_path).resolve()
    branch = args.branch or detect_git_branch(repo_path)
    context = args.context.strip() if args.context else "TBD"
    owner = args.owner.strip() if args.owner else "codex"

    session_path = SESSIONS_DIR / f"session-{session_id}.md"
    content = "\n".join(
        [
            f"# Session {session_id}",
            f"- Status: open",
            f"- Owner: {owner}",
            f"- Started: {now}",
            f"- Last Updated: {now}",
            f"- Repo Path: {repo_path}",
            f"- Branch: {branch}",
            f"- Goal: {args.goal.strip()}",
            "",
            "## Context",
            context,
            "",
            "## Next Steps",
            "- Capture the next concrete implementation step.",
            "",
            "## Timeline",
            f"- [{now}] Session opened.",
        ]
    )
    session_path.write_text(content + "\n", encoding="utf-8")
    set_current_session(session_path)
    return session_path


def note_session(args: argparse.Namespace) -> Path:
    session_path = resolve_session_path(args.session_file)
    now = utc_now()
    update_session_metadata(session_path, "Last Updated", now)
    append_session_lines(session_path, [f"- [{now}] {args.text.strip()}"])
    return session_path


def close_session(args: argparse.Namespace) -> Path:
    session_path = resolve_session_path(args.session_file)
    now = utc_now()

    summary = args.summary.strip()
    next_steps = args.next_step or []
    handoff_lines = [
        f"## Handoff ({now})",
        f"Summary: {summary}",
        "Next Steps:",
    ]
    if next_steps:
        handoff_lines.extend([f"- {item.strip()}" for item in next_steps])
    else:
        handoff_lines.append("- TBD")

    append_session_lines(session_path, handoff_lines)
    update_session_metadata(session_path, "Status", "closed")
    update_session_metadata(session_path, "Closed", now)
    update_session_metadata(session_path, "Last Updated", now)
    return session_path


def show_session(args: argparse.Namespace) -> Path:
    session_path = resolve_session_path(args.session_file)
    text = session_path.read_text(encoding="utf-8")
    print(f"# session_file\n{session_path}\n")
    print(text.rstrip())
    return session_path


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise ValueError("Provide only one of --prompt or --prompt-file")
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise ValueError("A prompt is required via --prompt or --prompt-file")


def enqueue_task(db_path: Path, args: argparse.Namespace) -> int:
    role = args.role.lower()
    provider = args.provider.lower()

    if role not in ALLOWED_ROLES:
        raise ValueError(f"Invalid role: {role}")

    prompt = read_prompt(args)
    repo_path = str(Path(args.repo_path).resolve())
    metadata_json = args.metadata_json or "{}"
    try:
        json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--metadata-json must be valid JSON") from exc

    now = utc_now()
    with connect_db(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO tasks (
              title, role, provider, repo_path, prompt, status,
              priority, attempts, max_attempts, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?, ?)
            """,
            (
                args.title,
                role,
                provider,
                repo_path,
                prompt,
                args.priority,
                args.max_attempts,
                now,
                metadata_json,
            ),
        )
        return int(cur.lastrowid)


def list_tasks(db_path: Path, status_filter: str | None, limit: int) -> list[sqlite3.Row]:
    query = """
      SELECT id, title, role, provider, status, priority, attempts, max_attempts,
             created_at, started_at, completed_at
      FROM tasks
    """
    params: list[Any] = []

    if status_filter:
        if status_filter not in ALLOWED_STATUS:
            raise ValueError(f"Invalid status filter: {status_filter}")
        query += " WHERE status = ?"
        params.append(status_filter)

    query += " ORDER BY priority ASC, id ASC LIMIT ?"
    params.append(limit)

    with connect_db(db_path) as conn:
        return list(conn.execute(query, params).fetchall())


def claim_next_task(conn: sqlite3.Connection, worker_id: str) -> sqlite3.Row | None:
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        """
        SELECT *
        FROM tasks
        WHERE status = 'queued'
        ORDER BY priority ASC, id ASC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        conn.commit()
        return None

    now = utc_now()
    updated = conn.execute(
        """
        UPDATE tasks
        SET status = 'running',
            started_at = ?,
            worker_id = ?,
            attempts = attempts + 1
        WHERE id = ? AND status = 'queued'
        """,
        (now, worker_id, row["id"]),
    )

    if updated.rowcount != 1:
        conn.rollback()
        return None

    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (row["id"],)).fetchone()
    conn.commit()
    return row


def render_command(template: list[str], context: dict[str, str]) -> list[str]:
    command = []
    for token in template:
        command.append(token.format(**context))
    return command


def finalize_task_success(
    conn: sqlite3.Connection,
    task_id: int,
    output_path: str,
) -> None:
    conn.execute(
        """
        UPDATE tasks
        SET status = 'succeeded',
            completed_at = ?,
            output_path = ?,
            last_error = NULL
        WHERE id = ?
        """,
        (utc_now(), output_path, task_id),
    )
    conn.commit()


def finalize_task_failure(
    conn: sqlite3.Connection,
    task_row: sqlite3.Row,
    output_path: str,
    error_text: str,
) -> None:
    will_retry = task_row["attempts"] < task_row["max_attempts"]
    next_status = "queued" if will_retry else "failed"

    conn.execute(
        """
        UPDATE tasks
        SET status = ?,
            completed_at = ?,
            output_path = ?,
            last_error = ?
        WHERE id = ?
        """,
        (next_status, utc_now(), output_path, error_text[:4000], task_row["id"]),
    )
    conn.commit()


def run_single_task(
    conn: sqlite3.Connection,
    task_row: sqlite3.Row,
    config: dict[str, Any],
) -> None:
    provider = task_row["provider"]
    provider_cfg = config.get("providers", {}).get(provider)
    if not provider_cfg:
        finalize_task_failure(
            conn,
            task_row,
            "",
            f"Provider '{provider}' is not configured in config.json",
        )
        return

    cmd_template = provider_cfg.get("command")
    if not isinstance(cmd_template, list) or not cmd_template:
        finalize_task_failure(
            conn,
            task_row,
            "",
            f"Provider '{provider}' command template is invalid",
        )
        return

    task_dir = RUNS_DIR / f"task-{task_row['id']}"
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = task_dir / "prompt.md"
    output_file = task_dir / "output.log"

    prompt_file.write_text(task_row["prompt"], encoding="utf-8")

    context = {
        "task_id": str(task_row["id"]),
        "role": task_row["role"],
        "provider": provider,
        "repo_path": task_row["repo_path"],
        "prompt": task_row["prompt"],
        "prompt_file": str(prompt_file),
        "output_file": str(output_file),
    }

    try:
        command = render_command(cmd_template, context)
    except KeyError as exc:
        finalize_task_failure(
            conn,
            task_row,
            str(output_file),
            f"Unknown command template token: {exc}",
        )
        return

    timeout_sec = int(config.get("runner", {}).get("task_timeout_sec", 1800))

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(f"# command\n{' '.join(command)}\n\n")
        handle.flush()

        try:
            proc = subprocess.run(
                command,
                cwd=task_row["repo_path"],
                stdout=handle,
                stderr=handle,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            handle.write(f"\n# exit_code\n{proc.returncode}\n")
            handle.flush()

            if proc.returncode == 0:
                finalize_task_success(conn, int(task_row["id"]), str(output_file))
                return

            finalize_task_failure(
                conn,
                task_row,
                str(output_file),
                f"Provider command failed with exit code {proc.returncode}",
            )
            return

        except subprocess.TimeoutExpired:
            handle.write("\n# timeout\nTask timed out\n")
            handle.flush()
            finalize_task_failure(
                conn,
                task_row,
                str(output_file),
                f"Task timed out after {timeout_sec} seconds",
            )
        except FileNotFoundError as exc:
            handle.write(f"\n# error\n{exc}\n")
            handle.flush()
            finalize_task_failure(
                conn,
                task_row,
                str(output_file),
                f"Command not found: {exc}",
            )


def worker_loop(
    db_path: Path,
    config: dict[str, Any],
    worker_id: str,
    watch: bool,
) -> None:
    poll_interval = float(config.get("runner", {}).get("poll_interval_sec", 2))
    conn = connect_db(db_path)
    try:
        while True:
            task = claim_next_task(conn, worker_id)
            if task is None:
                if watch:
                    time.sleep(poll_interval)
                    continue
                return

            with PRINT_LOCK:
                print(
                    f"[{worker_id}] running task #{task['id']} "
                    f"({task['provider']}/{task['role']})"
                )

            run_single_task(conn, task, config)

            status_row = conn.execute(
                "SELECT status, attempts, max_attempts FROM tasks WHERE id = ?",
                (task["id"],),
            ).fetchone()
            status = status_row["status"] if status_row else "unknown"

            with PRINT_LOCK:
                print(f"[{worker_id}] completed task #{task['id']} status={status}")
    finally:
        conn.close()


def run_workers(db_path: Path, config: dict[str, Any], workers: int, watch: bool) -> None:
    threads = []
    for idx in range(workers):
        worker_id = f"worker-{idx + 1}"
        thread = threading.Thread(
            target=worker_loop,
            args=(db_path, config, worker_id, watch),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swarm runner for Codex + Claude")
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help=f"SQLite database path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=f"Runner config path (default: {CONFIG_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize SQLite state")

    enqueue = subparsers.add_parser("enqueue", help="Enqueue a new task")
    enqueue.add_argument("--title", required=True, help="Human-readable task title")
    enqueue.add_argument("--role", required=True, help="planner|builder|reviewer|integrator")
    enqueue.add_argument("--provider", required=True, help="codex|claude")
    enqueue.add_argument("--repo-path", default=".", help="Target repo path")
    enqueue.add_argument("--prompt", help="Inline prompt text")
    enqueue.add_argument("--prompt-file", help="Path to prompt markdown file")
    enqueue.add_argument("--priority", type=int, default=100, help="Lower runs first")
    enqueue.add_argument("--max-attempts", type=int, default=2, help="Retry cap")
    enqueue.add_argument(
        "--metadata-json",
        help='Optional metadata JSON, e.g. {"branch":"wt-codex-1"}',
    )

    list_cmd = subparsers.add_parser("list", help="List queue items")
    list_cmd.add_argument("--status", help="queued|running|succeeded|failed")
    list_cmd.add_argument("--limit", type=int, default=50)

    run_cmd = subparsers.add_parser("run", help="Run queued tasks")
    run_cmd.add_argument("--workers", type=int, default=2)
    run_cmd.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling for new tasks",
    )

    session_cmd = subparsers.add_parser("session", help="Manage session handoff logs")
    session_sub = session_cmd.add_subparsers(dest="session_action", required=True)

    session_start = session_sub.add_parser("start", help="Start a new session log")
    session_start.add_argument("--goal", required=True, help="Session objective")
    session_start.add_argument("--repo-path", default=".", help="Repo path for this session")
    session_start.add_argument("--branch", help="Branch name (auto-detects if omitted)")
    session_start.add_argument("--owner", default="codex", help="Owner label")
    session_start.add_argument("--context", default="", help="Starting context summary")

    session_note = session_sub.add_parser("note", help="Append a note to the active session")
    session_note.add_argument("--text", required=True, help="Progress note")
    session_note.add_argument("--session-file", help="Specific session file to update")

    session_close = session_sub.add_parser("close", help="Close a session with handoff")
    session_close.add_argument("--summary", required=True, help="Final summary")
    session_close.add_argument(
        "--next-step",
        action="append",
        default=[],
        help="Action item for the next session (repeatable)",
    )
    session_close.add_argument("--session-file", help="Specific session file to close")

    session_show = session_sub.add_parser("show", help="Print the latest session log")
    session_show.add_argument("--session-file", help="Specific session file to show")

    return parser


def print_rows(rows: list[sqlite3.Row]) -> None:
    if not rows:
        print("No tasks found")
        return

    header = (
        "ID",
        "STATUS",
        "PROVIDER",
        "ROLE",
        "PRIORITY",
        "ATTEMPTS",
        "TITLE",
    )
    print("\t".join(header))
    for row in rows:
        attempts = f"{row['attempts']}/{row['max_attempts']}"
        print(
            "\t".join(
                [
                    str(row["id"]),
                    row["status"],
                    row["provider"],
                    row["role"],
                    str(row["priority"]),
                    attempts,
                    row["title"],
                ]
            )
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    config_path = Path(args.config).resolve()

    if args.command == "init":
        init_storage(db_path)
        print(f"Initialized swarm state at {db_path}")
        return

    init_storage(db_path)

    if args.command == "enqueue":
        task_id = enqueue_task(db_path, args)
        print(f"Enqueued task #{task_id}")
        return

    if args.command == "list":
        rows = list_tasks(db_path, args.status, args.limit)
        print_rows(rows)
        return

    if args.command == "session":
        if args.session_action == "start":
            session_path = start_session(args)
            print(f"Started session log: {session_path}")
            return

        if args.session_action == "note":
            session_path = note_session(args)
            print(f"Updated session log: {session_path}")
            return

        if args.session_action == "close":
            session_path = close_session(args)
            print(f"Closed session log: {session_path}")
            return

        if args.session_action == "show":
            show_session(args)
            return

        raise RuntimeError(f"Unknown session action: {args.session_action}")

    if args.command == "run":
        config = load_config(config_path)
        run_workers(db_path, config, args.workers, args.watch)
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
