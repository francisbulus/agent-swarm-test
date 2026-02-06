# Architecture

## Components

1. Queue state (`state/swarm.db`)
2. Worker pool (threads in `swarm_runner.py run`)
3. Provider adapters (`config.json` command templates)
4. Task artifacts (`runs/task-<id>/`)

## Lifecycle

1. `enqueue` inserts a task as `queued`.
2. A worker atomically claims the next queued task (`BEGIN IMMEDIATE` lock).
3. Runner writes prompt to `runs/task-<id>/prompt.md`.
4. Runner executes provider command in task `repo_path`.
5. Runner writes execution transcript to `runs/task-<id>/output.log`.
6. Task transitions to:
   - `succeeded` on exit code 0
   - `queued` for retry when attempts remain
   - `failed` when retry budget is exhausted

## Concurrency Model

- Each worker has its own SQLite connection.
- Task claiming is serialized with immediate transactions.
- Parallelism is per task, not per command step.

## Extension Points

- Add providers in `config.json`.
- Add richer metadata in `metadata_json`.
- Add event hooks around `run_single_task()` for notifications.
