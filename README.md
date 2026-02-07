# Swarm Runner (Codex + Claude)

Lightweight task-queue runner for coordinating a swarm of coding agents.

## What You Get

- SQLite-backed queue with retries
- Parallel workers (`--workers N`)
- Provider adapters for `codex` and `claude` CLIs
- Role-based task model (`planner`, `builder`, `reviewer`, `integrator`)
- Prompt templates in `prompts/`
- Per-task logs in `runs/task-<id>/output.log`

## Folder Layout

- `swarm/swarm_runner.py`: main CLI runner
- `swarm/config.json`: active provider command config
- `swarm/config.example.json`: example config
- `swarm/prompts/`: role prompt templates
- `swarm/state/swarm.db`: task database
- `swarm/runs/`: task artifacts and logs
- `swarm/sessions/`: persistent session handoff logs

## Quickstart

```bash
cd swarm
python3 swarm_runner.py init
python3 swarm_runner.py enqueue \
  --title "Plan feature rollout" \
  --role planner \
  --provider claude \
  --repo-path /absolute/path/to/repo \
  --prompt-file prompts/planner.md
python3 swarm_runner.py run --workers 2
python3 swarm_runner.py list
```

## Common Commands

```bash
# List queued tasks
python3 swarm_runner.py list --status queued

# Run in daemon mode (poll for new work)
python3 swarm_runner.py run --workers 4 --watch

# Enqueue an inline prompt
python3 swarm_runner.py enqueue \
  --title "Implement bug fix" \
  --role builder \
  --provider codex \
  --repo-path /absolute/path/to/repo \
  --prompt "Fix failing tests for payment retry logic"
```

## Session Handoff Flow

Use this when ending work so the next session can resume quickly.

```bash
# Start a new session log
python3 swarm_runner.py session start \
  --goal "Ship retry behavior for billing API" \
  --repo-path /absolute/path/to/repo \
  --context "Investigating flaky test in payment retries."

# Add progress notes during the session
python3 swarm_runner.py session note \
  --text "Confirmed failure occurs when retry header is missing."

# Close with handoff summary + next actions
python3 swarm_runner.py session close \
  --summary "Added retry header fallback and tests are green." \
  --next-step "Run reviewer pass on error handling paths" \
  --next-step "Merge after reviewer approval"

# Show latest session log for resume
python3 swarm_runner.py session show
```

Session logs are written to `sessions/session-<UTC timestamp>.md`.  
The active session pointer is `sessions/.current`.

## Configure Provider Commands

Edit `swarm/config.json` and change command arrays to match your local CLI syntax.

Supported template tokens:

- `{repo_path}`
- `{prompt}`
- `{prompt_file}`
- `{task_id}`
- `{role}`
- `{provider}`
- `{output_file}`

## Notes

- The runner does not create branches/worktrees for you; pass those paths via `--repo-path`.
- Failed tasks retry automatically until `max_attempts` is reached.
- Task logs include stdout, stderr, and exit code.

## Troubleshooting

If a run is interrupted and tasks remain in `running`, requeue them:

```bash
sqlite3 state/swarm.db \
"UPDATE tasks SET status='queued', started_at=NULL, completed_at=NULL, worker_id=NULL WHERE status='running';"
```

See `ARCHITECTURE.md`, `OPERATIONS.md`, and `TASK_SCHEMA.md` for details.
