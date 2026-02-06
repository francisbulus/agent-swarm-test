# Operations Guide

## Recommended Swarm Pattern

1. Enqueue one `planner` task (usually `claude`).
2. Enqueue multiple `builder` tasks (mix `codex` and `claude`) against separate worktrees.
3. Enqueue `reviewer` tasks to validate diffs and tests.
4. Enqueue one `integrator` task to merge final changes.

## Example Queue Session

```bash
python3 swarm_runner.py enqueue \
  --title "Plan release tasks" \
  --role planner \
  --provider claude \
  --repo-path /repo \
  --prompt-file prompts/planner.md

python3 swarm_runner.py enqueue \
  --title "Implement API retries" \
  --role builder \
  --provider codex \
  --repo-path /repo-worktrees/wt-codex-1 \
  --prompt-file prompts/builder.md \
  --metadata-json '{"branch":"wt-codex-1"}'

python3 swarm_runner.py enqueue \
  --title "Review retry patch" \
  --role reviewer \
  --provider claude \
  --repo-path /repo-worktrees/wt-review \
  --prompt-file prompts/reviewer.md

python3 swarm_runner.py run --workers 4 --watch
```

## Monitoring

- `python3 swarm_runner.py list`
- Inspect per-task logs under `runs/task-<id>/output.log`

## Failure Handling

- Check `last_error` in DB or task log output.
- Increase `--max-attempts` for unstable tasks.
- Tune timeout in `config.json` (`runner.task_timeout_sec`).

## Safety Guardrails

- Use isolated worktrees for each builder.
- Keep prompts scoped to one objective per task.
- Require reviewer/integrator gate before merge.
