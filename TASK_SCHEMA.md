# Task Schema

Each task row in `state/swarm.db` contains:

- `id`: integer primary key
- `title`: short human label
- `role`: `planner | builder | reviewer | integrator`
- `provider`: provider key in `config.json` (`codex`, `claude`, ...)
- `repo_path`: absolute path where provider command executes
- `prompt`: full prompt text
- `status`: `queued | running | succeeded | failed`
- `priority`: lower values run first
- `attempts`: number of started attempts
- `max_attempts`: retry cap
- `created_at`: UTC ISO timestamp
- `started_at`: UTC ISO timestamp (latest attempt)
- `completed_at`: UTC ISO timestamp (latest completion)
- `worker_id`: worker name that claimed the task
- `output_path`: path to run log
- `last_error`: latest error summary
- `metadata_json`: arbitrary JSON object for custom routing/tagging

## Metadata Suggestions

```json
{
  "epic": "billing-v2",
  "branch": "wt-codex-2",
  "depends_on": [12, 13],
  "labels": ["backend", "high-priority"]
}
```
