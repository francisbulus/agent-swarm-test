# Sessions Directory

This directory stores handoff logs created by:

```bash
python3 swarm_runner.py session start ...
python3 swarm_runner.py session note ...
python3 swarm_runner.py session close ...
```

## Files

- `session-<UTC timestamp>.md`: one session record.
- `.current`: pointer to the active/latest session file.

The intent is to keep enough context here so a new coding session can resume work without relying on chat history.
