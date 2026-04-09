# codex

## Reasoning harness

This repo now includes a lightweight reasoning harness inspired by our prior
iterations: it runs a deterministic multi-step reasoning pipeline, records a
trace for each stage, and returns a final answer plus confidence.

### Files

- `reasoning_harness.py`: core harness, stage implementations, and data models.
- `run_harness.py`: CLI wrapper.
- `tests/test_reasoning_harness.py`: unit tests for happy path and validation.

### Usage

```bash
python run_harness.py "How can we reduce deployment risk?"
python run_harness.py --json "How can we reduce deployment risk?"
```

### Test

```bash
python -m pytest -q
```
