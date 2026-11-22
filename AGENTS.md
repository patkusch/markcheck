# Working in markcheck

Find AI labels in your files and explain in plain English what they do and don't prove. Runs offline.

## Run the tests

You need Python 3.11 or newer.

```bash
pytest -q
```

Every test should pass before you start and after you finish.

## Where things are

- `src/`: the code
- `tests/`: the tests
- `docs/`: written documentation
- `scripts/`: helper scripts, not part of the tool itself

## Before you commit

- Make sure the tests pass.
- Put changes people will notice in `CHANGELOG.md`.
- Do not put real passwords, keys or private data in test files or examples. Use made-up ones.
