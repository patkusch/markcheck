# Contributing to markcheck

Thanks for helping. Small fixes and clearer wording are both welcome.

## Run the tests

You need Python 3.11 or newer. The full install pulls in PyTorch, so it is large.

```bash
pip install -e '.[all,dev]'
pytest -q
```

Every test should pass before you start and after you finish.

## Send a change

- Keep each pull request to one change, and say what it fixes and why.
- Add a test with your change where you can. A test that fails without your change is the best proof it works.
- Do not put real passwords, keys or private data in test files or examples. Use made-up ones.

## Report a problem

Open an issue and fill in the form. If it is a security problem, follow [SECURITY.md](SECURITY.md) and report it privately instead.
