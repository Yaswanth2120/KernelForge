# Contributing

Create a branch for each focused change: `feature/<name>`, `bug/<name>`, or `docs/<name>`.
Use descriptive conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
Push the branch, open a pull request, and require green CPU checks before squash merging.
Keep `main` releasable and delete merged branches. Never commit credentials or fabricated results.

Run `pytest`, `ruff check .`, and `kernelforge check-prereg` locally.
GPU correctness and performance claims require saved output from a real NVIDIA session.
Changes to frozen criteria need a dated entry in DECISIONS.md with old and new values.
