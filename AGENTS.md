# Fish Detector — Agent Coding Guide

## Project

A web app that accepts an uploaded image, runs a fine-tuned YOLOv8 fish-detection model on CPU, and returns bounding box coordinates for detected fish species. Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

## Architecture

Full architecture decisions and GCP service mapping are documented in:

**[docs/architecture-context.md](docs/architecture-context.md)**

- **Model:** YOLOv8n (Ultralytics), ONNX export for CPU inference
- **API:** Flask + Cloud Run (`--max-instances 1`)
- **Frontend:** Firebase Hosting (static HTML/CSS/JS)

Directories:
- `app/` — Flask API on Cloud Run; ONNX inference runs directly on CPU, returns bounding box coordinates (not an annotated image)
- `frontend/` — Static site on Firebase Hosting
- `terraform/` — GCP infrastructure (APIs, service accounts, Workload Identity, Artifact Registry, GCS bucket)
- `tests/` — pytest test suite

## Development Conventions

- Python 3.11+
- `requirements.txt` lives in `app/`
- Environment variables (never hardcoded): `GCP_PROJECT_ID`, `GCP_REGION`
- Image upload limit: 5MB enforced in both frontend JS and Flask
- `fish-id.onnx` is not committed to the repo — for local dev place it in `app/`; for CI/CD it is pulled from GCS at deploy time
- `terraform/.terraform.lock.hcl` is committed to pin provider versions for reproducible CI runs
- Infrastructure setup and deployment are documented in `README.md`

## Cost Guards

Always preserve `--max-instances 1` on Cloud Run. Do not raise this without explicit confirmation — it is the primary cost control for this demo.

## Development Workflow

Every coding session MUST follow this exact sequence without skipping steps: **create a worktree → make changes → run tests → open a PR**. Never commit directly to `main`.

### Workflow requirements
1. **Isolation:** You MUST spawn a sub-agent using Git worktree isolation (`isolation: worktree`) or manually use `git worktree add` for the new feature branch. When using Claude Code's `isolation: worktree`, the worktree path is managed automatically under `.claude/worktrees/`.
2. **Environment:** You MUST NEVER use the global Python environment. You MUST create a local virtualenv (`python3 -m venv .venv`) inside the worktree and activate it before installing dependencies or running code.
3. **Testing:** After implementing the change, you MUST run the test suite using `pytest tests/`. Do not proceed if tests fail.
4. **Pull Request:** Once tests pass, you MUST use the GitHub CLI (`gh pr create`) to open a pull request. Include a detailed description of the changes.

### Starting a Session

#### 1. Create a branch and worktree

From the root of the main repo checkout:

```bash
git fetch origin

git worktree add -b <branch-name> ../<repo>-<branch-name> origin/main
```

Use a descriptive branch name that reflects the work, e.g.:
- `feature/add-oauth`
- `fix/null-pointer-login`
- `chore/upgrade-dependencies`

> When using Claude Code's `isolation: worktree` option, the worktree is created automatically under `.claude/worktrees/` — no manual `git worktree add` needed.

#### 2. Install dependencies (if needed)

```bash
# Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
pip install -r tests/requirements.txt
```

---

### Making Changes

Always ask the user for confirmation before committing. Never commit automatically after making changes.

Work entirely inside the worktree directory. Commit regularly with clear messages:

```bash
git add <specific-files>
git commit -m "feat: describe what and why"
git push -u origin <branch-name>
```

Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:
- `feat:` — new feature
- `fix:` — bug fix
- `chore:` — maintenance/tooling
- `docs:` — documentation
- `refactor:` — code restructure with no behavior change
- `test:` — adding or updating tests

---

### Running Tests

Before opening a PR, install test dependencies and run the full test suite from the worktree root:

```bash
source .venv/bin/activate
pip install -r tests/requirements.txt
pytest tests/
```

Do not proceed if any tests fail. Fix failures before opening a PR.

---

### Opening a PR

Always ask the user for confirmation before opening a PR. Never open one automatically after tests pass.

Once the work is ready, open a PR using the GitHub CLI:

```bash
gh pr create \
  --title "<title>" \
  --body "<description>" \
  --base main
```

The PR description should include:
- **What** was changed
- **Why** it was changed
- **How to test** the change
- Any relevant issue numbers (`Closes #123`)

---

### Ending a Session

Leave the worktree in place if the PR is still under review. Once the PR is merged:

```bash
# From the main repo directory
git worktree remove ../<repo>-<branch-name>
git branch -d <branch-name>
git fetch origin --prune
```

---

### Worktree Hygiene

- Run `git worktree list` to see all active worktrees
- Run `git worktree prune` to clean up stale entries
- Keep worktree directory names consistent: `<repo>-<branch-name>`
- Never check out the same branch in two worktrees simultaneously

---

---

## Python Best Practices

### Module structure
- All imports at the top of the file — no imports inside functions, except inside `try/except ImportError` blocks used to give a clear error message when an optional package is missing
- Private helpers prefixed with `_` (e.g. `_load_config`, `_download_dataset`)
- A single `main()` function as the entry point, called via `if __name__ == "__main__"`

### Environment variables
- Read all environment variables once, at the top of `main()` — never call `os.environ` inside helper functions
- Use `os.environ["KEY"]` (raises `KeyError`) for required vars, `os.environ.get("KEY", default)` for optional ones

### Logging
- Use `logging` with a module-level `_logger = logging.getLogger(__name__)` — no `print()` statements in library code
- `main()` configures `logging.basicConfig` once

### Type annotations
- Annotate all function signatures — parameters and return types

### No hardcoded values
- No hardcoded bucket names, project IDs, regions, or model paths — all come from environment variables or arguments

### Idiomatic constructs
- Use list/dict/set comprehensions instead of `for` loops that build collections
- Use `with` for all file and resource operations — never open files without a context manager
- Use f-strings — not `.format()` or `%`
- Use `pathlib.Path` for file paths — not `os.path` string manipulation
- Use `enumerate()`, `zip()`, `any()`, `all()` instead of manual index tracking

### Exception handling
- Always catch specific exceptions — never bare `except:` or `except Exception:`
- Don't silence exceptions; if you catch and continue, log it
- Let exceptions propagate unless you can meaningfully handle them at that level
- Never use exceptions for normal control flow

### Functions and arguments
- Never use mutable default arguments (`def f(x=[])` is a bug — use `None` and assign inside)
- Functions should do one thing; if a function needs a comment to explain what each section does, split it
- Prefer returning values over mutating arguments

### Structured data
- Use `dataclasses` or `TypedDict` for structured data instead of plain dicts with implicit schemas
- Use tuples for fixed-shape immutable records

### Comments and docstrings
- No docstrings on functions whose name and type annotations already explain them
- No inline comments that restate what the code does — only comment the non-obvious *why*

### General
- Prefer `is None` / `is not None` over `== None`
- Use `if not x:` only when a falsy check is intentional — be explicit with `if x is None` vs `if len(x) == 0`
- Don't reassign variables to different types mid-function

---

### Quick Reference

```bash
# Start
git fetch origin
git worktree add -b <branch> ../<repo>-<branch> origin/main
python3 -m venv .venv && source .venv/bin/activate && pip install -r app/requirements.txt -r tests/requirements.txt

# Work
git add <specific-files> && git commit -m "..." && git push

# Test
source .venv/bin/activate && pytest tests/

# PR
gh pr create --title "..." --body "..." --base main

# Cleanup (after merge)
git worktree remove ../<repo>-<branch>
git branch -d <branch>
```
