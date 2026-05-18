# Fish Detector — Agent Coding Guide

## Project

A web app that accepts an uploaded image, runs a fine-tuned YOLOv8 fish-detection model on CPU, and returns an annotated result. Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

## Architecture

Full architecture decisions and GCP service mapping are documented in:

**[docs/architecture-context.md](docs/architecture-context.md)**

- **Model:** YOLOv8n (Ultralytics), ONNX export for CPU inference
- **API:** Flask + Cloud Run (`--max-instances 1`)
- **Frontend:** Firebase Hosting (static HTML/CSS/JS)

Directories:
- `app/` — Flask API on Cloud Run; ONNX inference runs directly on CPU, returns bounding box coordinates (not an annotated image)
- `frontend/` — Static site on Firebase Hosting
- `tests/` — pytest test suite

## Development Conventions

- Python 3.11+
- `requirements.txt` lives in `app/`
- Environment variables (never hardcoded): `GCP_PROJECT`, `GCP_REGION`
- Secrets via Secret Manager, not `.env` files in production
- Image upload limit: 5MB enforced in both frontend JS and Flask
- `best.onnx` is not committed to the repo — place it in `app/` before building the container
- GCP setup steps (APIs, service account, deploy commands) live in `README.md`

## Cost Guards

Always preserve `--max-instances 1` on Cloud Run. Do not raise this without explicit confirmation — it is the primary cost control for this demo.

## Development Workflow

Every coding session MUST follow this exact sequence without skipping steps: **create a worktree → make changes → run tests → open a PR**. Never commit directly to `main`.

### Workflow requirements
1. **Isolation:** You MUST spawn a sub-agent using Git worktree isolation (`isolation: worktree`) or manually use `git worktree add` for the new feature branch. 
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

### Quick Reference

```bash
# Start
git fetch origin
git worktree add -b <branch> ../<repo>-<branch> origin/main
python3 -m venv .venv && source .venv/bin/activate && pip install -r app/requirements.txt

# Work
git add <specific-files> && git commit -m "..." && git push

# Test
source .venv/bin/activate && pip install -r tests/requirements.txt && pytest tests/

# PR
gh pr create --title "..." --body "..." --base main

# Cleanup (after merge)
git worktree remove ../<repo>-<branch>
git branch -d <branch>
```
