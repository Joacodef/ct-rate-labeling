# Contributing Guidelines

Thanks for helping improve **ct-rate-labeling**! This document captures the expectations for contributors so changes stay reliable and easy to review.

## Getting Started
- Use [uv](https://github.com/astral-sh/uv) to manage the virtual environment.
- Install dependencies with `uv pip install -e .` (append `".[tests]"` when running pytest locally).
- Run the full suite before opening a PR: `uv run pytest tests`.

## Workflow Expectations
1. Fork or create a feature branch from `main`.
2. Keep changes focused; open separate PRs for unrelated fixes.
3. Add or update tests that cover the behavior you modify.
4. Update documentation (README, config comments, etc.) when behavior or usage changes.
5. Ensure `pytest` passes and lint tools (if enabled) succeed.

## Docstring & Comment Standards
To keep the code base self-documenting, every public function, method, and class **must** include a descriptive docstring using the Google style:

```python
def fetch_labels(volume_name: str) -> Dict[str, int]:
    """Summarize what the function does in one sentence.

    Args:
        volume_name: Human-readable scan identifier from the CSV.

    Returns:
        Mapping of label names to binary predictions.

    Raises:
        ValueError: If the volume is missing in the input table.
    """
    ...
```

Guidance:
- Keep the summary line short and imperative.
- List all parameters under **Args**, including types and how each is used.
- Document outputs under **Returns** and any exceptional paths under **Raises**.
- When a function performs side effects or has performance caveats, mention them briefly after the `Returns`/`Raises` sections.
- Private helpers (`_like_this`) still need docstrings when the logic is non-trivial; otherwise add a succinct inline comment.

Inline comments should be rare and only clarify intent that is not obvious from the code. Avoid restating what the code already communicates (e.g., “increment i by one”).

## Coding Style & Testing
- Prefer explicit, readable control flow over clever one-liners.
- Keep imports sorted logically (stdlib, third-party, local).
- Follow existing logging patterns and error handling approaches within `scripts/` and `src/ctr_labeling/`.
- When adding new dependencies, include them in `pyproject.toml` and mention why they are needed in the PR description.

## Pull Request Checklist
- [ ] Feature branch is up to date with `main`.
- [ ] `uv run pytest tests` succeeds locally.
- [ ] Docstrings/comments follow the style described above.
- [ ] Relevant documentation updated (README, configs, examples).
- [ ] Added or updated tests that cover new behavior.

By following these conventions, we keep CT-RATE Labeling maintainable and ready for downstream consumers.
