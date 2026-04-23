# mcp-latex-proofread

MCP server that constrains LLM agents to only insert/edit/remove managed `\llm{...}` comment macros in `.tex` files. Prevents agents from rewriting LaTeX source while proofreading.

## Commands

```bash
# Tests
uv run --group dev python -m pytest tests/

# Type checking
uv run mypy src/main.py tests/test_server.py

# Lint
uv run ruff check src/main.py tests/

# Run server (stdio transport, from the LaTeX project directory)
uv run mcp-latex-proofread
```

## Architecture

Single-file server: `src/main.py` using `fastmcp`.

Key concepts:
- **Workspace root** = `Path.cwd()` at server startup — only `.tex` files under this root may be modified
- **Managed macro format**: `\llm{<body>}% llm:id=<8-hex-char-id>` — must be its own line
- **Stable ID** survives line renumbering; used to update/remove a comment
- **Write verification**: every write is diffed against the exact expected transformation before hitting disk

## Tools exposed

| Tool | Purpose |
|------|---------|
| `list_llm_macros` | List all managed macros + validation errors |
| `insert_llm_macro_after_line` | Insert after line number |
| `insert_llm_macro_after_match` | Insert after literal substring match |
| `replace_llm_macro` | Replace body by stable ID |
| `remove_llm_macro` | Delete macro by stable ID |
| `validate_llm_macro_file` | Check all macros well-formed + unique IDs |

## Gotchas

- `match_text` in `insert_llm_macro_after_match` is **literal substring**, not regex; backslashes not unescaped — JSON callers pass `"\\section{Intro}"` to match `\section{Intro}`
- Server sandbox: absolute paths outside workspace root → `path_not_allowed` error
- `body` must be single-line; multiline → `invalid_argument`
- `validate_llm_macro_file` fails if **any** `\llm{` occurrence doesn't match the managed format — malformed macros block insert/replace ops

## Testing

Tests call tool functions directly (not via MCP transport). `WORKSPACE_ROOT` is monkeypatched per test via `main.WORKSPACE_ROOT = tmp_path`.
