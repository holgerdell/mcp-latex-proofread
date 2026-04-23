#!/usr/bin/env python3
"""MCP server for managing own-line LaTeX ``\\llm{...}`` comment macros.

Each managed macro has the form::

    \\llm{<body>}% llm:id=<id>

where ``<id>`` is a stable, server-generated hex token that survives line
renumbering.  The server exposes tools to read, insert, replace, remove, and
validate these macros inside ``.tex`` files rooted at the server process
startup directory.

All write operations are sandboxed: only ``.tex`` files under the configured
workspace root may be modified, and each write is verified to match the sole
permitted transformation before being committed to disk.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, NoReturn, TypedDict

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(
    "mcp-latex-proofread",
    instructions=(
        "Use this server to annotate LaTeX source files with LLM-authored comments.\n"
        "\n"
        "Each comment is a managed own-line macro of the form:\n"
        "    \\llm{<comment text>}% llm:id=<stable-id>\n"
        "The comment body is LaTeX. Use \\(...\\) for inline math, ``...'' for "
        "quoted strings, and escape special characters. Example:\n"
        "    \\llm{Consider renaming to ``coefficient''; the index \\(a_i\\) is "
        "never defined.}% llm:id=<stable-id>\n"
        "The stable ID survives line renumbering and is used to update or remove a "
        "comment later.\n"
        "\n"
        "CONSTRAINT: \\llm macros must be inserted in text mode only. Do not insert "
        "after a line that is inside a math environment (\\begin{equation}, "
        "\\begin{align}, \\[...\\], $$...$$, etc.). The server will reject such "
        "insertions with error code 'math_mode'.\n"
        "\n"
        "Invoke this server when you need to:\n"
        "  • Insert a new explanatory or review comment into a .tex file.\n"
        "  • Update or remove a comment you previously inserted.\n"
        "  • List or validate all managed comments in a file.\n"
        "\n"
        "Workflow:\n"
        "  1. Call insert_llm_macro_after_line or insert_llm_macro_after_match "
        "to add a comment.\n"
        "     For insert_llm_macro_after_match, match_text is an exact literal "
        "substring match against one source line; it is not regex-based and does "
        "not unescape backslashes. In JSON, pass a LaTeX command like "
        '\\section{Intro} as "\\\\section{Intro}" so the server receives a '
        "single leading backslash.\n"
        "  2. Record the returned stable ID if you may need to edit or remove the "
        "comment later.\n"
        "  3. Call replace_llm_macro or remove_llm_macro by ID to update or delete.\n"
        "\n"
        "Only .tex files inside the server startup directory may be modified."
    ),
)

WORKSPACE_ROOT = Path.cwd().resolve()
ID_RE = re.compile(r"llm:id=([A-Za-z0-9._-]+)")
OWN_LINE_RE = re.compile(
    r"^(?P<indent>\s*)\\llm\{(?P<body>.*)\}\s*%\s*llm:id=(?P<id>[A-Za-z0-9._-]+)\s*$"
)

MATH_ENV_NAMES = frozenset({
    "equation", "equation*", "align", "align*",
    "alignat", "alignat*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*",
    "displaymath", "math", "flalign", "flalign*",
    "subequations", "dmath", "dmath*",
})
_BEGIN_ENV_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_ENV_RE = re.compile(r"\\end\{([^}]+)\}")
_OPEN_DISPLAY_RE = re.compile(r"\\\[")
_CLOSE_DISPLAY_RE = re.compile(r"\\\]")


class MacroError(TypedDict):
    """A validation problem found in a single source line."""

    line: int
    """1-based line number where the problem was detected."""
    code: str
    """Machine-readable error code (e.g. ``"missing_id"``, ``"duplicate_id"``)."""
    message: str
    """Human-readable description of the problem."""


class MCPError(Exception):
    """Raised by helper functions to signal a structured tool error.

    Caught at the tool boundary and returned as ``{"code": ..., "message": ...}``
    so the MCP client receives a well-formed error object rather than a traceback.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ManagedMacro:
    """An immutable snapshot of one parsed ``\\llm{...} % llm:id=<id>`` line."""

    id: str
    """Stable identifier parsed from the trailing ``% llm:id=<id>`` comment."""
    line: int
    """1-based line number in the source file."""
    body: str
    """Text content inside ``\\llm{...}``."""
    raw: str
    """Full source line without its trailing newline."""
    line_start: int
    """Byte offset of the first character of this line in the file text."""
    line_end: int
    """Byte offset one past the last character of this line (including newline)."""
    body_start: int
    """Byte offset of the first character of ``body`` within the file text."""
    body_end: int
    """Byte offset one past the last character of ``body``."""


def fail(code: str, message: str) -> NoReturn:
    """Raise an :class:`MCPError` with the given structured error code and message."""
    raise MCPError(code, message)


def resolve_allowed_path(path: str) -> Path:
    """Resolve *path* to an absolute :class:`~pathlib.Path` inside the workspace.

    Raises :class:`MCPError` if the path escapes the workspace root, does not
    point to a ``.tex`` file, or the file does not exist.
    """
    if not path:
        fail("invalid_argument", "path is required")

    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (WORKSPACE_ROOT / p).resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        fail("path_not_allowed", "path escapes workspace root")

    if resolved.suffix != ".tex":
        fail("invalid_file_type", "only .tex files are allowed")

    if not resolved.exists():
        fail("file_not_found", "file does not exist")

    if not resolved.is_file():
        fail("file_not_found", "path is not a file")

    return resolved


def read_text(path: Path) -> str:
    """Read *path* as UTF-8 text and return its contents."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write *text* to *path* as UTF-8, preserving existing line endings."""
    path.write_text(text, encoding="utf-8", newline="")


def split_lines_with_endings(text: str) -> list[str]:
    """Split *text* into lines, keeping each line's terminator attached."""
    return text.splitlines(keepends=True)


def normalize_line_for_write(line: str, ending: str) -> str:
    """Return *line* with a line ending, using *ending* if none is present."""
    if line.endswith("\r\n") or line.endswith("\n") or line.endswith("\r"):
        return line
    return line + ending


def detect_default_newline(text: str) -> str:
    """Return the dominant newline sequence in *text*, defaulting to ``"\\n"``."""
    if "\r\n" in text:
        return "\r\n"
    if "\n" in text:
        return "\n"
    if "\r" in text:
        return "\r"
    return "\n"


def line_number_to_char_offset(text: str, line_number: int) -> int:
    """Return the byte offset of the first character of *line_number* (1-based).

    Raises :class:`MCPError` with code ``"line_out_of_range"`` when the line
    number is less than 1 or exceeds the number of lines in *text*.
    """
    if line_number < 1:
        fail("line_out_of_range", "line must be >= 1")

    if text == "":
        fail("line_out_of_range", "line out of range for empty file")

    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)

    if line_number > len(starts):
        fail("line_out_of_range", "line out of range")

    return starts[line_number - 1]


def get_line_texts(text: str) -> list[str]:
    """Return the file text as logical lines without trailing newline characters."""
    return text.splitlines()


def body_is_single_line(body: str) -> bool:
    """Return ``True`` if *body* contains no newline characters."""
    return "\n" not in body and "\r" not in body


def is_in_math_mode(lines: list[str], after_line: int) -> bool:
    """Return True if the position after after_line (1-based) is inside math mode."""
    env_depth = 0
    display_open = False
    dollar_display_open = False
    for raw in lines[:after_line]:
        line = raw.rstrip("\r\n")
        for m in _BEGIN_ENV_RE.finditer(line):
            if m.group(1) in MATH_ENV_NAMES:
                env_depth += 1
        for m in _END_ENV_RE.finditer(line):
            if m.group(1) in MATH_ENV_NAMES:
                env_depth -= 1
        opens = len(_OPEN_DISPLAY_RE.findall(line))
        closes = len(_CLOSE_DISPLAY_RE.findall(line))
        if opens > closes:
            display_open = True
        elif closes > opens:
            display_open = False
        if line.count("$$") % 2 == 1:
            dollar_display_open = not dollar_display_open
    return env_depth > 0 or display_open or dollar_display_open


def make_macro_line(body: str, macro_id: str) -> str:
    """Render a managed macro line for *body* and *macro_id*."""
    if not body_is_single_line(body):
        fail("invalid_argument", "body must be a single line")
    return f"\\llm{{{body}}}% llm:id={macro_id}"


def generate_id(existing_ids: set[str]) -> str:
    """Generate a fresh 8-hex-digit stable ID not present in *existing_ids*."""
    while True:
        candidate = secrets.token_hex(4)
        if candidate not in existing_ids:
            return candidate


def verify_only_allowed_change(modified: str, expected: str) -> None:
    """Fail if *modified* differs from the exact expected transformation."""
    if modified != expected:
        fail("internal_error", "write verification failed")


def get_managed_macros_or_fail(text: str) -> list[ManagedMacro]:
    """Scan *text* for managed macros and fail if validation errors are present."""
    parsed, errors = scan_macros(text)
    if errors:
        fail("invalid_macro_file", "file contains malformed \\llm macros")
    return parsed


def scan_macros(text: str) -> tuple[list[ManagedMacro], list[MacroError]]:
    """Scan *text* for ``\\llm{...}`` occurrences and validate each one.

    Returns the parsed managed macros plus any validation errors found.
    """
    macros: list[ManagedMacro] = []
    errors: list[MacroError] = []
    seen_ids: dict[str, int] = {}

    offset = 0
    for line_no, line in enumerate(split_lines_with_endings(text), start=1):
        raw = line.rstrip("\r\n")
        if "\\llm{" not in raw:
            offset += len(line)
            continue

        match = OWN_LINE_RE.match(raw)
        if match is None:
            if raw.lstrip().startswith("\\llm{"):
                code = "missing_id" if ID_RE.search(raw) is None else "malformed_macro"
            else:
                code = "not_own_line"
            errors.append(
                {
                    "line": line_no,
                    "code": code,
                    "message": "invalid managed \\llm macro format",
                }
            )
            offset += len(line)
            continue

        macro_id = match.group("id")
        if macro_id in seen_ids:
            errors.append(
                {
                    "line": line_no,
                    "code": "duplicate_id",
                    "message": (
                        f"duplicate llm:id={macro_id}; first seen on line {seen_ids[macro_id]}"
                    ),
                }
            )
            offset += len(line)
            continue
        seen_ids[macro_id] = line_no

        body = match.group("body")
        body_start = offset + raw.index(body)
        body_end = body_start + len(body)
        macros.append(
            ManagedMacro(
                id=macro_id,
                line=line_no,
                body=body,
                raw=raw,
                line_start=offset,
                line_end=offset + len(line),
                body_start=body_start,
                body_end=body_end,
            )
        )
        offset += len(line)

    return macros, errors


def relative_path_str(path: Path) -> str:
    """Return *path* as a string relative to ``WORKSPACE_ROOT``."""
    return str(path.relative_to(WORKSPACE_ROOT))


@mcp.tool(title="List managed LLM comments")
def list_llm_macros(file_path: str) -> dict[str, object]:
    """List all managed ``\\llm{...}`` macros in a ``.tex`` file.

    Returns ``{"path": ..., "macros": [...], "validation_errors": [...]}``.
    Each macro entry includes its stable ``id``, 1-based ``line`` number,
    ``body`` text, and full ``raw`` source line.  Validation errors are
    reported alongside valid macros rather than aborting the scan.
    """
    try:
        p = resolve_allowed_path(file_path)
        text = read_text(p)
        macros, errors = scan_macros(text)
        return {
            "path": relative_path_str(p),
            "macros": [
                {
                    "id": m.id,
                    "line": m.line,
                    "body": m.body,
                    "raw": m.raw,
                }
                for m in macros
            ],
            "validation_errors": errors,
        }
    except MCPError as e:
        return {"code": e.code, "message": e.message}


@mcp.tool(title="Insert LLM comment after line")
def insert_llm_macro_after_line(
    file_path: str, line: Annotated[int, Field(ge=1)], body: str
) -> dict[str, object]:
    """Insert a new managed ``\\llm{...}`` macro after the given line number.

    The server generates a fresh stable ID and writes the macro as its own
    line immediately after *line* (1-based).  Returns
    ``{"ok": true, "path": ..., "inserted": {"id": ..., "line": ..., "raw": ...}}``.
    """
    try:
        p = resolve_allowed_path(file_path)
        original = read_text(p)
        macros = get_managed_macros_or_fail(original)
        existing_ids = {m.id for m in macros}

        if not body_is_single_line(body):
            fail("invalid_argument", "body must be a single line")

        macro_id = generate_id(existing_ids)
        newline = detect_default_newline(original)
        new_line = make_macro_line(body, macro_id) + newline

        lines = split_lines_with_endings(original)
        if line < 1 or line > len(lines):
            fail("line_out_of_range", "line out of range")

        if is_in_math_mode(get_line_texts(original), line):
            fail("math_mode", "cannot insert \\llm macro inside a math environment")

        insert_at = sum(len(lines[i]) for i in range(line))
        expected = original[:insert_at] + new_line + original[insert_at:]
        modified = expected

        verify_only_allowed_change(modified, expected)
        write_text(p, modified)

        inserted_line = line + 1
        return {
            "ok": True,
            "path": relative_path_str(p),
            "inserted": {
                "id": macro_id,
                "line": inserted_line,
                "raw": new_line.rstrip("\r\n"),
            },
        }
    except MCPError as e:
        return {"code": e.code, "message": e.message}


@mcp.tool(title="Insert LLM comment after text match")
def insert_llm_macro_after_match(
    file_path: str,
    match_text: str,
    body: str,
) -> dict[str, object]:
    """Insert a new managed ``\\llm{...}`` macro after a line matching *match_text*.

    Matching is a literal substring check against each source line; it is not
    regex-based and does not unescape backslashes. JSON callers must escape
    backslashes normally, so to match a line containing ``\\section{Introduction}``
    the JSON string should be ``"\\section{Introduction}"``. If multiple lines
    match, the call fails and reports all matching line numbers. Returns
    ``{"ok": true, "path": ..., "line": ...}`` on success.
    """
    try:
        p = resolve_allowed_path(file_path)
        original = read_text(p)
        macros = get_managed_macros_or_fail(original)
        existing_ids = {m.id for m in macros}

        if not match_text:
            fail("invalid_argument", "match_text is required")

        lines = get_line_texts(original)
        matches = [i + 1 for i, t in enumerate(lines) if match_text in t]

        if not matches:
            fail("match_not_found", "no line contains match_text")
        if len(matches) > 1:
            return {
                "code": "multiple_matches",
                "count": len(matches),
                "lines": matches[:10],
            }

        matched_line = matches[0]

        if is_in_math_mode(lines, matched_line):
            fail("math_mode", "cannot insert \\llm macro inside a math environment")

        macro_id = generate_id(existing_ids)
        newline = detect_default_newline(original)
        new_line = make_macro_line(body, macro_id) + newline

        lines_with_endings = split_lines_with_endings(original)
        insert_at = sum(len(lines_with_endings[i]) for i in range(matched_line))
        expected = original[:insert_at] + new_line + original[insert_at:]
        modified = expected

        verify_only_allowed_change(modified, expected)
        write_text(p, modified)

        return {
            "ok": True,
            "path": relative_path_str(p),
            "line": matched_line + 1,
        }
    except MCPError as e:
        return {"code": e.code, "message": e.message}


@mcp.tool(title="Replace LLM comment body")
def replace_llm_macro(file_path: str, id: str, body: str) -> dict[str, object]:
    """Replace the body of an existing managed ``\\llm{...}`` macro.

    The macro is identified by its stable *id*; the ID and trailing metadata
    comment are preserved unchanged.  Returns
    ``{"ok": true, "path": ..., "updated": {"id": ..., "line": ..., "raw": ...}}``.
    """
    try:
        p = resolve_allowed_path(file_path)
        original = read_text(p)
        macros = get_managed_macros_or_fail(original)

        if not body_is_single_line(body):
            fail("invalid_argument", "body must be a single line")

        target = next((m for m in macros if m.id == id), None)
        if target is None:
            fail("macro_not_found", f"no managed macro with id={id}")

        replacement = (
            original[target.line_start : target.body_start]
            + body
            + original[target.body_end : target.line_end]
        )
        expected = original[: target.line_start] + replacement + original[target.line_end :]
        modified = expected

        verify_only_allowed_change(modified, expected)
        write_text(p, modified)

        return {
            "ok": True,
            "path": relative_path_str(p),
            "updated": {
                "id": id,
                "line": target.line,
                "raw": replacement.rstrip("\r\n"),
            },
        }
    except MCPError as e:
        return {"code": e.code, "message": e.message}


@mcp.tool(title="Remove LLM comment")
def remove_llm_macro(file_path: str, id: str) -> dict[str, object]:
    """Remove an existing managed ``\\llm{...}`` macro by stable *id*.

    The entire own-line macro (including its newline) is deleted from the file.
    Returns ``{"ok": true, "path": ..., "removed": {"id": ..., "line": ..., "raw": ...}}``.
    """
    try:
        p = resolve_allowed_path(file_path)
        original = read_text(p)
        macros = get_managed_macros_or_fail(original)

        target = next((m for m in macros if m.id == id), None)
        if target is None:
            fail("macro_not_found", f"no managed macro with id={id}")

        expected = original[: target.line_start] + original[target.line_end :]
        modified = expected

        verify_only_allowed_change(modified, expected)
        write_text(p, modified)

        return {
            "ok": True,
            "path": relative_path_str(p),
            "removed": {
                "id": id,
                "line": target.line,
                "raw": target.raw,
            },
        }
    except MCPError as e:
        return {"code": e.code, "message": e.message}


@mcp.tool(title="Validate LaTeX file for managed LLM comments")
def validate_llm_macro_file(file_path: str) -> dict[str, object]:
    """Validate all managed ``\\llm{...}`` macros in a ``.tex`` file.

    Checks that every ``\\llm{`` occurrence is on its own line with a valid
    ``% llm:id=<id>`` suffix, and that all IDs are unique.
    Returns ``{"path": ..., "valid": bool, "errors": [...]}``.
    """
    try:
        p = resolve_allowed_path(file_path)
        text = read_text(p)
        _, errors = scan_macros(text)
        return {
            "path": relative_path_str(p),
            "valid": len(errors) == 0,
            "errors": errors,
        }
    except MCPError as e:
        return {"code": e.code, "message": e.message}


def main() -> None:
    """Entry point: start the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
