"""Comprehensive tests for the mcp-latex-proofread MCP server."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict, cast

import pytest

import main
from main import (
    insert_llm_macro_after_line,
    insert_llm_macro_after_match,
    list_llm_macros,
    remove_llm_macro,
    replace_llm_macro,
    validate_llm_macro_file,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

HEX_ID_RE = re.compile(r"^[0-9a-f]{8}$")


class ReadLineEntry(TypedDict):
    line: int
    text: str


class MacroEntry(TypedDict):
    id: str
    line: int
    body: str
    raw: str


class InsertedEntry(TypedDict):
    id: str
    line: int
    raw: str


class MatchedEntry(TypedDict):
    line: int
    match_text: str
    occurrence: int


class UpdatedEntry(TypedDict):
    id: str
    line: int
    raw: str


class RemovedEntry(TypedDict):
    id: str
    line: int
    raw: str


SIMPLE = """\
\\documentclass{article}
\\begin{document}
\\section{Introduction}
Hello world.
\\end{document}
"""

# File with two well-formed managed macros.
WITH_MACROS = """\
\\documentclass{article}
\\begin{document}
\\llm{First comment}% llm:id=aaaa0001
\\section{Introduction}
\\llm{Second comment}% llm:id=bbbb0002
Hello world.
\\end{document}
"""


@pytest.fixture()
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporary workspace; WORKSPACE_ROOT is redirected here for every test."""
    monkeypatch.setattr(main, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


def write_tex(ws: Path, content: str, name: str = "doc.tex") -> str:
    """Write *content* to *ws/name* and return the relative path string."""
    (ws / name).write_text(content, encoding="utf-8")
    return name


def read_tex(ws: Path, name: str = "doc.tex") -> str:
    """Read the current content of *ws/name*."""
    return (ws / name).read_text(encoding="utf-8")


def expect_lines(result: dict[str, object]) -> list[ReadLineEntry]:
    lines = result["lines"]
    assert isinstance(lines, list)
    return cast(list[ReadLineEntry], lines)


def expect_macros(result: dict[str, object]) -> list[MacroEntry]:
    macros = result["macros"]
    assert isinstance(macros, list)
    return cast(list[MacroEntry], macros)


def expect_validation_errors(result: dict[str, object]) -> list[main.MacroError]:
    errors = result["validation_errors"]
    assert isinstance(errors, list)
    return cast(list[main.MacroError], errors)


def expect_errors(result: dict[str, object]) -> list[main.MacroError]:
    errors = result["errors"]
    assert isinstance(errors, list)
    return cast(list[main.MacroError], errors)


def expect_inserted(result: dict[str, object]) -> InsertedEntry:
    inserted = result["inserted"]
    assert isinstance(inserted, dict)
    return cast(InsertedEntry, inserted)


def expect_matched(result: dict[str, object]) -> MatchedEntry:
    matched = result["matched"]
    assert isinstance(matched, dict)
    return cast(MatchedEntry, matched)


def expect_updated(result: dict[str, object]) -> UpdatedEntry:
    updated = result["updated"]
    assert isinstance(updated, dict)
    return cast(UpdatedEntry, updated)


def expect_removed(result: dict[str, object]) -> RemovedEntry:
    removed = result["removed"]
    assert isinstance(removed, dict)
    return cast(RemovedEntry, removed)


# ---------------------------------------------------------------------------
# list_llm_macros
# ---------------------------------------------------------------------------


class TestListLlmMacros:
    def test_no_macros(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = list_llm_macros(p)
        assert result["macros"] == []
        assert result["validation_errors"] == []

    def test_two_macros_fields(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = list_llm_macros(p)
        macros = expect_macros(result)
        assert len(macros) == 2
        first = macros[0]
        assert first["id"] == "aaaa0001"
        assert first["line"] == 3
        assert first["body"] == "First comment"
        assert "\\llm{First comment}" in str(first["raw"])

    def test_macro_line_numbers(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = list_llm_macros(p)
        lines = [m["line"] for m in expect_macros(result)]
        assert lines == [3, 5]

    def test_missing_id_validation_error(self, ws: Path) -> None:
        content = SIMPLE.replace(
            "\\section{Introduction}",
            "\\llm{No id here}\n\\section{Introduction}",
        )
        p = write_tex(ws, content)
        result = list_llm_macros(p)
        errors = expect_validation_errors(result)
        assert len(errors) == 1
        assert errors[0]["code"] == "missing_id"

    def test_duplicate_id_validation_error(self, ws: Path) -> None:
        content = "\\llm{A}% llm:id=dup00001\n\\llm{B}% llm:id=dup00001\n"
        p = write_tex(ws, content)
        result = list_llm_macros(p)
        codes = [e["code"] for e in expect_validation_errors(result)]
        assert "duplicate_id" in codes

    def test_not_own_line_validation_error(self, ws: Path) -> None:
        # \llm{ appears mid-line (inline usage)
        content = "Some text \\llm{inline} more text\n"
        p = write_tex(ws, content)
        result = list_llm_macros(p)
        errors = expect_validation_errors(result)
        assert len(errors) == 1
        assert errors[0]["code"] == "not_own_line"

    def test_malformed_macro_validation_error(self, ws: Path) -> None:
        # Starts with \llm{ but id contains a space (invalid)
        content = "\\llm{body}% llm:id=bad id\n"
        p = write_tex(ws, content)
        result = list_llm_macros(p)
        errors = expect_validation_errors(result)
        assert len(errors) >= 1
        assert errors[0]["code"] == "malformed_macro"

    def test_returns_relative_path(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = list_llm_macros(p)
        assert result["path"] == "doc.tex"


# ---------------------------------------------------------------------------
# insert_llm_macro_after_line
# ---------------------------------------------------------------------------


class TestInsertLlmMacroAfterLine:
    def test_insert_after_line_1(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_line(p, 1, "A comment")
        assert result.get("ok") is True
        inserted = expect_inserted(result)
        assert inserted["line"] == 2
        assert HEX_ID_RE.match(inserted["id"])
        disk = read_tex(ws)
        assert "\\llm{A comment}" in disk
        # Original first line still at line 1
        assert disk.splitlines()[0] == "\\documentclass{article}"

    def test_insert_after_last_line(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        total = len(SIMPLE.splitlines())
        result = insert_llm_macro_after_line(p, total, "End note")
        assert result.get("ok") is True
        disk = read_tex(ws)
        assert "\\llm{End note}" in disk
        assert "\\llm{End note}" in disk.splitlines()[-1]

    def test_insert_preserves_surrounding_content(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        insert_llm_macro_after_line(p, 3, "middle")
        lines = read_tex(ws).splitlines()
        assert lines[2] == "\\section{Introduction}"
        assert "\\llm{middle}" in lines[3]
        assert lines[4] == "Hello world."

    def test_insert_generates_unique_ids_for_two_insertions(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        r1 = insert_llm_macro_after_line(p, 1, "first")
        r2 = insert_llm_macro_after_line(p, 2, "second")
        assert expect_inserted(r1)["id"] != expect_inserted(r2)["id"]

    def test_insert_line_zero_is_error(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_line(p, 0, "x")
        assert result["code"] == "line_out_of_range"

    def test_insert_line_beyond_end_is_error(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        total = len(SIMPLE.splitlines())
        result = insert_llm_macro_after_line(p, total + 1, "x")
        assert result["code"] == "line_out_of_range"

    def test_insert_file_unchanged_on_line_error(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        before = read_tex(ws)
        insert_llm_macro_after_line(p, 999, "x")
        assert read_tex(ws) == before

    def test_insert_body_with_newline_is_error(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_line(p, 1, "line1\nline2")
        assert result["code"] == "invalid_argument"

    def test_inserted_line_is_valid_macro_format(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_line(p, 1, "my body")
        raw = expect_inserted(result)["raw"]
        assert re.match(r"^\\llm\{my body\}% llm:id=[0-9a-f]{8}$", raw)


# ---------------------------------------------------------------------------
# insert_llm_macro_after_match
# ---------------------------------------------------------------------------


class TestInsertLlmMacroAfterMatch:
    def test_match_first_occurrence(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_match(p, "\\section{Introduction}", "sec comment")
        assert result.get("ok") is True
        assert result["line"] == 4
        disk = read_tex(ws)
        assert "\\llm{sec comment}" in disk.splitlines()[3]

    def test_multiple_matches_rejected(self, ws: Path) -> None:
        content = "alpha\nbeta\nalpha\ngamma\n"
        p = write_tex(ws, content)
        result = insert_llm_macro_after_match(p, "alpha", "comment")
        assert result["code"] == "multiple_matches"
        assert result["count"] == 2
        assert result["lines"] == [1, 3]

    def test_match_not_found_is_error(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_match(p, "nonexistent text", "x")
        assert result["code"] == "match_not_found"

    def test_empty_match_text_is_error(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = insert_llm_macro_after_match(p, "", "x")
        assert result["code"] == "invalid_argument"

    def test_file_unchanged_on_no_match(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        before = read_tex(ws)
        insert_llm_macro_after_match(p, "zzznomatch", "x")
        assert read_tex(ws) == before


# ---------------------------------------------------------------------------
# replace_llm_macro
# ---------------------------------------------------------------------------


class TestReplaceLlmMacro:
    def test_replace_body(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = replace_llm_macro(p, "aaaa0001", "Updated body")
        assert result.get("ok") is True
        disk = read_tex(ws)
        assert "\\llm{Updated body}% llm:id=aaaa0001" in disk
        assert "\\llm{First comment}" not in disk

    def test_replace_preserves_id(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = replace_llm_macro(p, "bbbb0002", "New text")
        updated = expect_updated(result)
        assert updated["id"] == "bbbb0002"

    def test_replace_line_number_in_response(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = replace_llm_macro(p, "bbbb0002", "x")
        assert expect_updated(result)["line"] == 5

    def test_replace_not_found_is_error(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = replace_llm_macro(p, "deadbeef", "x")
        assert result["code"] == "macro_not_found"

    def test_replace_body_with_newline_is_error(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = replace_llm_macro(p, "aaaa0001", "line1\nline2")
        assert result["code"] == "invalid_argument"

    def test_replace_file_unchanged_on_not_found(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        before = read_tex(ws)
        replace_llm_macro(p, "notexist", "x")
        assert read_tex(ws) == before

    def test_replace_other_macro_untouched(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        replace_llm_macro(p, "aaaa0001", "changed")
        disk = read_tex(ws)
        assert "\\llm{Second comment}% llm:id=bbbb0002" in disk


# ---------------------------------------------------------------------------
# remove_llm_macro
# ---------------------------------------------------------------------------


class TestRemoveLlmMacro:
    def test_remove_macro(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = remove_llm_macro(p, "aaaa0001")
        assert result.get("ok") is True
        disk = read_tex(ws)
        assert "aaaa0001" not in disk
        assert "\\llm{First comment}" not in disk

    def test_remove_returns_id_and_raw(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = remove_llm_macro(p, "aaaa0001")
        removed = expect_removed(result)
        assert removed["id"] == "aaaa0001"
        assert "\\llm{First comment}" in removed["raw"]

    def test_remove_returns_original_line_number(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = remove_llm_macro(p, "aaaa0001")
        assert expect_removed(result)["line"] == 3

    def test_remove_leaves_surrounding_lines_intact(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        remove_llm_macro(p, "aaaa0001")
        lines = read_tex(ws).splitlines()
        assert lines[0] == "\\documentclass{article}"
        assert lines[1] == "\\begin{document}"
        assert lines[2] == "\\section{Introduction}"

    def test_remove_other_macro_untouched(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        remove_llm_macro(p, "aaaa0001")
        disk = read_tex(ws)
        assert "\\llm{Second comment}% llm:id=bbbb0002" in disk

    def test_remove_not_found_is_error(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = remove_llm_macro(p, "deadbeef")
        assert result["code"] == "macro_not_found"

    def test_remove_file_unchanged_on_not_found(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        before = read_tex(ws)
        remove_llm_macro(p, "notexist")
        assert read_tex(ws) == before

    def test_remove_both_macros_sequentially(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        remove_llm_macro(p, "aaaa0001")
        remove_llm_macro(p, "bbbb0002")
        disk = read_tex(ws)
        assert "\\llm{" not in disk


# ---------------------------------------------------------------------------
# validate_llm_macro_file
# ---------------------------------------------------------------------------


class TestValidateLlmMacroFile:
    def test_clean_file_is_valid(self, ws: Path) -> None:
        p = write_tex(ws, WITH_MACROS)
        result = validate_llm_macro_file(p)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_no_macros_is_valid(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = validate_llm_macro_file(p)
        assert result["valid"] is True

    def test_missing_id_is_invalid(self, ws: Path) -> None:
        content = "\\llm{No id here}\n"
        p = write_tex(ws, content)
        result = validate_llm_macro_file(p)
        assert result["valid"] is False
        errors = expect_errors(result)
        assert errors[0]["code"] == "missing_id"

    def test_duplicate_id_is_invalid(self, ws: Path) -> None:
        content = "\\llm{A}% llm:id=dup00001\n\\llm{B}% llm:id=dup00001\n"
        p = write_tex(ws, content)
        result = validate_llm_macro_file(p)
        assert result["valid"] is False
        codes = [e["code"] for e in expect_errors(result)]
        assert "duplicate_id" in codes

    def test_not_own_line_is_invalid(self, ws: Path) -> None:
        content = "text \\llm{inline}\n"
        p = write_tex(ws, content)
        result = validate_llm_macro_file(p)
        assert result["valid"] is False
        assert expect_errors(result)[0]["code"] == "not_own_line"

    def test_malformed_id_is_invalid(self, ws: Path) -> None:
        content = "\\llm{body}% llm:id=bad id\n"
        p = write_tex(ws, content)
        result = validate_llm_macro_file(p)
        assert result["valid"] is False
        assert expect_errors(result)[0]["code"] == "malformed_macro"

    def test_error_reports_correct_line_number(self, ws: Path) -> None:
        content = "line one\nline two\n\\llm{no id}\nline four\n"
        p = write_tex(ws, content)
        result = validate_llm_macro_file(p)
        assert expect_errors(result)[0]["line"] == 3

    def test_returns_relative_path(self, ws: Path) -> None:
        p = write_tex(ws, SIMPLE)
        result = validate_llm_macro_file(p)
        assert result["path"] == "doc.tex"


# ---------------------------------------------------------------------------
# Path security — shared across all tools
# ---------------------------------------------------------------------------


class TestPathSecurity:
    def test_path_security_applies_to_all_write_tools(self, ws: Path) -> None:
        escape = "../secret.tex"
        (ws.parent / "secret.tex").write_text("s", encoding="utf-8")
        for tool_result in [
            list_llm_macros(escape),
            insert_llm_macro_after_line(escape, 1, "x"),
            insert_llm_macro_after_match(escape, "s", "x"),
            replace_llm_macro(escape, "abc", "x"),
            remove_llm_macro(escape, "abc"),
            validate_llm_macro_file(escape),
        ]:
            assert tool_result["code"] == "path_not_allowed", (
                f"expected path_not_allowed, got {tool_result}"
            )


# ---------------------------------------------------------------------------
# Math mode detection
# ---------------------------------------------------------------------------


class TestMathModeDetection:
    def test_insert_after_line_inside_equation_blocks(self, ws: Path) -> None:
        content = """\
\\begin{equation}
x = y
\\end{equation}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment inside")
        assert result["code"] == "math_mode"

    def test_insert_after_line_inside_align(self, ws: Path) -> None:
        content = """\
\\begin{align}
a &= b \\\\
c &= d
\\end{align}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment")
        assert result["code"] == "math_mode"

    def test_insert_after_line_inside_align_star(self, ws: Path) -> None:
        content = """\
\\begin{align*}
x = y
\\end{align*}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment")
        assert result["code"] == "math_mode"

    def test_insert_after_line_inside_display_math_brackets(self, ws: Path) -> None:
        content = """\
\\[
x = y
\\]
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment")
        assert result["code"] == "math_mode"

    def test_insert_after_line_inside_dollar_display_math(self, ws: Path) -> None:
        content = """\
$$
x = y
$$
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment")
        assert result["code"] == "math_mode"

    def test_insert_after_line_before_math_environment_succeeds(self, ws: Path) -> None:
        content = """\
text before
\\begin{equation}
x = y
\\end{equation}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 1, "comment")
        assert result.get("ok") is True

    def test_insert_after_line_after_math_environment_succeeds(self, ws: Path) -> None:
        content = """\
\\begin{equation}
x = y
\\end{equation}
text after
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 3, "comment")
        assert result.get("ok") is True

    def test_insert_after_match_inside_equation_blocks(self, ws: Path) -> None:
        content = """\
\\begin{equation}
x = y
\\end{equation}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_match(p, "x = y", "comment")
        assert result["code"] == "math_mode"

    def test_insert_after_match_before_math_succeeds(self, ws: Path) -> None:
        content = """\
\\section{Intro}
\\begin{equation}
x = y
\\end{equation}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_match(p, "\\section{Intro}", "comment")
        assert result.get("ok") is True

    def test_insert_nested_math_environments(self, ws: Path) -> None:
        content = """\
\\begin{align}
\\begin{aligned}
x = y
\\end{aligned}
\\end{align}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 3, "comment")
        assert result["code"] == "math_mode"

    def test_insert_gather_environment(self, ws: Path) -> None:
        content = """\
\\begin{gather}
x = y
\\end{gather}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment")
        assert result["code"] == "math_mode"

    def test_insert_eqnarray_environment(self, ws: Path) -> None:
        content = """\
\\begin{eqnarray}
x &=& y
\\end{eqnarray}
"""
        p = write_tex(ws, content)
        result = insert_llm_macro_after_line(p, 2, "comment")
        assert result["code"] == "math_mode"
