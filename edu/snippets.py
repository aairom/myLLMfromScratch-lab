"""Show the *real* source code of a symbol in the UI.

Displayed snippets are read from the live modules with ``inspect``, so the code a
learner reads is exactly the code that runs (it can never drift out of date) and
the comments in the source are the detailed explanations.
"""
from __future__ import annotations

import importlib
import inspect
import textwrap


class SnippetError(LookupError):
    """Raised when a dotted path does not resolve to a symbol with source code."""


def resolve(path: str):
    """Resolve ``"package.module.Class.method"`` to the Python object it names.

    The longest importable module prefix is imported, then the remaining parts
    are looked up as attributes.
    """
    parts = path.split(".")
    if len(parts) < 2:
        raise SnippetError(f"{path!r}: expected 'module.symbol'")
    obj = None
    attrs: list[str] = []
    for cut in range(len(parts) - 1, 0, -1):  # longest module prefix first
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
            attrs = parts[cut:]
            break
        except ModuleNotFoundError:
            continue
    if obj is None:
        raise SnippetError(f"{path!r}: no importable module in this path")
    for name in attrs:
        try:
            obj = getattr(obj, name)
        except AttributeError as exc:
            raise SnippetError(f"{path!r}: {name!r} not found") from exc
    return obj


def get_source(path: str) -> str:
    """Dedented source code (with comments and docstring) of the symbol at ``path``."""
    obj = resolve(path)
    if inspect.ismodule(obj):
        raise SnippetError(f"{path!r} is a module; point to a class or function")
    try:
        return textwrap.dedent(inspect.getsource(obj)).rstrip() + "\n"
    except (OSError, TypeError) as exc:
        raise SnippetError(f"{path!r}: source unavailable ({exc})") from exc


def get_docstring(path: str) -> str:
    """Cleaned docstring of the symbol at ``path`` (empty string if none)."""
    return inspect.getdoc(resolve(path)) or ""


def line_count(path: str) -> int:
    """Number of source lines of the symbol (handy for showing "N lines of code")."""
    return len(get_source(path).splitlines())
