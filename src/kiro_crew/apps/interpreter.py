"""Shared interpreter resolution for app spawn paths.

One policy, two consumers. The app BACKEND launcher (``backend.py``) and the
app stdio MCP SERVER registration (``bridges.py``) both spawn Python processes
on an app's behalf, and both must refuse to trust a bare ``python3``: a bare
name is resolved through PATH at spawn time, which is not guaranteed to exist
(some hosts ship only a versioned interpreter, so ``execvp("python3")`` raises
FileNotFoundError) and, even when present, may be an older system interpreter
than the one the app's dependencies were installed against — the process then
starts under the wrong interpreter and dies on import, with nothing surfaced
to the user.

The policy: prefer the app's OWN venv interpreter when the app ships one (a
venv is the strongest signal of which interpreter the app's code expects),
else fall back to the gateway's own ``sys.executable`` (always an absolute
path to a real interpreter). The gateway itself does not create app venvs:
an app's ``requirements.txt`` is provisioned with ``pip install --target``
into :func:`app_deps_dir` and reaches the child via ``PYTHONPATH`` — which
every interpreter this policy can resolve honors, venv or not. Keeping the
policy in one place is the point — two divergent copies is exactly the
defect class this module removes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from kiro_crew import platform_compat


def venv_python_path(root: Path) -> Path:
    """The path where ``root``'s venv interpreter would live (may not exist).

    POSIX venvs ship ``bin/python3``; native-Windows venvs ship
    ``Scripts\\python.exe`` and no ``python3`` at all (the same layout split
    ``cli_doctor`` and dev-fleet's ``_venv_python`` already handle).
    """
    if platform_compat.IS_WINDOWS:
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python3"


def app_deps_dir(root: Path) -> Path:
    """Directory an app's ``requirements.txt`` is provisioned into.

    Populated by ``pip install --target`` (``backend.py``'s spawn path) and
    exposed to processes spawned on the app's behalf via ``PYTHONPATH``. A
    plain directory rather than a venv, because venv creation needs
    ``ensurepip`` — which the packaged install's bundled interpreter does not
    ship — and the half-created skeleton a failed attempt leaves behind is
    runnable enough that :func:`resolve_app_python` would prefer it while it
    holds no dependencies at all. A ``--target`` install has no bootstrap
    step, so it cannot leave that trap.
    """
    return root / ".kirocrew-deps"


def _runnable(path: Path) -> bool:
    """Executable AND non-empty — the resolution-safety predicate.

    ``is_executable_file`` alone is not enough: on Windows it is an
    extension-allowlist check (there is no execute bit), so a zero-byte
    ``python.exe`` left by an interrupted copy/restore — the same shape as the
    Microsoft-Store reparse stub — would be accepted and then fail at spawn
    time with no diagnostic. An empty file cannot be a working interpreter or
    console script on any platform, so the size check is applied uniformly.
    """
    try:
        return platform_compat.is_executable_file(path) and path.stat().st_size > 0
    except OSError:
        return False


def _venv_version_matches(root: Path) -> bool:
    """Whether ``root``'s venv was created by the same Python minor version.

    Read from ``pyvenv.cfg`` (``version = X.Y.Z`` from the stdlib, or
    ``version_info = X.Y.Z...`` from virtualenv/uv). The deps the gateway
    provisions are built by ``sys.executable`` and reach the child via
    ``PYTHONPATH``, which sorts BEFORE a venv's ``site-packages`` — so running
    under a venv of a different minor version would import cp-tagged wheels of
    the wrong ABI and die on the first native extension. A venv the gateway
    cannot version-match buys nothing over ``sys.executable`` and carries that
    risk, so it is not preferred. Missing or unparsable ``pyvenv.cfg`` counts
    as a mismatch: a bare interpreter file without one is not a working venv.
    """
    cfg = root / ".venv" / "pyvenv.cfg"
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep or key.strip().lower() not in ("version", "version_info"):
            continue
        parts = value.strip().split(".")
        try:
            return (int(parts[0]), int(parts[1])) == sys.version_info[:2]
        except (IndexError, ValueError):
            return False
    return False


def resolve_app_python(root: Path | None) -> str:
    """Absolute interpreter for processes spawned on an app's behalf.

    Prefers ``<root>/.venv``'s interpreter when it exists as a runnable,
    non-empty executable AND its ``pyvenv.cfg`` names the same Python minor
    version as the gateway (see :func:`_venv_version_matches` — provisioned
    deps arrive via ``PYTHONPATH`` built by ``sys.executable``, so a
    version-mismatched venv would mix ABIs). Else the gateway's
    ``sys.executable`` — never a bare PATH-resolved name. The runnability
    check matters: a venv interpreter that lost its execute bit or was
    truncated to zero bytes (a partial copy, a restore that dropped content)
    would turn a working ``sys.executable`` fallback into a guaranteed spawn
    failure. ``root=None`` means "no app context" and resolves straight to
    ``sys.executable``.
    """
    if root is not None:
        venv_py = venv_python_path(root)
        if _runnable(venv_py) and _venv_version_matches(root):
            return str(venv_py)
    return sys.executable


def venv_provided_command(root: Path, name: str) -> str | None:
    """Absolute path of ``name`` if the app's venv or deps dir provides it.

    Covers console scripts a pip install creates: ``.venv/bin/<name>`` for an
    app-owned venv, and ``<deps dir>/bin/<name>`` for the gateway's
    ``pip install --target`` provisioning (``Scripts\\`` on Windows in both
    layouts; the ``.exe`` suffix is appended only when ``name`` does not
    already carry it). Both are invisible to PATH — a venv is never activated
    and a target dir has no activation at all. Only a runnable provided binary
    is a safe rewrite target: anything else a manifest names bare (``node``,
    ``docker``) was a deliberate PATH dependency and must be left alone, and a
    non-executable file (a data artifact, a partial pip install) must not
    displace a command that would otherwise work.

    Callers must pass a bare NAME (no path separators, no drive qualifier) —
    the caller-side guard in ``resolve_stdio_command`` enforces that, keeping
    the joins below inside the probed directories.
    """
    if platform_compat.IS_WINDOWS:
        scripts = "Scripts"
        exe_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    else:
        scripts = "bin"
        exe_name = name
    for base in (root / ".venv" / scripts, app_deps_dir(root) / scripts):
        candidate = base / exe_name
        if _runnable(candidate):
            return str(candidate)
    return None
