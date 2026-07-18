"""Sandboxed execution of LLM-written Python.

Isolation layers (documented honestly — see README "Tradeoffs & limits"):
  1. fresh subprocess per execution, killed as a process group on timeout
  2. resource limits via `resource.setrlimit`: CPU seconds, address space,
     file size, process count
  3. temp working directory containing ONLY a copy of the dataset
  4. network access removed from the environment (no proxies) and blocked
     best-effort; production isolation would be gVisor/Firecracker

The executed script is expected to print its findings to stdout and may write
artifacts (JSON chart specs, PNGs) into ./artifacts within its workdir.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from app.config import settings
from app.runtime.state import SandboxResult

MAX_OUTPUT_CHARS = 20_000

_PRELUDE = """
import resource, os
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_FSIZE, (50_000_000, 50_000_000))
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
except (ValueError, OSError):
    pass  # not permitted in some environments (e.g. containers); other limits still hold
os.makedirs("artifacts", exist_ok=True)
"""


def run_code(code: str, dataset_path: str | Path) -> SandboxResult:
    """Execute `code` in an isolated subprocess with the dataset copied to ./data.csv."""
    cfg = settings()
    workdir = Path(tempfile.mkdtemp(prefix="tracelab-sbx-"))
    try:
        shutil.copy(dataset_path, workdir / "data.csv")
        prelude = _PRELUDE.format(
            cpu=cfg.sandbox_cpu_seconds, mem=cfg.sandbox_memory_mb * 1024 * 1024
        )
        script = prelude + "\n" + textwrap.dedent(code)
        (workdir / "script.py").write_text(script)

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(workdir),
            "MPLBACKEND": "Agg",
            "NO_PROXY": "*",
            # no HTTP(S)_PROXY, no API keys — nothing sensitive crosses this boundary
        }

        proc = subprocess.Popen(
            [sys.executable, "script.py"],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group → clean kill on timeout
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=cfg.sandbox_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()

        artifacts = _collect_artifacts(workdir / "artifacts")
        return SandboxResult(
            code=code,
            stdout=stdout.decode(errors="replace")[:MAX_OUTPUT_CHARS],
            stderr=stderr.decode(errors="replace")[:MAX_OUTPUT_CHARS],
            exit_code=proc.returncode if not timed_out else -9,
            timed_out=timed_out,
            artifacts=artifacts,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _collect_artifacts(artifacts_dir: Path) -> list[dict]:
    out: list[dict] = []
    if not artifacts_dir.is_dir():
        return out
    for path in sorted(artifacts_dir.iterdir()):
        if path.suffix == ".json":
            try:
                out.append({"kind": "json", "name": path.name, "data": json.loads(path.read_text())})
            except (json.JSONDecodeError, OSError):
                out.append({"kind": "invalid", "name": path.name})
    return out
