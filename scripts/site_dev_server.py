#!/usr/bin/env python3
"""Serve the local site with an opt-in snippet runner.

This is for local development only. It serves files from `site/` and exposes
`POST /api/run-snippet` for short Python or JavaScript examples rendered in
lesson pages. The runner executes untrusted code with only a timeout and a
temporary working directory; do not expose this server beyond localhost.

这个脚本用于本地开发。它从`site/`提供文件，并公开`POST /api/run-snippet`接口，用于在课程页面中呈现的短Python或JavaScript示例。
运行器通过仅使用超时和临时工作目录来执行不受信任的代码；请勿将此服务器暴露在localhost之外。
"""


from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"
MAX_CODE_BYTES = 32_000
TIMEOUT_SEC = 5
DEFAULT_VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


class SiteDevHandler(SimpleHTTPRequestHandler):
    server_version = "AESiteDev/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/runner":
            self.write_json(
                {
                    "available": True,
                    "languages": ["python", "py", "javascript", "js"],
                    "timeoutSec": TIMEOUT_SEC,
                    "maxCodeBytes": MAX_CODE_BYTES,
                    "python": str(self.server.python_executable),
                }
            )
            return
        if self.path.startswith("/repo/"):
            self.serve_repo_file()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run-snippet":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid content length")
            return

        if length <= 0 or length > MAX_CODE_BYTES:
            self.write_json(
                {"ok": False, "error": f"code must be 1-{MAX_CODE_BYTES} bytes"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.write_json({"ok": False, "error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        code = payload.get("code")
        language = str(payload.get("language", "")).lower()
        if not isinstance(code, str) or not code.strip():
            self.write_json({"ok": False, "error": "code is required"}, HTTPStatus.BAD_REQUEST)
            return

        result = run_snippet(code, language, self.server.python_executable)
        self.write_json(result, HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_REQUEST)

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_repo_file(self) -> None:
        parsed = urlparse(self.path)
        rel = unquote(parsed.path.removeprefix("/repo/"))
        target = (ROOT / rel).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = "text/plain; charset=utf-8"
        if target.suffix == ".json":
            content_type = "application/json; charset=utf-8"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def run_snippet(code: str, language: str, python_executable: Path) -> dict[str, Any]:
    if language in {"python", "py"}:
        command = [str(python_executable), "-c", code]
    elif language in {"javascript", "js"}:
        command = ["node", "-e", code]
    else:
        return {"ok": False, "error": f"unsupported language: {language or 'unknown'}"}

    env = {
        **os.environ,
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }

    with tempfile.TemporaryDirectory(prefix="lesson-snippet-") as tmp:
        try:
            proc = subprocess.run(
                command,
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError:
            return {"ok": False, "error": f"runtime not found for {language}"}
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": True,
                "exitCode": None,
                "timedOut": True,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or f"Timed out after {TIMEOUT_SEC}s",
            }

    return {
        "ok": True,
        "exitCode": proc.returncode,
        "timedOut": False,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable),
        help="Python interpreter used for snippets; defaults to .venv/bin/python when present",
    )
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), SiteDevHandler)
    server.python_executable = args.python
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {SITE_DIR} at {url}")
    print(f"Snippet runner enabled with Python: {server.python_executable}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
