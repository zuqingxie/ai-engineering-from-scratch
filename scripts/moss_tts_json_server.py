#!/usr/bin/env python3
"""Local JSON adapter for MOSS-TTS-Nano paragraph playback.

The upstream MOSS-TTS-Nano demo API is form-based. The lesson site prototype
uses a small JSON contract instead, so this local-only server translates:

    POST /synthesize {"text": "...", "voice": "..."}

into a `moss-tts-nano generate --backend onnx ...` call and returns audio/wav.
Do not expose this server beyond localhost.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOSS_REPO = ROOT / "third_party" / "MOSS-TTS-Nano"
MAX_TEXT_BYTES = 20_000
SYNTH_TIMEOUT_SEC = 300


class MossJsonTTSHandler(BaseHTTPRequestHandler):
    server_version = "AIFS-MOSS-TTS/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(self.server.health_payload())
            return
        if self.path == "/voices":
            self.write_json(
                {
                    "available": self.server.health_payload()["available"],
                    "default": self.server.default_voice,
                    "voices": ["Junhao", "Ava"],
                    "byLanguage": {"en": self.server.english_voice, "zh": self.server.chinese_voice},
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.write_json({"error": "invalid content length"}, HTTPStatus.BAD_REQUEST)
            return

        if length <= 0 or length > MAX_TEXT_BYTES:
            self.write_json({"error": f"request must be 1-{MAX_TEXT_BYTES} bytes"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.write_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        text = str(payload.get("text") or "").strip()
        if not text:
            self.write_json({"error": "text is required"}, HTTPStatus.BAD_REQUEST)
            return

        health = self.server.health_payload()
        if not health["available"]:
            self.write_json({"error": health["message"]}, HTTPStatus.SERVICE_UNAVAILABLE)
            return

        voice = str(payload.get("voice") or self.server.default_voice).strip() or self.server.default_voice
        try:
            audio = self.server.synthesize(text=text, voice=voice, payload=payload)
        except subprocess.TimeoutExpired:
            self.write_json({"error": f"TTS timed out after {SYNTH_TIMEOUT_SEC}s"}, HTTPStatus.GATEWAY_TIMEOUT)
            return
        except RuntimeError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(audio)

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class MossJsonTTSServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[MossJsonTTSHandler],
        *,
        python_executable: Path,
        moss_repo: Path,
        english_voice: str,
        chinese_voice: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.python_executable = python_executable
        self.moss_repo = moss_repo
        self.english_voice = english_voice
        self.chinese_voice = chinese_voice
        self.default_voice = chinese_voice
        self._health_checked_at = 0.0
        self._health_cache: dict[str, Any] | None = None

    def health_payload(self) -> dict[str, Any]:
        now = time.time()
        if self._health_cache is not None and now - self._health_checked_at < 10:
            return self._health_cache

        available = False
        state = "unavailable"
        message = "MOSS-TTS-Nano is not ready."
        if not self.moss_repo.is_dir():
            message = f"Missing MOSS-TTS-Nano checkout: {self.moss_repo}"
        else:
            proc = subprocess.run(
                [
                    str(self.python_executable),
                    "-c",
                    "import onnxruntime, numpy; import moss_tts_nano.cli",
                ],
                cwd=self.moss_repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
                check=False,
            )
            available = proc.returncode == 0
            if available:
                state = "ready"
                message = "ok"
            else:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                hint = "Run `pip install -e third_party/MOSS-TTS-Nano` in the Python environment used by site_dev_server.py."
                message = (detail[-1] + " " if detail else "") + hint

        self._health_cache = {
            "status": "ok",
            "available": available,
            "state": state,
            "engine": "moss-tts-nano-json-adapter",
            "mossRepo": str(self.moss_repo),
            "python": str(self.python_executable),
            "message": message,
            "voices": {"en": self.english_voice, "zh": self.chinese_voice},
        }
        self._health_checked_at = now
        return self._health_cache

    def synthesize(self, *, text: str, voice: str, payload: dict[str, Any]) -> bytes:
        with tempfile.TemporaryDirectory(prefix="aifs-moss-tts-") as tmp:
            output = Path(tmp) / "speech.wav"
            command = [
                str(self.python_executable),
                "-m",
                "moss_tts_nano.cli",
                "generate",
                "--backend",
                "onnx",
                "--output",
                str(output),
                "--text",
                text,
                "--voice",
                voice,
                "--execution-provider",
                str(payload.get("execution_provider") or "cpu"),
                "--cpu-threads",
                str(int(payload.get("cpu_threads") or 4)),
                "--max-new-frames",
                str(int(payload.get("max_new_frames") or 375)),
                "--voice-clone-max-text-tokens",
                str(int(payload.get("voice_clone_max_text_tokens") or 75)),
                "--text-temperature",
                str(float(payload.get("text_temperature") or 1.0)),
                "--text-top-p",
                str(float(payload.get("text_top_p") or 1.0)),
                "--text-top-k",
                str(int(payload.get("text_top_k") or 50)),
                "--audio-temperature",
                str(float(payload.get("audio_temperature") or 0.8)),
                "--audio-top-p",
                str(float(payload.get("audio_top_p") or 0.95)),
                "--audio-top-k",
                str(int(payload.get("audio_top_k") or 25)),
                "--audio-repetition-penalty",
                str(float(payload.get("audio_repetition_penalty") or 1.2)),
            ]
            if payload.get("enable_text_normalization", True):
                command.append("--enable-wetext-processing")
            if payload.get("seed") not in (None, "", 0, "0"):
                command.extend(["--seed", str(int(payload["seed"]))])

            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            }
            proc = subprocess.run(
                command,
                cwd=self.moss_repo,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=SYNTH_TIMEOUT_SEC,
                check=False,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "TTS generation failed").strip()
                raise RuntimeError(detail[-2000:])
            if not output.is_file():
                raise RuntimeError("TTS generation finished without producing audio.")
            return output.read_bytes()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--moss-repo", type=Path, default=DEFAULT_MOSS_REPO)
    parser.add_argument("--english-voice", default="Ava")
    parser.add_argument("--chinese-voice", default="Junhao")
    args = parser.parse_args(argv)

    server = MossJsonTTSServer(
        (args.host, args.port),
        MossJsonTTSHandler,
        python_executable=args.python.absolute(),
        moss_repo=args.moss_repo.resolve(),
        english_voice=args.english_voice,
        chinese_voice=args.chinese_voice,
    )
    print(f"Serving MOSS-TTS JSON adapter at http://{args.host}:{args.port}")
    print(f"MOSS-TTS-Nano repo: {server.moss_repo}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MOSS-TTS JSON adapter.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
