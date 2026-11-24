"""Local page for one person to play, teach, and watch the agent use the lesson."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
from urllib.parse import urlparse

from rl_donut.play.arena import HumanArena


PAGE = Path(__file__).with_name("static") / "play.html"


def _handler(arena: HumanArena):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if self.path.startswith("/favicon"):
                return
            print(f"{self.address_string()} {fmt % args}", flush=True)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/play"):
                self._bytes(200, "text/html; charset=utf-8", PAGE.read_bytes())
                return
            if path == "/api/status":
                with lock:
                    self._json(200, arena.status())
                return
            if path == "/favicon.ico":
                self._bytes(204, "text/plain", b"")
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "the request was not JSON"})
                return
            if not isinstance(body, dict):
                self._json(400, {"error": "the request must be an object"})
                return
            try:
                with lock:
                    if path == "/api/match/start":
                        payload = arena.start(body.get("seed"), body.get("seconds"))
                    elif path == "/api/match/step":
                        payload = arena.step(
                            body.get("move", 0),
                            bool(body.get("jet", False)),
                            bool(body.get("shoot", False)),
                            body.get("aim", 0.0),
                        )
                    elif path == "/api/match/teach":
                        payload = arena.teach(body.get("player", "you"))
                    elif path == "/api/match/teach/begin":
                        payload = arena.begin_teach(body.get("player", "you"))
                    elif path == "/api/match/teach/step":
                        payload = arena.teach_step()
                    elif path == "/api/match/teach/finish":
                        payload = arena.finish_teach()
                    elif path == "/api/agent/restore":
                        payload = arena.restore_original()
                    elif path == "/api/demos/forget":
                        payload = arena.forget()
                    else:
                        self._json(404, {"error": "not found"})
                        return
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, payload)

        def _json(self, code: int, payload: dict) -> None:
            self._bytes(code, "application/json", json.dumps(payload).encode())

        def _bytes(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(arena: HumanArena, host: str = "127.0.0.1", port: int = 8765, *, background: bool = False):
    if not PAGE.exists():
        raise FileNotFoundError(PAGE)
    class Server(ThreadingHTTPServer):
        allow_reuse_address = True

    httpd = Server((host, port), _handler(arena))
    httpd.daemon_threads = True
    if background:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd
    print(f"Open http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        httpd.server_close()
