"""GEO sometimes refuses a request for a moment and answers the next one.
Both download scripts must wait that out, and must still fail fast on a real
error. A local server stands in for GEO: it refuses the first few requests."""

import gzip
import http.server
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import parse_geo_metadata as pgm  # noqa: E402

BODY = gzip.compress(b'!Series_title\t"x"\n' * 50)


def serve(refusals, status=403):
    """A server whose first `refusals` requests get `status`; /missing is always 404."""
    state = {"left": refusals, "seen": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def answer(self, with_body):
            state["seen"] += 1
            if self.path.endswith("/missing"):
                self.send_error(404)
                return
            if state["left"] > 0:
                state["left"] -= 1
                self.send_error(status)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            if with_body:
                self.wfile.write(BODY)

        def do_GET(self):
            self.answer(True)

        def do_HEAD(self):
            self.answer(False)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", state


def with_env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    return old


def restore(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---- scripts/parse_geo_metadata.py
def test_python_download_waits_out_refusals():
    server, url, state = serve(refusals=2)
    old = with_env(GEO_RETRY_WAIT="0", GEO_REFUSAL_TRIES="5")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "matrix.txt.gz"
            pgm.download(f"{url}/file", dest)
            assert dest.read_bytes() == BODY and state["seen"] == 3
            assert not dest.with_name(dest.name + ".part").exists()
    finally:
        restore(old)
        server.shutdown()


def test_python_download_gives_up_after_the_allowed_refusals():
    server, url, state = serve(refusals=100)
    old = with_env(GEO_RETRY_WAIT="0", GEO_REFUSAL_TRIES="4")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                pgm.download(f"{url}/file", Path(tmp) / "m.gz")
            except SystemExit as e:
                assert "could not download" in str(e)
            else:
                raise AssertionError("expected a stop")
            assert state["seen"] == 4 + 3  # the waits, then the 3 ordinary tries
    finally:
        restore(old)
        server.shutdown()


def test_python_download_does_not_wait_on_a_missing_file():
    server, url, state = serve(refusals=0)
    old = with_env(GEO_RETRY_WAIT="60", GEO_REFUSAL_TRIES="20")  # would take 20 minutes
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                pgm.download(f"{url}/missing", Path(tmp) / "m.gz")
            except SystemExit:
                pass
            else:
                raise AssertionError("expected a stop")
            assert state["seen"] == 3  # the 3 ordinary tries, no waiting
    finally:
        restore(old)
        server.shutdown()


# ---- scripts/download_idats.sh: its size check, run exactly as written
def remote_bytes(url, tries="5"):
    script = (ROOT / "scripts" / "download_idats.sh").read_text()
    start = script.index("remote_bytes() {")
    function = script[start : script.index("\n}\n", start) + 3]
    env = dict(
        os.environ,
        GEO_REFUSAL_TRIES=tries,
        GEO_RETRY_WAIT="0",
        no_proxy="127.0.0.1",
        NO_PROXY="127.0.0.1",
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -Eeuo pipefail; log() { printf "%s\\n" "$*"; }\n'
            + function
            + '\nremote_bytes "$1"',
            "_",
            url,
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def test_shell_size_check_waits_out_refusals():
    server, url, state = serve(refusals=2, status=403)
    try:
        r = remote_bytes(f"{url}/file")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == str(len(BODY))  # only the size on stdout
        assert "GEO answered 403" in r.stderr and state["seen"] == 3
    finally:
        server.shutdown()


def test_shell_size_check_gives_up_and_fails_fast_on_a_missing_file():
    server, url, state = serve(refusals=100)
    try:
        r = remote_bytes(f"{url}/file", tries="3")
        assert r.returncode != 0 and r.stdout.strip() == "" and state["seen"] == 4
    finally:
        server.shutdown()
    server, url, state = serve(refusals=0)
    try:
        r = remote_bytes(f"{url}/missing", tries="20")
        assert r.returncode != 0 and state["seen"] == 1  # no waiting on a 404
    finally:
        server.shutdown()


def test_the_big_download_retries_refusals_too():
    script = (ROOT / "scripts" / "download_idats.sh").read_text()
    assert "--retry-on-http-error=403,429,500,502,503,504" in script


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
