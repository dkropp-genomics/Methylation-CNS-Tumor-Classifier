"""Tests for scripts/download_idats.sh.

A tiny fake "GEO" is served from a local HTTP server, so the tests run in a
few seconds, need no internet, and exercise the real script end to end.

Run with pytest (CI) or directly:  python tests/test_download_idats.py
"""

from __future__ import annotations

import functools
import gzip
import http.server
import os
import random
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_idats.sh"
ACC = "GSE90496"
GSMS = [f"GSM24028{n:02d}" for n in range(10)]


# ----------------------------------------------------------------- helpers --
def make_idats(folder: Path, gsms=GSMS, channels=("Grn", "Red")) -> list[Path]:
    """Write small gzip files named like real GEO IDATs."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    paths = []
    for i, gsm in enumerate(gsms):
        for ch in channels:
            p = folder / f"{gsm}_57750410{i:02d}_R0{i % 6 + 1}C01_{ch}.idat.gz"
            p.write_bytes(gzip.compress(rng.randbytes(2000), mtime=0))
            paths.append(p)
    return paths


PLATFORM_FILES = [
    "GPL13534_HumanMethylation450_15017482_v.1.2.bpm.gz",
    "GPL13534_450K_Manifest_header_Descriptions.xlsx.gz",
]


def make_geo(root: Path, extra: tuple[str, ...] = (), **kw) -> Path:
    """Build <root>/geo/series/GSE90nnn/GSE90496/suppl/GSE90496_RAW.tar.

    `extra` = names of non-IDAT files to add to the tar (like GEO's GPL files).
    """
    files = make_idats(root / "src", **kw)
    for name in extra:
        p = root / "src" / name
        p.write_bytes(gzip.compress(b"platform file", mtime=0))
        files.append(p)
    suppl = root / "geo" / "series" / "GSE90nnn" / ACC / "suppl"
    suppl.mkdir(parents=True)
    tar_path = suppl / f"{ACC}_RAW.tar"
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as tar:
        for f in files:
            tar.add(f, arcname=f.name)
    return tar_path


class Server:
    """Serve a folder over HTTP on a free local port (quietly)."""

    def __init__(self, folder: Path):
        handler = functools.partial(Quiet, directory=str(folder))
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/geo"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def run(repo: Path, url: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(
        os.environ,
        GEO_BASE_URL=url,
        MIN_FREE_GB="0",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    return subprocess.run(
        ["bash", str(SCRIPT), ACC, *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class Case:
    """A throwaway repo folder plus a fake GEO server."""

    def __init__(self, **kw):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.tar = make_geo(root / "server", **kw)
        self.repo = root / "repo"
        self.repo.mkdir()
        self.server = Server(root / "server")

    def run(self, *args):
        return run(self.repo, self.server.url, *args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.close()
        self.tmp.cleanup()


def ok(res):
    assert res.returncode == 0, f"expected success:\n{res.stdout}\n{res.stderr}"


def fails(res, text):
    out = res.stdout + res.stderr
    assert res.returncode != 0, f"expected failure:\n{out}"
    assert text in out, f"expected {text!r} in output:\n{out}"


# ------------------------------------------------------------------- tests --
def test_fresh_download_then_skip():
    with Case() as c:
        ok(c.run())
        idats = sorted((c.repo / "data/idat" / ACC).glob("*.idat.gz"))
        assert len(idats) == 2 * len(GSMS)
        manifest = (c.repo / "results/download" / f"{ACC}_idat_md5.tsv").read_text()
        assert manifest.splitlines()[0] == "file\tmd5"
        assert len(manifest.splitlines()) == 1 + len(idats)
        summary = (c.repo / "results/download/download_summary.tsv").read_text().splitlines()
        assert len(summary) == 2 and summary[1].split("\t")[5] == str(len(GSMS))
        assert (c.repo / "data/raw" / f"{ACC}_RAW.tar").stat().st_size == c.tar.stat().st_size
        assert not list((c.repo / "data/raw").glob("*.part"))
        assert "already verified" in c.run().stdout  # second run skips


def test_existing_extraction_is_reused():
    """Dawson's case: tar and IDATs already on disk from manual commands."""
    with Case() as c:
        (c.repo / "data/raw").mkdir(parents=True)
        (c.repo / "data/raw" / f"{ACC}_RAW.tar").write_bytes(c.tar.read_bytes())
        with tarfile.open(c.tar) as t:
            t.extractall(c.repo / "data/idat" / ACC, filter="data")
        res = c.run("--offline")
        ok(res)
        assert "not extracting" in res.stdout


def test_partial_download_resumes():
    with Case() as c:
        part = c.repo / "data/raw" / f"{ACC}_RAW.tar.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(c.tar.read_bytes()[:5000])
        ok(c.run())


def test_missing_channel_fails_before_extraction():
    with Case(channels=("Grn",)) as c:
        fails(c.run(), "without both Grn and Red")
        assert not list((c.repo / "data/idat" / ACC).glob("*.idat.gz"))


def test_metadata_mismatch_fails():
    with Case() as c:
        meta = c.repo / "data/meta" / f"{ACC}_samples.tsv"
        meta.parent.mkdir(parents=True)
        rows = ["geo_accession\ttitle"] + [f"{g}\tx" for g in GSMS[:-1] + ["GSM9999999"]]
        meta.write_text("\n".join(rows) + "\n")
        fails(c.run(), "samples in metadata have no IDATs")


def test_corruption_is_caught_on_reverify():
    with Case() as c:
        ok(c.run())
        victim = sorted((c.repo / "data/idat" / ACC).glob("*.idat.gz"))[0]
        data = bytearray(victim.read_bytes())
        data[100] ^= 0xFF
        victim.write_bytes(bytes(data))
        # The tar still matches in size, so the file isn't re-extracted by name,
        # but its size is unchanged too -- only the gzip/MD5 checks can catch it.
        fails(c.run("--reverify", "--offline"), "")


def test_remove_tar_then_verify_from_manifest():
    with Case() as c:
        ok(c.run("--remove-tar"))
        assert not (c.repo / "data/raw" / f"{ACC}_RAW.tar").exists()
        res = c.run("--reverify")
        ok(res)
        assert "verifying" in res.stdout and "match the committed manifest" in res.stdout


def test_platform_files_are_skipped_not_extracted():
    """GEO's GSE90496 tar bundles GPL13534 manifest files next to the IDATs."""
    with Case(extra=tuple(PLATFORM_FILES)) as c:
        res = c.run()
        ok(res)
        assert "2 Illumina platform file(s)" in res.stdout
        idat_dir = c.repo / "data/idat" / ACC
        assert not list(idat_dir.glob("GPL*"))
        assert len(list(idat_dir.glob("*.idat.gz"))) == 2 * len(GSMS)


def test_platform_files_already_extracted_are_tolerated():
    """Dawson's case: a manual `tar -xf` put the GPL files in the IDAT folder."""
    with Case(extra=tuple(PLATFORM_FILES)) as c:
        (c.repo / "data/raw").mkdir(parents=True)
        (c.repo / "data/raw" / f"{ACC}_RAW.tar").write_bytes(c.tar.read_bytes())
        with tarfile.open(c.tar) as t:
            t.extractall(c.repo / "data/idat" / ACC, filter="data")
        res = c.run()
        ok(res)
        assert "not extracting" in res.stdout


def test_unknown_file_in_tar_fails():
    with Case(extra=("README.txt.gz",)) as c:
        fails(c.run(), "unexpected entries in tar")


def test_near_miss_idat_name_is_not_treated_as_idat():
    """`.idat.gz` must match a literal dot: 'Grn_idat.gz' is not an IDAT."""
    with Case(extra=("GSM2402899_5775041099_R01C01_Grn_idat.gz",)) as c:
        fails(c.run(), "unexpected entries in tar")


def test_remove_tar_after_earlier_verification():
    with Case() as c:
        ok(c.run())
        tar = c.repo / "data/raw" / f"{ACC}_RAW.tar"
        assert tar.exists()
        res = c.run("--remove-tar")
        ok(res)
        assert "verified earlier" in res.stdout and not tar.exists()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
