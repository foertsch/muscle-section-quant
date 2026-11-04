import hashlib

from msq.core import runinfo


def test_weights_checksum_and_missing_file(tmp_path):
    f = tmp_path / "cpsam_v2"
    f.write_bytes(b"not really weights" * 1000)
    w = runinfo.weights(f)
    assert w == {"file": "cpsam_v2", "bytes": 18000, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    assert runinfo.weights(tmp_path / "missing") is None


def test_code_version_and_packages():
    v = runinfo.code_version()
    assert v and v != "unknown"
    assert runinfo.repo_root() is not None  # a development checkout
    p = runinfo.packages()
    assert set(p) == {"python", *runinfo.PACKAGES} and all(p.values())
    assert runinfo.packages(("numpy", "no-such-package"), missing_ok=True)["no-such-package"] is None
    assert "-" in runinfo.platform_key() and runinfo.peak_memory_gb() > 0
