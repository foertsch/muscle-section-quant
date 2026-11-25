"""Pins for the data a report page embeds: a short digest per top-level key, so a refactor of the page code or of
the report base cannot change a number, an outline or an image unnoticed.

Floats are rounded to 3 decimals, embedded images are reduced to what survives a change of JPEG library (type, size,
mean colour of the decoded pixels; a PNG is lossless, so its pixels are hashed), and the volatile top-level fields
(dates, the run folder, the page title) are dropped before hashing."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image

VOLATILE = {"generated", "made", "run", "title"}


def image_signature(uri: str):
    head, _, body = uri.partition(",")
    mime = head[len("data:"):].split(";")[0]
    im = Image.open(io.BytesIO(base64.b64decode(body)))
    a = np.asarray(im)
    if mime == "image/png":
        return [mime, im.size, hashlib.sha1(a.tobytes()).hexdigest()[:12]]
    return [mime, im.size, [int(round(float(m))) for m in a.reshape(-1, a.shape[-1] if a.ndim == 3 else 1).mean(0)]]


def canon(obj):
    """The payload as plain JSON with rounded numbers and image signatures."""
    if isinstance(obj, dict):
        return {k: canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, str) and obj.startswith("data:image/"):
        return image_signature(obj)
    return obj


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(canon(obj), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def digests(payload: dict) -> dict[str, str]:
    return {k: digest(v) for k, v in payload.items() if k not in VOLATILE}


def embedded(html: str, name: str = "DATA"):
    """The JSON a page assigns to `const NAME = ...;` (the pinned data marker)."""
    return json.loads(re.search(rf"const {name} = (.*?);\n", html, re.S).group(1))


def assert_pinned(payload: dict, expected: dict[str, str]) -> None:
    got = digests(payload)
    assert set(got) == set(expected), f"keys differ: {sorted(set(got) ^ set(expected))}"
    changed = sorted(k for k in expected if got[k] != expected[k])
    assert not changed, f"payload changed under: {changed}"
