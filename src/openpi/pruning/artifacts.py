"""Hash-verified original and derived Future checkpoints."""
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ID = "Dream-LIN/pi05_libero_async_future"
ARTIFACT_TYPE = "future-static-mlp-v1"
CONFIG_NAME = "pi05_libero_async_future"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(root, path):
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))


def verify_checkpoint(checkpoint):
    root = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((root / "checkpoint_manifest.json").read_text())
    if manifest.get("format") != "Orbax JAX" or (root / "model.safetensors").exists():
        raise ValueError("Expected an Orbax JAX Future checkpoint")
    kind = manifest.get("artifact_type")
    if kind is None:
        if manifest.get("repo_id") != REPO_ID:
            raise ValueError("Expected the original Future checkpoint")
        width = 16384
    elif kind == ARTIFACT_TYPE:
        if manifest.get("source", {}).get("repo_id") != REPO_ID or manifest.get("config") != CONFIG_NAME:
            raise ValueError("Invalid derived checkpoint provenance")
        width = manifest.get("vlm_mlp_dim")
        if not isinstance(width, int) or not 0 < width <= 16384 or width % 64:
            raise ValueError("Invalid compact MLP width")
        if manifest.get("source_width") != 16384 or manifest.get("action_context_horizon") != 9:
            raise ValueError("Invalid derived model structure")
    else:
        raise ValueError("Unsupported artifact_type")
    entries = manifest.get("files", [])
    listed = set()
    for item in entries:
        rel = item["path"]
        path = root / rel
        if Path(rel).is_absolute() or rel in listed or not path.resolve().is_relative_to(root):
            raise ValueError(f"Invalid manifest path: {rel}")
        listed.add(rel)
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint integrity failure: {rel}")
    required = {"params/_METADATA", "assets/physical-intelligence/libero/norm_stats.json"}
    actual = {p.relative_to(root).as_posix() for folder in ["params", "assets"]
              for p in (root / folder).rglob("*") if p.is_file()}
    if not required <= listed or not actual <= listed:
        raise ValueError("Manifest must cover every parameter and asset file")
    if kind:
        if "channel_indices.npz" not in listed:
            raise ValueError("Missing retained indices")
        with np.load(root / "channel_indices.npz", allow_pickle=False) as archive:
            indices = archive["indices"]
        if (indices.shape != (18, width) or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= 16384) or np.any(np.diff(indices, axis=1) <= 0)):
            raise ValueError("Malformed retained indices")
    return root, manifest, width
