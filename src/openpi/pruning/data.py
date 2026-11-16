"""Read raw calibration inputs; never reuse another policy's activations."""
import json
from pathlib import Path

import numpy as np
from openpi_client import msgpack_numpy

from openpi.pruning.artifacts import sha256


def load_corpus(corpus=None, *, legacy_manifest=None, legacy_root=None, checkpoint_hash=None):
    if (corpus is None) == (legacy_manifest is None):
        raise ValueError("Select exactly one corpus format")
    if legacy_manifest is not None:
        manifest = Path(legacy_manifest).resolve()
        root = Path(legacy_root).resolve()
        source = json.loads(manifest.read_text())
        entries = []
        for item in source:
            path = (root/item["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Invalid legacy observation path")
            entries.append(dict(path=str(path),sha256=sha256(path),suite=item["suite"],
                task_id=int(item["task_id"]),trial=int(item["episode_index"]),format="legacy_npz"))
        provenance = dict(manifest=str(manifest),manifest_sha256=sha256(manifest),
            observation_policy="original pi05_libero dense rollouts; reused raw inputs only",
            scoring_policy="current Future checkpoint",format="legacy_npz")
    else:
        manifest = Path(corpus).resolve()
        source = json.loads(manifest.read_text())
        if source.get("checkpoint_manifest_sha256") != checkpoint_hash:
            raise ValueError("Future corpus checkpoint provenance mismatch")
        entries = []
        for item in source["observations"]:
            path = (manifest.parent/item["path"]).resolve()
            if not path.is_relative_to(manifest.parent) or sha256(path) != item["sha256"]:
                raise ValueError("Invalid or changed corpus observation")
            entries.append(dict(item,path=str(path),format="msgpack"))
        provenance = dict(manifest=str(manifest),manifest_sha256=sha256(manifest),
            observation_policy="Future dense checkpoint",format="msgpack")
    if not entries or len({x["path"] for x in entries}) != len(entries):
        raise ValueError("Empty or duplicate corpus")
    return entries,provenance


def read_observation(entry):
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Observation changed after corpus freezing")
    if entry["format"] == "msgpack":
        return msgpack_numpy.unpackb(path.read_bytes())
    with np.load(path,allow_pickle=False) as data:
        if str(data["task_suite"]) != entry["suite"] or int(data["task_id"]) != entry["task_id"]:
            raise ValueError("Observation task provenance mismatch")
        if int(data["episode_index"]) != entry["trial"]:
            raise ValueError("Observation episode provenance mismatch")
        return {"observation/image":data["observation_image"],
            "observation/wrist_image":data["observation_wrist_image"],
            "observation/state":data["observation_state"],"prompt":str(data["prompt"]),
            "action_context":np.zeros((0,7),dtype=np.float32)}
