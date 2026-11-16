"""Export physically compact Orbax parameters. CPU only; never edits the source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as model_lib
from openpi.pruning.artifacts import ARTIFACT_TYPE, CONFIG_NAME, REPO_ID, file_entry, sha256, verify_checkpoint
from openpi.pruning.channels import SCORE_NAME, channel_scores, compact_params, mlp_weights, select_channels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--keep",type=int,required=True)
    parser.add_argument("--selection",choices=["importance","prefix"],default="importance")
    parser.add_argument("--statistics",help=".npz produced by calibrate_future_channels.py")
    args=parser.parse_args()
    source, source_manifest, source_width=verify_checkpoint(args.checkpoint)
    if source_manifest.get("artifact_type") or source_width!=16384:
        raise ValueError("Export directly from the original dense Future checkpoint")
    output=Path(args.output).expanduser().resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a fresh output path separate from the source checkpoint")
    if not 0<args.keep<=source_width or args.keep%64:
        raise ValueError("keep must be a positive multiple of 64 <= 16384")
    source_hash=sha256(source/"checkpoint_manifest.json")
    statistics_provenance=None
    if args.selection=="importance":
        if not args.statistics:
            parser.error("importance selection requires --statistics")
        with np.load(args.statistics,allow_pickle=False) as stats:
            if str(stats["source_manifest_sha256"])!=source_hash or str(stats["score_name"])!=SCORE_NAME:
                raise ValueError("Statistics provenance or scoring definition mismatch")
            energy=stats["mean_energy"]
            statistics_provenance=dict(path=str(Path(args.statistics).resolve()),sha256=sha256(args.statistics),
                corpus_sha256=str(stats["corpus_sha256"]),score_name=SCORE_NAME)
    params=model_lib.restore_params(source/"params",restore_type=np.ndarray)
    gate,down=mlp_weights(params)
    if gate.shape!=(18,2,2048,16384) or down.shape!=(18,16384,2048):
        raise ValueError("Unexpected source VLM MLP layout")
    indices=(select_channels(channel_scores(energy,down),args.keep) if args.selection=="importance"
             else np.tile(np.arange(args.keep,dtype=np.int32),(18,1)))
    compact=compact_params(params,indices)
    output.mkdir(parents=True,exist_ok=False)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(output/"params",ocp.args.PyTreeSave({"params":compact}))
    shutil.copytree(source/"assets",output/"assets")
    np.savez(output/"channel_indices.npz",indices=indices)
    version=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "code-only-copy"
    code_root=Path(__file__).resolve().parents[1]
    code_hashes={str(p.relative_to(code_root)):sha256(p) for folder in ["src","scripts","packages"]
                 for p in (code_root/folder).rglob("*.py")}
    (output/"export_code_manifest.json").write_text(json.dumps(code_hashes,indent=2)+'\n')
    manifest=dict(format="Orbax JAX",artifact_type=ARTIFACT_TYPE,config=CONFIG_NAME,
        checkpoint_step=source_manifest["checkpoint_step"],created_at=time.time(),code_revision=version,
        source=dict(repo_id=REPO_ID,checkpoint=str(source),manifest_sha256=source_hash),
        source_width=16384,vlm_mlp_dim=args.keep,action_context_horizon=9,
        selection=args.selection,statistics=statistics_provenance,
        native_activation="GELU",online_pruning=False,files=[])
    manifest["files"]=[file_entry(output,p) for p in sorted(output.rglob("*")) if p.is_file()]
    (output/"checkpoint_manifest.json").write_text(json.dumps(manifest,indent=2)+'\n')
    verify_checkpoint(output)
    print(json.dumps(dict(checkpoint=str(output),width=args.keep,selection=args.selection,verified=True)))


if __name__=="__main__":
    main()
