"""Calibrate Future FFN energy, per-channel scores and rankings from raw inputs."""
import argparse
import csv
import json
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.policies.pruned_future_policy import DEFAULT_CHECKPOINT, create_pruned_future_policy
from openpi.pruning.artifacts import sha256
from openpi.pruning.calibration import collect_prefix_energy
from openpi.pruning.channels import SCORE_NAME, down_weight_factor, select_channels
from openpi.pruning.data import load_corpus, read_observation
from openpi.shared.nnx_utils import module_jit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus",help="corpus.json produced by eval_pruned_future.py")
    source.add_argument("--legacy-manifest",help="Existing static_pruning calibration_manifest.json")
    parser.add_argument("--legacy-root",help="Root against which legacy observation paths resolve")
    parser.add_argument("--output",required=True,help="New .npz statistics artifact")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--limit",type=int,help="Optional explicit pilot subset")
    parser.add_argument("--worker-count",type=int,default=1)
    parser.add_argument("--worker-index",type=int,default=0)
    args=parser.parse_args()
    if args.worker_count<1 or not 0<=args.worker_index<args.worker_count:
        parser.error("Invalid worker index/count")
    if args.legacy_manifest and not args.legacy_root:parser.error("--legacy-root is required")
    output=Path(args.output).resolve()
    if output.suffix!=".npz":parser.error("output must end in .npz")
    if output.exists() or output.with_suffix(".json").exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    source_hash=sha256(Path(args.checkpoint)/"checkpoint_manifest.json")
    entries,provenance=load_corpus(args.corpus,legacy_manifest=args.legacy_manifest,
        legacy_root=args.legacy_root,checkpoint_hash=source_hash)
    if args.limit is not None:
        if args.limit<1:parser.error("limit must be positive")
        entries=entries[:args.limit]
    binding=dict(source_manifest_sha256=source_hash,corpus=provenance,observations=entries,
        collector_sha256=sha256(Path(__file__).parents[1]/"src/openpi/pruning/calibration.py"),
        backend=jax.default_backend(),jax_version=jax.__version__,score_name=SCORE_NAME)
    binding_path=output.with_suffix(".inputs.json")
    if binding_path.exists():
        if not args.resume or json.loads(binding_path.read_text())!=binding:
            raise ValueError("Existing calibration inputs differ, or --resume missing")
    else:binding_path.write_text(json.dumps(binding,indent=2)+'\n')
    samples=output.parent/(output.stem+"_samples")
    samples.mkdir(exist_ok=True)
    policy=create_pruned_future_policy(args.checkpoint)
    if policy.metadata["artifact_type"]!="original-future":raise ValueError("Use the dense source Future checkpoint")
    model=policy._model
    embed=module_jit(model.embed_prefix)
    params=nnx.state(model.PaliGemma.llm,nnx.Param).to_pure_dict()["layers"]
    configs=model.PaliGemma.llm.module.configs
    collect=jax.jit(lambda weights,tokens,mask,positions:collect_prefix_energy(
        weights,tokens,mask,positions,configs)[0])
    group_totals,group_counts={},{}
    progress_path=(output.with_suffix(".progress.json") if args.worker_count==1 else
        output.with_suffix(f".worker{args.worker_index}.progress.json"))
    print(json.dumps(dict(event="ready",backend=jax.default_backend(),observations=len(entries),
        checkpoint=policy.metadata)),flush=True)
    for number,entry in enumerate(entries,1):
        if (number-1)%args.worker_count!=args.worker_index:continue
        before=time.monotonic()
        sample=samples/f"energy_{number:04d}.npz"
        reused=False
        if sample.exists() and args.resume:
            with np.load(sample,allow_pickle=False) as archive:
                if str(archive["input_sha256"])!=entry["sha256"]:raise ValueError("Changed sample")
                value=archive["energy"]
            reused=True
        else:
            raw=read_observation(entry)
            transformed=policy._input_transform(raw)
            batched=jax.tree.map(lambda x:jnp.asarray(x)[None,...],transformed)
            obs=model_lib.preprocess_observation(None,model_lib.Observation.from_dict(batched),train=False)
            tokens,valid,_=embed(obs)
            positions=jnp.cumsum(valid,axis=1)-1
            if int(jnp.sum(valid))==0:raise ValueError("No valid prefix tokens")
            value=np.asarray(collect(params,tokens,valid,positions),dtype=np.float32)
            if value.shape!=(18,16384) or not np.isfinite(value).all() or np.any(value<0):raise ValueError("Invalid energy")
            temporary=sample.with_name(sample.stem+".tmp.npz")
            np.savez_compressed(temporary,energy=value,input_sha256=np.asarray(entry["sha256"]))
            temporary.replace(sample)
        group=(entry["suite"],int(entry["task_id"]),int(entry["trial"]))
        group_totals[group]=group_totals.get(group,0)+value.astype(np.float64)
        group_counts[group]=group_counts.get(group,0)+1
        progress=dict(processed=number,total=len(entries),last_sample_s=time.monotonic()-before,
            elapsed_s=time.time()-started,reused=reused,last_input=entry["path"],status="running")
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress),flush=True)
    if args.worker_count>1:
        progress_path.write_text(json.dumps(dict(status="worker_complete",worker=args.worker_index,
            worker_count=args.worker_count,elapsed_s=time.time()-started),indent=2)+'\n')
        policy.close()
        return
    tasks={}
    for key,values in group_totals.items():tasks.setdefault(key[:2],[]).append(values/group_counts[key])
    suites={}
    for key,values in tasks.items():suites.setdefault(key[0],[]).append(np.mean(values,axis=0))
    mean_energy=np.mean([np.mean(values,axis=0) for values in suites.values()],axis=0).astype(np.float32)
    factor=down_weight_factor(params["mlp"]["linear"])
    score=mean_energy*factor
    if not np.isfinite(score).all():raise ValueError("Nonfinite channel importance")
    order=np.argsort(-score,axis=1,kind="stable").astype(np.int32)
    rank=np.argsort(order,axis=1).astype(np.int32)+1
    np.savez_compressed(output,mean_energy=mean_energy,weight_factor=factor,scores=score,
        descending_indices=order,rank_1based=rank,
        keep_14336=select_channels(score,14336),keep_12288=select_channels(score,12288),
        source_manifest_sha256=np.asarray(source_hash),corpus_sha256=np.asarray(provenance["manifest_sha256"]),
        score_name=np.asarray(SCORE_NAME))
    csv_path=output.with_suffix(".csv")
    with csv_path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['layer_0based','channel_0based','rank_1based','mean_energy','weight_factor','importance'])
        for layer in range(18):
            for channel in range(16384):
                writer.writerow([layer,channel,int(rank[layer,channel]),float(mean_energy[layer,channel]),
                    float(factor[layer,channel]),float(score[layer,channel])])
    report=dict(observations=len(entries),episodes=len(group_totals),tasks=len(tasks),suites=list(suites),
        channels_scored=int(score.size),shape=list(score.shape),backend=jax.default_backend(),
        weighting="equal suite/task/episode; equal sampled frame; equal valid token",corpus=provenance,
        checkpoint=str(Path(args.checkpoint).resolve()),source_manifest_sha256=source_hash,
        score_name=SCORE_NAME,statistics_sha256=sha256(output),csv_sha256=sha256(csv_path),
        elapsed_s=time.time()-started,
        layer_summary=[dict(layer=l,min=float(score[l].min()),median=float(np.median(score[l])),
            max=float(score[l].max()),zero_channels=int(np.sum(score[l]==0)),top10_channels=order[l,:10].tolist(),
            retained_score_fraction_14336=float(np.sum(np.sort(score[l])[-14336:])/max(float(score[l].sum()),1e-30)),
            retained_score_fraction_12288=float(np.sum(np.sort(score[l])[-12288:])/max(float(score[l].sum()),1e-30)))
            for l in range(18)])
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+'\n')
    progress_path.write_text(json.dumps(dict(status="complete",processed=len(entries),total=len(entries),
        elapsed_s=report["elapsed_s"],output=str(output)),indent=2)+'\n')
    print(json.dumps(dict(event="complete",output=str(output),channels=score.size,elapsed_s=report["elapsed_s"])),flush=True)
    policy.close()


if __name__=="__main__":main()
