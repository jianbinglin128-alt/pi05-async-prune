"""Measure warmed full-policy latency on a fixed corpus, with explicit flow noise."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy

from openpi.policies.pruned_future_policy import create_pruned_future_policy
from openpi.pruning.artifacts import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--corpus",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--samples",type=int,default=1000)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--overlap-steps",type=int,default=1)
    args=parser.parse_args()
    if args.samples<1 or args.warmup<1:
        parser.error("samples and warmup must be positive")
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    corpus_path=Path(args.corpus).resolve()
    entries=json.loads(corpus_path.read_text())["observations"]
    if not entries:
        raise ValueError("Empty corpus")
    observations=[]
    for entry in entries[:16]:
        path=(corpus_path.parent/entry["path"]).resolve()
        if not path.is_relative_to(corpus_path.parent) or sha256(path)!=entry["sha256"]:
            raise ValueError("Invalid corpus entry")
        obs=msgpack_numpy.unpackb(path.read_bytes())
        obs["action_context"]=np.zeros((args.overlap_steps,7),dtype=np.float32)
        observations.append(obs)
    policy=create_pruned_future_policy(args.checkpoint,overlap_steps=args.overlap_steps)
    noise=np.random.default_rng(17).standard_normal((10,32)).astype(np.float32)
    for i in range(args.warmup):
        policy.infer(observations[i%len(observations)],noise=noise)
    records=[]
    for i in range(args.samples):
        before=time.perf_counter_ns()
        response=policy.infer(observations[i%len(observations)],noise=noise)
        after=time.perf_counter_ns()
        records.append(dict(index=i,observation=i%len(observations),full_policy_ms=(after-before)/1e6,
            model_ms=response["policy_timing"]["infer_ms"]))
    (out/"samples.jsonl").write_text("".join(json.dumps(x)+'\n' for x in records))
    (out/"summary.json").write_text(json.dumps(dict(metadata=policy.metadata,args=vars(args),
        corpus_sha256=sha256(corpus_path),context="zero-valued valid context for shape timing only",
        latency={name:dict(zip(["p50","p95","p99"],map(float,np.percentile([x[name] for x in records],[50,95,99]))))
                 for name in ["full_policy_ms","model_ms"]}),indent=2)+'\n')
    policy.close()


if __name__=="__main__":
    main()
