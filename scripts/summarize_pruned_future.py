"""Summarize consumed steady-state requests using paired latency/window records."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    def read(name):
        path=root/name
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    episodes=read("episodes.jsonl")
    requests=read("inferences.jsonl")
    steady=[r for r in requests if not r["bootstrap"] and not r["terminal_prefetch"]
            and r.get("boundary_ns") is not None]
    def stats(name):
        values=[r[name] for r in steady if r.get(name) is not None]
        return dict(n=len(values),mean=float(np.mean(values)),**dict(zip(
            ["p50","p95","p99"],map(float,np.percentile(values,[50,95,99]))))) if values else None
    result=dict(episodes=len(episodes),successes=sum(e["success"] for e in episodes),
        error_episodes=sum(e["error"] is not None for e in episodes),steady_requests=len(steady),
        missed_deadline_rate=float(np.mean([r["missed_deadline"] for r in steady])) if steady else None,
        timing={name:stats(name) for name in ["submit_to_ready_ms","hide_window_ms","uncovered_ms",
            "join_wait_ms","boundary_to_join_ms","observation_age_ms"]},
        mean_rollout_wall_s=float(np.mean([e["rollout_wall_s"] for e in episodes])) if episodes else None)
    for suite in sorted({e["suite"] for e in episodes}):
        values=[e for e in episodes if e["suite"]==suite]
        result.setdefault("suites",{})[suite]=dict(episodes=len(values),successes=sum(e["success"] for e in values))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    root=Path(args.run).resolve()
    result=summarize(root)
    (root/"timing_summary.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
