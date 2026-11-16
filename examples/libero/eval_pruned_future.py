"""Paired LIBERO Future evaluation, configurable overlap, optional dense corpus."""
import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy, websocket_client_policy
from openpi_client.future_action_chunk_broker import FutureActionChunkBroker, validate_schedule

from openpi.pruning.artifacts import sha256
from scripts.summarize_pruned_future import summarize


class PairedClient:
    def __init__(self, client, seed):
        self.client=client
        self.rng=np.random.default_rng(seed)

    def infer(self, obs):
        return self.client.infer(dict(obs, _prune_flow_noise=self.rng.standard_normal((10,32)).astype(np.float32)))

    def reset(self):
        self.client.reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=18910)
    parser.add_argument("--suite",choices=["all","libero_spatial","libero_object","libero_goal","libero_10"],default="libero_spatial")
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--trial-start",type=int,default=0)
    parser.add_argument("--task-ids",default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--execute-steps",type=int,default=5)
    parser.add_argument("--overlap-steps",type=int,default=2)
    parser.add_argument("--output",required=True)
    parser.add_argument("--calibration-dir",help="Optional NEW corpus directory; dense checkpoint only")
    parser.add_argument("--snapshots-per-episode",type=int,default=4)
    args=parser.parse_args()
    validate_schedule(args.execute_steps,args.overlap_steps)
    task_ids=[int(x) for x in args.task_ids.split(',')]
    if (args.trials<1 or args.trial_start<0 or len(set(task_ids))!=len(task_ids)
            or not task_ids or any(x<0 or x>9 for x in task_ids) or args.snapshots_per_episode<1):
        parser.error("Invalid trials, task IDs or snapshot count")
    # Import simulator only after CLI validation; --help works without EGL initialization.
    from libero.libero import benchmark
    from examples.libero.prune_env import MAX_STEPS, LIBERO_DUMMY_ACTION, _get_libero_env, _observation
    logging.basicConfig(level=logging.INFO,force=True)
    np.random.seed(args.seed)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    client=websocket_client_policy.WebsocketClientPolicy(args.host,args.port)
    metadata=client.get_server_metadata()
    expected=dict(config="pi05_libero_async_future",backend="jax",action_target_alignment="future",
        predicted_steps=10,executed_steps=args.execute_steps,overlap_steps=args.overlap_steps,
        pre_inference_steps=args.execute_steps-args.overlap_steps,explicit_flow_noise=True)
    if any(metadata.get(k)!=v for k,v in expected.items()):
        client._ws.close()
        raise ValueError(f"Server/schedule mismatch: {metadata}")
    corpus_root=Path(args.calibration_dir).resolve() if args.calibration_dir else None
    corpus=None
    if corpus_root:
        if metadata["artifact_type"]!="original-future":
            raise ValueError("Record calibration with the original dense checkpoint")
        corpus_root.mkdir(parents=True,exist_ok=False)
        source=Path(metadata["checkpoint"])/"checkpoint_manifest.json"
        corpus=dict(checkpoint_manifest_sha256=sha256(source),metadata=metadata,observations=[])
    (output/"run.json").write_text(json.dumps(dict(args=vars(args),server_metadata=metadata,
        timing=dict(observation="env.step return; bootstrap after settling",
            boundary="env.step return of final queued action; excludes following preprocessing",
            latency="submit to client response; includes worker queue and network",
            excluded="reset, settling, warmup, terminal prefetch drain, disk writes")),indent=2)+'\n')
    suite_names=list(MAX_STEPS) if args.suite=="all" else [args.suite]
    warmed=False
    try:
        for suite_name in suite_names:
            suite=benchmark.get_benchmark_dict()[suite_name]()
            suite_index=list(MAX_STEPS).index(suite_name)
            for task_id in task_ids:
                task=suite.get_task(task_id)
                states=suite.get_task_init_states(task_id)
                if args.trial_start+args.trials>len(states):
                    raise ValueError("Requested initial states exceed benchmark")
                env,description=_get_libero_env(task,256,args.seed)
                try:
                    for trial in range(args.trial_start,args.trial_start+args.trials):
                        env.reset()
                        obs=env.set_init_state(states[trial])
                        for _ in range(10):
                            obs,_,_,_=env.step(LIBERO_DUMMY_ACTION)
                        obs_ns=time.perf_counter_ns()
                        client.reset()
                        if not warmed:
                            warm=dict(_observation(obs,description),action_context=np.zeros((args.overlap_steps,7),np.float32))
                            for _ in range(3): client.infer(warm)
                            client.reset()
                            warmed=True
                        paired=PairedClient(client,np.random.SeedSequence([args.seed,suite_index,task_id,trial]))
                        broker=FutureActionChunkBroker(paired,execute_steps=args.execute_steps,overlap_steps=args.overlap_steps)
                        actions=[]
                        snapshots=[]
                        done=False
                        error=None
                        start=time.perf_counter_ns()
                        try:
                            for step in range(MAX_STEPS[suite_name]):
                                prepared=_observation(obs,description)
                                if corpus_root and step%args.execute_steps==0:
                                    snapshots.append((step,msgpack_numpy.packb(dict(prepared,
                                        action_context=np.zeros((0,7),np.float32)))))
                                action=broker.infer(prepared,observation_ns=obs_ns)["actions"].tolist()
                                before=time.perf_counter_ns()
                                broker.mark_action_started(before)
                                obs,_,done,_=env.step(action)
                                after=time.perf_counter_ns()
                                obs_ns=after
                                actions.append(dict(step=step,start_ns=before,end_ns=after,duration_ns=after-before))
                                if done: break
                        except Exception as exception:
                            error=repr(exception)
                            raise
                        finally:
                            terminal=time.perf_counter_ns()
                            if actions and error is None:
                                terminal=actions[-1]["end_ns"]
                            try:
                                broker.reset()
                            finally:
                                broker.close()
                            entry=dict(suite=suite_name,task_id=task_id,trial=trial,task=description,
                                success=bool(done and error is None),error=error,steps=len(actions),
                                rollout_wall_s=(terminal-start)/1e9,switch_wait_ms=broker.switch_wait_ms,
                                start_ns=start,end_ns=terminal,overlap_steps=args.overlap_steps,
                                execute_steps=args.execute_steps)
                            for name,rows in [("episodes.jsonl",[entry]),("actions.jsonl",actions),
                                              ("inferences.jsonl",broker.inference_records)]:
                                with (output/name).open('a') as file:
                                    for row in rows:
                                        file.write(json.dumps(dict(row,suite=suite_name,task_id=task_id,trial=trial))+'\n')
                            if corpus_root and snapshots:
                                chosen=np.unique(np.linspace(0,len(snapshots)-1,
                                    min(args.snapshots_per_episode,len(snapshots)),dtype=int))
                                for index in chosen:
                                    step,payload=snapshots[index]
                                    path=corpus_root/f"{suite_name}_task{task_id}_trial{trial}_step{step}.msgpack"
                                    path.write_bytes(payload)
                                    corpus["observations"].append(dict(path=path.name,sha256=sha256(path),
                                        suite=suite_name,task_id=task_id,trial=trial,step=step,success=entry["success"]))
                                (corpus_root/"corpus.json").write_text(json.dumps(corpus,indent=2)+'\n')
                            logging.info("EPISODE %s",json.dumps(entry))
                finally:
                    env.close()
    finally:
        client._ws.close()
        if (output/"episodes.jsonl").exists():
            (output/"timing_summary.json").write_text(json.dumps(summarize(output),indent=2)+'\n')


if __name__=="__main__":
    main()
