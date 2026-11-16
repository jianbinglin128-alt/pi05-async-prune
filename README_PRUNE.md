# Static FFN pruning for JAX Future

Portable installation and quick start: [README.md](README.md).

Only VLM FFN intermediate channels are removed. Vision, token counts, attention,
action expert, Future context projection and ten denoising steps stay intact.
Weights are packed offline. Inference has no scoring, gathers, EMA or refresh.

## Entrypoints

Run all commands from this directory. Choose an available GPU and unused port.
Original scripts are retained; use these explicit new entrypoints for pruning.

```bash
cd $PWD
bash scripts/run_prune.sh test 0
```

Original dense Future, target schedule 4+1:

```bash
bash scripts/run_prune.sh server 0 --port 18910 --overlap-steps 1
```

In another terminal:

```bash
bash scripts/run_prune.sh eval 0 --port 18910 --overlap-steps 1 \
  --suite libero_spatial --task-ids 1,3,5,7,9 --trial-start 3 --trials 3 \
  --output $PWD/prune_runs/dense_o1_pilot_spatial
```

Use `--overlap-steps 2` on **both** server and evaluator for the current 3+2
baseline. `--execute-steps` defaults to 5; E=10/O=2 or 3 is also supported.
The client validates server metadata. New outputs start at action 0: never
discard O new actions. Context uses nine fixed slots and a validity mask.

## Record a calibration corpus

Use the original dense server. A 3+2 server is used in this example. Start it on
the selected port, then collect four suites, five tasks/suite, two initial states:

```bash
bash scripts/run_prune.sh eval 0 --port 18910 --overlap-steps 2 \
  --suite all --task-ids 0,2,4,6,8 --trial-start 0 --trials 2 \
  --output $PWD/prune_runs/calibration_dense \
  --calibration-dir $PWD/artifacts/corpus_v1 \
  --snapshots-per-episode 4
```

Corpora contain raw image/state/prompt observations and provenance. Up to four
frames are sampled evenly from each observed trajectory's candidate frames,
including failed trajectories. No evaluation outcome is used to rank channels.
Corpus collection packs candidate snapshots inside the rollout and therefore
is **not a production latency benchmark**. The source manifest hash is recorded.
Do not use these same trajectories for candidate quality selection.

Stop the evaluation server normally before GPU calibration to avoid competition.

```bash
bash scripts/run_prune.sh calibrate 0 \
  --corpus $PWD/artifacts/corpus_v1/corpus.json \
  --output $PWD/artifacts/energy_v1.npz
```

Calibration replays the native Gemma prefix blocks in a separate path, collecting
FP32 mean squared post-GELU-gate-times-up energy over valid tokens. It balances
sampled frames within episodes, then episodes, tasks and suites. No calibration
hooks are installed in the inference model.

Existing frozen dense observations can be reused without copying or repacking
images. Their original rollout policy is recorded separately from the Future
checkpoint used for scoring. The legacy NPZ loader verifies task/episode IDs and
hashes raw inputs; saved actions and old activations are not used.

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 bash scripts/run_prune.sh calibrate-cpu 0 \
  --legacy-manifest /path/to/legacy/calibration_manifest.json \
  --legacy-root /path/to/legacy \
  --output $PWD/artifacts/NEW_RUN/channel_importance.npz
```

`--resume` reuses per-observation energy checkpoints after verifying frozen input
bindings. Disjoint process shards may specify `--worker-count N --worker-index I`;
after all shards succeed, invoke once with `--resume` and no sharding to aggregate.
Do not run two processes for the same shard. Use the same backend and collector
version when resuming. Output contains energy, EcoVLA weight factor, importance,
descending channel IDs, one-based ranks and sorted keep indices for H14336/H12288.
The CSV has one row for each layer/channel pair. Score values are comparable for
within-layer selection; retained score fraction is not a policy-quality metric.

## Export compact weights

```bash
bash scripts/run_prune.sh export 0 \
  --checkpoint $PWD/checkpoints/pi05_libero_async_future \
  --statistics $PWD/artifacts/energy_v1.npz \
  --keep 14336 --selection importance \
  --output $PWD/artifacts/future_h14336
```

Repeat with a fresh output and `--keep 12288` only if needed. For a packing/shape
baseline, `--selection prefix` needs no statistics; use `--keep 16384` for the
identity export. Such prefix exports do not validate importance-based quality.
Export runs on CPU, preserves original parameter dtypes, copies normalization
assets, saves per-layer retained indices, and hashes every artifact. It rejects
LoRA MLP leaves; merge adapters before using this implementation.

The score is `energy[c] * sqrt(sum_d down[c,d]**4)`, an offline adaptation of
EcoVLA. The highest-scoring channels are retained, sorted in original channel
order, and the gate/up/down projections are packed together. Derived artifacts
carry a distinct type and source-manifest hash. Loading uses strict parameter
keys and shapes before construction of the monolithic JIT sampler.

Serve compact weights with the same server command plus:

```bash
--checkpoint $PWD/artifacts/future_h14336
```

## Measure and evaluate

Use an isolated GPU, stop the server, and benchmark each checkpoint separately:

```bash
bash scripts/run_prune.sh benchmark 0 \
  --checkpoint $PWD/artifacts/future_h14336 \
  --corpus $PWD/artifacts/corpus_v1/corpus.json \
  --overlap-steps 1 --warmup 20 --samples 1000 \
  --output $PWD/prune_runs/latency_h14336_o1
```

This benchmark cycles up to 16 observations and uses explicit fixed flow noise.
Its valid zero-valued context is for shape timing only. It reports model and
full-policy latency; it does not estimate closed-loop quality or WebSocket cost.

For closed-loop tests, use `eval` with no `--calibration-dir`. Each episode has
an explicit noise stream keyed by seed/suite/task/initial-state, independent of
previous episode length. Warmup uses the configured context shape. New run and
corpus directories must not exist, preventing accidental result overwrites.

```bash
bash scripts/run_prune.sh summary 0 --run $PWD/prune_runs/dense_o1_pilot_spatial
```

`inferences.jsonl` records observation/submit/worker-start/ready/boundary times.
The boundary is the final queued `env.step` return, before subsequent image
preprocessing. `uncovered_ms = max(ready - boundary, 0)`; `join_wait_ms` measures
actual blocking on the future, which may be smaller if client work intervenes.
The first new action is timestamped immediately before `env.step`, allowing an
observation-age measurement. Bootstrap and unused terminal prefetches are excluded
from steady-state summaries, including prefetches that completed before episode
termination. Disk writes and pending-request drain occur after rollout timing.

The observation clock is simulator `env.step` return, not camera exposure time.
LIBERO is not wall-clock paced at 20 Hz. Runtime logs are lightweight but are
still instrumentation; use a matched implementation for comparisons. No quality
equivalence or speedup is implied by successful loading or CPU tests.

Freeze a retained width after pilot tests, then compare dense/O2, compact/O2,
dense/O1 and compact/O1 with fixed E=5. Select the largest width meeting the
target deadline and quality criteria. Keep final confirmation data separate
from calibration and candidate selection; disclose any overlapping tasks.
