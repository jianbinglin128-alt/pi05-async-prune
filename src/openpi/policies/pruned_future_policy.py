"""Strict JAX Future loader; compact parameters are loaded before sampler JIT."""
import dataclasses
import os

import jax

from openpi.policies import policy_config
from openpi.pruning.artifacts import CONFIG_NAME, verify_checkpoint
from openpi.training import config
from openpi_client.future_action_chunk_broker import validate_schedule

DEFAULT_CHECKPOINT = os.environ.get("PRUNE_CHECKPOINT", "checkpoints/pi05_libero_async_future")


def create_pruned_future_policy(checkpoint=DEFAULT_CHECKPOINT, *, seed=7, execute_steps=5, overlap_steps=2):
    validate_schedule(execute_steps, overlap_steps)
    checkpoint, manifest, width = verify_checkpoint(checkpoint)
    cfg = config.get_config(CONFIG_NAME)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, vlm_mlp_dim=width))
    policy = policy_config.create_trained_policy(cfg, checkpoint,
        sample_kwargs={"num_steps": 10}, strict_params=True)
    policy._rng = jax.random.key(seed)
    policy._metadata = dict(policy.metadata, config=CONFIG_NAME, checkpoint=str(checkpoint),
        checkpoint_step=manifest.get("checkpoint_step"), vlm_mlp_dim=width,
        artifact_type=manifest.get("artifact_type", "original-future"),
        backend="jax", sampler="monolithic_jit", action_target_alignment="future",
        predicted_steps=10, executed_steps=execute_steps, overlap_steps=overlap_steps,
        pre_inference_steps=execute_steps-overlap_steps, output_skip_steps=0,
        num_denoise_steps=10, policy_seed=seed, online_pruning=False)
    return policy
