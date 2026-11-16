"""Serve original or statically compacted JAX Future checkpoints."""
import argparse
import json
import logging

import numpy as np
from openpi_client import base_policy

from openpi.policies.pruned_future_policy import DEFAULT_CHECKPOINT, create_pruned_future_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class ExplicitNoisePolicy(base_policy.BasePolicy):
    """Accept paired evaluation noise without modifying the production sampler."""
    def __init__(self, policy):
        self.policy = policy

    def infer(self, obs):
        obs = dict(obs)
        noise = obs.pop("_prune_flow_noise", None)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (10, 32) or not np.isfinite(noise).all():
                raise ValueError("Explicit flow noise must be finite [10,32]")
        return self.policy.infer(obs, noise=noise)

    def reset(self):
        self.policy.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18910)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--overlap-steps", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = create_pruned_future_policy(args.checkpoint, seed=args.seed,
        execute_steps=args.execute_steps, overlap_steps=args.overlap_steps)
    logging.info("PRUNED_FUTURE_READY %s", json.dumps(policy.metadata))
    try:
        WebsocketPolicyServer(ExplicitNoisePolicy(policy), host=args.host, port=args.port,
            metadata=dict(policy.metadata, explicit_flow_noise=True)).serve_forever()
    finally:
        policy.close()


if __name__ == "__main__":
    main()
