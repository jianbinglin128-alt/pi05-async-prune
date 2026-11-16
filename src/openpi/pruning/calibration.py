"""Calibration-only layer replay; production FFN and sampler have no hooks."""
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma, lora


def collect_prefix_energy(layer_params, tokens, valid_mask, positions, configs):
    """Return [L,H] mean squared post-gating energy and final unnormalized tokens.

    Replays the native Gemma block with a separate mutable statistics collection.
    Only reduced vectors leave each scan iteration; no full activation cache.
    """
    if any(c.lora_configs for c in configs):
        raise ValueError("Merge LoRA before calibration")
    if tokens.shape[0] != 1:
        raise ValueError("Calibration expects one observation at a time")
    attn_mask = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
    block = gemma.Block(configs=tuple(configs))

    def step(hidden, params):
        def intercept(next_fun, args, kwargs, context):
            result = next_fun(*args, **kwargs)
            module = context.module
            if isinstance(module, lora.FeedForward) and module.name == "mlp" and context.method_name == "__call__":
                x = args[0]
                # Use the exact native GELU and input dtype, then accumulate in FP32.
                z = nn.gelu(module._dot(x, module.w_gating[0], None)) * module._dot(x, module.w_gating[1], None)
                energy = jnp.sum(jnp.square(z.astype(jnp.float32)) * valid_mask[..., None], axis=(0, 1))
                energy /= jnp.maximum(jnp.sum(valid_mask), 1)
                module.sow("prune_stats", "energy", energy)
            return result
        with nn.intercept_methods(intercept):
            (outputs, _), stats = block.apply({"params": params}, [hidden, None], None,
                positions, attn_mask, [None, None], deterministic=True, mutable=["prune_stats"])
        return outputs[0], stats["prune_stats"]["mlp"]["energy"][0]

    final, energy = jax.lax.scan(step, tokens, layer_params)
    return energy, final
