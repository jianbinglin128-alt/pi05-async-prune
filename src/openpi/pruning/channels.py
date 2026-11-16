"""Selection and physical packing. This module does not participate in inference."""
import numpy as np

SCORE_NAME = "mean_valid_token_gate_up_energy_times_l2_squared_down_weights_v1"


def mlp_weights(params):
    mlp = params["PaliGemma"]["llm"]["layers"]["mlp"]
    if set(mlp) != {"gating_einsum", "linear"}:
        raise ValueError("Expected plain FFN weights; merge LoRA before pruning")
    gate, down = mlp["gating_einsum"], mlp["linear"]
    if gate.ndim != 4 or down.ndim != 3:
        raise ValueError("Expected scanned gate [L,2,D,H] and down [L,H,D]")
    if not (gate.shape[1] == 2 and gate.shape[0] == down.shape[0]
            and gate.shape[2] == down.shape[2] and gate.shape[3] == down.shape[1]):
        raise ValueError("Mismatched gate/up/down axes")
    return gate, down


def channel_scores(mean_energy, down):
    energy = np.asarray(mean_energy, dtype=np.float32)
    if energy.shape != down.shape[:2] or np.any(energy < 0) or not np.isfinite(energy).all():
        raise ValueError("Energy must be finite, nonnegative and shaped [layers,channels]")
    factor = down_weight_factor(down)
    score = energy * factor
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite channel score")
    return score


def down_weight_factor(down):
    """Compute EcoVLA's weight factor one layer at a time to limit host memory."""
    factors=[]
    for layer in down:
        weight=np.asarray(layer,dtype=np.float32)
        factors.append(np.sqrt(np.sum(np.square(np.square(weight)),axis=-1,dtype=np.float32)))
    result=np.stack(factors)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite down projection weight factor")
    return result


def select_channels(scores, keep):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all() or not 0 < keep <= scores.shape[1]:
        raise ValueError("Invalid scores or retained width")
    # Stable tie-breaking, then restore original order. Full width is identity.
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :keep]
    return np.sort(ranked, axis=1).astype(np.int32)


def compact_params(params, indices):
    gate, down = mlp_weights(params)
    indices = np.asarray(indices)
    layers, _, _, width = gate.shape
    if indices.ndim != 2 or indices.shape[0] != layers or not 0 < indices.shape[1] <= width:
        raise ValueError("Invalid retained indices shape")
    if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= width):
        raise ValueError("Invalid retained indices")
    if np.any(np.diff(indices, axis=1) <= 0):
        raise ValueError("Indices must be unique and sorted in original order")
    # Shallow-copy containers, retain unchanged arrays without a second full-model copy.
    def containers(tree):
        return {k: containers(v) for k, v in tree.items()} if isinstance(tree, dict) else tree
    out = containers(params)
    target = out["PaliGemma"]["llm"]["layers"]["mlp"]
    target["gating_einsum"] = np.stack([np.take(np.asarray(gate[l]), indices[l], axis=-1)
                                          for l in range(layers)])
    target["linear"] = np.stack([np.take(np.asarray(down[l]), indices[l], axis=0)
                                 for l in range(layers)])
    return out
