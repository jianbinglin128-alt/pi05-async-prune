import tempfile
from pathlib import Path
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models.pi0_config import Pi0Config
from openpi.pruning.calibration import collect_prefix_energy
from openpi.pruning.channels import channel_scores, compact_params, select_channels


class ChannelsTest(unittest.TestCase):
    def test_future_input_spec_has_fixed_masked_context_and_validated_width(self):
        obs,actions=Pi0Config(pi05=True,action_horizon=10,action_context_horizon=9,vlm_mlp_dim=12288).inputs_spec()
        self.assertEqual(obs.action_context.shape,(1,9,32))
        self.assertEqual(obs.action_context_mask.shape,(1,9))
        self.assertEqual(actions.shape,(1,10,32))
        for width in [0,123,16448]:
            with self.assertRaises(ValueError):Pi0Config(vlm_mlp_dim=width)

    def test_score_matches_ecovla_expression_and_ties_are_stable(self):
        down=np.array([[[1.,2.],[3.,4.],[0.,0.]]],np.float32)
        energy=np.array([[2.,1.,0.]],np.float32)
        expected=np.linalg.norm(energy[...,None]*down**2,axis=-1)
        np.testing.assert_allclose(channel_scores(energy,down),expected)
        np.testing.assert_array_equal(select_channels(np.ones((2,8)),3),[[0,1,2],[0,1,2]])
        np.testing.assert_array_equal(select_channels(np.arange(8)[None,:],8),[np.arange(8)])

    def test_physical_pack_matches_masked_native_gelu_and_keeps_other_parameters(self):
        rng=np.random.default_rng(1)
        gate=rng.standard_normal((2,2,4,8)).astype(np.float32)
        down=rng.standard_normal((2,8,4)).astype(np.float32)
        params={"PaliGemma":{"llm":{"layers":{"mlp":{"gating_einsum":gate,"linear":down},
            "mlp_1":{"sentinel":np.ones(3)}}}},"action_context_in_proj":{"kernel":np.ones(4)}}
        indices=np.array([[0,3,5,7],[1,2,4,6]],np.int32)
        out=compact_params(params,indices)
        self.assertIs(out["PaliGemma"]["llm"]["layers"]["mlp_1"]["sentinel"],params["PaliGemma"]["llm"]["layers"]["mlp_1"]["sentinel"])
        x=rng.standard_normal((1,3,4)).astype(np.float32)
        for layer in range(2):
            z=nn.gelu(jnp.asarray(x@gate[layer,0]))*jnp.asarray(x@gate[layer,1])
            mask=np.zeros(8,np.float32);mask[indices[layer]]=1
            reference=(z*mask)@down[layer]
            narrow=out["PaliGemma"]["llm"]["layers"]["mlp"]
            result=(nn.gelu(jnp.asarray(x@narrow["gating_einsum"][layer,0]))*
                jnp.asarray(x@narrow["gating_einsum"][layer,1]))@narrow["linear"][layer]
            np.testing.assert_allclose(result,reference,atol=1e-5,rtol=1e-5)
        self.assertEqual(gate.shape[-1],8)
        with self.assertRaises(ValueError):compact_params(params,np.array([[1,1],[2,3]]))

    def test_calibration_replay_matches_native_prefix_and_ignores_padding(self):
        cfg=gemma.Config(width=16,depth=2,mlp_dim=64,num_heads=2,num_kv_heads=1,head_dim=8)
        module=gemma.Module(configs=[cfg,cfg],embed_dtype="float32")
        tokens=jax.random.normal(jax.random.key(1),(1,4,16))
        valid=jnp.array([[True,True,False,False]])
        positions=jnp.cumsum(valid,axis=1)-1
        mask=valid[:,:,None]&valid[:,None,:]
        _,variables=module.init_with_output(jax.random.key(2),[tokens,None],positions,mask)
        native,_=module.apply(variables,[tokens,None],positions,mask)
        energy,hidden=collect_prefix_energy(variables["params"]["layers"],tokens,valid,positions,[cfg,cfg])
        final,_=gemma.RMSNorm().apply({"params":variables["params"]["final_norm"]},hidden,None)
        np.testing.assert_allclose(final,native[0],atol=1e-5,rtol=1e-5)
        modified=tokens.at[:,2:].set(1000.)
        energy2,_=collect_prefix_energy(variables["params"]["layers"],modified,valid,positions,[cfg,cfg])
        np.testing.assert_allclose(energy,energy2,atol=1e-5,rtol=1e-5)
        self.assertEqual(energy.shape,(2,64))
        self.assertTrue(np.isfinite(energy).all())

    def test_bfloat16_calibration_replay_preserves_native_prefix(self):
        cfg=gemma.Config(width=16,depth=2,mlp_dim=64,num_heads=2,num_kv_heads=1,head_dim=8)
        module=gemma.Module(configs=[cfg,cfg],embed_dtype="bfloat16")
        tokens=jax.random.normal(jax.random.key(10),(1,4,16)).astype(jnp.bfloat16)
        valid=jnp.array([[True,True,False,False]])
        positions=jnp.cumsum(valid,axis=1)-1
        mask=valid[:,:,None]&valid[:,None,:]
        _,variables=module.init_with_output(jax.random.key(11),[tokens,None],positions,mask)
        native,_=module.apply(variables,[tokens,None],positions,mask)
        energy,hidden=collect_prefix_energy(variables["params"]["layers"],tokens,valid,positions,[cfg,cfg])
        final,_=gemma.RMSNorm().apply({"params":variables["params"]["final_norm"]},hidden,None)
        np.testing.assert_allclose(np.asarray(final,dtype=np.float32),np.asarray(native[0],dtype=np.float32),
            atol=0.02,rtol=0.02)
        self.assertEqual(energy.dtype,jnp.float32)
        self.assertTrue(np.isfinite(energy).all())


if __name__=="__main__":unittest.main()
