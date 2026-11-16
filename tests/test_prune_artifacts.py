import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import orbax.checkpoint as ocp

from openpi.models.model import restore_params
from openpi.pruning.artifacts import ARTIFACT_TYPE, CONFIG_NAME, REPO_ID, file_entry, verify_checkpoint
from openpi.pruning.channels import compact_params


class ArtifactTest(unittest.TestCase):
    def fixture(self,root):
        (root/'params').mkdir()
        (root/'params/_METADATA').write_text('fixture')
        norm=root/'assets/physical-intelligence/libero/norm_stats.json'
        norm.parent.mkdir(parents=True)
        norm.write_text('{}')
        np.savez(root/'channel_indices.npz',indices=np.tile(np.arange(64,dtype=np.int32),(18,1)))
        manifest=dict(format='Orbax JAX',artifact_type=ARTIFACT_TYPE,config=CONFIG_NAME,
            source={'repo_id':REPO_ID},source_width=16384,vlm_mlp_dim=64,action_context_horizon=9,
            files=[file_entry(root,p) for p in root.rglob('*') if p.is_file()])
        (root/'checkpoint_manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_derived_hash_verification_rejects_changed_or_unlisted_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root)
            self.assertEqual(verify_checkpoint(root)[2],64)
            (root/'params/extra').write_text('extra weight')
            with self.assertRaisesRegex(ValueError,'every parameter'):verify_checkpoint(root)
            (root/'params/extra').unlink()
            (root/'params/_METADATA').write_text('changed')
            with self.assertRaisesRegex(ValueError,'integrity'):verify_checkpoint(root)

    def test_width_and_retained_indices_must_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=self.fixture(root)
            manifest['vlm_mlp_dim']=128
            (root/'checkpoint_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'indices'):verify_checkpoint(root)

    def test_compact_orbax_roundtrip_uses_native_restore_format(self):
        gate=np.arange(2*2*4*8,dtype=np.float32).reshape(2,2,4,8)
        down=np.arange(2*8*4,dtype=np.float32).reshape(2,8,4)
        source={'PaliGemma':{'llm':{'layers':{'mlp':{'gating_einsum':gate,'linear':down}}}}}
        packed=compact_params(source,np.array([[0,2,5,7],[1,3,4,6]],np.int32))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'params'
            with ocp.PyTreeCheckpointer() as ckpt:
                ckpt.save(path,ocp.args.PyTreeSave({'params':packed}))
            restored=restore_params(path,restore_type=np.ndarray)
            np.testing.assert_array_equal(restored['PaliGemma']['llm']['layers']['mlp']['linear'],
                packed['PaliGemma']['llm']['layers']['mlp']['linear'])


if __name__=='__main__':unittest.main()
