import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openpi.pruning.data import load_corpus,read_observation


class CalibrationDataTest(unittest.TestCase):
    def test_legacy_raw_inputs_keep_source_separate_and_detect_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'observation.npz'
            np.savez(path,observation_image=np.zeros((2,2,3),np.uint8),
                observation_wrist_image=np.zeros((2,2,3),np.uint8),observation_state=np.arange(8),
                prompt=np.asarray('pick up the cup'),task_suite=np.asarray('libero_object'),
                task_id=np.asarray(2),episode_index=np.asarray(1))
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps([dict(path=path.name,suite='libero_object',task_id=2,episode_index=1)]))
            entries,provenance=load_corpus(legacy_manifest=manifest,legacy_root=root)
            raw=read_observation(entries[0])
            self.assertEqual(raw['prompt'],'pick up the cup')
            self.assertEqual(raw['action_context'].shape,(0,7))
            self.assertIn('raw inputs only',provenance['observation_policy'])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):read_observation(entries[0])

    def test_duplicate_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'x.npz').write_bytes(b'fixture')
            item=dict(path='x.npz',suite='libero_object',task_id=2,episode_index=1)
            (root/'manifest.json').write_text(json.dumps([item,item]))
            with self.assertRaisesRegex(ValueError,'duplicate'):
                load_corpus(legacy_manifest=root/'manifest.json',legacy_root=root)


if __name__=='__main__':unittest.main()
