"""Download the source Future checkpoint and populate the tokenizer cache."""
import argparse, os
from pathlib import Path
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', default='checkpoints/pi05_libero_async_future')
a=p.parse_args()
snapshot_download(repo_id='Dream-LIN/pi05_libero_async_future', local_dir=a.output)
from openpi.shared.download import maybe_download
print('Tokenizer:',maybe_download('gs://big_vision/paligemma_tokenizer.model',gs={'token':'anon'}))
from openpi.pruning.artifacts import verify_checkpoint
root,manifest,width=verify_checkpoint(a.output)
print('Verified:',root,'width:',width)
