"""Write an isolated LIBERO path configuration without interactive prompts."""
import argparse, os
from pathlib import Path
import yaml
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', default=os.environ.get('LIBERO_SOURCE', 'third_party/libero'))
p.add_argument('--config-dir', default=os.environ.get('LIBERO_CONFIG_PATH', '.libero'))
a=p.parse_args();source=Path(a.source).expanduser().resolve();root=source/'libero/libero'
paths=dict(benchmark_root=root,bddl_files=root/'bddl_files',init_states=root/'init_files',assets=root/'assets',datasets=source/'datasets')
for key in ['bddl_files','init_states','assets']:
    if not paths[key].is_dir():raise FileNotFoundError(paths[key])
out=Path(a.config_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
dest=out/'config.yaml'
payload={k:str(v) for k,v in paths.items()}
if dest.exists() and yaml.safe_load(dest.read_text())!=payload:raise FileExistsError('Existing different configuration: '+str(dest))
dest.write_text(yaml.safe_dump(payload));print('export LIBERO_CONFIG_PATH='+str(out))
