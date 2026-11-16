#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gpu=$1; port=$2; suite=$3; trials=$4; checkpoint=$5; output=$6
mkdir "$output"
bash "$repo_dir/scripts/run_prune.sh" server "$gpu" --checkpoint "$checkpoint" --port "$port" --execute-steps 5 --overlap-steps 2 --seed 7 > "$output/server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
python_bin=${PRUNE_PYTHON:-$repo_dir/.venv/bin/python}
"$python_bin" - "$port" "$server_pid" <<'PY'
import os,socket,sys,time
port,pid=map(int,sys.argv[1:]);deadline=time.monotonic()+1200
while time.monotonic()<deadline:
    os.kill(pid,0)
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=1):break
    except OSError:time.sleep(2)
else:raise TimeoutError('Server startup exceeded 1200 seconds')
PY
rc=0
bash "$repo_dir/scripts/run_prune.sh" eval "$gpu" --port "$port" --suite "$suite" --trials "$trials" --trial-start 0 --execute-steps 5 --overlap-steps 2 --seed 7 --output "$output/eval" > "$output/eval.log" 2>&1 || rc=$?
echo "$rc" > "$output/exit_code.txt"
exit "$rc"
