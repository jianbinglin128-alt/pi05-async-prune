# π0.5 Future 静态通道剪枝

JAX π0.5 Future 异步动作上下文推理，离线校准 VLM FFN 通道重要性并导出物理紧凑权重。
保留完整视觉编码器、注意力、动作专家及 10 次去噪；在线没有评分、通道 gather 或动态剪枝。

本仓库包含运行源码、CPU 单元测试、安装配置、校准统计和四卡 LIBERO 启动器。
不含模型权重、数据集、虚拟环境或大型缓存。上游许可证见 LICENSE 和 LICENSE_GEMMA.txt。

## 1. 安装（Linux x86_64 / NVIDIA GPU）

需要 Python 3.11、Git、uv，以及支持所安装 CUDA/JAX 版本的 NVIDIA 驱动。

```bash
git clone --recurse-submodules https://github.com/jianbinglin128-alt/pi05-async-prune.git
cd pi05-async-prune
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv pip install --python .venv/bin/python -r requirements-libero.txt
uv pip install --python .venv/bin/python --no-deps -e third_party/libero
export LIBERO_CONFIG_PATH="$PWD/.libero"
.venv/bin/python scripts/configure_libero.py
```

私有仓库需先登录 `gh auth login` 或配置 Git 凭据。LIBERO 子模块应包含 BDDL、init_files 和 assets；若 Git LFS 指针尚未展开，按 LIBERO 上游说明下载对应资源。
无须下载训练图像数据集来运行这些评测。无显示器环境需要宿主机提供 EGL/OpenGL NVIDIA 驱动库。
本仓库 requirements-libero.txt 适配 Python 3.11；不要使用上游面向 Python 3.8 的旧版 requirements.txt 覆盖环境。

默认解释器为仓库 `.venv/bin/python`；已有环境可设置 `PRUNE_PYTHON=/absolute/path/to/python`。
已有 LIBERO 可设置 `LIBERO_SOURCE=/absolute/path/to/LIBERO`，并用同一路径运行 `scripts/configure_libero.py --source "$LIBERO_SOURCE"`。

## 2. 下载原始权重并导出相同 H8192

```bash
PYTHONPATH=src .venv/bin/python scripts/download_future_checkpoint.py
bash scripts/run_prune.sh export 0 \
  --checkpoint "$PWD/checkpoints/pi05_libero_async_future" \
  --statistics "$PWD/artifacts/channel_importance.npz" \
  --selection importance --keep 8192 \
  --output "$PWD/checkpoints/future_h8192"
```

源权重来自 [Dream-LIN/pi05_libero_async_future](https://huggingface.co/Dream-LIN/pi05_libero_async_future)。
随仓库提供的 channel_importance.npz 是已验证的离线校准统计，SHA256 为
`36ab9cc549f1e6e0274745fabcfe0ec08e12d7eeff6430b2c22b5404c32c6803`。
导出会核对源 checkpoint manifest；若上游文件改变导致 provenance 不匹配，应使用匹配的原始版本，不要绕过检查。
导出在 CPU 上进行，需要同时容纳 dense 和派生参数的主机内存，建议预留至少 32 GB 可用内存及 15 GB 磁盘。
首次 tokenizer 下载会写入 `~/.cache/openpi`，可设置 OPENPI_DATA_HOME。断网部署前先执行下载步骤并带走缓存。
完整 H8192 权重约 4.76 GB，未放入 Git；上述步骤使用相同的统计和切片方式重建相同保留通道。
若已有本次导出的完整 checkpoint，也可直接传输整个目录并跳过导出。

## 3. 检查与单套件评测

```bash
bash scripts/run_prune.sh test 0
PRUNE_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -c 'import jax; print(jax.devices())'
bash scripts/run_prune.sh server 0 --checkpoint "$PWD/checkpoints/future_h8192" \
  --port 18930 --execute-steps 5 --overlap-steps 2 --seed 7
```

另一个终端：

```bash
export LIBERO_CONFIG_PATH="$PWD/.libero"
bash scripts/run_prune.sh eval 0 --port 18930 --suite libero_spatial \
  --trials 50 --trial-start 0 --execute-steps 5 --overlap-steps 2 --seed 7 \
  --output "$PWD/prune_runs/spatial_50"
```

输出目录必须未使用。单套件 10 任务 × 50 初始状态 = 500 次。
服务器与评测器的 execute/overlap 必须相同。客户端心跳等待已设为 600 秒，允许首次 JAX 编译。

## 4. 四张卡并行跑完整 LIBERO

```bash
export LIBERO_CONFIG_PATH="$PWD/.libero"
bash scripts/run_libero_suites.sh "$PWD/checkpoints/future_h8192" \
  "$PWD/prune_runs/all4_h8192_50" 0,1,2,3 50
```

Spatial/Object/Goal/LIBERO-10 分别使用所传入的四张物理 GPU，合计 2,000 次。
GPU 参数请传物理编号，不要预先重新映射 CUDA_VISIBLE_DEVICES。默认端口 18930–18933，可设置 BASE_PORT。
启动器会等待所有套件结束并关闭自己的模型服务器。日志和退出码分别保存在各套件目录；需要长时间运行可在 tmux 中启动。
只有一张卡可在服务器启动后通过 `eval --suite all --trials 50` 顺序评测。

每套件输出 `episodes.jsonl`、`actions.jsonl`、`inferences.jsonl`、`timing_summary.json` 和运行 metadata。
```bash
bash scripts/run_prune.sh summary 0 --run "$PWD/prune_runs/all4_h8192_50/libero_spatial/eval"
```

## 调度与硬件迁移

E5O2：执行 3 个动作后，用当前真实观测和剩余 2 个排队动作发起推理，同时执行这 2 个动作。
新输出已对齐到上下文之后，直接执行新输出前 5 个动作，不再跳过前 2 步。
更少 overlap 需要目标机器的闭环验证；通道重要性可复用，实际耗时与遮盖窗口必须重新测量。
A6000 可先使用上述 x86_64 环境。DGX Spark 为 ARM64，当前 uv.lock 是现有环境的参考，尚未在 Spark 验证；
需要使用支持 ARM64 和该设备 GPU 的 JAX/CUDA、PyTorch wheel，不能宣称本仓库默认安装已适配 Spark。

## 已完成实验（2026-11-16，RTX 5090）

H8192 E5O2：Spatial 479/500（95.8%），Object 500/500（100%），Goal 492/500（98.4%），LIBERO-10 465/500（93%），合计 1936/2000（96.8%）。
同卡固定输入对照：dense 完整 policy 71.44 ms，H8192 61.50 ms；同步模型 57.52 → 45.90 ms。
不同设备不保证相同比例加速；尚无 2,000 次 dense 配对对照，校准包含部分相同任务及轨迹，不能认定剪枝无损。
E5O1 尚未验证。更多校准、导出和计时细节见 README_PRUNE.md；通用上游说明见 README.upstream.md。
