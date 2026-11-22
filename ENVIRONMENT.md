# 复现实验环境

本文档只记录本机环境事实和推荐运行方式，不要求修改共享 BrainCell 仓库。

## 当前发现的三个 Python 环境

| 环境 | Python | BrainCell | NEURON | 当前缺项 |
|---|---:|---|---:|---|
| 复现目录 `.venv` | 3.13.11 | 从 `/home/merveille/copneuron/braincell` 源码导入 | 9.0.1 | 当前集成测试可运行 |
| Conda `braincell` | 3.11.15 | 未安装为 package，但可通过 `PYTHONPATH` 从源码导入 | 8.2.6 | 缺 `h5py`，不能导入本复现包 |
| Conda `brain_cell` | 3.11.15 | 已安装 0.0.8 | 未安装 | 缺 `h5py`、NEURON 和 pytest |

版本是在 2026-08-08 用实际 import 得到的。两个名字相近的 Conda 环境不是同一个环境，不应混用其结论。

## 当前推荐命令

日常复现和 BrainCell 集成测试使用隔离环境，并明确让 Python 加载共享仓库中的当前源码：

```bash
cd /home/merveille/copneuron/braincell-cerebellar-reproduction
PYTHONPATH=src:/home/merveille/copneuron/braincell .venv/bin/pytest -q
```

NEURON 参考求解也使用该环境。`pyproject.toml` 的 `neuron-validation` 可选依赖固定为 9.0.1，避免参考 trace 在无人知情时换版本：

```bash
.venv/bin/pip install -e '.[neuron-validation]'
.venv/bin/python scripts/build_neuron_goc_mechanisms.py
```

## 关于已有 Conda `braincell` 环境

它确实同时具备 NEURON 8.2.6 和 BrainCell 所需的 JAX/BrainState/BrainUnit，并可以通过以下方式加载 BrainCell 当前源码：

```bash
PYTHONPATH=/home/merveille/copneuron/braincell \
  /home/merveille/miniconda3/envs/braincell/bin/python -c \
  'import braincell, neuron; print(braincell.__file__, neuron.__version__)'
```

但它当前没有 `h5py`，所以还不能直接执行本复现仓库。为避免在维护者未确认时修改共享环境，本阶段没有向该 Conda 环境安装包。若以后补齐依赖，应把 NEURON 8.2.6 与 9.0.1 生成的参考 trace 分开保存并比较，不能覆盖当前金标准后默认二者完全一致。
