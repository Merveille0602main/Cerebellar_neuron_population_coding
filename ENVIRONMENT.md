# 复现实验环境

在 Linux x86-64 或 WSL 中使用 Python 3.13。仓库内的 [BrainCell 源码](vendor/braincell/README.md)通过 pip 安装到项目环境，CPU 依赖按 [requirements-cpu.txt](requirements-cpu.txt) 固定。从仓库根目录执行。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -c 'import braincell; print(braincell.__version__, braincell.__file__)'
```

输出应显示 `0.1.0+reproduction.1` 和当前环境的 `site-packages/braincell`。前向仿真使用[子网命令](Simplified-Network/README.md#运行)。依赖由 pip 首次安装时下载，接线、细胞形态和 BrainCell 源码随仓库提供。

## 验证原生 Junction 与钙通道

```bash
python -m pip install pytest
JAX_PLATFORMS=cpu python -m unittest -v braincell.network.engine_test
JAX_PLATFORMS=cpu python -m pytest -q tests/test_bundled_braincell.py tests/test_paper_synapses.py
```

## 与 NEURON 对照

安装参考求解器，再用 `nrnivmodl` 编译仓库附带的 Cav2.3 参考机制。需要系统的 C/C++ 编译工具。对照默认使用 64 位精度。

```bash
python -m pip install '.[neuron-validation]'
comparison=vendor/braincell/examples/neuron_compare
(cd "$comparison/Cerebellum_mod/GoC/channel" && nrnivmodl)
JAX_PLATFORMS=cpu python "$comparison/channel_no_conc/engine/run.py" \
  "$comparison/channel_no_conc/configs/ma20_goc/cav2p3_ma20_goc.json" \
  --no-plot --out-dir Simplified-Network/runs/cav23-parity
```

本次用 NEURON 9.0.2 核对了 3 个交流、4 个直流及 3 个初始电位和温度用例。10 个用例均完成，最大膜电压差低于十亿分之一毫伏。

全柱形态回归测试可通过 `BRAINCELL_VALIDATION_ASSETS` 指向带有完整形态目录的 BrainCell 模型示例。常规颗粒层运行使用仓库内的资产。

早期原型的集成测试使用旧版 `Network.add_edges()` 和 `Network.add_projection()`，需要提供该旧接口的 BrainCell 环境。当前子网和[颗粒层运行入口](GRANULAR_LAYER_0_2.md)使用 `Network.connect()`。
