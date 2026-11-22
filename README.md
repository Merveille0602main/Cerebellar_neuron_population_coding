# 小脑颗粒层网络的 BrainCell 复现

本项目把 De Schepper 等人在 2022 年发表的小脑皮层柱连接组导入 BrainCell 0.2.0。当前仿真范围是颗粒层回路，包括苔藓纤维、小球中继、全房室颗粒细胞和全房室 Golgi 细胞。颗粒细胞的轴突包含升支和分叉后的两臂平行纤维，均参加电缆求解。

- [小脑皮层柱结构图](docs/network-structure.html)。下载 HTML 后用浏览器打开，可折叠查看原论文全网络的层次、细胞数量和密度、连接度、形态及通道。该图保留了论文全柱范围，包括浦肯野细胞层和分子层。当前 BrainCell 仿真范围见下表。
- [运行与验证](GRANULAR_LAYER_0_2.md)
- [与原始 NEURON 网络的差异](docs/neuron-alignment.md)

## 当前回路

| 通路 | 发布连接记录 | BrainCell 机制 |
| --- | ---: | --- |
| 苔藓纤维 → 小球 | 2,336 | 将父纤维的预定发放分发给各小球 |
| 小球 → 颗粒细胞 | 114,460 | AMPA、NMDA |
| 小球 → Golgi | 3,939 | AMPA、NMDA |
| 颗粒细胞升支 → Golgi 基底树突 | 22,431 | Masoli AMPA 替代、DBBS NMDA |
| 颗粒细胞平行纤维 → Golgi 顶端树突 | 63,552 | AMPA |
| Golgi → 颗粒细胞 | 96,887 | GABA-A α1 和 α6 |
| Golgi 轴突 → Golgi 基底树突 | 181,476 | GABA-A α1，参数见对齐说明 |
| Golgi 基底树突之间 | 2,082 | 保留缝隙连接的解剖端点，电耦合求解待实现 |

全量颗粒层包含 117 条苔藓纤维、2,336 个小球、28,615 个颗粒细胞及 70 个 Golgi 细胞。默认逐接触放置突触，每个化学接触独立保存释放和受体状态。连接数按解剖记录计，一个接触可以挂多种受体。181,476 条 Golgi 化学接触中含发布数据的 5,441 条自接触。

## 快速运行

从仓库根目录运行。测试环境使用 Python 3.13 和 CPU 版 JAX。安装包提供 BrainCell 运行时，`assets/braincell/` 提供所需的细胞模板与形态数据。

```bash
python -m venv .venv-braincell-0.2.0
source .venv-braincell-0.2.0/bin/activate
python -m pip install -e '.[braincell-runtime]'
python scripts/fetch_reference.py data/reference/balanced.hdf5
cerebellar-export export data/reference/balanced.hdf5 data/export/balanced --verify-published
JAX_PLATFORMS=cpu python scripts/run_granular_layer_chemical_subnetwork.py --exercise-circuit
```

发布 HDF5 约 124 MiB，下载脚本会校验原文件的 MD5。已有本地文件时可直接放到上述路径并执行导出。默认运行全房室小回路。全量形态定位和原生接线可分别用 `--full --plan-only` 和 `--full --build-only` 验证。

## 来源

- 网络、细胞位置和接触点来自 De Schepper 等（2022）的 BSB 重建。 [原论文](https://doi.org/10.1038/s42003-022-04213-y) · [模拟配置](https://github.com/Helveg/deschepper-etal-2022-source/blob/897eb74e942ce64a90b67090cae0da7e467978f8/configs/balanced.json) · [发布数据](https://doi.org/10.5281/zenodo.7230288)
- Golgi 的升支 AMPA 替代机制取自 2022 年网络论文所引用的 Masoli 等（2020）单细胞模型。 [Masoli 论文](https://doi.org/10.1371/journal.pcbi.1007937) · [ModelDB 源码](https://github.com/ModelDBRepository/266806/blob/2dedfcd3d6ea0163cee913be6baff2608bd3449b/Morphology_1/Synapses.py#L65-L76)
- 细胞构建采用 BrainCell 示例的分区膜机制与参数。附带模板和数据取自 BrainCell 提交 `7fe0019fe9756add751926037468b5733d8598b4`，保留原始许可。 [BrainCell 0.2.0](https://pypi.org/project/braincell/0.2.0/) · [示例资产许可](assets/braincell/LICENSE)

迁移过程中的早期记录见 [参数来源审计](PARAMETER_PROVENANCE.md)、[颗粒细胞形态迁移](PAPER_GRANULE_CELL.md) 和 [单突触对照](SYNAPSE_PORT.md)。当前运行入口和结果以本页及链接的验证说明为准。
