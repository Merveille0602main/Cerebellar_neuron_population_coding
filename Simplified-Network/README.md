# 256 个颗粒细胞和 12 个 Golgi 的子网

这是此前用于本机试跑的同一组细胞。从 De Schepper 等（2022）的 `balanced.hdf5` 中，按原始 ID 排序后，用 NumPy 随机种子 318 依次无放回选取 256 个颗粒细胞和 12 个 Golgi。保留原有 117 条苔藓纤维和 2,336 个小球，以及两端都在子网内的接触。原始 ID、三维位置和形态接触位置保存在 `data/spec/`。 [论文](https://doi.org/10.1038/s42003-022-04213-y) · [发布数据](https://doi.org/10.5281/zenodo.7230288)

颗粒细胞和 Golgi 分别保留 114 和 317 个计算房室。化学通路包括小球到颗粒细胞和 Golgi、升支和平行纤维到 Golgi、Golgi 到颗粒细胞以及 Golgi 间抑制，共 **7,521 条受体事件路由**。每条化学接触采用现有 BrainCell 的独立释放和受体状态。裁切后的网络只包含选中细胞之间的反馈。

## Junction 电耦合

`gap_goc.npz` 的 **74 行有向记录**来自原模型的镜像处理。按原始细胞 ID 和形态位置配对后得到 **37 个双端接触**，其中 25 个跨两个 Golgi，12 个连接同一 Golgi 的两个不同形态位置。每个物理接触统一采用 DBBS 并行缝隙连接机制的默认电导 **1 nS**。 [原模型配置](https://github.com/Helveg/deschepper-etal-2022-source/blob/897eb74e942ce64a90b67090cae0da7e467978f8/configs/balanced.json) · [DBBS 并行 gap junction 源码](https://github.com/dbbs-lab/dbbs-mod-collection/blob/f896c16f963042331a481a07a03240e4cc059f2c/dbbs_mod_collection/mod/glia__dbbs_mod_collection__gap_junction__parallel.mod)

默认的[原生 Junction 接线](junction.py)从 `/home/merveille/copneuron/braincell-junction` 加载 BrainCell。接线保留原始两端的细胞与形态位置，用 `Network.connect_junction()` 注册双端 `mech.Junction`。每个接触端点绑定到其所在的计算房室。原生求解器在同一个隐式步骤中联立轴向电流、膜电流和缝隙连接，得到相互耦合的 Golgi 膜电压。

[历史显式耦合版](explicit_junction.py)使用发布版 BrainCell 0.2.0。它从每步起始的形态电节点电压计算等量反向的电流，并在本步内将其作为恒定点电流注入两端。两种模式使用相同的解剖接触和电导。离散化后，落到同一细胞同一电节点的两端之间没有电压差。

## 运行

在 WSL 的 Git 仓库中进入 `Simplified-Network`，运行以下命令。安装父项目会安装子网共用的细胞与化学突触代码，以及用来重现旧结果的 BrainCell 0.2.0。默认模式从指定的本地工作树导入原生 BrainCell，可用 `--braincell-junction-root` 指向其他位置。目录自带接线、形态映射和细胞模板，运行时直接读取子网数据。

```bash
python -m pip install -e '..[braincell-runtime]'
JAX_PLATFORMS=cpu python run_forward.py
python compare_forward.py
```

默认模拟 18 毫秒，步长 0.05 毫秒，沿用原基线的同步苔藓纤维刺激。可用 `--duration-ms`、`--dt-ms` 和 `--output` 修改时长、步长和结果目录。[原生结果摘要](results/with_native_junction/forward_summary.json)记录了工作树路径、Git HEAD 和实际运行的 Python 源码指纹。本次工作树含有尚未提交的原生 Junction 实现。精确复跑需要同一份工作树源码。

下面的命令使用发布版 BrainCell 0.2.0，将显式耦合和无耦合两组复跑结果写到新目录。若只拷贝本目录来重跑这两种历史模式，可以用 `python -m pip install -r requirements.txt` 安装历史运行代码。

```bash
JAX_PLATFORMS=cpu python run_forward.py --junction-mode explicit --output results/explicit_rerun
JAX_PLATFORMS=cpu python run_forward.py --junction-mode off --output results/without_junction
python compare_forward.py --baseline results/without_junction --explicit results/explicit_rerun
```

如需重新提取，传入发布数据的原文件。

```bash
python extract.py ../data/reference/balanced.hdf5
```

## 输出

- 无耦合基线保存在 [摘要](results/forward_summary.json) 和 [胞体电压及发放数据](results/forward_traces.npz)。
- 显式耦合版保存在 [摘要](results/with_junction/forward_summary.json) 和 [电压、发放及注入电流数据](results/with_junction/forward_traces.npz)。[原始两组对照](results/junction_comparison.json)仍可单独查阅。
- 原生耦合版保存在 [摘要](results/with_native_junction/forward_summary.json) 和 [电压、发放及 Junction 数据](results/with_native_junction/forward_traces.npz)。其 `junction_current_pA` 由每步求解后的两端房室电压和电导重建。`junction_endpoint_voltage_mV` 和 `junction_endpoint_cv_ids` 分别保存所用电压和房室索引。
- 两种耦合模式的电流数组按时间步、物理接触和端点排列。`junction_contact_endpoints` 保存每个端点的原始细胞 ID 与 BSB 形态索引。
- [三组对照](results/native_junction_comparison.json)核对刺激表、输入事件数、时间轴、细胞 ID、化学接线和电耦合端点，汇总胞体电压差、发放数及发放时刻。配对发放时移只统计两组发放次数相同的细胞，并按细胞内的发放顺序配对。

## 本机验证

三组均在 WSL CPU 上运行 360 步，收到 2,340 条苔藓纤维事件和 46,720 条小球事件。268 个胞体的电压记录均为有限值，九种按通路记录的受体电导峰值均大于零。每组中 256 个颗粒细胞各发放一次，12 个 Golgi 中有 11 个发放，共 19 次胞体发放。每个细胞的发放次数在三组中一致。

| 运行 | 整个前向进程耗时 | 峰值常驻内存 | 有非零电流的 Junction 接触 | 单端最大绝对电流 |
| --- | ---: | ---: | ---: | ---: |
| 无耦合 | 11 分 9 秒 | 5.38 GiB | — | — |
| 显式耦合 | 10 分 10 秒 | 5.47 GiB | 37 | 64.47 pA |
| 原生 Junction | 8 分 40 秒 | 5.11 GiB | 37 | 63.71 pA |

| 对照 | 群体 | 最大配对发放时移 | 胞体电压均方根差 | 同时刻最大胞体电压差 |
| --- | --- | ---: | ---: | ---: |
| 显式与无耦合 | 颗粒细胞 | 0.00 毫秒 | 0.200 mV | 11.75 mV |
| 显式与无耦合 | Golgi | 1.65 毫秒 | 6.836 mV | 75.47 mV |
| 原生与无耦合 | 颗粒细胞 | 0.00 毫秒 | 0.208 mV | 12.40 mV |
| 原生与无耦合 | Golgi | 1.65 毫秒 | 6.857 mV | 75.51 mV |
| 原生与显式 | 颗粒细胞 | 0.00 毫秒 | 0.058 mV | 2.77 mV |
| 原生与显式 | Golgi | 0.05 毫秒 | 0.224 mV | 8.24 mV |

电压差按同一细胞、同一时间步计算，动作电位的时间偏移也会反映在瞬时电压差上。原生版与两组历史结果使用不同的 BrainCell 运行代码。原生端点取所在房室电压，显式端点取形态电节点电压。对照反映了这些差异与积分方式的共同影响。两组电流峰值分别取自步末端点电压重建值和步初注入电流。 [完整对比数据](results/native_junction_comparison.json)

在 BrainCell 0.2.0 中运行 `python -m unittest -v test_junction` 可核对镜像配对、并联接触和显式双向电流。本次还运行了原生工作树中的 `braincell.network.engine_test`，其中的双端耦合测试涵盖跨细胞、同细胞不同房室和与直接联立求解的数值对照。
