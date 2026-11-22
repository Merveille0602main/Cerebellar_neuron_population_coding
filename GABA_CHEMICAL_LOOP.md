# GoC→GrC GABA 与无 gap 颗粒层化学回路

## 1. 论文机制来源

`balanced.json` 把 `golgi_to_granule` 声明为 `synapses: ["GABA"]`。论文时代的 `dbbs_models.GranuleCell` 进一步把该名称解析为 `dbbs_mod_collection` 的 `GABA.granule`，并覆盖 `U=0.35`。这不是通用双指数 GABA，也不是 `golgi_to_glomerulus` 上的一个虚构化学突触。

本项目固定使用 DBBS 历史提交：

```text
commit  8d21c4a4f888f19a6531224f1eb08d6c833c44e5
file    glia__dbbs_mod_collection__GABA__granule.mod
SHA256  f99a72c520860e01d77ca2d34b4f7bade4796a65b6abcb5c6f88f5d79695bfae
```

获取和编译：

```bash
.venv/bin/python scripts/fetch_dbbs_gaba_reference.py
.venv/bin/python scripts/build_neuron_goc_mechanisms.py --force
```

## 2. BrainCell 移植内容

`GrCGABA2020` 保留了：

- `U=0.35`、`tau_1=0.1 ms`、`tau_rec=36.169 ms`、`tau_facil=58.517 ms`；
- 100-event Tsodyks–Markram 释放与扩散历史；
- alpha1 和 alpha6 两套各 8 状态受体；
- alpha1 使用局部短脉冲加扩散，alpha6 只使用扩散；
- `gmaxA1=918.807 pS`、`gmaxA6=132.842 pS`；
- `Erev=-65 mV`、扩散和通道 Q10。

固定电压 `-40 mV`、单事件、`dt=0.05 ms` 的 NEURON 9.0.1 对照结果保存在 `data/export/grc_gaba_comparison.json`：

| 信号 | 峰值相对误差 | normalized RMSE | 结果 |
|---|---:|---:|---|
| transmitter | 0.841% | 0.109% | 通过 |
| alpha1 open | 0.379% | 0.181% | 通过 |
| alpha6 open | 0.000437% | 0.000596% | 通过 |
| total conductance | 0.577% | 0.411% | 通过 |
| current | 0.577% | 0.411% | 通过 |

所有信号通过当前 1% 峰值误差和 1% normalized RMSE 门限；alpha1/总电导的最佳对齐相差一个 0.05 ms step，与此前 AMPA 的事件/积分更新顺序差异一致。

## 3. Spike 输出语义修正

BrainCell 多区室 `Cell` 会为每个 CV 计算阈值上穿；网络的通用 `population_spike` 再对所有 CV 做 `any`。这意味着同一个沿轴突和树突传播的动作电位可能在不同时间步被当成多个 cell event。

强输入审计中，未限制的 GoC 输出在 30 ms 内产生了 56 个 CV 聚合事件。复现层新增 `_restrict_spikes_to_soma()`，只保留 soma CV 的向上阈值穿越；同一测试随后得到 2 个 soma 动作电位事件。该适配只位于复现仓库，没有修改共享 BrainCell。它不改变膜电位或通道动力学，只定义哪些 CV 可以成为网络 presynaptic event source。

## 4. 可运行化学回路

入口：

```python
build_granular_layer_chemical_subnetwork(...)
```

它在共同输入网络上增加：

```text
MF schedule
    ↓ topology-expanded mossy_to_glomerulus
glomerulus ──AMPA/NMDA──→ GrC
    │
    └────────AMPA/NMDA──→ GoC soma spike
                              ↓ exact golgi_to_granule contacts
                         GrC GABA alpha1/alpha6
```

最小自动测试选择原始 ID：

```text
MF 214 → glomerulus 732 → GoC 24
                         → GrC 4130
GoC 24 → GrC 4130: 1 edge, 2 contacts, target compartments 120/129
```

10 个 1 ms 间隔输入事件在 18 ms pilot 中触发 1 个 GoC soma event，并在两个发布 contact 上得到非零 GABA conductance。

运行命令：

```bash
PYTHONPATH=src:/home/merveille/copneuron/braincell \
  .venv/bin/python scripts/run_granular_layer_chemical_subnetwork.py
```

## 5. 当前边界

- 这是无 gap junction 的化学回路；`gap_goc` 没有被偷偷替换成 event synapse。
- MF→glomerulus 仍是固定 schedule 的拓扑预展开，不是在线 relay。
- GABA 接触机制已通过固定电压对照，GoC 的 intrinsic whole-cell baseline 仍未与 NEURON 完全一致。
- 当前 pilot 证明因果事件可以穿过整个化学链，不代表全规模网络、论文背景协议或群体统计已经复现。

## 6. 分块资源基准

已完成 20 ms CPU 分块基准，机器可读结果位于
`data/export/balanced/benchmarks/granular_chemical_loop_20ms_cpu.json`。配置为
117 个 MF 输入表、2 个 glomerulus、8 个 GrC、2 个 GoC、8 个
glomerulus→GrC contacts、2 个 glomerulus→GoC contacts 和 8 个
GoC→GrC contacts；20 ms 按 5 ms 分为 4 块。

- 400 steps，连续事件状态跨块保留；
- 输出量估计与实测均为 340,000 bytes；每块最多 85,000 bytes；
- CPU wall time 75.44 s；
- peak RSS 2,190,991,360 bytes，约 2.04 GiB；
- 0–20 ms 内 MF/glomerulus 没有事件，因为所选论文输入最早从 200 ms 开始；
- 当前 BrainCell GoC 产生 2 个自发 soma spikes，并由发布 GoC→GrC contacts 产生非零 GABA 电导。

最后一项是内禀 GoC 活动，不应被误写成 MF 驱动响应；它也再次说明完整论文协议基准前必须解决或量化 GoC intrinsic whole-cell 差异。
