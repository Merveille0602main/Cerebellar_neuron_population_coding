# glomerulus → Granule 突触机制移植记录

## 1. 参数与方程来源

网络受体配置来自论文仓库：

- `deschepper-etal-2022-source/configs/balanced.json`
- simulation：`stim_on_MFs`
- connection：`glomerulus_to_granule`
- synapses：`["AMPA", "NMDA"]`
- simulation resolution：`0.05 ms`
- temperature：`32 °C`

受体和递质释放方程来自 BrainCell 仓库保留的 NEURON 模型源文件：

- `examples/neuron_compare/Cerebellum_mod/GrC/synapse/GRANULE_Ampa_det_vi.mod`
- `examples/neuron_compare/Cerebellum_mod/GrC/synapse/GRANULE_Nmda_det_vi.mod`

最终实例参数还必须经过论文时代 `dbbs_models` v1.1.2（提交
`5f920d5`）核对。该适配层把 `GranuleCell` 的 AMPA/NMDA 映射到上述
`AMPA.granule`/`NMDA.granule` 方程，并把两者的 MOD 默认 `U=0.42`
覆盖为 `U=0.43`。因此这里的 0.43 是论文 GrC 的最终有效值，不是对
MOD 文件的任意改动。GoC 对相同机制没有该覆盖，仍使用 0.42；完整
来源链见 `PARAMETER_PROVENANCE.md`。

BrainCell 移植实现位于：

- `src/cerebellar_reproduction/paper_synapses.py::GrCAMPA2020`
- `src/cerebellar_reproduction/paper_synapses.py::GrCNMDA2020`

## 2. 共同的突触前短时释放

AMPA 和 NMDA 的原始 MOD point process 各自保存一份相同的释放状态。移植版保持这一结构。

| 参数 | 数值 | 含义 |
|---|---:|---|
| `U` | 0.43 | GrC 的基础释放利用率；MOD 默认 0.42 被论文适配层覆盖 |
| `tau_rec` | 8 ms | 恢复时间常数 |
| `tau_facil` | 5 ms | 易化变量衰减时间常数 |
| `tau_1` | 1 ms | 活跃资源衰减时间常数 |
| `Cdur` | 0.3 ms | 局部递质脉冲时长 |
| `Tmax` | 1 mM | 局部递质标度 |
| `M` | 21.515 | 单囊泡分子数参数，沿用 MOD 数值和换算式 |
| `Rd` | 1.03 µm | 扩散距离 |
| `Diff` | 0.223 µm²/ms | 扩散系数 |
| `lambda` | 20 nm | 突触间隙高度 |

下面的连续系统给出事件之间的解析衰减关系：

```text
dy/dt = -y / tau_1
dz/dt =  y / tau_1 - z / tau_rec
du/dt = -u / tau_facil
```

实现中不逐步数值积分这三个变量，而是在新事件到达时按经过时间计算解析指数衰减，然后使用与 MOD `NET_RECEIVE` 等价的更新：

```text
u ← u + U(1-u)
x ← 1-y-z
y ← y + xu
```

与 MOD 一样，`Cdur` 尚未结束时到达的第二个事件被忽略。最近 100 次接受的释放幅度和时间被保留，用原 MOD 的三维扩散 Green 函数计算 `Trelease`。

## 3. AMPA

| 参数 | 数值 |
|---|---:|
| `gmax` | 1200 pS = 0.0012 µS |
| `Erev` | 0 mV |
| `r1FIX` | 5.4 /ms |
| `r2` | 0.82 /ms |
| `r5` | 0.013 /ms |
| `r6FIX` | 1.12 /ms |
| `kB` | 0.44 mM |
| `Q10_diff` | 1.5 |
| `Q10_channel` | 2.4 |

移植了 `C/O/D` 三状态动力学、局部 `Cdur` 脉冲、扩散尾迹和 32 °C Q10。电流按 NEURON 符号约定作为总点电流处理。

## 4. NMDA

| 参数 | 数值 |
|---|---:|
| `gmax` | 18800 pS = 0.0188 µS |
| `Erev` | −3.7 mV |
| `Rb` | 5 /ms/mM |
| `Ru` | 0.1 /ms |
| `RdRate` | 0.012 /ms |
| `Rr` | 0.009 /ms |
| `Ro` | 0.03 /ms |
| `Rc` | 0.966 /ms |
| `v0_block` | −20 mV |
| `k_block` | 13 mV |

移植了 `C0/C1/C2/D/O` 五状态 Westbrook scheme 和：

```text
MgBlock = 1 / (1 + exp(-(V - v0_block) / k_block))
```

## 5. 为什么当前可以把状态放在 BrainCell postsynaptic point

对 `balanced.hdf5` 的全部 `glomerulus_to_granule` contacts 检查得到：

- contacts：114,460；
- 唯一 `(post cell, target compartment)`：114,460；
- 最大重复数：1；
- 每个 Granule Cell 恰有 4 个 contacts。

所以本通路中一个 BrainCell 颗粒细胞树突终末突触点恰好对应一个 BSB contact，短时释放状态不会被多个 presynaptic contact 共享。这个结论不能自动推广到其他连接类型；接入每条新通路前必须重新检查。

## 6. 当前验收与剩余差距

已经完成：

- 第一事件释放量为 0.43 的测试；
- `Cdur` 内重复事件抑制测试；
- AMPA/NMDA 耦合 Markov 更新后的概率质量守恒测试；
- NMDA 在 −20 mV 时 `MgBlock=0.5` 测试；
- 双受体真实多区室子网络端到端运行测试；
- 4 glomerulus、16 Granule、16 个论文 contacts 的 5 ms CPU 运行。
- NEURON 9.0.1 与 BrainCell 的 `−65 mV、dt=0.05 ms` 自动轨迹比较；
- AMPA 单脉冲峰值：`393.831817 pS` 对 `392.888993 pS`，相对误差约 `0.239398%`；
- NMDA 单脉冲峰值：`262.657926 pS` 对 `262.658170 pS`，相对误差约 `0.000093%`；
- transmitter、open state、conductance、current 均通过峰值误差 `<1%` 和 normalized RMSE `<1%` 的当前门限。
- 三脉冲 `1/11/21 ms` 条件通过同一门限；
- 论文 sensory burst 相对时序 `1/4/7/32/52 ms` 也通过：AMPA 电导峰值误差约 `0.026490%`，NMDA 约 `0.000431%`；AMPA transmitter 峰值误差约 `0.685045%`，NMDA 约 `0.000003%`。
- 该 sensory-burst 时间序列已接入真实 `mossy_to_glomerulus` 分支和 `glomerulus_to_granule` contacts；默认 4 mossy、8 glomerulus、32 GrC 运行中，每个 mossy 与其子 glomerulus 都记录到 5 个 spike。

仍需完成：

1. 检查不同保持电位下的 NMDA Mg 阻塞曲线；
2. `initial_mf` 和 `background` 的 BSB 3.10.3 Poisson 语义及显式替代 seed 已实现并接入小网络；仍需完成全 117 个 source、完整 10 s 的资源基准。原模拟使用未保存的系统熵，因此不能恢复历史随机 realization；
3. 当前可称为“固定电压下单脉冲、三脉冲和论文 sensory-burst 数值门限通过，并在 GrC 小网络中闭合输入链”；在真实细胞电压轨迹对照完成前，不扩大为整细胞响应等价。

参考轨迹与比较命令：

```bash
# 项目环境中的 NEURON 9.0.1 生成 MOD 参考轨迹；脚本会显式应用
# dbbs_models 对 GrC 的 U=0.43 覆盖
.venv/bin/python scripts/run_neuron_grc_synapse_reference.py ampa
.venv/bin/python scripts/run_neuron_grc_synapse_reference.py nmda

# 项目环境生成 BrainCell 固定电压轨迹
.venv/bin/python scripts/run_braincell_grc_synapse_reference.py ampa
.venv/bin/python scripts/run_braincell_grc_synapse_reference.py nmda

# 1% 门限比较
.venv/bin/python scripts/compare_grc_synapse_traces.py \
  data/reference/neuron_grc_ampa.npz data/reference/braincell_grc_ampa.npz
.venv/bin/python scripts/compare_grc_synapse_traces.py \
  data/reference/neuron_grc_nmda.npz data/reference/braincell_grc_nmda.npz
```
