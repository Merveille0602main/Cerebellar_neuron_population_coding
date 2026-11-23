# glomerulus→GoC 通路审计与第一版迁移

## 1. 目标

本阶段把论文网络的第二条化学突触通路 `glomerulus_to_golgi` 迁移为 BrainCell 可运行子网络。原则是依次对齐：

1. 论文 BSB 配置声明；
2. `balanced.hdf5` 中实际保存的 cell pair、contact 和目标 compartment；
3. `dbbs_models.GolgiCell` 对配置中突触名称的实际解析；
4. 对应 NMODL 方程及默认参数；
5. BrainCell GoC 形态、电生理模型和 contact 位置；
6. 最小运行与回归测试。

论文专用构建器和机制别名全部保存在本复现仓库，没有因此继续修改共享 BrainCell 仓库。

## 2. 配置层结论

`deschepper-etal-2022-source/configs/balanced.json` 声明：

```text
connection: glomerulus_to_golgi
pre:  glomerulus / soma
post: golgi_cell / basal_dendrites
detailed: true

simulation cell model: dbbs_models.GolgiCell
simulation synapses: AMPA_MF, NMDA
temperature: 32 °C
```

因此 `glomerulus_to_golgi` 是真实的兴奋性化学突触通路，不是 `mossy_to_glomerulus` 那种无化学突触的 relay edge。

## 3. `balanced.hdf5` 实际拓扑

导出的 `connectivity/glomerulus_to_golgi.npz` 包含：

- 3,939 个 contact；
- 3,939 条唯一 cell-level edge，即每条 edge 恰有一个 contact；
- 1,808 个有该类输出的 glomerulus；
- 69 个收到该类输入的 GoC；发布 population 总数为 70；
- 787 个唯一 postsynaptic compartment；
- 全部 3,939 个 contact 和 787 个唯一 compartment 都能映射到 BrainCell GoC，覆盖率 100%。

绝大部分 contact 位于 basal dendrite。按发布 HDF5 的 branch label 和 BrainCell GoC 的源 region 规则重新统计：

- `dend_basal`: 3,938 个 contact；
- `dend_apical`: 1 个 contact。

配置规则要求 basal dendrite，但发布 HDF5 中确实存在一个 apical contact。当前迁移以已发布 `balanced.hdf5` 为金标准，保留该 contact，不在导入时静默删除或改写；后续若要严格重跑 BSB 重建，应另行调查这个单点差异的产生过程。

## 4. 突触机制来源审计

### 4.1 名称不能仅凭文件名推断

BrainCell 的 ModelDB 266806 GoC 资源目录中存在：

- `GOLGI_Ampa_mossy_det_vi.mod`，其本地版本默认 `gmax=2000 pS`、`U=0.2`；
- `PC_NMDA_NR2B.mod`，其本地版本默认 `gmax=5000 pS`、`U=0.2`，使用 8 状态 NR2B 模型并写入 Ca current。

这些文件属于独立的 Masoli et al. 2020 GoC 模型资源，但文件存在于 GoC 目录并不意味着 De Schepper et al. 网络配置通过 `dbbs_models.GolgiCell` 使用了它们。

### 4.2 论文配置的实际名称解析

论文使用的 `dbbs_models.GolgiCell` 定义可在 [dbbs-lab/dbbs-models](https://github.com/dbbs-lab/dbbs-models) 的 paper-era v1.1.2 提交 `5f920d5` 中核对。其映射为：

```python
"AMPA_MF": {"point_process": ("AMPA", "granule")}
"NMDA":    {"point_process": ("NMDA", "granule")}
```

对应 [dbbs-lab/dbbs-mod-collection](https://github.com/dbbs-lab/dbbs-mod-collection) 的机制在 2020-07-07 的重命名提交 `21e7d48`/`551c170` 后为：

```text
AMPA.granule: gmax=1200 pS, U=0.42, 3-state receptor
NMDA.granule: gmax=18800 pS, U=0.42, 5-state receptor
```

这正是本项目已经从 `GRANULE_Ampa_det_vi.mod` 和 `GRANULE_Nmda_det_vi.mod` 移植并与 NEURON 对照过的方程。需要区分的是，论文 `GranuleCell` 另把这两份机制的 `U` 覆盖为 0.43，而 `GolgiCell` 没有该覆盖，所以 GoC 的最终有效值仍为 0.42。

所以本阶段没有另行移植 2 nS/5 nS 的 standalone GoC 文件。那样做会改变论文实际运行的受体方程、最大电导、短时释放参数和 NMDA 状态数。

### 4.3 BrainCell 中的命名处理

为了让网络声明仍能清楚表达 postsynaptic cell type，本项目新增两个注册名称：

```text
GoCAMPA_MF2020 → 继承 GrCAMPA2020
GoCNMDA2020    → 继承 GrCNMDA2020
```

它们是同一套 DBBS NMODL 方程的 GoC 语义别名，不复制动力学代码；但会显式把从 GrC 类继承到的 `U` 恢复为 GoC 有效值 0.42。测试明确检查 GrC=0.43、GoC=0.42，以及 `gmax=0.0012/0.0188 µS`。

## 5. GoC 细胞和形态映射

使用 BrainCell 已有正式模板：

```text
examples/neuron_compare/cell/goc_ma2020/goc_braincell.py
```

构建条件：

- 发布 GoC ASC morphology；
- 1 soma、151 dendrite branches、75 axon branches；
- 共 227 branches；
- 当前 CV 规则生成 317 个 control volumes；
- 温度设为论文配置的 32 °C；
- population 维度由选中的原始 GoC ID 数决定。

BSB `GolgiCell` morphology 到 BrainCell morphology 的最佳变换是 `rotate_z_minus_90`。全部形态 compartment 均按 exact geometry 映射，没有采用 GrC 那样的 semantic synthetic mapping。

每个发布 target compartment 先转换成 `(BrainCell branch_id, normalized x)`，再由 BrainCell discretization 定位到实际 CV 和 synapse pool index。BSB compartment ID 没有被错误地直接当作 BrainCell synapse index。

## 6. 新增可运行构建器

入口：

```python
build_glomerulus_golgi_subnetwork(
    spec_path,
    source_hdf5,
    braincell_repo,
    glomerulus_spike_times_ms=None,
    max_glomeruli=4,
    max_golgi=4,
)
```

构建流程：

1. 读取 `glomerulus_to_golgi.npz` 的原始 cell pairs 和 target compartments；
2. 选择指定 glomerulus ID 及与其相连的 GoC ID；
3. 只把 population ID 紧凑重编号为 BrainCell 本地 index，metadata 仍保存原始 ID；
4. 对选中的 contact 去重得到 cell edge，并保留 `contact_to_edge`；
5. 构建 32 °C 的多区室 GoC population；
6. 将每个 target compartment 映射到 GoC CV；
7. 在同一 contact 位置放置 `GoCAMPA_MF2020` 和 `GoCNMDA2020`；
8. 为两个 mechanism 建立显式 projection；
9. 记录 AMPA/NMDA conductance、GoC soma voltage 和 population spikes。

未提供 schedule 时，前 `max_glomeruli` 个已连接 glomerulus 各在 1 ms 发放一次；提供 schedule 时，其 key 必须是发布拓扑中的原始 glomerulus ID。当前输入仍是明确 schedule，不声称已实现 MF→glomerulus 的在线因果 relay。

## 7. 已完成的最小验证

新增测试包括：

- GoC 机制别名与论文 DBBS 机制的继承及默认参数一致；
- 选择 1 个发布 glomerulus→GoC contact；
- metadata 中 edge/contact 都为 1，目标为 basal dendrite；
- glomerulus schedule 产生 1 次 spike；
- GoC contact 上 AMPA_MF 和 NMDA conductance 都大于 0；
- GoC soma voltage 全部为有限值。

聚焦测试命令：

```bash
.venv/bin/pytest -q tests/test_paper_synapses.py tests/test_runnable_subnetwork.py
```

当前结果：7 个测试通过。

## 8. 单 contact 的 NEURON 对照

### 8.1 可复现环境和输入

NEURON 作为独立参考求解器安装在项目环境中。下文历史结果使用 9.0.1，当前 [NEURON 对照环境](ENVIRONMENT.md#与-neuron-对照) 使用 9.0.2。

```bash
.venv/bin/pip install -e '.[neuron-validation]'
.venv/bin/python scripts/fetch_dbbs_gaba_reference.py
.venv/bin/python scripts/build_neuron_goc_mechanisms.py
```

构建脚本收集 GoC 的全部离子通道、`CdpStC_MA20_GoC.mod`，以及论文实际使用的 DBBS granule AMPA/NMDA MOD 文件。生成目录位于 `data/reference/neuron_goc_mechanisms/`，manifest 记录 NEURON 版本、全部源路径和 SHA-256，因此不会把来源不明的预编译机制当成金标准。

对照选取的是可运行构建器确定的第一个发布 contact，而非另造的抽象位置：

```text
glomerulus global ID = 732
Golgi global ID      = 24
BSB target compartment = 471
BrainCell location   = dendrite_129, x=0.299347724...
```

刺激组在 1 ms 输入一个 presynaptic event；基线组使用相同细胞、初始化和 3 ms 运行时间但不输入 event。两边均记录 AMPA/NMDA conductance、电流、contact voltage 和 soma voltage。

```bash
.venv/bin/python scripts/run_neuron_goc_contact_reference.py
.venv/bin/python scripts/run_braincell_goc_contact_reference.py
.venv/bin/python scripts/compare_goc_contact_traces.py
```

### 8.2 结果和判定边界

机器可读报告为 `data/export/goc_contact_comparison.json`。当前结果：

| 信号 | 峰值相对误差 | normalized RMSE | 判定 |
|---|---:|---:|---|
| AMPA conductance | 0.244% | 0.369% | 通过；最佳对齐相差 1 个 0.05 ms step |
| AMPA current | 0.729% | 0.673% | 通过；最佳对齐相差 1 step |
| NMDA conductance | 0.0000485% | 0.0000125% | 通过 |
| NMDA current | 3.117% | 0.925% | 通过当前 5%/1% 阈值 |
| 基线扣除后的 soma response | 1.256% | 0.353% | 通过 |
| 基线扣除后的 contact response | 1.014% | 0.826% | 通过 |

这说明论文的 AMPA/NMDA 接触机制以及它们引起的局部和 soma 增量响应已经通过对照。但不能据此宣称整个 GoC 模型等价：无输入时 3 ms 内的 soma 最大差异为 0.582 mV，contact 最大差异为 0.554 mV。报告明确把这部分标为 `intrinsic_whole_cell_parity: unresolved`。在完成离子通道、初始化和积分时序审计以前，它不能被突触增量对照掩盖。

## 9. 尚未完成的等价性验证

这版构建器证明了发布拓扑、精确 contact 位置、论文实际 DBBS 突触方程和真实多区室 GoC 可以在 BrainCell 中共同运行，但还不能称为完整论文 GoC 动力学复现：

1. GoC 单 contact 的突触及增量电压响应已经通过，但无输入的 intrinsic whole-cell baseline 尚未通过。
2. 网络事件现已限制为 soma CV 的向上阈值穿越，避免一个传播动作电位被多个 CV 重复发送；但 GoC 自发节律尚未按论文运行环境单独校准。
3. MF schedule 已经同时展开到 GrC 和 GoC，形成共同输入的 `MF→glomerulus→{GrC, GoC}` 子网络。
4. `GoC→GrC GABA` 已加入并通过固定电压 NEURON 对照，形成不含 gap junction 的颗粒层化学回路；详见 `GABA_CHEMICAL_LOOP.md`。
5. `gap_goc` 仍缺 BrainCell 双端电耦合 runtime 支持。
6. 尚未对 70 个 GoC、3,939 个双机制 contact 做资源基准。

## 10. 下一步

按风险顺序继续：

1. 并行审计 GoC intrinsic baseline 的初始化、离子通道和积分时序；
2. 对共同输入和 GABA 化学回路做资源估算及分块长时运行；
3. 把论文 background/sensory 协议接入化学回路并确定最小统计对照；
4. 最后单独设计 `gap_goc` 双端耦合扩展。
