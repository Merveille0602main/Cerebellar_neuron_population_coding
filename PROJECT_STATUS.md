# De Schepper et al. (2022) 小脑网络：复现说明与项目状态

更新日期：2026-08-07  
目标：以 BrainCell 复现论文所使用的小脑皮层网络及其仿真实验。本文把“网络重建”和“网络运行”分开说明，并记录当前实现的边界。

本文的首要科学依据是论文 PDF：[De Schepper et al., *Communications Biology* 5, 1240 (2022)](s42003-022-04213-y.pdf)，DOI: [10.1038/s42003-022-04213-y](https://doi.org/10.1038/s42003-022-04213-y)。论文给出研究问题、网络/实验方法和结果；`deschepper-etal-2022-source` 中的 JSON、Python 与 HDF5 文件仅用于补充其具体实现和已发布网络实例。两者若有数值或版本差异，以论文的对应实验描述为准，并在复现记录中标注差异。

## 概述

论文实验由两个层次组成：

```text
BSB：定义三维组织、放置细胞、加载形态、检测空间接触、写出网络 HDF5
                                      ↓
NEURON + dbbs_models：实例化细胞/突触机制、施加输入、时间积分、记录结果
```

在我们的复现中，已经从论文发布的 `balanced.hdf5` 无损导出了细胞群、细胞位置、17 类连接和每个解剖接触的信息，并将其创建为 BrainCell 的 7 个 `Population` 和 17 个 `EdgeSet`。这表示**网络拓扑已被导入 BrainCell**，但它还是“结构网络”，不是可仿真的完整网络。

要得到可运行且与论文相同的网络，还必须补齐：真实细胞模型、突触池与形态位置映射、mossy-fiber/glomerulus relay 输入、论文的突触参数和刺激协议，以及化学突触的 Tsodyks–Markram 短时释放动力学。Gap junction 是当前 BrainCell 的明确功能缺口。

---

## 1. 论文从网络搭建到实验的完整流程

论文复现仓库为 De Schepper 等发布的源码。 [论文源码](https://github.com/Helveg/deschepper-etal-2022-source)

基础网络配置为 `configs/balanced.json`；已构建的发布网络为 `balanced.hdf5`。论文 Methods 将流程概括为三个可迭代阶段：**configuration → reconstruction → simulation**，之后进行数据分析和可视化。

### 1.1 BSB 网络搭建的完整流程

BSB（Brain Scaffold Builder）在这里是**三维网络重建器/编译器**，不是积分膜电位的仿真器。它按下列阶段工作。

#### A. 定义组织空间

论文重建的是 `300 × 295 × 200 μm³` 的小脑皮层薄片，体积为 `17.7 × 10⁻³ mm³`。其中第二个空间维度由三个层厚度相加得到；`balanced.json` 分别配置其横向范围与分层：

| 层 | 厚度 | 作用 |
|---|---:|---|
| granular layer | 130 μm | 颗粒细胞、Golgi 细胞、glomerulus、mossy-fiber entities |
| Purkinje layer | 15 μm | Purkinje 细胞胞体层 |
| molecular layer | 150 μm | 平行纤维、Basket/stellate 细胞及 Purkinje 树突区域 |

#### B. 放置细胞与实体

配置为每一种细胞给出 placement 算法、密度/阵列参数、胞体大小和层约束。

| 类型 | BSB placement | 在发布网络中的数量 |
|---|---|---:|
| Golgi cell | 粒子放置 | 70 |
| Purkinje cell | 平行阵列放置 | 99 |
| Mossy fibers | 相对 glomerulus 放置的 entity | 117 |
| Basket cell | 分子层粒子放置（下部） | 147 |
| Stellate cell | 分子层粒子放置（上部） | 299 |
| Glomerulus | 粒子放置、relay | 2,336 |
| Granule cell | 粒子放置 | 28,615 |

`mossy_fibers` 是 `entity: true`，且是 relay；它没有普通细胞的形态和位置数组。它代表外部输入通道，而不是一个具有膜电位方程的 Granule/Golgi 类细胞。

#### C. 关联细胞形态与功能区段

BSB 为细胞指定形态库中的名称（如 `GranuleCell`、`GolgiCell`、`PurkinjeCell`），并按功能把形态划分为诸如 `basal_dendrites`、`apical_dendrites`、`axon`、`parallel_fiber`、`ascending_axon`、`pf_targets` 等区域。

这一步很重要：连接不只是“细胞 A 连到细胞 B”，而是“细胞 A 的某一形态区段与细胞 B 的某一形态区段相遇”。

#### D. 生成空间连接和接触

每一类连接都由 BSB 的空间算法生成。例如：

| 连接类型 | 主要生成方式 | 前 → 后 |
|---|---|---|
| mossy_to_glomerulus | 专用 connectome 规则 | mossy fiber → glomerulus |
| glomerulus_to_granule | 专用 connectome；Granule 收敛数为 4 | glomerulus → Granule |
| golgi_to_glomerulus / golgi_to_granule | 专用 connectome 规则 | Golgi → glomerulus / Granule |
| gap_goc | `VoxelIntersection` | Golgi basal dendrite ↔ Golgi basal dendrite |
| ascending_axon_to_* | `TouchDetector` | Granule ascending axon → Golgi / Purkinje |
| parallel_fiber_to_* | `FiberIntersection` | Granule parallel fiber → Golgi / Purkinje / Basket / Stellate |
| GoC/MLI 局部化学连接 | `VoxelIntersection` | 轴突 → 目标树突或胞体 |

这些算法以空间相交、纤维相交或生物学收敛/发散规则决定：

1. 哪些细胞对存在连接；
2. 每对细胞具有几个接触（contact）；
3. 接触落在两个形态的哪个 compartment；
4. 所用形态实例的标识。

对 `gap_goc`，配置还执行 `bsb.postprocessing.BidirectionalContact`（`mirroring_gap`），使电耦合具有双向关系。

#### E. BSB 输出已构建网络

BSB 将实际生成的网络写入 `balanced.hdf5`。这不是规则配置，而是一次确定的构建结果，关键内容包括：

```text
cells/placement/<cell type>/identifiers  每群的仿真全局 ID 起点和数量
cells/placement/<cell type>/positions    已放置细胞的三维坐标（entity 除外）
cells/connections/<connection>           每个接触的 [pre_global_id, post_global_id]
cells/connection_compartments/<connection> 每个接触的 [pre_compartment, post_compartment]
cells/connection_morphologies/<connection> 每个接触的 [pre_morphology, post_morphology]
morphologies/                            形态相关数据
entities/mossy_fibers                    relay entity 的 ID
```

因此，`balanced.hdf5` 可以被视为论文网络的“编译产物”或金标准网络实例。只要保持这些表不变，就不必先在 BrainCell 重写 BSB 的空间相交算法，也能获得同一张连接图。

### 1.2 后续实验（NEURON 仿真）的完整流程

BSB 配置文件的 `simulations` 段负责把已建网络交给 NEURON。完整步骤如下。

#### A. 为每个非 relay 群实例化细胞模型

配置指定 `dbbs_models.GranuleCell`、`GolgiCell`、`PurkinjeCell`、`BasketCell`、`StellateCell`。这些模型包含细胞形态、离子通道和膜动力学；`mossy_fibers`、`glomerulus` 则作为 relay。

#### B. 为每类连接绑定 NEURON 突触/机制

例如：

| 网络连接 | 仿真突触/机制 |
|---|---|
| glomerulus → Granule | AMPA + NMDA |
| glomerulus → Golgi | AMPA_MF + NMDA |
| Golgi → Granule/Golgi | GABA |
| Granule AA/PF → 靶细胞 | AMPA、AMPA_AA 或 AMPA_PF，部分另有 NMDA |
| Basket/Stellate → Purkinje 及 MLI 互连 | GABA |
| gap_goc | `gap`，source 为 `vgap` |

这里 BSB 只负责把“哪个接触使用哪个机制”传给 NEURON；实际的离子通道、突触动力学和电流积分由 NEURON 及 `dbbs_models` 完成。

#### C. 设置数值与输入协议

论文 Methods 报告所有正式仿真均通过 BSB 的 NEURON adapter、MPI 并行运行，时间分辨率为 `0.025 ms`。每次运行先经历 `5 s` 稳定期，随后有 `100 ms` 的随机 mf 输入初始化以使网络去同步；记录所有神经元胞体的 spike 和 voltage，部分分析另记录突触电流、胞内钙等变量。

论文用下列协议验证网络。它们是 BrainCell 实验复现的权威清单：

| 论文实验 | 输入/干预 | 主要观测与目的 |
|---|---|---|
| 静息态 | 所有 mossy fibers 上 4 Hz Poisson 背景，持续 4 s | 各群基础放电率；作为网络平衡的功能校验 |
| 颗粒层振荡与同步 | 同样的 4 Hz 背景；分别保留、禁用或增加 GoC gap-junction 密度 | GoC/GrC FFT、GoC spike cross-correlation、同步比例和空间范围 |
| 感觉 burst / 垂直柱传播 | 对 4 条相邻 mf（约经 80 个 glomeruli）施加短 burst | GrC 活动簇、向 aa/pf 的传播、GoC/MLI/PC 响应及 PC burst–pause |
| 侧向抑制 | 背景上叠加：一侧 24 条 mf、50 Hz Poisson、100 ms；另做 MLI→PC 断连 | PC 活动调制随离活动簇距离的变化 |
| 去除 PC 上的 MLI 抑制 | 关闭 PC 上 GABA-A 抑制；向 13 条 mf 施加单次脉冲 | PC simple-spike latency jitter，模拟 PC-Δγ2 条件 |
| PF–PC LTD 的学习后状态 | 4 条 mf，50 Hz、160 ms；PF→PC AMPA `g_syn` 从 1200 pS 分 10 步降至 30%，每档 5 次 | PC simple-spike suppression；论文预测约 35% PF–PC LTD 可得到约 15% PC 放电抑制 |

仓库 `balanced.json` 与 `plasticity/` 配置包含具体开发/执行参数（例如温度、设备名称、部分时长和突触别名）。它们对实现有价值，但在与论文 Methods 的正式实验参数出现差别时，必须先建立“配置文件—论文图号”的对应关系，不能直接把任一 JSON 当作所有论文图的唯一协议。

#### D. 运行并分析

NEURON 对所有膜电位、离子通道、化学突触和电耦合进行时间积分，输出记录数据。论文仓库中的 `plots/`、`tables/`、`tools/` 脚本再从结果 HDF5 提取峰值、放电频率、突触电流及相关图表。

### 1.3 论文中的短时突触动力学与长时可塑性实验

这两者必须区分：

1. **短时突触动力学（原模型运行时存在）**：论文 Methods 明确指出，化学递质释放采用 Tsodyks–Markram scheme，受体激活采用动力学方案。释放概率、囊泡循环、恢复时间常数、扩散与受体参数均参与运行时的突触响应。这是每次输入事件发生时的短时状态变化，不是“学习后永久改变 PF 权重”。
2. **PF–PC 长时 LTD（论文 Fig. 8 的实验操作）**：论文没有在单次仿真内训练或更新 PF 突触。它以多个独立仿真预设学习后的状态：将 AMPA 介导的 PF→PC `g_syn` 从默认 `1200 pS` 逐步降到 `30%`，共 10 个下降步骤；每个条件运行 5 次。然后把各条件与 0% LTD 的 pre-learning 基线相比较。

因此，论文复现所需的是“**短时释放模型 + 多个固定 PF–PC LTD 条件**”，不是在线 PF-STDP。

仓库的 `plasticity/` 还生成了 PF–MLI 与 PF–PC 的 11 × 11 组合配置（`AMPA_0`…`AMPA_10`、`NMDA_0`…`NMDA_10`）。这是源码中的扩展扫描；论文正文所报告的 Fig. 8 实验只明确讨论 PF–PC LTD，不能在文档中把 PF–MLI 扫描误写成论文已报告的核心可塑性实验。

---

## 2. BrainCell 平台功能与 NEURON 网络搭建的比较

这里要避免一个常见误解：NEURON 本身不是 BSB 那种三维网络重建器。论文的“网络搭建”由 BSB 完成，NEURON 是底层细胞/网络仿真器。BrainCell 则同时提供细胞声明、化学突触和显式网络连接 API，但目前没有 BSB 同级的空间建网模块。

| 能力 | BSB + NEURON（论文流程） | BrainCell 当前版本 | 对本项目的含义 |
|---|---|---|---|
| 三维层、密度、细胞放置 | BSB 原生支持 | 无 BSB 式 placement compiler | 复用 `balanced.hdf5` 的位置和连接，而非从零重建几何规则 |
| 形态驱动空间相交建连 | BSB 原生支持 Touch/Fiber/Voxel 等算法 | 无对应通用建连器 | 直接导入已生成的接触表 |
| 细胞群 | BSB placement sets；NEURON 中逐个实例 | `Network.add_population()`，同类 batched `Cell` | 7 个群已创建为 BrainCell populations |
| 细胞对连接图 | BSB 写入 HDF5；适配器转 NEURON | `Network.add_edges()` / `EdgeSet` | 17 类边表已精确导入 |
| 每个解剖接触 | BSB 保存 morphology/compartment 对 | `Projection` 可用 `explicit_contacts()` | 已保存，但尚未绑定 BrainCell 突触池 |
| 化学突触与短时释放 | NEURON/NMODL 的论文机制：Tsodyks–Markram 释放 + 受体动力学 | 已有 AMPA、NMDA、GABAa、ExpSyn/Exp2Syn 等；投射支持权重、延迟；未发现现成 Tsodyks–Markram 实现 | 简化化学网络可做；严格复现须新增/迁移短时释放模型 |
| 外部输入 / relay | BSB–NEURON adapter 的 relay 与 spike generator | 有网络运行和事件投递基础设施 | 需专门实现 mossy fiber/glomerulus relay 方案 |
| Gap junction | NEURON 可计算两端电压差引起的电流 | `braincell.mech.Junction` 仅声明占位；没有 partner wiring | 不能原样运行 `gap_goc` |
| 在线 STDP/LTP/LTD | 可用 NMODL 和自定义状态变量实现 | 没有内置 STDP；需实现可变权重的自定义突触 | 不属于本文公开配置的必需复现范围；仅为可能的后续扩展 |
| 大规模并行数值计算 | NEURON 的传统事件/ODE 仿真生态 | JAX 向量化网络运行 | 需要先进行规模、内存和数值一致性基准测试 |

### 2.1 BrainCell 中的核心对象

```text
Population  = 一类同构、批量化的 Cell，例如 28,615 个 Granule cells
EdgeSet     = 细胞到细胞的唯一 pre/post 对
Projection  = 将 EdgeSet 展开成每个解剖接触，并指定后突触池、权重、延迟
```

这恰好对应我们的导出格式：

```text
BSB cell_pairs (每个 contact 的全局 ID)
        ↓ 转为群内局部编号
local_pairs (每个 contact)
        ↓ 去重
edge_pairs (BrainCell EdgeSet)
        ↓ contact_to_edge 保留“一条边上的多个接触”
Projection + explicit_contacts + synapse_index
```

普通化学突触适合这一模型：先由 presynaptic spike 触发，再向目标 synapse pool 投递具有权重和延迟的事件。

但 gap junction 不适合普通 `Projection`：它没有“pre spike 事件”，而是在每个积分步读取两个目标位置的电压，并施加双向电流。因此不能用一个 GABA/AMPA projection 冒充 gap junction。

---

## 3. 本项目的论文复现完整流程

### 3.1 总体路线

```text
发布的 balanced.hdf5
        ↓ 读取并验证
可移植网络规格（population + contact + morphology metadata）
        ↓
BrainCell 的 Population + EdgeSet
        ↓
真实 Cell + 形态位置映射 + chemical Projection + relay
        ↓
基线刺激实验
        ↓
论文 Fig. 8 的 PF–PC 固定 LTD 状态扫描
        ↓
与论文记录和图表逐项比较
        ↓
（扩展目标）在线可塑性与 GoC gap junction
```

### 3.2 分阶段工作、验收标准和 BrainCell 覆盖范围

| 阶段 | 产物与验收标准 | BrainCell 当前能否完成 |
|---|---|---|
| 0. 固定原始参照 | 核对下载文件 MD5、尺寸、HDF5 结构 | 可以，已完成 |
| 1. 导入网络拓扑 | 7 个群、17 类连接的细胞数/接触数/边数均与 HDF5 一致 | 可以，已完成 |
| 2. 导入空间接触元数据 | 保留每 contact 的 BSB compartment 和 morphology 标识 | 可以，已完成保存；尚未转换为 BrainCell 点位 |
| 3. 可运行细胞和化学网络 | 真实细胞模型、AMPA/NMDA/GABA、Tsodyks–Markram 短时释放、权重、延迟、relay、记录器 | 可做，但需要适配和参数校准；未完成 |
| 4. 基线实验 | 复现基础输入下的放电/电压/突触电流统计 | 原理上可做；依赖阶段 3 |
| 5. 固定 PF–PC LTD 状态扫描 | 复现论文 Fig. 8：1200 pS 到 30% 的 10 档下降、每档 5 次 | 原理上可做；每个条件一轮独立仿真，不需要在线学习 |
| 6. 在线突触可塑性（可选扩展） | 在同一运行中按 pre/post/teaching 信号改每个接触的权重 | 需新增 BrainCell 突触机制；不属于本论文公开配置的必需复现内容 |
| 7. GoC 电耦合 | 对 `gap_goc` 的所有接触施加双端电压耦合电流 | 当前不能直接完成；需扩展 BrainCell runtime |

### 3.3 “复现成功”应采用的三层标准

1. **拓扑复现**：细胞数、细胞坐标、每类连接的 contact 表和 cell-pair 图一致。这个层次已经完成。
2. **功能复现**：相同输入下，BrainCell 网络的放电率、膜电位、突触电流和时序统计与 NEURON 结果在预先规定的误差范围内一致。这个层次尚未开始运行验证。
3. **机制复现**：同一细胞机制、Tsodyks–Markram 短时释放、受体动力学、gap junction，以及论文所采用的固定 PF–PC LTD 条件等价。这是最高标准，当前主要受 BrainCell gap junction 和短时释放机制缺口限制；不以“在线 PF 学习”作为论文的机制要求。

不要把第 1 层的成功误称为第 2 或第 3 层成功；反过来也不应因为第 3 层尚缺功能而否定第 1 层已取得的可复用成果。

---

## 4. 目前已完成的工作和遇到的困难

### 4.1 已完成

项目目录为本仓库根目录。

#### A. 固定并验证论文发布网络

- 下载并验证 `data/reference/balanced.hdf5`；
- 文件大小：129,879,036 bytes；
- 发布 MD5：`f0e17f89718c05756eb07a5da77b5c4c`；
- 没有修改论文仓库和 BrainCell 原始仓库。

#### B. 实现 BSB HDF5 → 可移植网络规格导出器

实现位于 `src/cerebellar_reproduction/`：

| 文件 | 职责 |
|---|---|
| `bsb_legacy.py` | 读取旧版 BSB HDF5 结构 |
| `export.py` | 写出 population / connectivity `.npz` 和 `manifest.json` |
| `validation.py` | 校验计数、索引范围、contact 到 edge 的映射 |
| `braincell_adapter.py` | 将规格 lower 为 BrainCell populations、edge sets、projections |
| `cli.py` | `cerebellar-export export` 与 `validate` 命令 |

每个 population `.npz` 保存：

- `source_ids`：论文网络的原始 simulation global IDs；
- `positions_um`：有放置坐标的细胞三维位置；entity 群不虚构位置。

每个 connectivity `.npz` 保存：

- `cell_pairs`：BSB 原始全局 `[pre, post]` contact 表；
- `local_pairs`：转换后的群内局部编号；
- `edge_pairs`：去重后的 cell-level 边；
- `contact_to_edge`：每个 contact 指向其所属 edge；
- `compartment_pairs`、`morphology_pairs`：原始 BSB 接触定位元数据。

#### C. 完成全网络结构验证与 BrainCell 拓扑构建

导出结果为：

- 7 个 populations；
- 17 个 connectivity sets；
- 31,683 个 cells/entities；
- 1,542,295 个解剖 contact；
- 694,859 条唯一 cell-level edges。

`scripts/build_braincell_topology.py` 已成功构建真实的 `braincell.Network` 声明，包含全部 7 个群和 17 个 `EdgeSet`。目前使用 `TopologyOnlyCell(pop_size=(N,))` 占位对象，因此能验证网络结构和尺寸，不能调用网络动力学。

自动测试当前为 `2 passed`，覆盖 BSB ID 读取和 BrainCell adapter 的基础行为。

### 4.2 已解决的关键困难：全局 ID 偏移

`/cells/type_maps` 是 HDF5 内部的类型映射组，不是每个连接可直接使用的全局 ID 表。它省略了 117 个 `mossy_fibers` entity，因此若把它错误当成连接表的 ID 映射，Basket、Stellate、Glomerulus、Granule 的 ID 会整体偏移 117。

最终采用 `/cells/placement/<cell_type>/identifiers = [start_global_id, count]` 作为权威的 simulation global-ID 区间。这样导出的 `local_pairs` 与所有原始 connection 表保持一致，后续 population mapping 才是正确的。

### 4.3 尚未解决的困难和原因

| 问题 | 为什么会出现 | 当前处理状态 |
|---|---|---|
| BSB compartment ID 不能直接作为 BrainCell synapse index | 两套形态离散化/点位编号体系不同 | 已保留原 ID；待建立逐细胞模型的映射表 |
| BrainCell 示例细胞不等于论文的 `dbbs_models` | 版本、形态、离子通道和参数可能不同 | 必须逐个比较；不能直接宣称动力学等价 |
| 原论文的短时递质释放 | 论文使用 Tsodyks–Markram 释放、囊泡循环和受体动力学 | BrainCell 现有 AMPA/NMDA/GABAa 不能自动保证等价；需实现或迁移 |
| Mossy fiber 与 glomerulus 是 relay | 它们不是普通有膜电位细胞 | 待设计显式 spike source/relay population |
| 1.54M contacts 的资源成本 | 真实接触、形态点位和多区室状态规模很大 | 先做小规模/单类连接验证，再进行全规模基准 |
| Gap junction | BrainCell `Junction` 源码标注为 placeholder，尚无 partner wiring | 保留数据，不在基线运行中激活 |
| 在线 STDP | 现有 `weight` 是 projection 参数；没有通用动态学习模块 | 不属于论文复现必需项；仅为后续研究的可选扩展 |
| 论文 NEURON 机制的完整参数来源 | 公开网络仓库引用外部 `dbbs_models` | 需取得/定位该模型源码或已编译机制，才能做严格参数迁移 |

---

## 5. 接下来要做的工作

以下顺序优先保证每一步都有可测试产物，而不是一次性拼装完整网络后才发现不一致。

### 5.1 下一阶段：建立“接触位置映射”报告和原型

目标：将 BSB 的 `compartment_pairs` 与 `morphology_pairs` 转换为 BrainCell 细胞中实际放置的 synapse pool index。

具体工作：

1. 检查论文形态、每种 BSB compartment 的含义和编号范围；
2. 检查 BrainCell 的 Granule、Golgi、Purkinje、Basket、Stellate 示例细胞的 morphology/region 定义；
3. 对每种目标连接定义可追踪的映射策略，例如 `PF → PC` 映射到 PC 的 `pf_targets` 区域；
4. 先选一类连接实现端到端验证：`local_pairs → explicit_contacts → synapse_index → BrainCell Projection`；
5. 输出每类连接的映射覆盖率、丢失接触数和降级策略。任何接触映射的近似都必须写入报告。

验收：选定连接的每个 BSB contact 都能映射到一个明确的 BrainCell target，或被明确列为不可映射而非静默丢弃。

### 5.2 建立可运行的化学突触基线网络

1. 选择并审计五类 BrainCell 细胞模型；记录与 `dbbs_models` 的形态/机制差异；
2. 在每个目标区域放置 AMPA、NMDA、GABAa 突触池，并实现/验证 Tsodyks–Markram 短时释放状态；
3. 为 16 类非 gap 连接建立适配：`mossy_to_glomerulus` 是 relay branching，不应误建成化学突触；其余连接按论文受体机制建立 `ProjectionBinding`，且使用导入的接触而不是概率重连；
4. 为 mossy_fibers/glomerulus 实现确定性的 spike relay；
5. 加入论文的 `dt`、温度（若 BrainCell 模型支持）、权重、延迟、背景输入和记录器；
6. 从小型子网络开始，随后评估完整网络的 JAX 编译时间、内存与运行时间。

验收：在固定随机种子和固定输入下，网络可重复运行并输出每个群的 spike/电压/突触记录；所有 16 类非 gap 连接的 contact 数不变。

### 5.3 复现论文实验

1. 把论文基础刺激协议转为 BrainCell 输入；
2. 建立与论文 recorder 等价的数据输出格式或转换器；
3. 先比较单细胞响应，再比较局部回路，最后比较全网络统计；
4. 首先实施论文 Fig. 8 的 `PF–PC` 固定 LTD 状态扫描（1200 pS 至 30%、10 档、每档 5 次）；PF–MLI 扫描仅在确定其对应的源码实验后作为扩展；
5. 运行仓库中相应的分析逻辑或重写等价分析，比较放电率调制和图表趋势；
6. 为每一项比较预先写明允许误差、随机种子与未覆盖机制。

### 5.4 后续扩展目标（不阻塞基线复现）

#### Gap junction

需要在 BrainCell runtime 中实现一个双端耦合机制：每个 contact 持有 pre/post population、cell index、形态点位和 conductance，并在每一步把 `g_gap × (V_other - V_self)` 同时加入两端电流。它需要不同于 event projection 的图结构和数值求解验证。

#### 在线突触可塑性（论文复现不需要）

需要实现每个 contact 独立的可变权重状态，例如：

```text
pre spike + post spike / teaching signal / calcium state
                  ↓
plasticity update rule
                  ↓
contact.weight(t + dt)
                  ↓
后续 AMPA/NMDA 事件使用新权重
```

这不是复现本文公开实验配置所需的工作。只有项目将来研究“训练过程中在线改变 PF 权重”时才需要它；届时仍须从相应机制中确认精确公式和参数，不能以一个通用 STDP 规则替代后就称为论文复现。

---

## 常用命令与文件

在 `braincell-cerebellar-reproduction` 目录中：

```bash
# 从发布地址获取并校验原始网络
scripts/fetch_reference.py data/reference/balanced.hdf5

# 导出可移植网络规格
cerebellar-export export data/reference/balanced.hdf5 data/export/balanced --verify-published

# 验证导出结果
cerebellar-export validate data/export/balanced

# 构建 BrainCell 的结构网络（非可运行模型）
python scripts/build_braincell_topology.py data/export/balanced
```

关键文件：

- `data/reference/balanced.hdf5`：发布的 BSB 网络金标准；
- `data/export/balanced/manifest.json`：导出内容、计数和源文件校验信息；
- `data/export/balanced/populations/*.npz`：群 ID 与坐标；
- `data/export/balanced/connectivity/*.npz`：边、接触、compartment/morphology 信息；
- `scripts/build_braincell_topology.py`：把当前规格构建为 BrainCell `Network` 的最小示例；
- `src/cerebellar_reproduction/braincell_adapter.py`：未来加入真实细胞和 Projection 时要继续扩展的适配层。
