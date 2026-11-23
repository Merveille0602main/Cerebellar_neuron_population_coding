# 论文参数与 BrainCell ModelDB 参数的来源审计

## 1. 结论

BrainCell `examples/neuron_compare/Cerebellum_mod` 中的文件不能整体视为 De Schepper et al. 2022 论文的最终参数表。它们主要是为 BrainCell/NEURON 对照保存的独立 ModelDB 模型资源；论文网络则通过 BSB 配置加载论文时代的 `dbbs_models`，再由 `dbbs_models` 把逻辑名称解析到 `dbbs_mod_collection` 的机制，并可能在实例化后覆盖 MOD 默认值。

因此“同名细胞”或“某个细胞目录中存在某个 MOD 文件”都不足以证明论文使用了该文件及其默认参数。复现时采用的权威链是：

```text
balanced.json 中的连接/模型名称
        ↓
论文时代 dbbs_models v1.1.2 的名称映射和属性覆盖
        ↓
dbbs_mod_collection 对应机制的方程与 MOD 默认值
        ↓
NEURON 中实例化后的最终参数和参考轨迹
        ↓
BrainCell 移植参数与数值对照
```

`balanced.hdf5` 是已发布网络的细胞、位置、边和 contact 的金标准，但它不保存所有细胞膜和突触动力学参数；动力学仍必须沿上面的软件来源链恢复。

## 2. 为什么会有出入

### 2.1 资源来源不同

论文配置中的细胞模型是 `dbbs_models.GranuleCell`、`GolgiCell`、`PurkinjeCell`、`BasketCell` 和 `StellateCell`。BrainCell 仓库中的 `Cerebellum_mod/GrC`、`GoC`、`PC`、`BC`、`SC` 则汇集了各个 ModelDB 细胞模型的形态、通道和突触源文件。这两者有重叠，但不是同一个封闭版本的软件包。

例如，BrainCell standalone GoC 资源中有：

- `GOLGI_Ampa_mossy_det_vi.mod`：`gmax=2000 pS`、`U=0.2`；
- `PC_NMDA_NR2B.mod`：`gmax=5000 pS`、`U=0.2`，使用另一套 8 状态 NMDA 方程。

但论文的 `dbbs_models.GolgiCell` 并不把 `AMPA_MF`/`NMDA` 映射到这两份文件，而是映射到 `AMPA.granule`/`NMDA.granule`。所以论文在这条通路的有效值是 1200/18800 pS、`U=0.42`，方程也不同。

### 2.2 MOD 默认值不是最终实例值

MOD 文件定义机制和默认值，但 Python/HOC 适配层可以在每次创建 point process 后重新赋值。论文时代的 `dbbs_models.GranuleCell` 对 AMPA 和 NMDA 显式设置：

```text
gmax=1200/18800 pS
U=0.43
tau_rec=8 ms
tau_facil=5 ms
tau_1=1 ms
```

对应的两份 MOD 文件自身写的是 `U=0.42`。如果只读取 MOD，就会漏掉 0.43 这个最终有效覆盖。相反，`dbbs_models.GolgiCell` 对同一套机制没有这组属性覆盖，因此 GoC 保留 MOD 的 `U=0.42`。

### 2.3 版本、形态和数值实现也会造成差异

即使离子通道方程来自同一模型，以下差异仍可能改变整细胞结果：

- 模型仓库提交和依赖版本；
- 使用的 ASC/SWC 形态文件及坐标变换；
- branch 到 section、compartment 到 control volume 的离散方式；
- 通道密度的区域分配；
- 温度和 Q10；
- 初始电压、离子浓度和 reversal potential；
- NEURON 的事件边界语义与 BrainCell 的离散时间更新顺序；
- NEURON 8/9 与 BrainCell/JAX 数值积分器的实现差异。

这解释了为什么形态、单机制和整细胞必须分别验证，不能因为文件名一致就直接认定等价。

还要进一步区分两种“形态”：BSB 在网络重建阶段用来求 contact 的 scaffold
morphology，和 `dbbs_models` 在 NEURON 运行阶段实例化的 cell morphology。
它们在原流程中也可能来自不同资源，坐标不一定逐点相同；能否对应必须通过
branch topology、区域标签、compartment 顺序或显式位置映射证明，不能只比较
文件名。当前 StellateCell 就体现了这一问题：共享 BrainCell `SC.asc` 与
paper-era `dbbs_models/stellate.asc` 的哈希不同，而后者也不能通过当前
exact-geometry 算法映射发布 HDF5 contacts。该问题暂不影响当前 GrC/GoC
颗粒层路径，但进入 MLI 通路前必须单独解决。

## 3. 当前已确认的参数来源

| 论文通路 | `balanced.json` 名称 | `dbbs_models` 最终解析 | 关键最终值 | BrainCell 处理 |
|---|---|---|---|---|
| glomerulus→GrC | `AMPA` | `AMPA.granule`，Granule adapter 覆盖 | `gmax=1200 pS`, `U=0.43` | `GrCAMPA2020` 默认 0.43 |
| glomerulus→GrC | `NMDA` | `NMDA.granule`，Granule adapter 覆盖 | `gmax=18800 pS`, `U=0.43` | `GrCNMDA2020` 默认 0.43 |
| glomerulus→GoC | `AMPA_MF` | `AMPA.granule`，无 GoC 属性覆盖 | `gmax=1200 pS`, `U=0.42` | `GoCAMPA_MF2020` 显式默认 0.42 |
| glomerulus→GoC | `NMDA` | `NMDA.granule`，无 GoC 属性覆盖 | `gmax=18800 pS`, `U=0.42` | `GoCNMDA2020` 显式默认 0.42 |
| GoC→GrC | `GABA` | `GABA.granule`，Granule adapter 覆盖 | `U=0.35`，alpha1/alpha6 两套 8 状态受体 | `GrCGABA2020` 默认 0.35 |

共同的 AMPA/NMDA 释放参数除 `U` 外为 `tau_rec=8 ms`、`tau_facil=5 ms`、`tau_1=1 ms`、`Cdur=0.3 ms`、`Tmax=1 mM`。论文仿真环境为 `dt=0.05 ms`、`32 °C`。

### 3.1 细胞模板审计：并不是所有“看起来不同”的值都真的不同

新增的机器审计 `data/export/dbbs_model_parameter_audit.json` 直接读取
`dbbs_models` v1.1.2 的两个 cell model 源文件，并与当前 BrainCell
模板比较，得到：

- GoC：ASC 文件 SHA-256 完全相同；25 个分区通道密度/通透率逐值完全相同，最大绝对差为 0；
- GrC：当前 soma 和 dendrite 使用的 12 个通道密度逐值完全相同，最大绝对差为 0；
- GrC 结构仍不相同：论文 `dbbs_models.GranuleCell` 程序化构建合成 soma、4 条 dendrite、hillock、AIS、ascending axon 和两条 PF chain；当前可运行原型采用 BrainCell standalone `GrC.asc` 的 soma+dendrite 模板，没有把 axonal load 加入当前 `grc_ma2020` cell。

所以 GoC 当前约 0.58 mV/3 ms 的无输入基线偏差不能再归咎于“25 个密度参数抄错”或“形态文件不同”。其排查范围已经缩小到机制状态初始化、电流方程、离子池、更新顺序、记录时刻和积分器。相反，GrC 确实还存在几何与完整结构差异；这部分只能称为“真实 BrainCell morphology 上的已发布 contact 可运行迁移”，不能称为论文整细胞完全等价。

当前尚未完成其余化学通路以及 Purkinje/Basket/Stellate 全部膜机制的同等级来源审计，因此这些部分仍标为“待核对”，不会直接采用 ModelDB 目录默认值并宣称论文等价。

## 4. 本项目如何处理差异

每新增一条可运行通路，按以下门禁执行：

1. 从 `balanced.json` 确认 connection model、postsynaptic cell model、逻辑突触名、温度和步长；
2. 从 `balanced.hdf5` 读取真实 edge/contact 和目标 compartment；
3. 固定到论文时代 `dbbs_models` 版本，查清逻辑名解析和实例属性覆盖；
4. 固定对应 `dbbs_mod_collection` 源文件及提交，记录 SHA-256；
5. 在 NEURON 中显式设置所有适配层覆盖，生成单机制或整细胞参考轨迹；
6. 在复现仓库实现 BrainCell adapter，不为了论文复现继续污染共享 BrainCell 模型默认值；
7. 比较 transmitter、受体状态、conductance、current 和 cell voltage，而不只检查“能跑”；
8. 参数来源尚不明确时保持未完成状态，不用最相近文件猜值。

## 5. 本次发现与修正

2026-08-08 的逐层审计发现：此前 `GrCAMPA2020` 和 `GrCNMDA2020` 错把 MOD 默认 `U=0.42` 当成 GrC 最终值，遗漏了 `dbbs_models.GranuleCell` 的 `U=0.43` 覆盖。已完成以下修正：

- 两个 GrC 机制默认值由 0.42 改为 0.43；
- GoC 两个语义别名显式保持 0.42，防止继承 GrC 默认值后被连带改变；
- NEURON GrC 参考脚本显式设置 `synapse.U=0.43`；
- 新增测试，分别锁定 GrC=0.43、GoC=0.42；
- 重新生成 AMPA/NMDA 的单脉冲、三脉冲和 sensory-burst 六组对照。

六组对照均通过峰值相对误差与 normalized RMSE 小于 1% 的门限。当前结果如下：

| 条件 | NEURON 电导峰值 | BrainCell 电导峰值 | 峰值相对误差 | normalized RMSE |
|---|---:|---:|---:|---:|
| AMPA 单脉冲 | 393.831817 pS | 392.888993 pS | 0.239398% | 0.155177% |
| AMPA 三脉冲 | 393.831817 pS | 392.888993 pS | 0.239398% | 0.167354% |
| AMPA sensory burst | 430.732512 pS | 430.618413 pS | 0.026490% | 0.115991% |
| NMDA 单脉冲 | 262.657926 pS | 262.658170 pS | 0.000093% | 0.000167% |
| NMDA 三脉冲 | 390.045764 pS | 390.048226 pS | 0.000631% | 0.000523% |
| NMDA sensory burst | 460.669422 pS | 460.671406 pS | 0.000431% | 0.000820% |

这次修正仅改变复现仓库中的论文专用 adapter 和参考脚本，没有更改共享 BrainCell 仓库的 ModelDB 源文件或通用细胞默认值。

## 6. 可复核来源

- [论文网络配置](https://github.com/Helveg/deschepper-etal-2022-source/blob/897eb74e942ce64a90b67090cae0da7e467978f8/configs/balanced.json)
- paper-era `dbbs_models`：v1.1.2，提交 `5f920d5`
- GABA 对应 `dbbs_mod_collection`：提交 `8d21c4a4f888f19a6531224f1eb08d6c833c44e5`
- GABA 源 SHA-256：`f99a72c520860e01d77ca2d34b4f7bade4796a65b6abcb5c6f88f5d79695bfae`
- 当前 NEURON 9.0.1 编译输入及 SHA-256：`data/reference/neuron_goc_mechanisms/manifest.json`
- 可重复参数审计脚本：`scripts/audit_dbbs_model_parameters.py`
- 参数审计报告：`data/export/dbbs_model_parameter_audit.json`
- AMPA/NMDA 详细移植：`SYNAPSE_PORT.md`
- glomerulus→GoC 名称解析：`GLOMERULUS_GOLGI.md`
- GoC→GrC GABA：`GABA_CHEMICAL_LOOP.md`
