# 论文完整颗粒细胞模板

> BrainCell 0.2.0 的颗粒层模型以历史 Arborize 初始化后的 114 个房室和发布形态的 0.15 微米平行纤维直径为准。下文记录较早的 125 房室移植。新版用法见 [颗粒层回路](GRANULAR_LAYER_0_2.md)。

## 1. 为什么增加这个模板

论文运行时使用的颗粒细胞不是当前 BrainCell 网络原型中使用的
`GrC.asc` soma+dendrite 简化模板，而是 `dbbs_models` v1.1.2（提交
`5f920d5`）中的程序化 `GranuleCell`。后者还包含 axon hillock、AIS、
126 µm ascending axon 和总长 2000 µm 的 parallel fiber。

参数审计已经证明两者的 soma/dendrite 12 个通道密度一致，但缺少轴突负载
仍会改变整细胞膜电位和放电传播。因此，固定电压下突触机制通过 NEURON
对照并不等价于整细胞已经与论文相同。

本阶段在复现仓库新增
`src/cerebellar_reproduction/paper_granule_cell.py`，没有修改共同维护的
BrainCell 仓库。它复用 BrainCell 已有 Ma et al. 2020 通道实现和经过审计的
六分区参数表，只替换为论文的程序化结构、离散化和 population 支持。

## 2. 历史源码的真实结构

论文结构按“一个 NEURON section 对应一个 BrainCell branch”迁移：

| 分区 | NEURON section / BrainCell branch 数 | section 内离散 | 长度和直径 |
|---|---:|---:|---|
| soma | 1 | `nseg=1` | 5.62232 µm × 5.8 µm |
| dendrite | 4 | 每 branch 1 CV | 设置 `L=15 µm`、直径 0.75 µm |
| axon hillock | 1 | `nseg=1` | 1 µm × 1.5 µm |
| AIS | 1 | `nseg=1` | 10 µm × 0.7 µm |
| ascending axon | 1 | `nseg=18` | 126 µm × 0.3 µm |
| parallel fiber | 100 | 每 section 1 CV | 每段 20 µm × 0.3 µm |

总计是 **108 branches、125 CVs**。100 个 PF section 不是一条串联链，而是
偶数、奇数 section 分别构成两个反向的 50 段分支，因此每个 PF arm 为
1000 µm。

历史 dendrite 构造器先写 `L=15`，但随后只生成 `j=0..9` 的十个 pt3d 点，
坐标覆盖 0 至 13.5 µm。NEURON pt3d 会以点路径重算 section 长度，因此本文
在几何层保留实际的 **13.5 µm**，并单独记录原始 15 µm 设置，避免把源码的
off-by-one 行为悄悄“修正”掉。

## 3. 与 BrainCell 现有 `GrCFull` 示例的差异

检查发现现有实验示例并不是论文结构：

- 它从 standalone `GrC.asc` 载入 soma/dendrite，而不是论文程序化结构；
- 它创建 4 个、每个 7 µm 的 AA branches，轴突长度只有 28 µm，却把 PF
  起点坐标放到论文的 126 µm 末端；
- 它创建左右各 142 个、每段 7 µm 的 PF branches；
- 论文实际是 1 个 126 µm AA section（`nseg=18`）和总计 100 个 20 µm PF
  sections。

因此这个示例不能作为论文整细胞参考。当前没有改动或删除该共享示例，
而是在复现仓库建立 `PaperGranuleCell`，以免论文专用判断未经共同维护者
评审就进入 BrainCell 主仓库。

## 4. 已实现内容

`PaperGranuleCell` 当前完成：

1. 精确重建论文六个结构分区和 PF 左右臂拓扑；
2. 保留 AA 为一个 branch，并分配 18 个 BrainCell CV；
3. 总 branch/CV 数运行时断言；
4. 支持 `pop_size` 和 `name`，可用于后续 BrainCell network population；
5. 复用已审计的 soma、dendrite、hillock、AIS、AA、PF 通道参数；
6. 保留四个 dendrite 的稳定 ordinal，使发布 HDF5 的 synthetic
   GranuleCell compartment 能继续使用语义位置映射。

自动测试位于 `tests/test_paper_granule_cell.py`，检查 branch 数、长度、PF
左右链父子关系、region 集合和 population axis。当前结果为 2 passed。

运行命令：

```bash
PYTHONPATH=/home/merveille/copneuron/braincell:src \
  .venv/bin/python -m pytest tests/test_paper_granule_cell.py -q
```

## 5. 尚未宣称等价的部分

结构和参数一致仍不足以宣称动力学等价。进入发布网络之前还需要：

1. 用相同程序化结构建立 NEURON reference cell；
2. 对无刺激静息、胞体电流注入和树突单突触分别比较 voltage trace；
3. 检查 NEURON 初始化、离子反转电位、钙状态及 BrainCell staggered solver
   的离散差异；
4. 检查 spike 阈值与 reset/事件输出语义，而不只比较峰值电压；
5. 评估 125 CV/GrC 的资源成本。发布网络有 88,158 个 GrC，不能未经基准
   就把当前 5-CV soma+dendrite 原型整体替换为完整模板。

所以当前采用双层策略：完整模板是整细胞等价验证的金标准；现有简化模板
继续用于网络接线和资源原型。只有对照误差和资源边界明确后，才决定完整
网络采用完整模型、经验证的降阶模型，还是分层实验方案。
