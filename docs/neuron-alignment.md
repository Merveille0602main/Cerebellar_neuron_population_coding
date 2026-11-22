# 与原始 BSB / NEURON 网络的对齐情况

原始参考为 De Schepper 等（2022）的 `balanced.hdf5` 与历史 BSB、Arborize 和 DBBS 模型。BrainCell 0.2.0 保留了所选颗粒层通路的发布细胞 ID、连接记录与形态位置。全柱结构见 [交互式结构图](network-structure.html)。 [原论文](https://doi.org/10.1038/s42003-022-04213-y) · [网络配置](https://github.com/Helveg/deschepper-etal-2022-source/blob/897eb74e942ce64a90b67090cae0da7e467978f8/configs/balanced.json)

## 树突和计算房室

| 细胞 | 形态分段 | 历史 NEURON 分段 | 当前 BrainCell 分段 |
| --- | --- | --- | --- |
| 颗粒细胞 | 108 条 section / branch，含 4 条树突、1 条升支和 100 段平行纤维 | Arborize 1.1.1 重算后共 114 个 segment。四条树突各一个，升支七个。原细胞构造器初设升支为 18 个，随后被覆盖 | 共 114 个控制体积，四条树突各一个，升支七个 |
| Golgi | 227 条 section / branch，含 151 条树突、75 条轴突与胞体 | 用同一份 ASC 经 NEURON Import3d 导入，按 Arborize 1.1.1 规则复算得 317 个 segment，其中树突 207、轴突 109、胞体 1 | 共 317 个控制体积，按 branch 长度应用同一套分段数规则 |

BSB 用于记录接触位置的形态线段是相邻形态采样点之间的区间。它们与 NEURON 的数值 segment 和 BrainCell 的控制体积属于不同的离散。当前代码把 BSB 线段中点换成目标 branch 上的归一化位置，再落到 BrainCell 的控制体积。普通 NEURON 路由的接收器经 Arborize 设在目标 section 的中点。即使数值段总数一致，当同一树突被划成多个控制体积时，两端的点电流仍可能注入不同位置。膜面积、分叉处轴向电导、求解与事件调度也需要配对电压和电流轨迹验证。 [接触位置映射](../src/cerebellar_reproduction/morphology_mapping.py) · [Arborize 1.1.1](https://pypi.org/project/arborize/1.1.1/)

## 突触实例与释放

默认采用 BrainCell 的 `contact` 模式。不同接触各自维护释放历史及受体状态，电流进入目标控制体积。在历史 NEURON 普通路由中，多个 NetCon 可以共用同一 section 的同类型受体，释放状态存于各自的 NetCon。BrainCell 可选的 `section` 模式会进一步合并发布接触，在同一路径内共享释放历史和受体。此设置用于近似试验。密集输入时，受体饱和、去敏化和短时可塑性可导致不同的响应。 [BrainCell 路由](../src/cerebellar_reproduction/granular_layer.py) · [NEURON NetCon](https://www.neuronsimulator.org/en/latest/progref/modelspec/programmatic/network/netcon.html#NetCon.weight)

### AMPA_AA

2022 年网络的配置为升支同时请求 `AMPA_AA` 和 `NMDA`。已核对的历史 Golgi 模板暴露了 `AMPA_MF`、`AMPA_PF` 和 `NMDA`，未找到该运行的 `AMPA_AA` 最终绑定。当前使用 **2022 年网络论文所引用的 Masoli 等（2020）** 的 `syn_AA2GoC_det` 作为替代。该实现使用 `Golgi_PF_syn` 的动力学，将释放参数设为 `tau_1 = 30 ms`、`tau_rec = 35.1 ms`、`tau_facil = 54 ms`、`U = 0.4`，最大电导为 `1200 pS`。BrainCell 使用发布网络的接触坐标与原网络路由，独立保留发布配置中的升支 NMDA。 [Masoli 论文](https://doi.org/10.1371/journal.pcbi.1007937) · [AA 定义](https://github.com/ModelDBRepository/266806/blob/2dedfcd3d6ea0163cee913be6baff2608bd3449b/Morphology_1/Synapses.py#L65-L76)

### Golgi → Golgi 化学抑制

发布配置指定由 Golgi 轴突连接 Golgi 基底树突，论文指定 GABA-A α1。历史 Golgi 模板里未找到 `GABA` 的实际机制绑定。本版沿用已移植的 DBBS GABA-A α1 八状态动力学与短时释放、扩散历程，使用论文报告的约 `3.2 nS` 作为名义最大电导。反转电位 `-65 mV` 和释放、扩散参数继承自 DBBS 颗粒细胞 GABA 模板。这些参数属于替代假设，论文的 Golgi 抑制电流、静息放电率及其群体效应还需要 NEURON 或实验轨迹对照。Golgi → 颗粒细胞通路继续保留 α1 和 α6。 [原论文](https://doi.org/10.1038/s42003-022-04213-y) · [DBBS GABA 原始机制](https://github.com/dbbs-lab/dbbs-mod-collection/blob/8d21c4a4f888f19a6531224f1eb08d6c833c44e5/dbbs_mod_collection/mod/glia__dbbs_mod_collection__GABA__granule.mod) · [BrainCell 移植](../src/cerebellar_reproduction/paper_synapses.py)

### Gap junction

`gap_goc` 保留 **2,082 条**发布接触记录，包括两端的 Golgi 细胞 ID 及基底树突位置，其中 114 条记录的两端属于同一细胞。全量形态检查会解析两端。当前运行器未接入双端电流与电压耦合，因此这些记录不会生成运行时电连接。原论文的 Golgi 电同步与颗粒层振荡不能据此复现。 [发布数据](https://doi.org/10.5281/zenodo.7230288) · [原论文图 3 和图 4](https://doi.org/10.1038/s42003-022-04213-y)

## 其余范围

- 只运行颗粒层回路。原论文中的浦肯野、篮状和星状细胞及其投射在结构图中可查阅。
- 苔藓纤维到小球按发布配置做事件中继。当前预先展开已知发放时间表，输入背景噪声可生成同规则的新序列。原始随机实现与全网逐时刻轨迹未复现。
- 神经元输出以对应轴突 section 中点的电压阈值产生事件。小回路用 0.05 毫秒步长做诊断，论文用 0.025 毫秒步长运行网络。整细胞发放时间、网络电压轨迹与论文的群体统计仍待同条件对照。
