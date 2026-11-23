# 256 个颗粒细胞和 12 个 Golgi 的子网

这是此前用于本机试跑的同一组细胞。从 De Schepper 等（2022）的 `balanced.hdf5` 中，按原始 ID 排序后，用 NumPy 随机种子 318 依次无放回选取 256 个颗粒细胞和 12 个 Golgi。保留原有 117 条苔藓纤维和 2,336 个小球，以及两端都在子网内的接触。原始 ID、三维位置和形态接触位置保存在 `data/spec/`。 [论文](https://doi.org/10.1038/s42003-022-04213-y) · [发布数据](https://doi.org/10.5281/zenodo.7230288)

颗粒细胞和 Golgi 分别保留 114 和 317 个计算房室。化学通路包括小球到颗粒细胞和 Golgi、升支和平行纤维到 Golgi、Golgi 到颗粒细胞以及 Golgi 间抑制，共 **7,521 条受体事件路由**。每条化学接触采用现有 BrainCell 的独立释放和受体状态。裁切后的网络只包含选中细胞之间的反馈。

## Junction 电耦合

`gap_goc.npz` 的 **74 行有向记录**来自原模型的镜像处理。按原始细胞 ID 和形态位置配对后得到 **37 个双端接触**，其中 25 个跨两个 Golgi，12 个连接同一 Golgi 的两个不同形态位置。每个物理接触统一采用 DBBS 并行缝隙连接机制的默认电导 **1 nS**。 [原模型配置](https://github.com/Helveg/deschepper-etal-2022-source/blob/897eb74e942ce64a90b67090cae0da7e467978f8/configs/balanced.json) · [DBBS 并行 gap junction 源码](https://github.com/dbbs-lab/dbbs-mod-collection/blob/f896c16f963042331a481a07a03240e4cc059f2c/dbbs_mod_collection/mod/glia__dbbs_mod_collection__gap_junction__parallel.mod)

BrainCell 0.2.0 的公开 `mech.Junction` 是参数声明。子网内的 [Junction 实现](junction.py)使用该版本的点机制布局，将接触两端绑定到形态映射得到的电节点。每个积分步从同一步起始的端点电压计算一对等量、反向的电流，并将其作为点电流送入现有膜求解器。步内保持电流恒定，因此电耦合采用显式步进。离散化后若两个端点落在同一电节点，该接触的电压差和净电流恒为零。 [BrainCell 机制接口](https://brainx.chaobrain.com/braincell/apis/braincell.mech.html)

## 运行

进入本目录。第一次使用时，用 `requirements.txt` 安装和原基线一致的项目运行代码及 BrainCell 0.2.0。目录内自带接线、两类形态映射数据和细胞模板，可以单独拷贝使用。前向脚本只读取子网数据。

```bash
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu python run_forward.py
python compare_forward.py
```

默认启用 Junction，使用和已保存基线相同的同步苔藓纤维输入，模拟 18 毫秒，步长 0.05 毫秒。可用 `--duration-ms` 和 `--dt-ms` 修改时长与步长。可用 `--output` 指定结果目录。重新生成无电耦合基线时使用单独的输出目录。

```bash
JAX_PLATFORMS=cpu python run_forward.py --no-junctions
python compare_forward.py --baseline results/without_junction
```

如需重新提取，传入发布数据的原文件。

```bash
python extract.py ../data/reference/balanced.hdf5
```

## 输出

- 无耦合基线保存在 [摘要](results/forward_summary.json) 和 [胞体电压及发放数据](results/forward_traces.npz)。
- 启用 Junction 的前向结果保存在 [摘要](results/with_junction/forward_summary.json) 和 [胞体电压、发放及 Junction 电流数据](results/with_junction/forward_traces.npz)。`junction_current_pA` 按时间步、物理接触和端点排列，端点的原始细胞 ID 和形态位置在 `junction_contact_endpoints` 中。
- [配对比较](results/junction_comparison.json)核对两次运行的刺激表、输入事件数、时间轴、细胞 ID、化学接线和运行版本，汇总胞体电压差、发放数和发放时刻。发放时刻的配对统计只使用前后发放次数相同的细胞。

## 本机验证

原无耦合基线在 WSL CPU 上完成了 360 步积分。268 个细胞的胞体电压记录均为有限值，九种按通路记录的受体电导峰值均大于零。256 个颗粒细胞各发放一次，12 个 Golgi 中有 11 个发放，共 19 次胞体发放。整个前向进程耗时约 11 分 9 秒，峰值常驻内存约 5.38 GiB。

启用 Junction 后，在相同的 18 毫秒刺激和 0.05 毫秒步长下完成了 360 步。两组均收到 2,340 条苔藓纤维事件和 46,720 条小球事件。37 个接触均产生非零电流，单端最大绝对电流为 **64.47 pA**，所有时间步的配对电流和均为零。耦合版进程耗时约 10 分 10 秒，峰值常驻内存约 5.47 GiB。

| 群体 | 无耦合 → 有耦合胞体发放数 | 最大配对发放时移 | 胞体电压均方根差 | 同时刻最大胞体电压差 |
| --- | ---: | ---: | ---: | ---: |
| 颗粒细胞 | 256 → 256 | 0.00 毫秒 | 0.20 mV | 11.75 mV |
| Golgi | 19 → 19 | 1.65 毫秒 | 6.84 mV | 75.47 mV |

表中的电压差按同一细胞、同一时间步计算。因此动作电位的时间偏移也会反映在瞬时电压差上。发放时移按同一细胞的发放顺序配对，本次所有细胞的发放次数一致。 [对比数据](results/junction_comparison.json)

运行 `python -m unittest -v test_junction` 可验证镜像配对、并联接触，以及两个被动细胞之间的双向电流及膜电压响应。
