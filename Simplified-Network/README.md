# 256 个颗粒细胞和 12 个 Golgi 的子网

这是此前用于本机试跑的同一组细胞。从 De Schepper 等（2022）的 `balanced.hdf5` 中，按原始 ID 排序后，用 NumPy 随机种子 318 依次无放回选取 256 个颗粒细胞和 12 个 Golgi。保留原有 117 条苔藓纤维和 2,336 个小球，以及两端都在子网内的接触。原始 ID、三维位置和形态接触位置保存在 `data/spec/`。 [论文](https://doi.org/10.1038/s42003-022-04213-y) · [发布数据](https://doi.org/10.5281/zenodo.7230288)

颗粒细胞和 Golgi 分别保留 114 和 317 个计算房室。化学通路包括小球到颗粒细胞和 Golgi、升支和平行纤维到 Golgi、Golgi 到颗粒细胞以及 Golgi 间抑制，共 **7,521 条受体事件路由**。每条化学接触采用现有 BrainCell 的独立释放和受体状态。裁切后的网络只包含选中细胞之间的反馈。

**Junction 状态。** 接触表保留了 **74 条 Golgi 间缝隙连接**的解剖端点。运行时未装配 `Junction` 电耦合机制，因此这次前向仿真没有计算缝隙连接电流。

## 运行

进入本目录。第一次使用时，在 Python 环境中安装上一级仓库的 BrainCell 运行依赖，再执行前向脚本。

```bash
python -m pip install -e '..[braincell-runtime]'
JAX_PLATFORMS=cpu python run_forward.py
```

原工作区可以直接使用已有的虚拟环境。

```bash
JAX_PLATFORMS=cpu PYTHONPATH=../src \
  ../.venv-braincell-0.2.0/bin/python run_forward.py
```

本目录包含运行所需的接线、两类形态映射数据和细胞模板。前向脚本不读取全量 `balanced.hdf5`。默认使用同步苔藓纤维输入，模拟 18 毫秒，步长 0.05 毫秒。可用 `--duration-ms` 和 `--dt-ms` 修改时长与步长。可用 `--output` 指定结果目录。

如果把本目录单独拷贝到其他位置，用 `python -m pip install -r requirements.txt` 安装固定版本的项目运行代码和 BrainCell，再运行同一条前向命令。

如需重新提取，传入发布数据的原文件。

```bash
python extract.py ../data/reference/balanced.hdf5
```

## 输出

- [前向仿真摘要](results/forward_summary.json)包含事件计数、各通路的峰值电导、胞体电压范围和耗时。
- [前向仿真数组](results/forward_traces.npz)包含时间轴、两类细胞的胞体电压、胞体发放时刻和原始细胞 ID。电压矩阵的轴为时间步和细胞，列顺序由相关的 `source_ids` 数组给出。

## 本机验证

BrainCell 0.2.0 在 WSL CPU 上完成了 360 步积分。268 个细胞的胞体电压记录均为有限值，九种按通路记录的受体电导峰值均大于零。256 个颗粒细胞各发放一次，12 个 Golgi 中有 11 个发放，共 19 次胞体发放。整个前向进程耗时约 11 分 9 秒，包含接线、初始化、编译和写盘，峰值常驻内存约 5.38 GiB。
