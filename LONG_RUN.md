# BrainCell 长时分块运行与 10 秒资源基准

## 1. 为什么不能循环调用 `Network.run()`

旧的 `Network.run()` 每次调用都会新建事件 delivery ring buffer。细胞膜、离子通道和突触状态虽然继续存在，但块末尾已经入队、尚未送达的延迟事件会丢失。因此简单写成十次 `run(duration=1*s)` 并不等价于一次 `run(duration=10*s)`。

BrainCell 现新增 `Network.run_chunks(...)`：所有块共享一个 delivery state，并连续保存细胞、突触、绝对时间和 ring cursor。调用方每次只接收一个 `NetworkRunResult`，处理或落盘后再请求下一块，因此不需要让整个时间序列同时驻留内存。

核心回归测试特意把 presynaptic spike 放在第一块最后一步；该事件在第二块第一步正确到达，拼接后的时间、电导和 spike 与单次运行完全一致。

## 2. 输出内存估算

`estimate_network_output_memory(...)` 根据初始化后的真实 probe/spike shape 和 dtype 计算返回数组下限。它不包含 JAX 编译缓存、runtime state、临时数组和 allocator slack，因此是输出量估算，不是进程 RSS 预测。

117 MF、8 glomerulus、32 GrC、`dt=0.05 ms`、10 秒、population-level spike recording 的结果：

| 输出 | shape/step | dtype | 10 秒大小 |
|---|---:|---:|---:|
| time | scalar | float32 | 0.8 MB |
| MF spikes | 117 | float32 | 93.6 MB |
| glomerulus spikes | 8 | float32 | 6.4 MB |
| GrC soma voltage | 32 | float32 | 25.6 MB |
| GrC AMPA conductance | 4 × 32 | float32 | 102.4 MB |
| GrC NMDA conductance | 4 × 32 | float32 | 102.4 MB |
| GrC population spikes | 32 | bool | 6.4 MB |

合计 `337.6 MB`。使用 1 秒块时，每块输出上限为 `33.76 MB`。

## 3. 完整 10 秒 CPU 基准

命令：

```bash
PYTHONPATH=src:/home/merveille/copneuron/braincell \
  .venv/bin/python scripts/benchmark_mossy_pathway.py \
  --duration-ms 10000 --chunk-duration-ms 1000
```

环境：AMD Ryzen 9 8940HX（16 cores/32 threads）、CPU JAX、Python 3.13.11、JAX 0.11.0、BrainState 0.5.3、BrainUnit 0.5.2。

结果：

- 200,000 个时间步、10 个连续块；
- 总墙钟时间 `914.576 s`，约 15.24 分钟；
- 峰值 RSS `1,456,046,080 bytes`，约 1.36 GiB；
- 逐块处理的未压缩数组总量 `337.6 MB`；本次不落盘；
- MF spikes 2286、保留的 glomerulus spikes 152、GrC spikes 0；
- AMPA/NMDA 峰值约 `0.00042701/0.00039756 µS`；
- GrC 胞体电压范围约 `−64.971 至 −49.873 mV`。

逐块耗时在 45–261 秒之间。背景输入开始后，具有较多实际突触事件的第 7–9 块明显变慢，说明当前主要瓶颈是多区室 GrC 与 AMPA/NMDA 状态更新，而不是输出数组分配。RSS 在运行后半段保持约 1.3–1.4 GiB，没有随块数持续线性增长。

原配置转换器共生成 3459 个 `initial_mf + background` 事件，但背景生成窗口延伸到 12500 ms；本次 10000 ms 仿真只执行其中落在仿真窗口内的事件，因此记录到 2286 个 MF spikes。

## 4. 边界与下一步

该基准覆盖 117 个输入 source，但下游仍是 8 glomerulus、32 GrC 的关键通路子网络，不是 31,683 entities/cells、17 类连接的完整论文网络。MF→glomerulus 仍是固定 schedule 的拓扑预展开，不是原生 runtime relay。

下一步资源工作包括：

1. 为 `Network.run_chunks()` 增加 probe 选择和降采样，避免不需要的全分辨率电导轨迹；
2. 为逐块 NPZ/HDF5 写出增加统一 manifest 和恢复/拼接工具；
3. 修复或审计 GrC `reset_state()` 与首次 `init_state()` 的离子初态差异；
4. 在接入 glomerulus→GoC 和 GoC→GrC GABA 前，先用相同估算器给出新增状态量和预期运行时间。

机器可读结果保存在 `data/export/balanced/benchmarks/mossy_pathway_10s_cpu.json`。
