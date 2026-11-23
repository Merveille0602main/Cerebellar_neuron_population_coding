# BrainCell 上游仓库修改说明

## 1. 文档目的与代码边界

本文记录小脑论文复现早期在独立 BrainCell 工作树中完成的网络运行改动。当前运行环境见 [环境说明](ENVIRONMENT.md)。

记录日期：2026-08-08。BrainCell 当前基准提交为 `c803e77`。

本次明确修改的已跟踪文件只有：

- `braincell/network/engine.py`
- `braincell/network/runtime_test.py`

BrainCell 工作树中未跟踪的 `GrC/` 和 `examples/neuron_compare/cell/GrC/` 不属于本次修改，本文也不对其来源或内容作归属判断。

## 2. 修改解决的问题

论文协议需要秒级到 10 秒级运行。以当前 117 个 mossy-fiber source、8 个 glomerulus、32 个多区室 GrC 子网络为例，`dt=0.05 ms` 的 10 秒仿真包含 200,000 步。BrainCell 的 `Network.run()` 会一次返回整个时间窗口的时间、膜电位、电导和 spike 数组，仅这些返回数组就约为 337.6 MB。

简单地循环调用十次 `Network.run(duration=1*s)` 不能安全代替一次 10 秒连续运行。每次 `run()` 都会创建新的事件 delivery ring；细胞膜、离子通道和突触状态虽然会继续存在，但前一块末尾已经入队、尚未到达突触的延迟事件不会进入新 ring，因此可能在块边界丢失。对于依赖传导延迟和反馈回路的网络，这会改变仿真语义。

本次修改的目标是提供一个有界输出的连续运行接口：让调用方逐块消费或保存结果，同时保持同一次运行会话中的绝对时间、细胞状态、突触状态和延迟事件投递状态连续。

## 3. `engine.py` 的具体修改

### 3.1 新增公开接口 `Network.run_chunks(...)`

接口参数为：

```python
network.run_chunks(
    dt=0.05 * u.ms,
    duration=10_000 * u.ms,
    chunk_duration=1_000 * u.ms,
    delay_quantization="ceil",
    event_backend="auto",
    brainevent_backend="jax_raw",
    spike_recording="full",
)
```

它返回惰性迭代器，每次迭代产生一个 `NetworkRunResult`。调用方可以统计或落盘当前块，并在请求下一块前释放当前块，从而避免整个时间序列同时驻留内存。

同一个迭代器中的所有块共享：

- 一个 delivery ring state，包括尚未投递的延迟事件和当前 cursor；
- 各 population 的细胞膜、离子通道与 spike/reset 状态；
- 突触机制状态；
- 绝对仿真时间。

最后一块不足 `chunk_duration` 时，只运行剩余步数。`chunk_duration <= 0` 会抛出 `ValueError`。

### 3.2 抽取 `_prepare_run_session(...)`

原 `Network.run()` 中的参数验证、事件后端归一化、拓扑编译和 delivery state 创建被抽取为 `_prepare_run_session(...)`。`run()` 和 `run_chunks()` 使用同一准备路径，避免两套初始化逻辑发生差异。

### 3.3 抽取 `_run_segment(...)`

原 `run()` 中执行一次 `brainstate.compile.for_loop`、整理 trace/spike 并生成 `NetworkRunResult` 的部分被抽取为 `_run_segment(...)`。

- `run()` 仍只准备一次会话并执行一个完整 segment；
- `run_chunks()` 只准备一次会话，然后用相同的 setup 和 delivery state 连续执行多个 segment。

因此原有 `Network.run()` 的调用形式和返回类型没有改变。该重构的预期是保持旧接口行为兼容，而 `run_chunks()` 是显式选择的新能力。

### 3.4 新增 `_step_count(duration, dt)`

该辅助函数使用与运行时间网格相同的 `brainunit.math.arange(0, duration, dt)` 语义计算步数。`run_chunks()` 先把总时长和块时长换算成离散步数，再按步数切分，避免浮点除法造成最后一块多一步或少一步。

## 4. `runtime_test.py` 的具体修改

新增测试：

```text
test_run_chunks_preserves_boundary_delivery_and_matches_single_run
```

测试分别构建两个相同的最小突触网络：

1. 参考网络连续运行 `0.3 ms`；
2. 待测网络以 `0.1 ms` 为块运行三个块。

presynaptic spike 被安排在第一块边界，使其延迟投递必须跨块完成。测试拼接三个块的结果，并与连续运行逐项比较：

- postsynaptic conductance；
- presynaptic spikes；
- absolute time。

这个测试不是只检查“可以分三次运行”，而是直接覆盖此次修改要解决的块边界事件丢失问题。

## 5. 复现项目中的配套代码

以下文件没有修改 BrainCell API，而是复现项目对新接口的使用和验证：

- `src/cerebellar_reproduction/long_run.py`：依据初始化后的真实 probe/spike shape 和 dtype 估算返回数组大小，并逐块消费结果；
- `scripts/benchmark_mossy_pathway.py`：运行 MF→glomerulus→GrC 长时资源基准；
- `tests/test_long_run.py`：在真实多区室 GrC 通路上比较连续运行和分块运行；
- `LONG_RUN.md`：记录内存估算方法、10 秒 CPU 基准和适用边界；
- `data/export/balanced/benchmarks/mossy_pathway_10s_cpu.json`：机器可读的基准结果。

内存估算器属于复现实验工具，没有放入共享 BrainCell 仓库，因为它目前依赖本项目的记录需求和输出统计方式。

## 6. 已完成验证

BrainCell 网络运行回归：

```bash
cd path/to/historical-braincell-checkout
pytest braincell/network/runtime_test.py
```

结果：43 个测试通过，另有 3 个参数化 subtests 通过。

复现项目回归：

```bash
.venv/bin/pytest
```

结果：14 个测试通过。真实 GrC 分块测试验证拼接后的胞体电压、AMPA/NMDA 电导和 glomerulus spikes 与连续运行一致。

10 秒 CPU 基准使用 117 个 MF source、8 个 glomerulus 和 32 个 GrC，共 200,000 步：

- 总墙钟时间约 914.576 秒；
- 峰值 RSS 约 1.456 GB；
- 全部未压缩输出约 337.6 MB；
- 1 秒块的单块输出约 33.76 MB；
- RSS 没有随块数持续线性增长。

## 7. 当前限制与维护风险

1. `run_chunks()` 只保证同一个迭代器生命周期内的状态连续，尚未实现跨进程 checkpoint/resume。
2. 如果调用方提前停止迭代，剩余运行不会自动执行；会话状态也没有持久化。
3. 当前没有在迭代期间修改 population、connection 或 projection 的显式防护。调用方应把网络拓扑视为冻结状态。
4. 当前仍记录每一步的完整 probe；分块限制了峰值返回数组，但不会减少总计算量或总落盘量。probe 选择与降采样尚未实现。
5. JAX 编译缓存、runtime state、临时数组和 allocator slack 不包含在输出大小估算中，因此估算值不能替代进程 RSS 实测。
6. API 目前直接暴露 generator。若共享项目希望支持暂停、恢复或显式关闭，后续可能更适合设计公开的 `RunSession` 对象。
7. 该修改尚未提交到 BrainCell git 历史，也未经过其他共同维护者的 API 审查；在形成上游提交或 PR 前，应先确认命名、返回协议和长期兼容策略。

## 8. 建议的上游处理方式

在共同维护者完成评审前，建议将这两处修改视为复现项目所需的本地 BrainCell 补丁。评审时重点确认：

- `run_chunks` 是否作为 `Network` 的长期公开 API；
- 是否需要以 `RunSession` 替代 generator；
- 是否同时加入 probe 选择、降采样和 checkpoint 语义；
- 是否要求运行期间的拓扑冻结检查；
- 文档和类型标注应达到的项目标准。

如果决定不把该能力合入共享仓库，这个补丁是局部且可逆的：只需恢复上文列出的两个已跟踪文件。执行任何恢复命令前应先检查 `git diff`，避免覆盖共同维护者随后写入这两个文件的内容。

## 9. 与论文复现范围的关系

`run_chunks()` 只改变长时仿真的执行与结果消费方式，不改变从 `balanced.hdf5` 导入的 population、edge、contact、形态映射或突触方程。它也没有让当前子网络自动变成完整论文网络。

它是继续迁移 glomerulus→GoC、GoC→GrC 等通路前所需的运行基础设施：加入更多多区室细胞和突触状态后，输出和运行成本会继续增长，而跨块事件投递必须保持正确。后续论文专用机制仍优先实现在独立复现仓库；只有被证明为通用 BrainCell 能力的部分，才建议另行提交共享仓库评审。

## 10. 后续边界确认

本文完成后进行的 glomerulus→GoC 第一版迁移没有继续修改共享 BrainCell。GoC 机制语义别名、发布 contact 构建器、运行脚本、测试和来源审计均保存在 `braincell-cerebellar-reproduction`。因此截至本文更新时，BrainCell 侧仍只有第 1 节列出的两个已跟踪文件属于本复现工作的修改。

## 11. 共同输入、GABA 和 soma spike 适配

这一阶段也没有继续修改共享 BrainCell。以下新增内容全部位于本复现目录：

- `build_mossy_glomerulus_granule_golgi_subnetwork()`；
- `GrCGABA2020` 与其 NEURON 固定电压对照；
- `build_granular_layer_chemical_subnetwork()`；
- `_restrict_spikes_to_soma()`。

最后一项用于解决多区室 `Cell` 的每个 CV 都可产生 threshold-crossing buffer、而网络又会把任意 CV crossing 当成 cell event 的语义差异。它只在论文细胞实例上屏蔽非 soma CV 的网络输出，不改共享 BrainCell 的 `Cell`、`Network` 或 delivery 源码，也不改膜动力学。详细证据见 `GABA_CHEMICAL_LOOP.md`。

最新聚焦回归：

```bash
.venv/bin/pytest -q tests/test_paper_synapses.py tests/test_runnable_subnetwork.py
```

结果：10 个测试通过。

## 12. 论文完整 GrC 模板的代码边界

整细胞等价阶段发现共享 BrainCell 的实验性 `GrCFull` 示例与论文历史
`dbbs_models.GranuleCell` 在 AA/PF section 构造上不同。当前没有直接修改
该共享示例，也没有新增 BrainCell 已跟踪文件的改动。论文准确的 108-branch、
125-CV 模板位于复现仓库
`src/cerebellar_reproduction/paper_granule_cell.py`，其证据、限制和测试见
`PAPER_GRANULE_CELL.md`。因此截至本节，共享 BrainCell 仍只有第 1 节列出的
两个已跟踪文件属于本复现工作的修改。
