# Reproduction 项目内置 BrainCell

这个包在 reproduction 的虚拟环境中安装为 `braincell`，版本号为 `0.1.0+reproduction.1`。基础源码来自 BrainCell 提交 `7fe0019fe9756add751926037468b5733d8598b4`。包内包含本项目此前验证的原生双端 Junction 实现，以及经核对移植的 `braincell_revised` 对照改动。许可证与源文件的版权声明保留。 [许可](LICENSE) · [BrainCell 基础提交](https://github.com/chaobrain/braincell/commit/7fe0019fe9756add751926037468b5733d8598b4) · [NEURON 对照改动](https://github.com/Merveille0602main/Cerebellar_neuron_population_coding/commit/c4c881b61e975044a235f6cf4c90c93006e6a7f2)

- `Network.connect_junction()` 注册双端电耦合。求解器在同一时间步联立更新相连房室的电压。
- 钙通道移植了 DCN CaHVA 和 CaLVA 的零电压解析极限，以及 Cav3.1 和 Cav3.3 的自动微分保护。
- [单通道对照目录](examples/neuron_compare/channel_no_conc)提供 64 位对照入口和 Golgi Cav2.3 非零测试电导。对照使用当前内置求解器和 NEURON 共有的步初正弦采样约定，按配置的原相位生成刺激。常规前向仿真沿用原先的计算精度。

CPU 环境安装和子网运行见[项目 README](../../README.md)。
