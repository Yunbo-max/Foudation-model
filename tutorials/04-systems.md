# 04　计算、Kernel 与并行：数学等价为什么还不够快

> 前置：[02 架构](02-architecture.md)、[03 Scaling](03-scaling.md)；实践：[作业二](../assignments/02-systems.md)。

## 4.1 场景一：GPU 很强，为什么一个小模型仍很慢

你把 CPU 训练改成 CUDA，发现时间没有按标称 FLOPs 比例下降。
可能是模型太小、CPU 准备数据占时、算子启动开销大，或者数据反复搬运。
GPU 的矩阵运算能力只是上界；训练需要把数据从存储、主存、设备显存传到执行单元。
系统优化的第一步是测出瓶颈，第二步才是决定融合、重算、并行或更换数据管道。

本项目 [model.py](../src/fm_tutorial/model.py) 显式计算 score 与 softmax，便于看懂 mask。
它会物化 `[B,H,T,T]` 中间量，不是 FlashAttention 实现。
[train.py](../scripts/train.py) 是单设备基础训练器，提供梯度累积和可选 CUDA 混合精度；没有实现 DDP/FSDP。
本章解释完整系统问题，并把尚未实现的工作准确指向官方项目。

## 4.2 计算强度与 roofline：区分算力瓶颈和带宽瓶颈

设一次操作的计算量为 $F$ FLOPs，设备内存传输量为 $M$ 字节，计算强度为 $I=F/M$。
若硬件峰值算力为 $P$ FLOPs/s，带宽为 $W$ 字节/s，理想下界为：

$$
t\ge\max(F/P,M/W),\qquad
\mathrm{throughput}\le\min(P,WI).
$$

低 $I$ 的运算常受带宽限制，高 $I$ 的运算可能更接近计算限制。
这只是上界模型；缓存、kernel 启动、依赖、同步、占用率和通信会让实际更慢。
例如两个逐元素操作之间落盘一个大中间张量，融合后可能减少传输，而 FLOPs 几乎没变。
这解释了为什么“相同计算量”不等于“相同运行时间”。

GPU 的 HBM 容量大、访问代价高；片上 SRAM/寄存器容量小、访问快。
具体层级和容量随设备变化，应记录你的硬件型号，不把一种 GPU 的调优结果推广到所有 GPU。
Tile 过小会重复读数据或启动太多程序；过大可能占用过多寄存器并降低并发。
最优 tile 是形状、dtype、架构共同决定的实验结果。

## 4.3 显式注意力的内存成本

单个 batch/head 的 score 矩阵为 $T\times T$，全批中间量为 $BHT^2s$ 字节。
当 $T$ 翻倍，score 项增长四倍；Q/K/value 项只按 $T$ 增长。
前向保存概率，反向还产生梯度与 workspace，总峰值不能只用一份 score 估计。
本仓库 score 用 float32 计算，若输入激活是低精度，不能仍按两字节估算它。

减少中间量的目标是保留相同注意力数学定义，同时改变数据读写与计算顺序。
FlashAttention 是精确注意力的 IO-aware 方法，不是将长序列近似成短序列。
这里“精确”指没有稀疏或低秩近似；浮点重排仍可能带来数值差异。
如果算法改了 mask、窗口或 softmax 归一化范围，那就不只是系统优化。

## 4.4 Online softmax：不用保存完整概率矩阵

对一条 query，分数 $s_j$ 的稳定 softmax 可用三个状态概括已处理 keys：

$$
m=\max_j s_j,\quad \ell=\sum_j e^{s_j-m},\quad
u=\sum_j e^{s_j-m}v_j.
$$

$m,\ell$ 为标量，$u\in\mathbb R^{d_h}$，最后输出 $o=u/\ell$。
把 keys 分成块 A、B，分别计算 $(m_A,\ell_A,u_A)$ 和 $(m_B,\ell_B,u_B)$。
合并时用共同最大值 $m'=\max(m_A,m_B)$：

$$
\ell'=e^{m_A-m'}\ell_A+e^{m_B-m'}\ell_B,
\quad u'=e^{m_A-m'}u_A+e^{m_B-m'}u_B.
$$

旧指数权重需要重新缩放，否则新块出现较大分数后，旧块与新块不在同一归一化基准。
这个合并等式直接来自指数恒等式，是理解分块注意力的核心。
逐 query block 更新状态，就不必把全体 `[T,T]` 概率写回 HBM。
仍然需要遍历所有允许的 query-key 组合，因此一般密集注意力的算术二次量级没有自动消失。

手工用两个短分数块检查合并等式，是算法正确性练习，不是 GPU benchmark。
真正 kernel 还需处理 tile 边界、因果区、dtype、布局和线程协作。
本仓库未提供原创 Triton 前向或反向 kernel，不能以这个公式宣称已经实现 FlashAttention。

## 4.5 反向传播：为什么省内存的方法会重算

忽略 batch/head，设 $O=PV_h$，上游梯度 $G=\partial L/\partial O\in\mathbb R^{T\times d_h}$。
链式法则给出：

$$
dV_h=P^\top G,
\quad dP=GV_h^\top,
\quad dS_{ij}=P_{ij}\left(dP_{ij}-\sum_kP_{ik}dP_{ik}\right).
$$

令 $S=QK^\top/\sqrt{d_h}$ 后：

$$
dQ=dSK/\sqrt{d_h},\qquad dK=dS^\top Q/\sqrt{d_h}.
$$

softmax Jacobian 在行内耦合所有 key；只对每个概率独立求导会漏掉归一化项。
masked 概率应为零，相应禁止路径不能产生有效梯度。
IO-aware 反向可以借保存的行归一化信息重新计算某块 $P$，用更多计算换更少的 HBM 中间量。
这与 activation checkpointing 都包含重算，但重算边界与算法设计不同。

GPU backward 不只是把以上公式抄进 Python，需要决定梯度累积、并行分工和可能的原子写入。
本项目的梯度来自 PyTorch autograd，没有原创 GPU backward kernel，也没有该 kernel 的性能成绩。
要完成这一部分，进入 [CS336 systems 官方作业](https://github.com/stanford-cs336/assignment2-systems) 与 [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
先通过官方数值检查，再测真实 GPU 性能；CPU 能验证公式，不能证明 kernel 加速。

## 4.6 使用 PyTorch SDPA 时必须检查的语义

`scaled_dot_product_attention` 根据输入与平台选择可用后端，调用它不保证一定用了 FlashAttention。
记录框架版本、dtype、head dimension、mask、设备，并用 profiler 或官方 backend 控制确认选择。
函数按 `dropout_p` 应用 dropout；eval 时传 0，而不是仅依赖外层 `model.eval()`。
GQA 支持条件与后端随版本变化，先查安装版本文档，不复制某个版本的设备兼容结论。

替换本项目显式 attention 时，保留 causal/key padding/all-PAD 的行为，并分别比较前向与梯度。
使用浮点误差容限 `atol + rtol*abs(reference)`，不要要求所有后端逐位相同。
容限应由 dtype 与问题尺度决定，不能不断放宽直到任何错误都过关。
固定实际语料 batch 的验证 loss 也应接近；只有一个随机张量比较并不能发现数据接口错误。

## 4.7 场景二：单卡放不下，不一定先上多卡

训练状态包括权重、梯度、优化器状态、激活与临时 workspace。
假设使用两字节权重和梯度、四字节 master weights 与两个 Adam moment，参数状态的示意合计约 $16N$ 字节。
这是特定精度策略下的账本示例；PyTorch 实际状态可能不同，必须查看实际 dtype 与分配。
激活又受 batch、context、层数和 kernel 影响，不能用参数状态公式估计全部显存。

优先问是哪一项占主导：

| 主导项 | 可考虑的手段 | 主要代价 |
| --- | --- | --- |
| microbatch 激活 | 减 microbatch、梯度累积 | 更多串行 microbatch |
| 保存的层激活 | activation checkpointing | 反向重算 |
| 参数/优化器状态 | 分片、offload | 通信或主机传输 |
| 注意力中间量 | SDPA/FlashAttention | 后端与形状约束 |
| 序列 padding | 按长度分桶或 packing | 数据边界与 mask 更复杂 |

把 batch 降到能运行只是可行性措施，不意味着训练目标不变。
要保持有效 batch，可调整累积，并核对损失分母、学习率和更新次数。
原训练器没有 activation checkpointing/offload 开关，需在扩展分支实现并检查；不要在命令里编造不存在的选项。

## 4.8 数据并行：复制模型，分配样本，聚合梯度

有 $p$ 个设备，每个设备读不同数据，得到本地梯度 $g_r$。
等权 batch 时全局平均梯度为 $g=\frac1p\sum_{r=1}^pg_r$。
DDP 使用通信原语聚合梯度，每个设备更新同一份参数副本。
若本地有效 token 数不同，平均本地均值未必等于全局 token 均值，应显式按计数加权。
数据 sampler 还要避免各卡读取相同样本，并在每 epoch 更新随机排列规则。

梯度累积时可在前 $A-1$ 个 microbatch 用 `no_sync()` 避免不必要的 all-reduce，最后一次同步。
forward 与 backward 应在对应 context 中；必须遵照当前 DDP 文档。
损失仍需按目标全局 batch 正确缩放，否则优化步长改变。
DDP 不会自动让单份权重或优化器状态变小，每卡仍保留模型副本。

## 4.9 其他并行方式：切样本、切参数、切层的区别

Tensor parallel 把大矩阵运算与参数切到多个设备，层内常需要通信。
Pipeline parallel 把不同层分到设备，多个 microbatch 流水执行，但可能产生 bubble。
ZeRO/FSDP 分片参数、梯度或优化器状态，在需要时通信恢复相应部分。
这些方式可以组合，却带来更复杂的状态保存、布局与通信拓扑。

流水线有 $p$ 个 stage、$m$ 个 microbatch 时，简单无交错调度的利用率直觉约为 $m/(m+p-1)$。
这忽略 stage 不均衡、通信和具体调度，不应直接当成实测利用率。
增加 microbatch 降低 bubble 的同时可能增加延迟和缓冲压力。
跨节点 tensor parallel 若通信慢，可能不如主要用节点内通信；需按真实互联测量。

## 4.10 场景三：四卡跑更快，为什么训练总成本反而高

强扩展固定总工作量，从一卡增加到 $p$ 卡，speedup 为 $S_p=t_1/t_p$，效率为 $S_p/p$。
若四卡只快两倍，单位工作耗费的 GPU 时间增加了一倍。
弱扩展则固定每卡工作，考察总规模扩大时吞吐；它回答不同问题。
不注明哪种扩展，读者无法判断加速是来自更多硬件还是系统改善。

通信、慢设备、数据管道和尾部等待都可使效率下降。
平均每卡忙碌不代表所有设备同时做有效计算，profiler 时间线能帮助发现等待。
成本应按完整资源占用时间计算，含失败重试与验证，而不是只计 kernel 时间。
本文 GPU 数量与实验预算均为学习建议，不是唐杰课程官方硬件要求。

## 4.11 在真实工作负载上测量

从固定真实 TinyStories/WikiText/OWT 验证文档构造 batch，保存 token ID 与 tokenizer 哈希。
对 kernel 正确性可补充随机张量和极端输入，但性能主表应注明真实 batch 的 shape、长度分布与 mask。
先预热，包括 JIT/编译；再重复测量，报告中位数与分位数，保留原始记录。
CUDA 默认异步，CPU `perf_counter` 包住一次调用可能只量到发射时间。
用 CUDA events 或 PyTorch benchmark 的正确同步机制，分开编译时间与稳态运行时间。

峰值显存应在相同生命周期区间测，区分 allocated 与 reserved。
端到端训练步骤需包含 forward、backward、optimizer；注意力前向 kernel 快不代表训练整体按相同比例变快。
真实 benchmark 没运行时，表格留空并列待执行配置，不引用文档示例的 TFLOPS 当自己的结果。
交付协议详见 [作业二](../assignments/02-systems.md)。

## 4.12 检查理解

1. 从 online softmax 的合并式解释为什么不能简单相加两个块的输出 $o_A,o_B$。
2. 手推 softmax 反向的行内耦合项，写出 `dQ/dK/dV` 的 shape。
3. 梯度累积减少哪种显存，为什么不能减少全部优化器状态？
4. 两卡本地有效目标数不同时，如何得到全局 token 平均损失的正确梯度？
5. 设计一个四卡强扩展实验，包含相同数据量、计时区间与失败成本。

## 原始来源与阅读目的

- [FlashAttention](https://arxiv.org/abs/2205.14135)：IO-aware 分块与重算的原始算法。
- [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：官方 GPU kernel 实践，前后向与硬件条件。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：mask、dropout、后端约束。
- [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[profiler](https://docs.pytorch.org/docs/stable/profiler.html)：计时与瓶颈证据。
- [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)：官方分布式接口。
- [ZeRO](https://arxiv.org/abs/1910.02054)、[Megatron-LM](https://arxiv.org/abs/1909.08053)、[GPipe](https://arxiv.org/abs/1811.06965)：分片、张量并行与流水线的原始方法。
- [CS336 systems](https://github.com/stanford-cs336/assignment2-systems)：完整系统作业扩展入口，按访问时 handout 与 commit 固定版本。
