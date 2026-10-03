# 05　推理经济学：延迟、吞吐、状态与答案质量一起优化

> 前置：[04 系统](04-systems.md)；实现：[generate.py](../scripts/generate.py)；实践：[作业二](../assignments/02-systems.md)。

## 5.1 场景一：长资料刚提交，用户为何等不到第一个字

一个请求包含几页资料，随后要求生成短回答。
服务器先处理全部 prompt，建立每层状态并算出第一个输出分布，这一阶段叫 prefill。
之后每次生成一个新 token，把它加入上下文，逐步继续，这一阶段叫 decode。
两阶段的工作形状不同；优化一项可能损害另一项。
用户看到的首次响应还包括排队、网络传输与 tokenization，不能只测一次模型调用。

先建立指标词典：

| 指标 | 定义 | 使用注意 |
| --- | --- | --- |
| TTFT | 请求到首个输出 token 的时间 | 区分服务端/客户端起点 |
| TPOT | 后续输出 token 的平均间隔 | 第一个 token 通常不计 |
| ITL | 相邻输出 token 的逐次间隔 | 看分位数，别只看均值 |
| E2E latency | 请求到完整输出的时间 | 含排队与停止条件 |
| output tokens/s | 总输出 token / 测量时间 | 标明是否含 prompt |
| goodput | 满足质量与延迟约束的请求量 | 需先定义约束 |

同一个系统可能吞吐更高却 p95 延迟更差。
如果用户要求实时互动，应先设延迟目标，再比较满足目标的吞吐，而非只追求最大 batch。
本章不提供当前云价格或硬件报价；实验报告使用你实际获得的资源费用与日期。

## 5.2 Prefill 与 decode 的计算差别

长度 $T$ 的 prefill 对多个 token 进行矩阵运算，容易形成较大的 GEMM。
密集注意力的 score 仍为 `[B,H,T,T]`，使用 FlashAttention 主要降低中间量读写。
带 KV cache 的 decode 只计算当前 token 的 Q/K/value，并读取历史 K/value。
此时 query 长度为 1，key 长度为历史长度，score shape 为 `[B,H,1,T]`。

每步仍要处理模型层权重，并读取随上下文增长的 KV 状态。
小 batch decode 常难以充分利用矩阵计算能力，可能更受带宽影响；这是需测量的常见机制，不是所有硬件的定理。
增大 batch 能让多个请求共享权重读取，但增加 KV 占用、排队与调度复杂度。
因此“训练快”不等于“单请求首字快”，两者的工作负载不同。

## 5.3 KV cache：缓存哪些量，为什么有效

因果 decoder 中，过去位置的表示不依赖未来 token。
固定参数、eval 模式与相同位置规则下，每层过去的 K/value 可保存并在后续复用。
当前 query 对历史 K 计算 score，再混合历史 value；不需重新投影全部过去 token。
要缓存每一层的 KV，而不是只缓存最后一层隐藏状态。

单层缓存 shape 通常是 `[B,H_kv,T,d_h]` 的 K 与 value。
逻辑占用为：

$$
M_{KV}=2LBTH_{kv}d_hs\quad\text{字节}.
$$

对不同请求长度，应按 $\sum_iT_i$ 而不是 $BT_{\max}$ 估计逻辑有效量，实际分配策略可能仍有 padding。
GQA 通过减少 $H_{kv}$ 降低缓存项；量化 KV 通过降低 $s$ 进一步改变它。
两者都会涉及架构或数值选择，质量与可用 kernel 必须分别检查。
权重缓存、prompt prefix cache 与每步 KV cache 是不同层次的复用。

位置偏移很容易写错：第 $T$ 个新 token 的 RoPE 位置应延续前文，而不是每次从 0 开始。
非方阵 causal mask 也必须允许当前 query 读完整历史。
Dropout 开启时旧激活不再与每次重新计算严格一致，因此缓存生成应采用 eval 语义。

## 5.4 本仓库的生成器：一个清楚但有意简化的基线

[generate.py](../scripts/generate.py) 每次取最新 `context_length` 个 token，重新执行完整 forward，再从最后位置采样。
它校验 tokenizer 哈希与 checkpoint，禁止输出 PAD，以 EOS 或 `max_new_tokens` 终止。
它没有持久 KV cache、continuous batching、分页分配、量化或 speculative decoding。
所以本项目生成器适合验证权重/词表链路与采样行为，不能用于宣称生产 serving 性能。

上下文超限后采用滑动截断，还会把当前窗口的位置重新从 0 计算。
这是教学基线的具体语义，不等同于保留全历史 KV 的无限长度推理。
实验必须报告截断率，避免把被丢弃的信息仍算成模型已处理的有效上下文。
如果自行实现 cache，应先在 context 内比较 cached/uncached logits，再定义超长上下文策略。

```bash
python scripts/generate.py --checkpoint runs/cpu/last.pt --tokenizer data/processed/tokenizer.json --prompt 'Once upon a time' --max-new-tokens 64 --temperature 0.8 --seed 42
```

命令使用你已经训练的真实 checkpoint；短训练后输出可能不连贯，这是需要观察和记录的结果。
没有 checkpoint 时命令失败，不会自动下载别人模型或产生伪训练结果。

## 5.5 采样：分布、可复现性与正确性是三件事

温度 $\tau>0$ 的采样分布为：

$$
p_i(\tau)=\frac{e^{z_i/\tau}}{\sum_je^{z_j/\tau}}.
$$

较低温度通常更集中，较高温度通常更分散；它改变选择分布，不能凭空增加知识。
本项目 `temperature=0` 采用 greedy：选最大 logit，不执行除以 0。
Greedy 每步局部最大不一定最大化整条序列概率，也不保证任务答案正确。
固定 seed 有助于在相同环境下复现随机采样，不保证跨设备逐位相同。

Top-k 保留分数最高的 $k$ 个候选后重新归一化；top-p 保留累计概率达到阈值的最小前缀。
它们是常见扩展方法，基础脚本尚未实现，不要传入不存在的 CLI 参数。
不同采样方案会改变生成长度、任务成绩与成本，serving 对比要固定这些条件。
只说“两个模型都生成 64 token”也不够：EOS 可能使实际输出更短，需要记录实际数量。

## 5.6 场景二：长短请求混合为什么浪费资源

一个静态 batch 里，短请求早早完成，长请求还在继续。
若系统等待整个 batch 完成才接新请求，空闲槽位被浪费。
Continuous batching 在迭代边界重新调度，让新请求加入，完成请求离开。
它提高资源利用率的机会，同时需要维护各请求的 KV、长度、采样状态与取消行为。

长 prompt 的 prefill 可能阻塞正在 decode 的短请求。
Chunked prefill 把长 prompt 分段排入调度，可以改善交互延迟，但代价与最佳 chunk 大小依系统而变。
这些策略没有消除计算，只改变谁先获得设备时间。
公平性、最大等待、超时与服务等级也属于调度目标。

固定长度合成请求可以隔离某个性能因素，真实负载报告还应保留真实请求的长度分布与到达模式。
本教材不编造请求日志；可从公开评估数据形成可审计的文本请求，并明确它是离线回放。
离线回放不是已经部署的线上业务流量，不能据此声称生产 SLA。

## 5.7 分页 KV：把连续逻辑序列映射到离散物理块

每个请求的长度动态增长，如果预留最大连续缓存，会浪费未用空间。
PagedAttention 使用类似页表的映射，把逻辑 token block 对应到物理 KV block。
分配随需求增长，多个请求还可在明确条件下共享相同前缀块。
末块可能未填满，仍存在内部碎片；分页不是零字节管理开销。

Prefix sharing 需要保证模型、adapter、tokenizer、位置和输入 token 一致。
“字符串看起来相同”不够，例如不同 chat template 可能生成不同 token 序列。
缓存还需考虑生命周期、回收和请求隔离，不能跨不兼容配置复用。
本仓库没有分页实现；阅读 [vLLM 原论文](https://arxiv.org/abs/2309.06180) 与官方项目后再做扩展。
原论文性能结果属于其硬件与工作负载，本项目不复制其加速比例作为自己的结果。

## 5.8 量化：把数值精度换成内存与带宽预算

简单对称量化可写为：

$$
q=\mathrm{clip}(\mathrm{round}(x/s),q_{\min},q_{\max}),\qquad \hat x=sq.
$$

$s>0$ 为 scale，整数 $q$ 用更少位保存，解量化 $\hat x$ 近似原值。
量化误差会通过网络传播；outlier、分组大小和 scale 选择影响误差。
按通道/分组量化比单个全局 scale 更灵活，但需要元数据和适合的 kernel。
低位存储不保证运行更快：若频繁解量化或 kernel 不匹配，额外代价可能抵消收益。

Weight-only、activation、KV 量化改变不同数据项，不应混称“模型是 4-bit”。
校准集只能来自训练/开发数据，不能用最终测试答案来调量化。
比较时固定 prompt 与采样，既测 NLL/任务成功率，也测峰值显存与端到端延迟。
本项目无量化脚本；先采用框架/模型官方支持路径，在报告中记录方法版本与精度设置。

## 5.9 Speculative decoding：草稿快，验证需要纠正分布

小草稿模型 $q$ 一次提出一段候选，大目标模型 $p$ 并行验证这些候选。
对某位置候选 $x\sim q$，一种保持目标分布的接受概率为：

$$
A(x)=\min(1,p(x)/q(x)).
$$

若拒绝，需从归一化的正残差 $[p-q]_+$ 采样，而不是随意挑目标模型最高 token。
接受/拒绝与残差纠正一起，才能在算法条件下保持目标采样分布。
完整算法还处理首次拒绝后的草稿截断与额外目标 token，详见原论文。
用“目标模型同意就接受”的口头描述，不足以保证分布正确。

收益取决于草稿成本、接受率、目标并行验证效率、batch 和输出任务。
高接受率不自动意味着高加速；草稿本身可能太慢，或目标 batch 已充分利用硬件。
对 deterministic greedy 的加速与对随机目标分布的精确采样，应分开说明。
本项目没有 speculative 实现，也没有实测接受率；此处为算法学习与扩展设计。

## 5.10 场景三：更长的“思考”是否值得付钱

让模型生成更多推理 token，或生成多个候选再验证，都会增加 test-time compute。
如果真实成功率从 $s_1$ 变为 $s_2$，平均每题费用从 $c_1$ 变为 $c_2$，应比较每个成功任务的成本，而非只看总分。
简化指标为 $c/s$；它假设任务独立且成功定义固定，复杂服务应直接汇总真实成本与成功次数。
失败重试、verifier 调用、工具与等待也都计入费用。

预算增长曲线应在真实公开基准上测，比如后续 RLVR 章节中的数学或代码任务。
固定题集、prompt、解析器、超时与 verifier，扫描候选数或最大输出预算。
不能先用答案挑出好候选，再宣称系统自己已经识别好候选。
Self-judge 的准确率也需要独立标注；评分器可能偏向更长、更自信的回答。

## 5.11 费用账本与 serving 实验

若使用 $n_g$ 块 GPU、总时间 $t$ 秒、实际每 GPU 小时费用为 $c_g$，设备费用估算为：

$$
C_{\mathrm{device}}=n_g\cdot(t/3600)\cdot c_g.
$$

补充 CPU、存储、网络、模型加载、空闲与工具成本，单位与计费条件来自真实账单或明确预算假设。
平均费用/请求用完整区间费用除以完成请求数；失败请求不能悄悄从资源时间中删掉。
按百万输出 token 报价可作一种口径，但需明确 prompt token 成本与 tokenizer。
没有实际报价时只报告 GPU 秒与 token 数，不伪造货币价格。

建议用公开真实验证段落做 prompt 长度分组，比较 1、2、4 等不同并发；具体值取决于设备容量。
保存请求文本/ID、数据版本、生成配置、每请求时间戳、实际输入输出长度、错误与截断率。
在带 GPU 的机器上测本项目重算基线，以及你独立部署的优化服务；先比较语义与质量再比较吞吐。
相同 checkpoint 若不能直接导入外部服务，需明确模型格式不同，不能归因成纯 serving 对比。
完整协议见 [作业二](../assignments/02-systems.md)。

## 5.12 检查理解

1. 长 prompt、短答案与短 prompt、长答案，分别主要考验哪个阶段？
2. 为什么缓存 K/value 而不缓存一个“最终答案向量”？
3. 推导 GQA 的 KV 字节量，并列出实际占用超过逻辑量的三个原因。
4. 为什么只用接受概率而漏掉拒绝后的残差采样，会改变目标分布？
5. 为真实任务定义 goodput，说明如何处理超时、格式错误和错误答案。

## 原始来源与阅读目的

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：动态 KV 内存管理与 serving 负载。
- [vLLM 官方项目](https://github.com/vllm-project/vllm)：当前安装、模型支持与 benchmark 工具，按版本查文档。
- [Speculative decoding 原论文](https://proceedings.mlr.press/v202/leviathan23a.html)：接受与残差纠正保证目标分布的算法。
- [GQA 原论文](https://arxiv.org/abs/2305.13245)：KV head 共享的架构动机。
- [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：避免异步计时失真。
- [CS336 systems 官方作业](https://github.com/stanford-cs336/assignment2-systems)：GPU 性能与系统实践的扩展路径。
