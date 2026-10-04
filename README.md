# Foundation Model：从基础组件到长程 Agent

**一套原创中文教程：理解原理，亲手实现，使用真实数据验证，再读懂大型开源训练体系。**

课程主线参考唐杰《高级机器学习》公开课件照片及个人整理的 2026 年 16 周地图，结合 Stanford CS336、课程团队教材、Nemotron、OLMo、DCLM、Pile 与一手论文。尚未找到 2026 整套官方实验 handout；这里是独立学习项目，每章的代码、实验选择和验收方法由本项目提供。公开来源及版本边界见 [来源表](references/sources.md)。

## 从哪里开始

- 不熟悉张量、梯度和概率：先读 [00 基础预备](tutorials/00-prerequisites.md)。
- 知道 Transformer 名称，但没有从头实现：读 [02 架构](tutorials/02-architecture.md)，配合 `src/fm_tutorial/model.py` 逐项看 shape、因果遮罩、RoPE、GQA、RMSNorm 和 SwiGLU。
- 想直接跑一次完整的基础训练链路：按下面“第一条可运行链路”，然后做 [作业 1](assignments/01-foundations.md)。
- 想读工业级训练仓库：读 [官方开源训练体系导读](references/open-training.md)，先定位数据、模型、训练、后训练和评估各自的代码。
- 想研究 self-judge / 长程 Agent：先读 [10 RLVR](tutorials/10-rlvr.md)、[11 Agent RL](tutorials/11-agent-rl.md)、[14 自评与自演化](tutorials/14-self-evolution.md)，再做 [作业 5](assignments/05-agent.md)。
- 想系统理解 Diffusion：从 [17 DDPM、DDIM 与 score-based](tutorials/17-diffusion.md) 开始，接着读 [连续 latent 与离散文本](tutorials/17a-diffusion-text.md)、[图片、表格、视频与 4D](tutorials/17b-diffusion-modalities.md)，配合 [实践六](assignments/06-diffusion.md)。

每章按“基础与场景 → 数学 → 实现 → 实验 → 误区 → 理解检查”展开。解释中的例子用于理解；模型评价使用公开数据集与既有 benchmark，不把手造题、单元测试或合成曲线当作研究结果。

## 16 周主线

| 周 | 教程 | 主要要理解的问题 |
|---|---|---|
| 1 | [三次范式转移](tutorials/01-paradigms.md) | 学习信号如何从标签扩展到 next-token、偏好、验证器、环境与自评？ |
| 2 | [重新理解架构](tutorials/02-architecture.md) | Tokenizer 与 Transformer 的每个组件怎样影响信息流？ |
| 3 | [训练动力学与 Scaling Law](tutorials/03-scaling.md) | 参数、数据、计算和 loss 如何联系，什么外推有证据？ |
| 4 | [算子、并行与训练系统](tutorials/04-systems.md) | FLOPs、内存和通信各自在哪里成为瓶颈？ |
| 5 | [推理成本与服务](tutorials/05-inference.md) | Prefill、decode、KV cache、batch 和延迟怎样相互制约？ |
| 6 | [预训练数据](tutorials/06-data.md) | 如何从 raw dump 得到无泄漏、可追踪的训练集？ |
| 7 | [合成数据与治理](tutorials/07-synthetic-data.md) | 生成数据怎样增加有用信息，怎样识别污染与反馈退化？ |
| 8 | [SFT 与蒸馏](tutorials/08-sft.md) | 哪些 token 被监督，教师输出怎样变成学习信号？ |
| 9 | [偏好学习](tutorials/09-preferences.md) | DPO 在比较什么，参考模型和偏好数据起什么作用？ |
| 10 | [RLVR 与推理](tutorials/10-rlvr.md) | 可验证奖励、group advantage、PPO/GRPO 和探索怎样连接？ |
| 11 | [长程 Agent 强化学习](tutorials/11-agent-rl.md) | 多步轨迹的状态、动作、奖励和信用分配如何定义？ |
| 12 | [Agent 基础设施](tutorials/12-agent-foundations.md) | 环境、工具、harness、终止条件和轨迹日志怎样搭建？ |
| 13 | [记忆与持续学习](tutorials/13-memory.md) | Context、KV、外部记忆和权重更新分别保存什么？ |
| 14 | [自评与自演化](tutorials/14-self-evolution.md) | Self-judge 何时有帮助，如何验证它没有制造自我确认？ |
| 15 | [评估与安全](tutorials/15-evaluation-safety.md) | 如何比较模型、发现泄漏，并识别 reward hacking？ |
| 16 | [下一阶段与研究问题](tutorials/16-frontier.md) | 怎样把前沿方向转成可证伪、有真实基准的研究问题？ |

补充：[预备知识](tutorials/00-prerequisites.md) · [课程与实践覆盖表](docs/coverage.md) · [源码导读](references/open-training.md) · [Diffusion 专题与原始来源](references/diffusion.md)。

## Diffusion 专题：从概率算法到六类数据

本专题是原有 16 周主线的扩展，按“场景 → 基础 → 公式 → 代码 → 失败原因”学习。

| 阅读顺序 | 内容 | 对应代码 |
|---|---|---|
| [17 基础算法](tutorials/17-diffusion.md) | Gaussian、Markov、DDPM、DDIM、score、VP/VE、reverse SDE、probability-flow ODE、CFG 与 flow matching | [continuous.py](src/fm_tutorial/diffusion/continuous.py)、[models.py](src/fm_tutorial/diffusion/models.py) |
| [17A 文本](tutorials/17a-diffusion-text.md) | embedding/latent 与 token 解码、D3PM 转移和后验、absorbing MASK、双向网络与解掩码 | [discrete.py](src/fm_tutorial/diffusion/discrete.py) |
| [17B 各模态](tutorials/17b-diffusion-modalities.md) | 图像 U-Net/DiT/latent、混合类型表格、时空视频、动态 3D 与多视角约束 | [diffusion_shapes.py](scripts/diffusion_shapes.py)；作者源码导读 |
| [17C 条件生成](tutorials/17c-diffusion-conditioning.md) | cat、相加、FiLM、AdaLN/Zero、cross-attention、条件 token、空间控制，以及 CFG/classifier guidance | [conditioning.py](src/fm_tutorial/diffusion/conditioning.py)、[条件诊断](scripts/conditioning_check.py)、[条件生成演示](scripts/conditioning_demo.py) |

```bash
# 环境沿用本页的 pip install -e .；CPU 可运行
python scripts/diffusion_shapes.py
python scripts/diffusion_demo.py --mode gaussian --sampler ddim --train-steps 300 --out-dir runs/diffusion-gaussian
python scripts/diffusion_demo.py --mode score --sampler ode --train-steps 300 --sample-steps 100 --out-dir runs/diffusion-score
python scripts/diffusion_demo.py --mode masked --train-steps 300 --out-dir runs/diffusion-masked
python scripts/conditioning_check.py
python scripts/conditioning_demo.py --method adaln --train-steps 300 --out-dir runs/conditioning-adaln
```

演示保存模型、实际损失、生成样本和配置；默认使用明确标注的合成教学输入。连续模式可通过 `--data-npy` 接入预处理的真实数据。小演示与 shape 检查用于理解算法；没有在此复现大型图像、视频、4D 模型的训练或基准结果。详细实验路线见 [实践六](assignments/06-diffusion.md)。

## 五项实践交付

| 实践 | 内容 | 交付入口 |
|---|---|---|
| 1 | BPE + Transformer + 基础训练；逐步扩展至约 0.1B | [Foundations](assignments/01-foundations.md) |
| 2 | Attention kernel、profiling、分布式训练与推理测量 | [Systems](assignments/02-systems.md) |
| 3 | 真实语料清洗、数据记录、训练测量与 held-out scaling 外推 | [Data + Scaling](assignments/03-data-scaling.md) |
| 4 | 同基座、同评估协议比较 SFT / DPO / RLVR | [Alignment](assignments/04-alignment.md) |
| 5 | 既有公开环境中的长程 Agent；外部 verifier 与 self-judge 消融 | [Agent](assignments/05-agent.md) |

这些是本教程的学习交付协议，和唐杰课程公开方向对应；不冒充未公开的官方 handout 或评分细则。

## 第一条可运行链路

需要 Python 3.10+。CPU 可以检查实现并跑小配置；约 100M 训练应在自己的 GPU 上依据实际吞吐与预算规划。以下命令从仓库根目录运行。

```bash
git clone https://github.com/Yunbo-max/Foudation-model.git
cd Foudation-model
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data]'

# 下载 TinyStories 官方原始文本版，保留 train / validation
python scripts/download_tinystories.py --out-dir data/raw --max-train 1000 --max-validation 100

# 规范化、过滤、去重；保留官方验证集，删除 train/validation 交叉重复
python scripts/prepare_data.py --input data/raw/train.jsonl --validation-input data/raw/validation.jsonl --out-dir data/clean --min-chars 80

# 只在训练集学习 BPE，转换为 uint32 token 文件
python scripts/train_tokenizer.py --data-dir data/clean --out-dir data/processed --vocab-size 512

# 小配置，先核对运行与 checkpoint；100 步不能证明模型能力
python scripts/train.py --config configs/cpu.json --data-dir data/processed --out-dir runs/cpu --steps 100 --device cpu
python scripts/generate.py --checkpoint runs/cpu/last.pt --tokenizer data/processed/tokenizer.json --prompt 'Once upon a time' --max-new-tokens 64 --temperature 0.8

# 同一目录、配置和数据，恢复到累计 200 个 optimizer updates
python scripts/train.py --config configs/cpu.json --data-dir data/processed --out-dir runs/cpu --steps 200 --device cpu --resume runs/cpu/last.pt
```

TinyStories 是论文作者真实发布的**合成训练语料**，这里获取官方 `TinyStories-train.txt` 与 `TinyStories-valid.txt`，固定解析后的数据 revision 并记录来源文件、哈希和选择规则。原始文本版保留单段落换行，不能与 Parquet 表示混用哈希；也不选用另一个 GPT4V2 版本。第一次用 1000 篇训练文本和 512 词表跑通流程；它不是正式预训练数据规模。真实网页/领域文本可转换为含 `text` 的 JSONL，再使用同一清洗入口。BPE 的清晰 Python 实现适合学习与小语料，处理整个网络语料时应读生产实现和数据工具，不能据此估算工业效率。

约 100M 的架构配置在 [configs/100m.json](configs/100m.json)。训练程序依据实际 tokenizer 覆盖词表大小并打印真实参数量；训练时间、显存和效果必须实际测量。不要直接照抄 CPU 的 100 步作为预训练配方。

## 代码对应什么

| 文件 | 可使用的功能 |
|---|---|
| [tokenizer.py](src/fm_tutorial/tokenizer.py) | 原创 byte-level BPE、训练/编码/解码/持久化、EOS/PAD |
| [data.py](src/fm_tutorial/data.py) | JSONL 规范化、全局去重、官方 split 保留或稳定哈希 split、数据报告 |
| [model.py](src/fm_tutorial/model.py) | Decoder-only Transformer、RoPE、GQA、RMSNorm、SwiGLU、next-token loss |
| [train.py](scripts/train.py) | 真实 token 文件训练、验证、梯度累积、学习率、checkpoint 和恢复 |
| [alignment.py](src/fm_tutorial/alignment.py) | Completion log-prob、SFT/DPO、group advantage 与 GRPO/PPO 核心损失 |
| [scaling.py](src/fm_tutorial/scaling.py) | 对真实训练测量作单轴拟合与 held-out 预测 |
| [agent.py](src/fm_tutorial/agent.py) | 有界工具循环、外部 verifier、advisory self-judge、轨迹日志 |
| [diffusion/](src/fm_tutorial/diffusion/) | 原创 DDPM/DDIM、VP SDE/ODE、categorical 与 masked diffusion，以及小型去噪网络 |

后训练损失是可调用组件；完整 rollout 分布式对齐和真实长程 Agent RL 参照作业指南及官方框架。GPU Triton kernel 的完整 forward/backward、多 GPU、部署压测和研究基准结果需要对应硬件运行，不能由 CPU 单元测试替代。

```bash
python -m unittest discover -s tests -v
python scripts/alignment_check.py
python scripts/check_links.py
```

具体本地验证证据和未运行范围见 [验证记录](docs/validation.md)。

## 如何读 Nemotron、OLMo、DCLM、Pile

它们承担不同职责。Nemotron 是模型族及其发布配方；OLMo/OLMo-core 帮助理解可公开追踪的模型与训练；DCLM 重点是数据质量和可比较的训练评价协议；Pile 帮助理解语料组成与来源。先读 [源码导读](references/open-training.md)，再选择与你当前章节对应的文件。当前 Nemotron 规范入口为 [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)，旧名称不应自动当作当前源码位置。

## 实验记录的最低要求

固定原始数据 revision 与清洗规则、tokenizer、基座权重、代码 commit、评估脚本和成本口径。记录参数数目、训练 tokens、optimizer updates、验证 loss、吞吐和 peak memory。模型 loss 下降、单元测试通过、实际 benchmark 提升是三种不同证据，要分别报告。

Scaling CSV 应来自真实运行，不提供“看起来像训练结果”的预置曲线。SFT/DPO/RLVR 比较保留独立评估集并记录输入模板、completion mask、长度归一化、采样预算、verifier 与 reward 的版本。失败和无效组同样记录，不静默丢弃。

## 来源与授权

[完整来源表](references/sources.md)保留用户提供的新闻链接、课程教材、课程存档、个人复现和官方项目入口。本教程原创新增的代码与文字采用 Apache-2.0；外链数据、权重及仓库使用各自许可。本仓库没有整包搬运第三方个人仓库。
