# 开放训练源码导读：从数据到模型，再到后训练与 Agent

本导读在 2026-10-03 核验公开仓库 README、目录树和关键文件名。
链接指向当时默认分支的真实入口；分支会更新，正式实验需另行记录 commit 与依赖锁。
下面只给原创阅读路线与源链接，不批量复制第三方正文或代码。
本仓库中的小型实现不等于复现这些项目的规模、硬件、数据或发布成绩。

## 1. 先用角色地图避免选错仓库

| 项目 | 主要角色 | 适合学习 | 不应直接推断 |
| --- | --- | --- | --- |
| Nemotron | 模型家族配方与开发流程 | 数据→预训练→SFT→RL→部署的组织 | 公开配方包含完整报告所有数据 |
| NeMo / 当前拆分库 | GPU 训练和后训练基础设施 | 并行、转换、采样服务与运行配置 | 一个历史 NeMo 入口覆盖当前全部组件 |
| OLMo | 早期模型与训练实现 | 模型、配置和旧训练流程 | 当前最新发布仍在原库维护 |
| Olmo-core | 当前 OLMo 生态训练模块 | 模型构造、数据加载、训练器和官方脚本 | 下载代码即可在 CPU 复现发布模型 |
| open-instruct | 后训练与评估 | SFT、DPO、RLVR、数据与污染检查 | 后训练配方等于从零预训练 |
| DCLM | 数据筛选与控制训练评测 | 数据清洗、去重及固定预算比较 | 各版本聚合分数可直接比较 |
| The Pile | 多来源语料构建与复制代码 | 数据谱系、混合、清洗和分割 | 代码许可证覆盖所有原始语料 |
| Stanford CS336 | 官方课程与作业接口 | 从原理到实现及行为检查 | 学生作业完成等于工业训练已复现 |

数据仓库、模型仓库、训练框架和后训练仓库互相依赖，但回答的问题不同。
先确定你要审查哪个环节，再追踪其输入输出；不要试图从一份 README 获得全部训练事实。

用户原文中的 `numtron`、`prle` 未提供精确 URL；本次分别检索 Nemotron、The Pile 作为可能对应的候选。
这是根据名称与训练语境作出的检索推断，没有得到用户确认，不能把候选名称当作原引用的确定身份。
`project/public` 也尚未定位到唯一仓库；保留这个未解决项，不虚构链接或源码说明。

## 2. NVIDIA Nemotron：看完整配方如何分层

检索中尝试的候选路径 `NVIDIA/Nemotron` 在本次 API 核验返回 404。
GitHub 搜索和当前 README 确认规范入口为 [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)。
不要猜测路径补一个能打开的 URL，更不要把相似命名的第三方仓库当官方来源。

建议按以下顺序阅读：

1. [README.md](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/README.md)：区分开发步骤、训练配方、使用 cookbook 与用例。
2. [docs/steps/basics.md](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/steps/basics.md)：理解 step、配置、运行环境与产物的边界。
3. [Nano 3 配方](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/README.md)：将一个模型的各阶段连接起来。
4. 分别读 [pretrain.md](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/pretrain.md)、[sft.md](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/sft.md)、[rl.md](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/rl.md)，记录数据、初始化检查点与训练后端。
5. 到 [src/nemotron/steps/_runners/nemo_rl.py](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/_runners/nemo_rl.py) 看流程层如何调用 RL 后端。

阅读问题：每阶段产物是什么？下一阶段怎样选 checkpoint？评测是训练过程内反馈还是独立验收？
这些问题把“有一段 RL 代码”与“有可复现分阶段训练流程”区分开。
[Super 3](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md)、[Ultra 3](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/ultra3/README.md) 和 [Lightning 3.5](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/lightning35/README.md) 提供其他真实配方入口。

官方 README 明确某些配方使用公开数据子集，发布技术报告还使用额外数据，结果会不同。
因此记录“读过配方”或“运行公开子集”时，不能声称完整复现技术报告成绩。
代码 [LICENSE](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/LICENSE) 为 Apache-2.0；模型权重、数据和附属资源仍应读取各自卡片与许可证。

## 3. NeMo：历史入口与当前拆分必须一起读

[NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) 与旧 `NVIDIA/NeMo` 的 API 均返回迁移提示。
当前规范仓库是 [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech)。
其 [README](https://github.com/NVIDIA-NeMo/Speech/blob/main/README.md) 说明原仓库已聚焦音频、语音和多模态语音，拆分前最后发布是 v2.7.3。
复现历史 NeMo 2.x 训练流程时应使用对应 tag 与容器，而不是照旧教程安装最新主分支。

当前 LLM 训练源码应沿官方拆分角色阅读：

| 入口 | 重点 | 一个可查文件 |
| --- | --- | --- |
| [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Megatron 训练与 Hugging Face 转换 | [docs/training/README.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/training/README.md) |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | 分布式后训练、生成与优化 | [nemo_rl/algorithms/grpo.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/algorithms/grpo.py) |
| [Speech](https://github.com/NVIDIA-NeMo/Speech) | ASR、TTS 与语音模型 | [README.md](https://github.com/NVIDIA-NeMo/Speech/blob/main/README.md) |

Megatron-Bridge 的 [src/megatron/bridge/models/README.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/README.md) 帮助理解模型封装与转换层。
NeMo RL 从 [examples/run_grpo.py](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_grpo.py) 进入，再阅读 [grpo_math_1B.yaml](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml) 和 [loss_functions.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/algorithms/loss/loss_functions.py)。
把入口、资源配置、采样与损失连接起来，检查推理后端与训练后端是否使用相同权重版本。
这些 GPU 运行栈与本仓库 CPU 数学演示的依赖不同；不建议未经核对混装所有最新版。
各库分别查看 LICENSE 和第三方 notices，模型与数据许可不能仅从框架许可推断。

## 4. OLMo 原库：理解历史实现，也尊重维护状态

[allenai/OLMo](https://github.com/allenai/OLMo) 在 API 中不是 archived，但其 [README](https://github.com/allenai/OLMo/blob/main/README.md) 明确写出原库已过时、不再活跃，并指向 OLMo-core。
“未归档”与“最新模型在这里维护”不是同一事实。
原库仍适合读早期预训练配置与模型结构，不应被当成当前新模型的唯一入口。

建议阅读：

1. [olmo/config.py](https://github.com/allenai/OLMo/blob/main/olmo/config.py)：模型、数据和训练配置之间如何连接。
2. [olmo/model.py](https://github.com/allenai/OLMo/blob/main/olmo/model.py)：注意力、归一化、块和输出头的实际组织。
3. [olmo/data/memmap_dataset.py](https://github.com/allenai/OLMo/blob/main/olmo/data/memmap_dataset.py)：token 数据如何加载与切片。
4. [scripts/train.py](https://github.com/allenai/OLMo/blob/main/scripts/train.py) 与 [olmo/train.py](https://github.com/allenai/OLMo/blob/main/olmo/train.py)：入口脚本和训练器如何分工。
5. [configs/official-0724/OLMo-1B.yaml](https://github.com/allenai/OLMo/blob/main/configs/official-0724/OLMo-1B.yaml)：从实际配置核对实现参数。

阅读时分别标出输入 token、标签移位、损失归约、数据游标和 checkpoint 状态。
对照本仓库模型时，寻找设计差异及其理由，不复制规模相关配置后假定 CPU 可运行。
代码 [LICENSE](https://github.com/allenai/OLMo/blob/main/LICENSE) 为 Apache-2.0；模型、语料和版本说明另行核对。

## 5. Olmo-core：模型、数据、训练模块与官方脚本

规范大小写入口为 [allenai/Olmo-core](https://github.com/allenai/Olmo-core)；`OLMo-core` URL 能解析到同一项目。
[README](https://github.com/allenai/Olmo-core/blob/main/README.md) 将其定位为 OLMo 生态的 PyTorch 构件，列出当前官方发布训练脚本。
大型系统把模型计算、分布式训练模块、数据来源和运行器分开，便于替换并行方式而不重写模型。

可依次阅读：

1. [src/examples/llm/train.py](https://github.com/allenai/Olmo-core/blob/main/src/examples/llm/train.py)：先获得整体装配图。
2. [src/olmo_core/nn/transformer/config.py](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/nn/transformer/config.py) 与 [model.py](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/nn/transformer/model.py)：配置到模块。
3. [src/olmo_core/data/data_loader.py](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/data_loader.py)：数据状态、批次与训练进度。
4. [src/olmo_core/train/train_module/transformer/train_module.py](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/train/train_module/transformer/train_module.py)：计算与训练行为的边界。
5. [src/olmo_core/train/trainer.py](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/train/trainer.py)：训练循环、回调与恢复。

再读 [OLMo2 官方目录](https://github.com/allenai/Olmo-core/tree/main/src/scripts/official/OLMo2) 和 [OLMo3 官方目录](https://github.com/allenai/Olmo-core/tree/main/src/scripts/official/OLMo3)。
已核验实际文件包括 [OLMo-2-0325-32B-train.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py) 与 [OLMo-3-1025-7B-midtrain.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py)。
后训练可读 [src/scripts/train/sft/README.md](https://github.com/allenai/Olmo-core/blob/main/src/scripts/train/sft/README.md)。
代码 [LICENSE](https://github.com/allenai/Olmo-core/blob/main/LICENSE) 为 Apache-2.0，运行资源要求与容器依赖以该版本 README 为准。

## 6. open-instruct：从监督到偏好，再到可验证奖励

[allenai/open-instruct](https://github.com/allenai/open-instruct) 自称 AllenAI 后训练代码库。
它承接基座模型与训练数据，关注指令、偏好、RLVR 和评估；与 OLMo 预训练库的职责不同。
[README](https://github.com/allenai/open-instruct/blob/main/README.md) 还说明支持模型的 SFT 可使用 Olmo-core 实现，表明组件边界并非仓库名称边界。

阅读顺序：

1. [open_instruct/dataset_processor.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_processor.py)：原始记录如何变成训练实例。
2. [open_instruct/finetune.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py)：聊天模板、掩码和 SFT。
3. [open_instruct/dpo.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo.py) 与 [dpo_utils.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_utils.py)：偏好对、参考策略和归约。
4. [open_instruct/grpo.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/grpo.py) 与 [grpo_fast.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/grpo_fast.py)：采样、异步资源与策略更新。
5. [open_instruct/ground_truth_utils.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/ground_truth_utils.py)：奖励怎样连接任务答案和程序判定。

阅读问题：prompt、completion 和工具观察的边界在哪里？谁产生奖励？无效样本怎么处理？训练后评测是否独立？
命名为 GRPO 的实现也可能有版本和目标函数差异，要按实际代码写公式。
本仓库 [alignment.py](../src/fm_tutorial/alignment.py) 只演示核心损失，不能替代上述采样和分布式系统。
代码 [LICENSE](https://github.com/allenai/open-instruct/blob/main/LICENSE) 为 Apache-2.0；README 明确不同模型发布还有各自许可及基座模型条件。

## 7. DCLM：把数据选择变成控制实验

[mlfoundations/dclm](https://github.com/mlfoundations/dclm) 提供 DataComp-LM 的数据处理、训练和评测流程。
核心问题是相同训练协议与计算规模下，如何选择更好的数据，而不是把模型和预算都换掉后归因于数据。
先读 [README](https://github.com/mlfoundations/dclm/blob/main/README.md) 的 filtering/mixing 轨道与工作流。

源码入口：

1. [baselines/README.md](https://github.com/mlfoundations/dclm/blob/main/baselines/README.md) 与 [dclm_baseline_refinedweb.yaml](https://github.com/mlfoundations/dclm/blob/main/baselines/baselines_configs/dclm_baseline_refinedweb.yaml)：清洗流水线配置。
2. [baselines/mappers/filters/content_filters.py](https://github.com/mlfoundations/dclm/blob/main/baselines/mappers/filters/content_filters.py)：过滤判据如何进入代码。
3. [dedup/README.md](https://github.com/mlfoundations/dclm/blob/main/dedup/README.md)：去重方法与实现依赖。
4. [rust_processing/tokshuf-rs/README.md](https://github.com/mlfoundations/dclm/blob/main/rust_processing/tokshuf-rs/README.md)：tokenization 和 shuffle 的系统实现。
5. [training/train.py](https://github.com/mlfoundations/dclm/blob/main/training/train.py) 与 [eval/aggregated_metrics.py](https://github.com/mlfoundations/dclm/blob/main/eval/aggregated_metrics.py)：训练入口及聚合分数。

注意 README 在 2025-09 说明 centered CORE/EXTENDED 的基线修正，提供 v1/v2 字段与聚合迁移方式。
新旧聚合值不能不加说明直接比较；报告必须标出指标版本。
训练数据规模、固定模型配置和 held-out 评估共同支持数据筛选结论。
本仓库 scaling 脚本只消费真实测量，不用其他教学库的合成 presets 代替 DCLM 训练证据。
代码 [LICENSE](https://github.com/mlfoundations/dclm/blob/main/LICENSE) 为 MIT；下载数据时读取相应数据发布条件。

## 8. The Pile：看数据谱系，不把代码许可当语料许可

[EleutherAI/the-pile](https://github.com/EleutherAI/the-pile) 的默认分支为 `master`，不是 `main`。
[README](https://github.com/EleutherAI/the-pile/blob/master/README.md) 将其定位为 Pile 复制代码，旧下载入口与处理流程需逐项验证可用性。
代码存在不意味着原始组件今天仍可下载，也不意味着每个组件允许任意用途。

建议阅读：

1. [the_pile/datasets.py](https://github.com/EleutherAI/the-pile/blob/master/the_pile/datasets.py)：每个来源怎样加载与解析。
2. [the_pile/pile.py](https://github.com/EleutherAI/the-pile/blob/master/the_pile/pile.py)：来源、权重及混合过程。
3. [processing_scripts/README.md](https://github.com/EleutherAI/the-pile/blob/master/processing_scripts/README.md)：历史处理步骤。
4. [processing_scripts/dedupe_train.py](https://github.com/EleutherAI/the-pile/blob/master/processing_scripts/dedupe_train.py) 与 [pass2_shuffle_holdout.py](https://github.com/EleutherAI/the-pile/blob/master/processing_scripts/pass2_shuffle_holdout.py)：去重和分割。

对每个组件记录来源、时间、采样权重、清洗、许可和不可用情况。
代码 [LICENSE](https://github.com/EleutherAI/the-pile/blob/master/LICENSE) 为 MIT，不能据此断言来源书籍、代码、论文和邮件全部同为 MIT。
本教程不分发原语料；使用合法可取得的数据子集时，仍需说明与原 Pile 的差异。

## 9. Stanford CS336：官方作业作为实现检验入口

课程主页：[Language Modeling from Scratch](https://cs336.stanford.edu/)；官方 GitHub 组织：[stanford-cs336](https://github.com/stanford-cs336)。
本教程中的 CS336 指这套官方课程；同名个人复现库与官方来源需要分开。
本次目录核验的实际手册如下，文件名不可把旧年份机械替换为新年份。

| 作业 | README | 当前真实手册文件 |
| --- | --- | --- |
| 1 基础 | [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics/blob/main/README.md) | [cs336_assignment1_basics.pdf](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf) |
| 2 系统 | [assignment2-systems](https://github.com/stanford-cs336/assignment2-systems/blob/main/README.md) | [cs336_assignment2_systems.pdf](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf) |
| 3 Scaling | [assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling/blob/main/README.md) | [cs336_assignment3_scaling.pdf](https://github.com/stanford-cs336/assignment3-scaling/blob/main/cs336_assignment3_scaling.pdf) |
| 4 数据 | [assignment4-data](https://github.com/stanford-cs336/assignment4-data/blob/main/README.md) | [cs336_assignment4_data.pdf](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf) |
| 5 对齐 | [assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment/blob/main/README.md) | [cs336_spring2026_assignment5_alignment.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf) |

Assignment 5 README 明确为 Spring 2026，并链接 [safety/RLHF supplement](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)。
先读手册规范，再写自己的实现，最后通过 adapters 连接课程行为检查。
课程测试可以帮助验证数学与接口，不能被用来证明工业训练或公开能力分数。
不同作业仓库的 LICENSE 与附属手册权利应逐项查阅；本次作业 5 根目录未见 LICENSE，不能推定与前四个库相同。

## 10. 唐杰课程、教材和个人实践分别怎么用

[dujh22/AML-LLM](https://github.com/dujh22/AML-LLM) 是唐杰、杜晋华及合作者教材项目。
阅读 [README_ZH.md](https://github.com/dujh22/AML-LLM/blob/main/README_ZH.md) 的内容结构，再看 [latex/latex/book.pdf](https://github.com/dujh22/AML-LLM/blob/main/latex/latex/book.pdf) 与 [book.tex](https://github.com/dujh22/AML-LLM/blob/main/latex/latex/book.tex) 对应章节。
它帮助建立基础概念和课程脉络，不应直接视为 2026 全部实验的已运行记录。
README 声称 MIT，但本次根目录树未见其链接的 LICENSE，另有 [latex/latex_raw/LICENSE](https://github.com/dujh22/AML-LLM/blob/main/latex/latex_raw/LICENSE)；不要据此整包复制教材。

[AML 2024 课程归档](https://dujh22.github.io/Dujinhua_wiki/aml2024/) 是 TA 主页发布的历史课程页面。
页面说明授课教授为 Jie Tang、核心 TA 为 Jinhua Du，列出基础、进阶、应用专题及原课件入口。
它能核对历史课程内容；2024 的作业、课时和主题不能自动当作 2026 的安排。
[主页](https://dujh22.github.io/Dujinhua_wiki/) 的 AML 链接也指向该归档。

[jackmcgradylee/from-token-to-agent](https://github.com/jackmcgradylee/from-token-to-agent) 是基于 2026 课程地图的个人复现。
读 [COURSE.md](https://github.com/jackmcgradylee/from-token-to-agent/blob/main/COURSE.md) 获取其整理的学习路线，再按 [README](https://github.com/jackmcgradylee/from-token-to-agent/blob/main/README.md) 和报告核查实际进展。
规划、部分实现和实际实验分开：约 100M 正式训练仍待执行，后训练及 Agent 工作不能从目录名字推断完成。
本教程仅将它作为课程地图和个人实现对照，不沿用其分数为本项目结果。

[LuckVd/aml-llm-lab](https://github.com/LuckVd/aml-llm-lab) 是 CPU 优先教学脚手架。
读 [README](https://github.com/LuckVd/aml-llm-lab/blob/master/README.md)、[labs/08-agent/README.md](https://github.com/LuckVd/aml-llm-lab/blob/master/labs/08-agent/README.md) 与 [presets/README.md](https://github.com/LuckVd/aml-llm-lab/blob/master/presets/README.md)。
其 README 明确 presets 为合成数据，后训练/Agent 小测试与 demo 使用离线替身；这些适合接口教学，不能充当真实训练或公开任务证据。
本次根目录未发现许可证，保留链接学习，不整包复制代码。

## 11. 用户提供的四篇报道：保留来源，限定用途

| 来源 | 入口 | 在本教程中的角色 |
| --- | --- | --- |
| OSCHINA | [news/502554](https://www.oschina.net/news/502554) | 课程与行业讨论的报道线索；本次抓取未取得完整可核正文 |
| CSDN techforward | [165839597](https://blog.csdn.net/techforward/article/details/165839597) | 全链路课程报道背景，算法仍核对原论文与官方实现 |
| 观点网 | [603304](https://www.guandian.cn/m/show/603304) | 2026-09-17 关于记忆、自我改进与长程任务愿景的报道 |
| 搜狐 | [1077904253_258768](https://www.sohu.com/a/1077904253_258768) | 用户指定报道线索；本次正文抓取失败，未据其建立技术结论 |

新闻可以说明某个观点被报道，不能证明一种训练算法有效、某个模型达到能力水平或本项目复现成功。
本教程中的算法事实来自原论文或官方代码；本项目的完成与性能结论只能来自本项目实际验证记录。

## 12. 阅读产物：每个项目写一页源码笔记

记录版本、角色、输入、输出、关键控制流、一个数学目标、一个实现边界及一个待验证问题。
把模型结构、数据处理、训练循环、后训练和评测各选一个入口，不要求把所有框架都安装一遍。
遇到历史路径、重命名或 README 与树不一致时，优先核对目录树，并在笔记写出差异。
提交原创实现与笔记，引用源链接；没有许可的内容不因公开可见就自动可再分发。

返回 [课程导航](../README.md)；Agent 实践见 [作业 05](../assignments/05-agent.md)。
