# 来源、版本与使用边界

核对日期：2026-10-03。正文是原创教程，公式与技术事实以论文、官方文档和实际源码为依据；新闻用于课程背景。源码路径、默认分支、依赖和数据授权可能变化，运行实验前固定 commit、数据 revision、评估版本及配置。

## 用户提供的入口

| 原入口 | 本教程中的位置与用途 |
|---|---|
| [OSCHINA 报道](https://www.oschina.net/news/502554) | 唐杰课程新闻与作业方向的公开报道；不是完整作业 handout |
| [CSDN 文章](https://blog.csdn.net/techforward/article/details/165839597) | 用户给出的二手报道入口，保留溯源；算法解释不依赖它 |
| [观点网报道](https://www.guandian.cn/m/show/603304) | 课程背景，不作为评分细则或算力要求依据 |
| [搜狐报道](https://www.sohu.com/a/1077904253_258768) | 课程背景，不作为完整课程材料 |
| `numtron` | 结合此前讨论解释为 **Nemotron**；规范入口见下表 |
| `olmo` / `DCLM` | 规范入口见下表 |
| `prle` | 此前讨论按 **Pile** 资料展开；这一拼写本身不是已确认的仓库名 |
| `project/public` | 不能唯一定位项目，保留为未确认词，不编造 GitHub 链接 |

## 课程与个人复现

| 来源 | 性质 | 正确使用方法 |
|---|---|---|
| [dujh22/AML-LLM](https://github.com/dujh22/AML-LLM) | 唐杰、杜晋华等的教材项目，Word / PDF / LaTeX | 学习概念；不是 2026 全套作业代码，README 更新日期为 2025-08-09 |
| [2024 课程存档](https://dujh22.github.io/Dujinhua_wiki/aml2024/) / [Wiki 源码](https://github.com/dujh22/Dujinhua_wiki) | 助教公开保存的旧课程网页，列授课教授 Jie Tang、助教及课件链接 | 课程背景与旧版讲义；2024 四项作业不同于 2026 实操路线 |
| [jackmcgradylee/from-token-to-agent](https://github.com/jackmcgradylee/from-token-to-agent) | 个人参考 2026 地图开展复现与扩展；Apache-2.0 | 课程地图、部分基础代码及实验组织；不能把其扩展当官方要求 |
| [课程照片拼图](https://github.com/jackmcgradylee/from-token-to-agent/blob/main/docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg) | 公开课堂照片，能看到 16 周地图 | 课程主线证据；不是完整 PPT 下载，本仓库只链接 |
| [HW1 实验报告](https://github.com/jackmcgradylee/from-token-to-agent/blob/main/assignments/hw1-foundations/report.md) | 个人提交的具体进度 | 明写正式 0.1B 训练待完成；与 README 的“完成”标记有出入，依具体报告判断 |
| [LuckVd/aml-llm-lab](https://github.com/LuckVd/aml-llm-lab) | CPU 优先教学脚手架；默认分支 master，未确认独立许可证文件 | 比较练习组织；其 scaling/system presets 明示为合成，05–08 测试/demo 使用离线替身；本项目不复制其代码、不引用为能力结果 |
| [sunccchengze/zixue2026](https://github.com/sunccchengze/zixue2026) | 个人自学计划，映射唐杰方向与 CS336 | 路线参考；多个题目未实现，不当作原课程代码 |

完整 2026 官方课件＋完整 handout＋统一 starter code 仓库在此前公开检索中未找到。本教程的验收协议、数据选择和实现都是本项目明确给出的学习设计，不冒充官方评分标准。

## 官方开源训练体系

详细文件导读见 [open-training.md](open-training.md)。

| 项目 | 当前入口 | 主要角色 |
|---|---|---|
| Nemotron | [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) | NVIDIA 模型族的数据、训练配方、评估等发布；代码 Apache-2.0，模型/数据另查 model/data card |
| NeMo 生态 | [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) | 当前按组件拆分；具体 LLM 训练入口见官方源码导读，不假设旧 NeMo 主仓库仍代表整条 LLM 链路 |
| OLMo | [allenai/OLMo](https://github.com/allenai/OLMo) | 早期完全开放模型与训练实现 |
| OLMo-core | [allenai/Olmo-core](https://github.com/allenai/Olmo-core) | 新的训练基础设施、模型及配置，Apache-2.0 |
| Open-Instruct | [allenai/open-instruct](https://github.com/allenai/open-instruct) | SFT、偏好对齐及 RL 的实现和配方 |
| DCLM | [mlfoundations/dclm](https://github.com/mlfoundations/dclm) | 数据整理、数据集评估与训练/评价协议，代码 MIT；数据另查授权 |
| Pile | [EleutherAI/the-pile](https://github.com/EleutherAI/the-pile) | 多来源文本集的采集、组成与处理代码，代码 MIT；不是一个模型训练框架，也不是各数据源授权的总许可 |

## 课程作业与算法一手材料

- [CS336 课程](https://stanford-cs336.github.io/)；[A1 basics](https://github.com/stanford-cs336/assignment1-basics)、[A2 systems](https://github.com/stanford-cs336/assignment2-systems)、[A3 scaling](https://github.com/stanford-cs336/assignment3-scaling)、[A4 data](https://github.com/stanford-cs336/assignment4-data)、[A5 alignment](https://github.com/stanford-cs336/assignment5-alignment)。不同作业 README 标注年份可能不同，具体以仓库文件及 commit 为准。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[Chinchilla](https://arxiv.org/abs/2203.15556)。
- [InstructGPT](https://arxiv.org/abs/2203.02155)、[DPO](https://arxiv.org/abs/2305.18290)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[PPO](https://arxiv.org/abs/1707.06347)。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)、[TRL 文档](https://huggingface.co/docs/trl/index)。
- [TinyStories 数据集](https://huggingface.co/datasets/roneneldan/TinyStories) / [论文](https://arxiv.org/abs/2305.07759)：真实发布、可下载的合成训练语料；“公开合成语料”与本地伪造评估结果不是一回事。
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)、[GSM8K](https://github.com/openai/grade-school-math)、[MATH](https://github.com/hendrycks/math)、[SWE-bench](https://github.com/SWE-bench/SWE-bench)、[AgentBench](https://github.com/THUDM/AgentBench)。精确 task/config/版本应和模型实验一起锁定。

## 引用原则

新增 Diffusion 专题的原始论文与作者源码集中于 [diffusion.md](diffusion.md)，核对日期为 2026-10-04；覆盖连续/离散文本、图像、表格、视频和动态 3D。它是本项目扩展，不改写原课程公开要求。

每章末给出相关来源。不会把只有目录的项目说成已实现，也不会把单元测试通过说成模型有效，更不会把原文中的硬件预算和教学扩展改写成唐杰课程要求。外链仓库不会自动随本教程锁定；复现实验记录其 commit 即可。
