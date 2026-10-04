# 课程方向、教程与实践覆盖

本表区分“讲解了”“提供了代码”“实际运行过”。实际运行证据只在 [validation.md](validation.md) 中给出，不能把计划写成完成。

| 公开课程方向 | 原创教程 | 本仓库实践与边界 |
|---|---|---|
| Tokenizer + Transformer | 00–02；作业1 | BPE、Decoder-only Transformer、基本训练；近100M配置不等于已经训练100M |
| Training dynamics / scaling | 03；作业3 | 真 CSV 单轴拟合与 holdout；非联合 Chinchilla 定律估计；无伪曲线 |
| Compute / kernels / parallelism | 04；作业2 | 解释 IO-aware attention、online softmax、通信；完整 Triton backward 与多卡实践指向官方CS336/框架 |
| Inference economics | 05；作业2 | 基础生成入口；KV cache、serving、压测在生产源码中阅读并按指南实测 |
| Raw pretraining data | 06；作业3 | 正规化、文档去重与split记录；规模化 near-dedup/quality classifiers 阅读DCLM等 |
| Synthetic data / governance | 07 | 公共数据及生成数据分析；无自制benchmark或教师生成评估标签代替真实评估 |
| SFT / distillation | 08；作业4 | Completion-masked SFT核心损失；蒸馏和全训练配方指导 |
| Preference learning | 09；作业4 | DPO核心损失与reference detach；真实对齐需完整训练数据和官方runner |
| RLVR / reasoning | 10；作业4 | group-relative advantage、clipped policy objective与KL；rollout/奖励/成本协议指导 |
| Long-horizon Agent RL | 11；作业5 | 多步轨迹、信用分配及真实benchmark实验设计；不声称长程RL已训练 |
| Agent foundations | 12；作业5 | 原创有界harness，可传入policy、tools、verifier，记录轨迹 |
| Memory / continual learning | 13 | 各记忆机制解释、来源和公开benchmark；需要任务实验确认有效性 |
| Self-evaluation / evolution | 14；作业5 | Self-judge作为建议，external verifier裁定；不能把自信当正确性 |
| Evaluation / safety | 15 | 评估集隔离、污染、reward hacking及公开工具；测试不证明基准能力 |
| Frontier | 16 | 从真实任务和已知失败出发设计可证伪研究，无拼接式“新颖性”承诺 |
| 本项目新增 Diffusion 专题 | 17、17A、17B；实践6 | DDPM/DDIM、VP score-SDE/ODE、dense categorical 与随机解掩码可运行；六类数据表示与原论文源码导读；未训练大型多模态模型 |
| Diffusion 条件生成扩展 | 17C；实践6 | 八个条件注入教学模块、CFG、结构检查与二维条件生成；SPADE/ControlNet/IP-Adapter/joint attention 有原理与作者源码导读，不冒充完整复现 |

教材按课程地图组织，但讲解顺序、例子、数据选择、验收要求和实现细节属于本项目。新闻报道和个人计划不会覆盖这些明确的边界。
