# 作业 04：同 Base、同 Eval 的 SFT／DPO／RLVR 对照

本作业对应公开课程地图 HW4 的 Controlled Comparison，并连接第 08–11 章。
目标是研究不同学习信号在同一个基座和同一个真实任务上的作用，保存可审计数据、训练和评价记录。
本仓库实现 SFT/DPO/GRPO 数学核心，完整真实模型训练需要串接官方训练器与评测代码。

## A. 三个必须先回答的问题

**问题一：是示范格式改善，还是解题能力改善？** 格式有效率与答案准确率分开记录。
模型能输出最终答案标记，并不代表答案正确。

**问题二：是算法改善，还是起点更强？** 所有分支从同一个 Base 起步，DPO 与 RLVR 使用同一个共享 SFT checkpoint。
不能用 Instruct 模型作一个分支、Base 模型作另一个分支，再称为方法对照。

**问题三：是学习效率改善，还是花了更多生成成本？** 训练步数相同不能保证总成本相同。
教师推理、候选生成、reference 评分、rollout、verifier 和最终评价都要记账。

主线使用公开数学任务，避免用自制十题、脚本化假回答或合成能力表充当 benchmark。
数学目标的结论不自动推广到安全、对话偏好或长程 Agent。

## B. 预注册实验协议

在生成训练数据之前保存如下选择，后续变更要版本化并说明理由。

| 项目 | 主线选择与需要固定的内容 |
| --- | --- |
| Base | [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) Base，固定 revision |
| 数据 | [MATH](https://github.com/hendrycks/math) 官方 train，固定来源与内容摘要 |
| 开发集 | 从 train 按 prompt ID 固定 dev，所有分支共用 |
| 最终评价 | MATH 官方 test 与 [GSM8K](https://github.com/openai/grade-school-math) test |
| evaluator | 同一 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) commit/任务配置 |
| 提示与输出 | 固定 prompt、答案格式、few-shot、停止词、输出长度及解码 |
| 参数更新 | 全参数或 LoRA；若 LoRA，rank/target modules 对各分支一致 |
| 总预算 | 数据准备、学生更新与推理成本分开，声明控制的成本口径 |

模型卡只是来源说明，不是本机训练可行性的保证。先在自己的机器上实测单 batch 内存与 throughput。
所有性能数字只能记录实际测量；本教材不保证“某张卡几小时完成”。
公开模型上游可能见过公开 benchmark，记录已知训练信息并报告污染限制。
不要声称本作业 test 完全不可能出现在模型预训练中。

## C. 分支结构与公平起点

保存 Base 为 $\theta_0$，按固定示范训练得到共享 $\theta_S$。

| 分支 | 起点 | 额外信号 | 解释范围 |
| --- | --- | --- | --- |
| Base | $\theta_0$ | 无 | 训练前行为与成本基线 |
| SFT | $\theta_0$ | 原始公开训练解答 | 示范学习作用 |
| SFT-continued | $\theta_S$ | 同题池更多示范更新 | 控制更多学生更新的影响 |
| SFT→DPO | $\theta_S$ | 固定离线 chosen/rejected | 偏好阶段的额外作用 |
| SFT→RLVR | 同一 $\theta_S$ | 在线 rollout 与 verifier | 在线探索阶段的额外作用 |

数学对照中，DPO pair 可由答案正确性生成，因此研究的是正确性派生偏好，而非通用人类价值对齐。
RLVR 的训练 prompt 池与 DPO/SFT 的源题池一致，但其候选会在线改变，这是算法差异的一部分。
若另比较直接从 Base RLVR，单独列分支，不能混入共享 SFT 起点的比较。
共同 SFT 的 checkpoint hash 必须在 DPO 与 RLVR run metadata 中完全相同。

## D. 数据整理和防泄漏

先按题目 ID/内容摘要划分 train/dev，再生成示范、候选或 rollout。
同题不同回答不能跨 split；官方 test 不参与 teacher prompt、pair 构造或 reward 阈值调优。
保留公开 dataset 的原始 record ID、类型、难度、题目、标准解答和最终答案。
MATH 原始 solution 的数学结构应由可信处理器解析，不能仅取最后一个数字。

SFT 可用原始训练 solution；若用教师解答，记录 teacher revision、采样预算、验证规则和接受率。
DPO 每条记录包含相同 prompt、chosen、rejected、两个候选 ID 和标签来源。
构造正确/错误 pair 时保留平局，并报告无正确候选、全正确和有效 pair 题目的比例。
不能为全错题捏造 chosen，也不能随机给全对候选贴正确性偏好。

RLVR 标准答案只作为训练 verifier 的输入，不拼进 policy prompt。
答案条件修复或蒸馏是另一种信号，若使用必须另列实验条件。
生成日志和失败请求要完整保存，避免隐藏多次试错成本。
数据层不得用教学脚手架的离线剧本替身补齐缺失候选。

## E. 响应遮罩与 tokenization 验收

输入 ID 与 completion mask 均为 `[B,T]`，logits 为 `[B,T,V]`。
预测目标来自 `input_ids[:,1:]`，对应 logits `[:,:-1,:]` 与 mask `[:,1:]`。
prompt/system/user/PAD 为零，目标回答与协议规定的第一个 EOS 为一；mask 第一列必须为零。
prompt 不算损失，仍通过因果 attention 作为回答条件。

在真实 collator 上抽若干**训练侧**样例，解码并标出以下位置：

- prompt 最后一个 token 与第一个 completion token。
- assistant header、结束符、真实 EOS 和 PAD。
- 最长样本的截断点，最终答案是否仍保留。
- 多轮样本中每个 assistant/tool 区间。

不能用字符数推断 token 边界；完整序列 BPE 可能在字符串边界合并。
检查训练与评分共用模板、tokenizer、截断及 mask。
这些小样例用于程序正确性，不构成研究 benchmark 或能力测量。

## F. 损失接口和数学诊断

本仓库 [`alignment.py`](../src/fm_tutorial/alignment.py) 提供：

| 接口 | 输入核心尺寸 | 返回及约定 |
| --- | --- | --- |
| `sft_loss` | logits `[B,T,V]`、IDs/mask `[B,T]` | 有效回答 token 平均 NLL |
| `completion_log_probs` | 同上 | `[B]` sequence SUM，已做 next-token shift |
| `dpo_loss` | 四个 `[B]` sequence SUM | pair 平均；reference detach |
| `group_advantages` | reward `[G,K]` | advantage `[G,K]`、valid groups `[G]` |
| `grpo_loss` | token log-prob/mask `[G,K,T]` 或 `[G*K,T]` | active-token 平均 PPO 代理损失 |

```bash
python -m unittest discover -s tests -p 'test_alignment.py' -v
python scripts/alignment_check.py --device cpu
```

这些检查只证明数学接口与数值行为，不加载真实基座，不产生对齐训练结果。
完整训练可使用固定版本的 [TRL](https://github.com/huggingface/trl)、[Open-Instruct](https://github.com/allenai/open-instruct) 或 [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)。
本仓库不复制第三方完整训练器，不承诺一个命令自动完成真实三路对齐。
安装外部工具后记录版本、依赖和实际命令，禁止把多个版本的默认行为混在同一方法名下。

## G. SFT 分支的实施要求

使用真实 prompt/solution，构造回答 mask，记录有效训练 token 数和截断率。
选择 token 平均或序列平均，并在所有相关比较中保持一致。
本仓库默认 token 平均：长回答的贡献与长度成正比。
若采用序列平均作为消融，另列分支，不悄悄替换目标。

训练日志包括 NLL、学习率、grad norm、有效 token、真实参数数、checkpoint hash 与开发集 loss。
多 microbatch 长度不同时，确认梯度累积实现的全局归一化口径。
先用 dev 选学习率、长度与 epoch，再冻结最终方案。
不要通过反复查看官方 test 选择 SFT epoch。

## H. DPO 分支的实施要求

从共享 $\theta_S$ 生成离线 pair，冻结参考模型为同一 $\theta_S$。
policy 和 reference 对相同 token 序列、同一完成区间评分，使用序列 log-prob **和**。
原始 DPO 的 margin 为 policy chosen/rejected 差减 reference 差，再乘 $\beta$。
平均 token log-prob 是另一种目标，若测试它需另列变体。

reference 权重不更新，评分不留梯度，缓存键包含样本、模板、mask 和 reference revision。
检查 adapter 共享不会使 reference 跟着 policy 更新。
记录 chosen/rejected log-prob、margin、pair accuracy、长度差、截断率和训练 pair 覆盖。
监控绝对 chosen likelihood，避免 margin 提升掩盖两支概率共同退化。
使用 `logsigmoid` 保持稳定，初始 policy=reference 时 loss 应与 $\log2$ 相符。

## I. RLVR 分支的实施要求

从相同 $\theta_S$ 开始，每题采样 $K$ 个回答，对真实训练答案运行冻结 verifier。
记录 correctness reward 与 format reward，明确权重；报告 accuracy 时只使用对应正确性指标。
GRPO 组内标准差采用 population convention；constant reward 组与 $K=1$ 组无相对信号。
记录全对、全错和有效组比例，将无效组从 loss mask 中排除。
全 batch 无效时跳过更新，避免 optimizer momentum/weight decay 在零 loss 下仍改变 policy。

对每条 rollout 保存 policy revision、IDs、old token log-prob、采样参数和终止原因。
old policy 是该 rollout 的分母，reference 是 KL 锚点，二者不能互换。
同 batch 多轮 PPO 更新时，old 分数保持固定，不用 current 分数覆盖。
若使用异步推理引擎，记录 policy lag 与 rollout/training log-prob 差异及可信修正设置。

记录 ratio、clip fraction、sampled KL surrogate、回答长度和 truncation。
本仓库 loss 为 token 平均；原始 GRPO 的回答长度归一/组平均是不同 reduction。
可选 k3 KL 在 old-policy 样本上未经修正不是当前 policy KL 的无偏估计。
不得把 reward 上升当作 benchmark 提升；高 reward 样本要检查奖励漏洞。

## J. 可信评价命令与协议

固定 harness commit，并安装其 HF backend；按该版本官方文档核对命令。
下列命令使用当前公开 CLI，运行的是既有任务，没有自造评分题。

```bash
lm-eval ls tasks
lm-eval run --model hf \
  --model_args pretrained=Qwen/Qwen2.5-Math-1.5B \
  --tasks gsm8k hendrycks_math --device cuda:0 --batch_size 1 \
  --output_path runs/eval/base --log_samples --show_config
```

正式复现实验在 `model_args` 中固定 revision，各训练分支把 `pretrained` 换成保存的 HF checkpoint 路径。
LoRA 分支使用该 harness 版本支持的 adapter 方式，或者可靠合并后的 checkpoint，并记录相应处理。
SFT/DPO/RLVR 的 checkpoint 命名只是标识，必须能对应到训练日志和实际权重摘要。

固定 `gsm8k` 的 strict-match/flexible-extract 选择，报告实际 few-shot、生成长度与停止词。
`hendrycks_math` 与 `minerva_math` 的 prompt/数学判断协议不同，`hendrycks_math500` 的集合也不同。
选择一个并贯穿各分支，不能挑每个分支最有利的版本。
以上 Base 命令不自动应用 chat template；若选择 chat 评价，所有分支需采用同一预先声明模板。
用 `--limit` 做环境检查时，结果仅为 smoke check，不提交为完整 benchmark 得分。

## K. 等成本比较与原始结果表

提交逐 run 的实际成本，而不是只写“每种方法训练 100 步”。

| 记录项 | 为什么需要 |
| --- | --- |
| teacher/pair/rollout 推理 tokens | 训练信号获取并非免费 |
| reference/policy scoring tokens | DPO 与 RLVR 有额外前向成本 |
| 更新 token 与优化步 | 不同回答长度使每步工作量不同 |
| GPU/CPU 时间、硬件与峰值内存 | 把实际效率与算法效果分开 |
| verifier 时间与失败率 | 环境故障不能冒充任务错误 |
| eval tokens 与样本数 | best-of-K/voting 成本不同于单答 |

若预算难以严格匹配，展示多个预算下的性能—成本测量，并明确比较不是精确等成本。
端到端成本包含共享 SFT；额外阶段成本可以另报，但不能取代总成本。
greedy、sampling、best-of-K 和 voting 分栏，禁止把多答提升归为训练提升。
保存全部官方逐题输入、输出、停止原因、verdict 与汇总 JSON。
没有运行的分支写“未运行”，没有结果的单元格留空，不填论文、预置脚手架或估计分数。

## L. 错误分析、交付与扩展

按 MATH 官方类型/难度分析错误：格式失败、计算错误、数学错误、截断、拒答、verifier 异常。
检查高 reward 低 test 的样本，寻找列举答案、冗长模板、解析攻击与训练题记忆。
对齐收益和遗忘分别评价；数学准确率提高不证明通用指令或安全能力提高。
扩展一般偏好时使用现有 HH-RLHF 等数据与已有可信 benchmark，另写协议。
扩展长程 Agent 时进入 [作业 05](05-agent.md)，增加轨迹、工具和环境预算，不把单轮数学 rollout 称为长程 Agent RL。

交付数据卡、冻结实验协议、mask 检查、共享 Base/SFT 摘要、三路训练配置与日志、成本表、官方评价原始输出和错误分析。
验收允许结果不提升，要求结论与测量一致。
如只完成核心损失验证，写明“数学核心已验证；真实数据三路训练和 benchmark 尚未运行”。

思考：为什么相同 Base 仍不足以保证公平？零方差组比例过高是能力变强，还是学习信号失效？
如果模型只学会判分格式，应该怎样从训练 reward 与官方评价中发现？

## 一手阅读

- [InstructGPT](https://arxiv.org/abs/2203.02155)、[DPO](https://arxiv.org/abs/2305.18290)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)。
- [TRL SFT](https://huggingface.co/docs/trl/en/sft_trainer)、[DPO](https://huggingface.co/docs/trl/en/dpo_trainer)、[GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)。
- [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)、[Open-Instruct](https://github.com/allenai/open-instruct)、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
