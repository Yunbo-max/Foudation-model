# 08 SFT 与蒸馏：从语言续写转向条件示范学习

对应公开课程地图 W8：SFT & Distillation。
预训练让模型学习文本如何继续；监督微调用选定示范告诉它，在某类输入之后应如何回答。
SFT 是一个有明确标签的条件概率学习问题。它可以教格式、语气、工具调用和解题方式，但示范中的错误也会被学习。

## 8.1 三个场景中的 SFT

**场景一：数学回答格式。** Base 模型会继续“题目—解答”文本，却未必遵守“最后一行给出答案”的指令。
SFT 的示范让该格式成为高概率行为；格式成功与数学正确性应分别评价。
如果只学到 `####` 或 `\\boxed{}`，解析成功率会上升，但解题准确率未必上升。

**场景二：客服对话。** 人工示范包含用户提问和客服答复。
通常希望模型学习客服答复，而不是替用户编造下一轮问题。
这决定哪些 token 进入损失，也是 response mask 的实际用途。

**场景三：教师解法蒸馏。** 教师给出同一道公开训练题的多步解答。
学生拟合教师回答，可以学习一种解题路径；它并未直接优化最终验证奖励。
教师很长的错误推理也会产生很多监督 token，因此筛选和权重不能省略。

本章目标是从条件似然推导 mask 和损失，连接可运行核心代码，再设计同 Base 的真实实验。

## 8.2 条件语言模型的监督目标

设 prompt 为 $x$，回答为 $y=(y_1,\ldots,y_L)$。
自回归分解为：

$$p_\theta(y\mid x)=\prod_{t=1}^{L}p_\theta(y_t\mid x,y_{<t}).$$

对单条示范的负对数似然是 $-\sum_t\log p_\theta(y_t\mid x,y_{<t})$。
training 时使用真实先前 token 作为条件，这称为 teacher forcing。
它不保证自由生成时所有步骤仍正确；生成中的一次错误会改变后续条件。

如果标签是人写的，叫人工示范学习；如果标签由教师模型生成，叫序列级蒸馏的一种形式。
SFT 指优化目标，蒸馏指监督信号的来源或教师—学生关系，两者不是互斥算法。
[InstructGPT](https://arxiv.org/abs/2203.02155) 的研究流程先进行示范 SFT，再学习偏好和强化学习。
这说明阶段关系，不表示每个项目必须复用同一数据量和超参数。

## 8.3 张量尺寸与一位移位

在 batch 中，将 prompt、回答和必要的结束 token 拼成 $U\in\mathbb N^{B\times T}$。
模型输出 $Z\in\mathbb R^{B\times T\times V}$，其中 $B$ 是 batch、$T$ 是最大长度、$V$ 是词表大小。
$M\in\{0,1\}^{B\times T}$ 标记**目标 token**是否属于可学习回答。
这不是 attention mask：prompt 虽不计入损失，仍是回答预测必须读取的条件。

令 $j$ 表示输入位置，预测目标 $U_{b,j}$ 由前一位置 $Z_{b,j-1,:}$ 给出。
因此有效位移是：

```python
prediction_logits = logits[:, :-1, :]   # [B, T-1, V]
target_ids = input_ids[:, 1:]           # [B, T-1]
target_mask = completion_mask[:, 1:]    # [B, T-1]
```

例子只是索引演示，不是 benchmark 数据：

| 输入位置 | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| token | BOS | 题目 | 回答前缀 | 答案甲 | 答案乙 | EOS |
| completion mask | 0 | 0 | 0 | 1 | 1 | 1 |
| 负责预测该 token 的 logits 位置 | 无 | 0 | 1 | 2 | 3 | 4 |

回答第一个 token 必须被回答前缀位置的 logits 预测。
若不移位 mask，就会把该 token 错位屏蔽，并可能把前缀纳入监督。
PAD 位置必须为零；回答后的第一个 EOS 通常为一，以学习停止。
EOS 后的重复 PAD/EOS 填充不能全部当真实结束 token 学习。
本仓库要求 `completion_mask[:, 0]` 为零，因为序列首 token 没有前一个 logits 负责预测。

## 8.4 Token 平均、序列平均与序列和

本仓库 `sft_loss` 使用所有有效回答 token 的平均 NLL：

$$
\mathcal L_{\mathrm{token}}
=-\frac{\sum_{b,j\geq1}M_{b,j}\log p_\theta(U_{b,j}\mid U_{b,<j})}
         {\sum_{b,j\geq1}M_{b,j}}.
$$

另一种目标是先除每条回答的长度 $L_b=\sum_jM_{b,j}$，再对 $B$ 条回答平均：

$$\mathcal L_{\mathrm{sequence}}=-\frac1B\sum_b\frac1{L_b}\sum_jM_{b,j}\log p_\theta(U_{b,j}\mid U_{b,<j}).$$

两条回答分别有 10 和 100 个 token 时，token 平均让长回答的总贡献约大十倍。
序列平均让两条回答拥有相等的总权重。没有普遍正确的归一化，必须与任务和采样器一起说明。
完全不归一化的序列和，还会让 batch token 数变化影响梯度尺度。
DPO 后续需要序列 log-prob **和**来表示回答概率，不能把它与这里的平均 NLL混淆。

梯度累积时，多个 microbatch 的平均损失简单再平均，只有各自有效 token 数相同时才等于全局 token 平均。
长度变化很大时，应累计 NLL 分子和有效 token 分母，或按有效 token 数加权。
多卡训练也应确认全局分母，避免每卡平均造成不一致权重。

## 8.5 Tokenization 边界不是字符串长度

不能用 prompt 字符数决定 mask 的 token 边界。
BPE 的合并可能跨字符串拼接边界；分别 encode(prompt) 与 encode(prompt+completion) 的前缀不一定一致。
应使用完整序列的可靠 token offset、带 generation 区间的 chat template，或经过验证的 token 拼接策略。
无论哪种方法，都输出几条解码后的带 mask 样例人工检查。

多轮对话应明确学习所有 assistant 回答，还是只学习最后一轮。
工具结果可以作为条件而不作模型文本监督；模型发出的工具调用是否纳入 mask，则取决于目标行为。
assistant header、结束符和系统提示也须按模型模板固定规则处理。
把 EOS 当 PAD 时尤其要依据有效长度构造 mask，不能靠 token ID 一刀切。

截断优先级也是协议：先限制 prompt，还是删掉回答尾部？
若保留了题目而截掉全部回答，该样本没有 SFT 信号，应拒绝或明确剔除。
若删除最终答案却留下推理，可能教出永不完结的回答；记录截断率及最终答案保留率。

## 8.6 本仓库的核心接口

[`alignment.py`](../src/fm_tutorial/alignment.py) 提供：

```python
from fm_tutorial.alignment import sft_loss, completion_log_probs

logits = model(input_ids)["logits"]
loss = sft_loss(logits, input_ids, completion_mask)
sequence_logp = completion_log_probs(logits, input_ids, completion_mask)
```

`loss` 是标量；`sequence_logp` 为 `[B]`，是同一区间内的 log-prob 和。
外层训练循环执行 zero_grad、backward、梯度裁剪和 optimizer.step。
这里没有完整真实数据 collator、分布式对齐训练器或教师服务，应由实践者串接。
[`TransformerLM`](../src/fm_tutorial/model.py) 的内置 `labels` 损失用于 next-token 预训练；做回答 SFT 时应取 logits 后调用上述函数。
不应同时把 prompt 全序列损失与回答损失相加，却称为 completion-only SFT。

`sft_loss` 对全空回答 mask 拒绝计算；空 mask 的 sequence log-prob 可返回零，但不是有意义的回答似然。
浮点计算使用稳定 log-softmax，低精度训练中应检查累加与损失精度。
不能先 softmax 再 log；极小概率可能先下溢为零。

## 8.7 从核心损失进入真实训练器

可使用 [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) 或 [Open-Instruct](https://github.com/allenai/open-instruct) 进行完整模型实验。
先固定工具版本，再确认数据格式、completion-only/assistant-only 选项与 chat template 支持。
默认值随版本变化，不能只写“使用默认设置”。
特别检查 packing 后是否保留回答 mask、文档边界和正确的监督 token 数。

实施顺序是：真实数据整理 → collator 可视检查 → 小 batch 梯度诊断 → dev 集选择超参数 → 完整训练 → 官方评测。
本仓库损失可以作为小 batch 的数学对照；不应替换可信 harness 为临时字符串判分。
选择 LoRA 时记录 rank、target modules、可训练参数数与 adapter revision。
LoRA 降低可训练参数量，并不让所有模型加载、激活和生成成本同时消失。

## 8.8 与蒸馏的进一步联系

硬标签蒸馏只监督教师实际输出的 token。
若教师还能提供同一词表上的分布 $p_T(v\mid h)$，可最小化 soft-target 交叉熵：

$$\mathcal L_{\mathrm{soft}}=-\sum_{t,v}M_t p_T(v\mid h_t)\log p_S(v\mid h_t).$$

$p_T$ 与 $p_S$ 均在 $V$ 维词表上，必须对齐 tokenizer 与条件历史。
教师学生词表不同，则不能逐 token 直接做这个 KL/交叉熵，应另行设计对齐方法。
温度改变目标分布；若引入温度缩放，应记录公式与梯度尺度处理，不能默默改训练目标。
教师输出样本更容易获得，但没有教师完整分布中的不确定性信息。

## 8.9 同 Base、同 Eval 的第一条实验分支

作业建议采用固定 revision 的 [Qwen2.5-Math-1.5B Base](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)，不是 Instruct 版本。
模型选择来自公开模型卡；本教材未运行其微调，也不保证任何具体硬件时长。
使用 MATH 官方 train 解答，并在 train 内固定 dev 子集；最终 test 不参与配置选择。

| 分支 | 起点 | 更新信号 | 用途 |
| --- | --- | --- | --- |
| Base | $\theta_0$ | 无 | 衡量训练前行为 |
| SFT | $\theta_0$ | 同一训练池示范 | 衡量条件示范效果 |
| SFT→DPO | 固定 $\theta_S$ | 第 09 章偏好对 | 衡量额外偏好阶段 |
| SFT→RLVR | 同一 $\theta_S$ | 第 10 章在线验证奖励 | 衡量额外探索阶段 |

后两分支共享完全相同的 SFT checkpoint，不能一个用更强的 Instruct 模型，一个用 Base。
固定 tokenizer、评价 prompt、解码、最大输出 token、答案提取和 evaluator commit。
若比较端到端成本，后两分支必须计入共同 SFT 阶段和额外生成成本。
不同方法的训练步数相同，不代表总成本相同。

## 8.10 可信评测与失败定位

评价采用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的固定 GSM8K/MATH 任务，或固定官方 Qwen 数学评价流程。
本作业选择一种协议贯穿各分支，不混合不同 prompt 的结果。
例如 harness 的 `gsm8k` 有自己的 few-shot、停止词和严格/宽松答案提取规则；报告必须标出实际 filter。
保存逐题生成输出、token 数、停止原因和评价 verdict，才能区分以下失败。

- dev NLL 下降但 test 不升：可能过拟合示范风格、教师错误或任务分布。
- 数字正确但 strict extraction 失败：属于格式问题，不能任意改官方 scorer 来掩盖。
- 回答反复续写：检查 EOS 是否进入 mask、训练模板与生成模板是否一致。
- 用户句子也被生成：检查 assistant 区间和 prompt 是否错误纳入标签。
- 部分 batch 无梯度：检查回答全部截断、空 mask 和 loss 是否误 detach。

训练 token accuracy 不能代替自由生成的解题准确率。
结果没有运行时写“未测”，不要填教师论文、课程仓库或预置脚手架的数字。

## 8.11 交付与思考

完整交付见 [作业 04](../assignments/04-alignment.md)：示范来源、split、模板、mask 样例、checkpoint 与真实评测输出。

1. 为何不计 prompt 损失，仍能让 prompt 影响模型梯度？
2. 回答长度相差十倍时，如何决定用 token 平均还是序列平均？
3. 最终答案正确的教师解法，哪些情况下仍不适合作为示范？
4. packing 后出现跨样本注意力，是否必然是 bug？应如何说明与控制？

## 主要一手来源

- [InstructGPT 论文](https://arxiv.org/abs/2203.02155)：SFT、偏好学习与 RLHF 的研究流程。
- [TRL SFTTrainer 官方文档](https://huggingface.co/docs/trl/en/sft_trainer)：数据格式、mask 与 packing 实现入口。
- [Open-Instruct 官方仓库](https://github.com/allenai/open-instruct)：公开数据上的 SFT/DPO/RLVR 工作流。
- [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)：对齐训练实践与测试入口。
- [课程团队教材](https://github.com/dujh22/AML-LLM)：SFT、RLHF 与科学评价的系统背景。

下一章不再只模仿一条答案，而是从同一输入下的回答比较中学习。
