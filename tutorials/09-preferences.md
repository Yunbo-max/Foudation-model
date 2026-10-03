# 09 偏好学习：从成对比较到 DPO

对应公开课程地图 W9：Preference Learning。
SFT 要求提供示范；偏好学习只需表达同一 prompt 下两个回答哪个更符合目标。
标签可能来自人、模型或程序，三种来源的证据强度不同，必须在数据卡中区分。

## 9.1 三个场景解释比较标签

**场景一：两条数学解答。** 一个答案正确，一个错误，可靠答案验证器可以提供比较。
这属于正确性派生偏好；它没有告诉我们哪条解答更易懂、更安全或步骤更严谨。
若两个最终答案都正确，不能凭随机顺序把一个叫 chosen、另一个叫 rejected。

**场景二：两条咨询答复。** 一条简洁回答问题，另一条很长但回避问题。
人类比较可能比从头写专家示范便宜，但标注人可能偏好礼貌、长度或熟悉措辞。
需固定评价维度，并记录一致性和分歧，不能把多数偏好称为客观事实。

**场景三：模型给自己打分。** 同一模型生成两条答案，再选择自己喜欢的一条。
这可以构造训练信号，但偏好可能只是风格自强化。
独立评价应由既有 benchmark 或另一个可靠证据来源承担。

本章从 reward model 推导 DPO，并说明参考模型、序列概率与偏好数据的实现边界。

## 9.2 偏好对的数据定义

每条记录为 $(x,y_w,y_l)$：$x$ 是同一个 prompt，$y_w$ 是 chosen，$y_l$ 是 rejected。
比较只有在相同任务条件下才有意义。
若一个候选拥有工具结果而另一个没有，标签混入了信息可得性差异。
若 prompt 在两支中不一致，DPO 无法把它们解释为同一条件分布的比较。

应保存生成 policy revision、采样参数、候选 ID、标注准则、标注源与平局。
将训练/开发划分按 prompt 进行，而不是按 pair 行随机进行。
同题的不同回答对跨 split，会使开发集过于容易，也可能泄露答案。
重复对或同一 chosen 配上很多 rejected，会隐含增加某题的权重。

偏好数据不是“一个正确答案加任意乱码”就足够。
过于明显的错误只能教模型回避表面形式，难以提供细粒度判断信号。
真实 hard negative 是模型确实容易生成、但未满足任务标准的候选。

## 9.3 Bradley–Terry 模型与 reward learning

令 $r_\phi(x,y)\in\mathbb R$ 为标量 reward model。
一种常用比较模型写为：

$$P(y_w\succ y_l\mid x)=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).$$

其中 $\sigma(u)=1/(1+e^{-u})$。
训练 reward model 使用 $-\log\sigma(r_w-r_l)$；两个评分整体加同一常数不会改变比较概率。
所以偏好标签识别的是同 prompt 内的差异，绝对分数并非天然校准。
跨 prompt 比较两个 reward 数字大小时应格外小心。

InstructGPT 流程在 SFT 后收集比较，训练 reward model，再用强化学习优化 policy。
完整 RLHF 还涉及在线生成、KL 控制和训练稳定性；它不是“给交叉熵加一个分数”这么简单。
reward model 可能在训练分布之外失效，policy 恰好会向高 reward 的新区域移动。
这就是 reward overoptimization 和 reward hacking 的重要来源。

## 9.4 KL 正则化的最优 policy

设 $\pi_{\mathrm{ref}}(y\mid x)$ 是固定参考 policy，$\beta>0$ 为 KL 系数。
考虑每个 prompt 下的目标：

$$\max_\pi\ \mathbb E_{y\sim\pi}[r(x,y)]
-\beta D_{\mathrm{KL}}(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)).$$

在分布可自由优化、支持集允许等理想条件下，最优解为：

$$\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}.$$

$Z(x)$ 是归一化常数；这不是有限模型训练必然达到的状态。
整理得到 $r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)$。
同一个 $x$ 下做两个回答的 reward 差，$\log Z(x)$ 正好抵消。
这是 [DPO 原论文](https://arxiv.org/abs/2305.18290) 将 reward 参数化改写成 policy 参数化的关键。

## 9.5 DPO 损失与梯度方向

记 $\ell^\theta_w=\log\pi_\theta(y_w\mid x)$，其余 $\ell^\theta_l,\ell^{\mathrm{ref}}_w,\ell^{\mathrm{ref}}_l$ 类似。
定义参考校正后的 margin：

$$\Delta=(\ell^\theta_w-\ell^\theta_l)-(\ell^{\mathrm{ref}}_w-\ell^{\mathrm{ref}}_l).$$

$$\mathcal L_{\mathrm{DPO}}=-\mathbb E_{(x,y_w,y_l)}\log\sigma(\beta\Delta).$$

其单对梯度为：

$$\nabla_\theta\mathcal L=-\beta\sigma(-\beta\Delta)
\left(\nabla_\theta\ell^\theta_w-\nabla_\theta\ell^\theta_l\right).$$

最小化损失推动 chosen 相对 rejected 的 log-prob 差上升；已经容易满足的偏好对权重会降低。
这不保证 chosen 的绝对概率一定增加，两个回答的概率都可能下降，但 rejected 下降更多。
因此只监控 preference accuracy 或 margin，可能漏掉生成质量恶化。

数值演示只是公式诊断：当 policy=reference 时，$\Delta=0$，单对 loss 为 $\log2$。
若 chosen 的 log-prob 相对增加 1、其余不变，$\beta=0.1$，loss 约为 0.644。
这是计算结果，不是训练效果、benchmark 得分或已训练模型的观测。

## 9.6 为什么必须使用序列 log-prob 和

回答概率按 token 概率相乘，因此：

$$\ell_\theta(x,y)=\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t}).$$

实现中应使用第 08 章同一完成区间，只包括回答和协议规定的结束 token。
prompt token 不能混进 `chosen` 和 `rejected` 的评分；两支的 padding 不应进入任何 log-prob。
completion mask 的移位必须与 logits/targets 相同。

把上述和除以回答长度，会将目标改成平均 token 概率差，已经不是原始 DPO 的序列概率形式。
长度归一可能是有意的变体，但必须改名说明目标，不能“为稳定”悄悄加入。
原始 DPO 也不能自动消除长度偏差：标签中的长回答偏好、EOS 行为和序列分布都会影响学习。
应报告 chosen/rejected 长度分布及两者差值，并按长度分组检查 margin 和生成结果。

## 9.7 固定参考模型与 detach

参考模型通常来自共享 SFT checkpoint，负责定义相对改动的锚点。
固定权重、`eval()` 与 `torch.no_grad()` 分别控制更新、训练态随机性与梯度记录，作用不同。
仅调用 `eval()` 不会禁止梯度；仅把 reference 参数从 optimizer 移除，也仍可能浪费计算图内存。
本仓库 `dpo_loss` 会 detach reference log-prob，但完整训练器仍须正确处理参考模型。

若 reference 与 policy 共享同一可训练 adapter，更新 policy 会改变 reference，锚点失效。
可冻结独立副本，或在经过验证的 adapter-disabled/reference-adapter 模式下评分。
缓存参考分数时，缓存键包含 tokenizer、模板、截断、mask、reference revision 和样本 ID。
换模板后继续使用旧 reference cache，会制造虚假的 margin。

reference 不是 rollout old policy：DPO 使用固定锚点和离线比较，不需要 PPO 的旧采样分布分母。
下一章会明确区分这三个模型/分布角色。

## 9.8 本仓库的实现连接

[`alignment.py`](../src/fm_tutorial/alignment.py) 的最小组合为：

```python
policy_chosen = completion_log_probs(chosen_logits, chosen_ids, chosen_mask)
policy_rejected = completion_log_probs(rejected_logits, rejected_ids, rejected_mask)
loss = dpo_loss(policy_chosen, policy_rejected,
                reference_chosen, reference_rejected, beta=0.1)
```

四个 sequence log-prob tensor 均为 `[B]`，`loss` 为 batch 对平均标量。
`reference_chosen/rejected` 必须来自相同 token 序列和 mask。
使用稳定 `logsigmoid`；先算 sigmoid 再 log 在大负 margin 下可能下溢。
空回答应在 collator 拒绝，不能利用 `completion_log_probs` 的空和为零制造不公平比较。

两支可以合并成 `2B` 个样本一起前向，再按索引拆开，但数据对应关系必须保持。
长候选的截断会改变偏好含义：chosen 可能被截掉正确答案，rejected 却完整保留。
应统一长度政策并记录两支截断率；不能把后处理后的任意文本继续沿用原标签。

本模块只有数学核心；真实模型训练可串接 [TRL DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer) 或 Open-Instruct。
冻结版本并记录 `loss_type`、beta、reference 设置、batch 和 token budget。
现有训练器可能支持多种偏好损失，其名称和默认值要逐项核对，不能都叫原始 DPO。

## 9.9 两类真实偏好实验

**正确性派生实验。** 使用 MATH 官方 train 内的训练题，从固定共享 SFT policy 采样候选。
由固定训练侧数学验证器构造“一条正确、一条错误”的 pair，平局不造假标签。
按 prompt 限制 pair 数，报告每类题目可构造 pair 的比例。
没有正确候选的难题与全部正确的易题都会被排除，这是样本选择偏差。

**一般偏好实验。** 使用公开、已有的数据集，例如 [Anthropic HH-RLHF](https://github.com/anthropics/hh-rlhf)。
其 GitHub README 已标记迁移到 [Anthropic 官方 Hub 数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)，获取时固定该数据 revision。
保留原始 helpful/harmless 来源区分，理解其 dialogue 格式和 chosen/rejected 语义。
在固定的既有安全/指令 benchmark 与公开评测工具上评价，而不是自行写十个问题证明安全对齐。
数学派生偏好实验不能替代一般偏好或安全实验，二者问题不同。

作业 04 的主线选择第一类，以便与 SFT、RLVR 使用同 Base 和同数学评价协议。
将一般偏好分支作为扩展，需要单独的数据卡与评价协议。

## 9.10 同 Base、同 Eval、同成本的比较

从第 08 章保存的 $\theta_S$ 开始 DPO，同时另一路从相同 $\theta_S$ 开始 RLVR。
SFT-only 分支可以继续用示范训练到同额外预算，避免把“更多更新”误认为偏好方法优势。
公开 Base 的既有预训练成本对所有分支相同，不计为本项目新增成本；共同 SFT 成本在端到端比较中计入。

| 成本项 | SFT | DPO | RLVR |
| --- | --- | --- | --- |
| 示范或标签准备 | 数据整理/教师费用 | 候选生成/标注 | 训练提示与验证器 |
| 学生更新 | 回答 NLL | 两候选 policy 评分 | 在线 rollout 与 policy 更新 |
| 辅助模型 | 可有教师 | 固定 reference 评分 | old policy、可选 reference |
| 评价 | 同一冻结协议 | 同一冻结协议 | 同一冻结协议 |

比较训练 token 相同、GPU 时间相同和总费用相同，会得到不同控制条件。
无法严格等成本时，报告实际成本差异与性能—成本曲线，不声称已完成等成本实验。
所有分支保存官方 test 逐题输出及 checkpoint hash。

## 9.11 失败模式与思考

- 偏好训练准确率上升、benchmark 下降：可能是离线 pair 过拟合、长度捷径或绝对似然退化。
- loss 初始不是接近 $\log2$：检查 policy/reference 起点、dropout、缓存和 token mask。
- reference 出现梯度：检查副本、adapter 共享、评分 no_grad 和 detach。
- 难题完全不参与训练：检查 pair 构造覆盖，不能仅报留下样本的正确率。
- 模型只拒绝回答：可能标签偏好保守，需在既有 helpfulness 与 safety 评价上分别检查。

1. 为何 reward 的 prompt 常数不可识别，却不妨碍成对比较？
2. chosen 和 rejected 都变得更不可能时，DPO loss 为什么仍可能下降？
3. 为什么平均 token log-prob 不能直接替代原始 DPO 的序列和？
4. 如何利用平局比例判断候选生成策略是否提供了有效学习信号？

## 主要一手来源

- [DPO 论文](https://arxiv.org/abs/2305.18290)：KL 约束推导、Bradley–Terry 与原始目标。
- [InstructGPT 论文](https://arxiv.org/abs/2203.02155)：示范、比较和 RLHF 流程。
- [TRL DPOTrainer 官方文档](https://huggingface.co/docs/trl/en/dpo_trainer)：实现选项与数据格式。
- [Anthropic HH-RLHF](https://github.com/anthropics/hh-rlhf)：公开人类偏好数据来源。
- [Open-Instruct](https://github.com/allenai/open-instruct)：公开 SFT、DPO、RLVR 配方；其 README 建议新评价使用 [OLMES](https://github.com/allenai/olmes)。

下一章把静态 pair 变成在线采样—验证—更新闭环，学习信号也从相对标签转为程序化奖励。
