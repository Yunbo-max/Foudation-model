# 10 RLVR 与 GRPO：让在线生成接受可验证反馈

对应公开课程地图 W10：RLVR & Reasoning。
RLVR 是强化学习中的奖励来源选择：回答经程序化验证后产生奖励。
GRPO 是一种 policy 优化方法：用同题多个回答的相对奖励构造 advantage，再优化受约束的 policy 更新。
二者常组合，但不能当成同一个概念；GRPO 也可以使用 learned reward，RLVR 也可以采用其他优化器。

## 10.1 三个场景说明可验证奖励

**场景一：数学题。** 判分器提取最终答案，检查是否与标准答案等价。
正确奖励可以规模化，但不证明推理链逐步正确，也不证明模型在未覆盖的数学领域可靠。
如果判分器接受“列出所有可能答案”，模型可能学会穷举而非解题。

**场景二：程序题。** 执行候选程序，奖励可信测试的通过情况。
测试不足时，模型会学会训练测试的特例；超时、沙箱异常和答案错误需区分。
程序应在受限执行环境运行，训练任务与隐藏评测保持隔离。

**场景三：格式奖励。** 正确的 JSON 或数学结尾标记容易判定。
它可以帮助模型接入系统，但格式正确的错误答案仍是错误答案。
当格式分超过正确性分，reward 上升可能主要来自格式，而不是目标能力。

学习闭环始终是：训练题 → policy rollout → verifier → reward/advantage → 更新 → 独立评价。
第 11 章将把回答级反馈推广到有多步工具和环境状态的 Agent 轨迹。

## 10.2 先写出要优化的期望

令 $x\sim q$ 为训练 prompt，$y\sim\pi_\theta(\cdot\mid x)$ 为完整回答，$R(x,y)$ 为 verifier 奖励。
目标可以写为 $J(\theta)=\mathbb E[R(x,y)]$，也可以加参考 policy 的 KL 正则。
reward 通常由外部程序计算，不对 token 直接可微。
policy gradient 使用 log-prob 的梯度：

$$\nabla_\theta J=\mathbb E_{y\sim\pi_\theta}
\left[R(x,y)\sum_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\right].$$

reward 在这里是一条完整回答的标量，梯度来自每个被采样 token 的概率。
减去适当 baseline 可以降低方差；将 reward 直接 `.backward()` 不能训练离散采样器。
SFT 则用固定示范 token 做交叉熵，两种信号的产生机制不同。
有限 group 的相对归一化是一种具体估计方式，不应宣称仍是原始期望梯度的无条件无偏估计。

## 10.3 GRPO 的组内相对 advantage

每个 prompt 采样 $K$ 个回答。batch 中有 $G$ 个 prompt，reward tensor 为 $R\in\mathbb R^{G\times K}$。
本章的 $G$ 表示组数，$K$ 表示每组回答数；原论文常用 $G$ 表示组内候选数，阅读时注意记号转换。
对第 $g$ 组：

$$\mu_g=\frac1K\sum_{k=1}^{K}R_{gk},\qquad
\sigma_g=\sqrt{\frac1K\sum_{k=1}^{K}(R_{gk}-\mu_g)^2}.$$

$$A_{gk}=\frac{R_{gk}-\mu_g}{\max(\sigma_g,\varepsilon_{\mathrm{adv}})}.$$

这使用 population standard deviation，即 PyTorch 的 `unbiased=False`。
对 outcome reward，可把 $A_{gk}$ 广播给该回答的有效 token。
它表示“该回答比同题其他候选好多少”，不是对该 token 单独正确性的判断。
[DeepSeekMath](https://arxiv.org/abs/2402.03300) 引入 GRPO，以组内 reward baseline 替代 PPO 中额外训练的 value model。
省掉 critic 不表示 rollout 或 policy 更新没有成本。

## 10.4 零方差组为什么没有相对学习信号

三个二值 reward 组的数学诊断如下；这不是模型测量。

| rewards | 均值/标准差 | 相对 advantage | 相对学习有效性 |
| --- | --- | --- | --- |
| `[0,0,0,0]` | $0/0$ | 全零 | 无，所有回答都失败 |
| `[1,1,1,1]` | $1/0$ | 全零 | 无，所有回答都成功 |
| `[1,0,0,0]` | $0.25/\sqrt{0.1875}$ | 约 `[1.732,-0.577,-0.577,-0.577]` | 有差异信号 |

加 epsilon 只避免除零，并不会凭空制造排序信息。
$K=1$ 必然没有组内差异，因此不能用 group-relative objective 学习该样本。
本仓库返回 `valid_groups`，要求从损失 mask 中剔除零方差组。
否则这些组虽贡献零 policy 项，却仍占平均分母，稀释有效梯度。

全对组意味着题目可能太容易，全错组意味着任务太难、探索不足或格式/verifier 故障。
两类都不应只写成“跳过比例”，应分别统计。
某些训练器会为无相对信号组单独施加 KL 正则，这是可声明的设计；本仓库教学方案将其整体排除。
所有组无效时应跳过 optimizer step：零梯度 loss 不保证 momentum/weight decay 不更新权重。

## 10.5 三个 policy 角色与 rollout 日志

| 名称 | 用途 | 是否更新 |
| --- | --- | --- |
| current policy $\pi_\theta$ | 当前计算梯度的模型 | 每次 optimizer step 更新 |
| old policy $\pi_{\mathrm{old}}$ | 采样 batch 的旧 policy 与 PPO 分母 | 一轮 rollout 内冻结 |
| reference policy $\pi_{\mathrm{ref}}$ | 限制相对锚点的偏移 | 按声明策略冻结或阶段性重设 |

old 与 reference 可以一开始相同，但角色不同，不能随意互换。
rollout 还可能由另一个推理引擎的 $\pi_{\mathrm{rollout}}$ 执行。
即使权重名相同，采样温度、top-p、精度和权重同步延迟都可能令它不等于训练侧 old policy。

每次采样保存 policy revision、实际 token IDs、目标位置 log-prob、采样参数和停止原因。
PPO 的 `old_log_probs` 应固定为这一批样本对应的 old 评分。
不能每次更新后用 current 重算并覆盖 old，否则 ratio 被重置，掩盖偏移。
若采样由异步 worker 完成，还要保存权重版本及 policy lag。

教学实验先采用同步权重、相同评分实现与明确采样分布，减少不匹配因素。
若使用 top-p/temperature，应说明概率属于原始 softmax 还是变换后的行为分布。
真实行为分布与训练 policy 不同，需要可信训练器支持的修正；不能把两者当作相等。
[TRL GRPO 官方文档](https://huggingface.co/docs/trl/en/grpo_trainer) 专门讨论训练—推理引擎不匹配及修正选项。

## 10.6 PPO clipping 的 token 代理目标

对某个回答 token，定义当前与旧 policy 的条件概率比：

$$\rho_{gkt}=\exp\left(\ell^\theta_{gkt}-\ell^{\mathrm{old}}_{gkt}\right).$$

$$s_{gkt}=\min\left(\rho_{gkt}A_{gk},
\operatorname{clip}(\rho_{gkt},1-\epsilon,1+\epsilon)A_{gk}\right).$$

最小化负 $s$，再加入可选 KL 项。
当 $A=1$、$\rho=1.4$、$\epsilon=0.2$ 时，代理奖励为 1.2，继续放大概率不再获得该项收益。
当 $A=-1$、$\rho=0.6$ 时，两支为 -0.6 与 -0.8，取 -0.8，限制有利方向上的过大改变。
这是 PPO 的局部代理约束，不是精确保证每次更新都处于某个全分布 KL 球内。

逐 token ratio 与完整序列 importance ratio 不同。
完整回答比值是所有 token ratio 的乘积，长序列中会有高方差。
PPO 使用局部条件概率比的代理目标；不要把它解释为对任意旧轨迹都精确无偏的整序列重加权。
复用同一 rollout 多轮后，clip fraction、KL 和有效样本行为都可能改变，应限制并记录更新轮数。

## 10.7 长度归一化与组平均

原始 GRPO 的常见写法先对每条回答的有效 token 平均，再对组内候选平均。
本仓库 `grpo_loss` 明确采用**所有有效 token 的平均**，这是便于检查的教学选择。
对于长短回答混合，两者权重不同；不能声称本模块完整复现论文训练目标。

记 $M_{gkt}$ 为有效 token mask，$L_{gk}=\sum_tM_{gkt}$。
本仓库负代理目标为：

$$\mathcal L_{\mathrm{token}}=-\frac{\sum_{g,k,t}M_{gkt}s_{gkt}}{\sum_{g,k,t}M_{gkt}}.$$

另一种等回答权重的目标为 $-\frac1{GK}\sum_{g,k}\frac1{L_{gk}}\sum_tM_{gkt}s_{gkt}$，并需排除无效组/空回答。
长回答贡献更多有效 token，与每条回答权重相等是不同实验条件。
更换 reduction、reward 标准差归一化或 token-level/sequence-level ratio，都改变了目标。
必须记录这些选择，而不是只写一个算法名 `GRPO`。

组内标准差还会改变题目权重：相同原始 reward 差，在小标准差组中的归一化幅度不同。
因此应按难度观察 valid group 比例、reward 分布和 advantage 分布。
它们是训练诊断，不能替代独立 benchmark。

## 10.8 KL 项与 sampled surrogate 的边界

本仓库可选的 k3 型项使用 $d=\ell^{\mathrm{ref}}-\ell^\theta$：

$$k_3=e^d-d-1\geq0.$$

在当前 policy 采样、支持集等条件成立时，其期望等于 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$。
但 PPO batch 来自 old policy；未经适当 importance 修正，样本均值不是当前 policy KL 的无偏估计。
因此本仓库称其为可选 sampled surrogate penalty，不把日志数字当完整分布 KL 的精确测量。
同一篇论文或训练器中的 KL 位置也可能不同：加入 reward 与直接加到 loss 的 advantage 含义不同。

reference、old log-prob 与 advantage 均应 detach；current log-prob 必须保留计算图。
先按 mask 选择有效项再指数运算，防止被屏蔽极值仍导致溢出。
有效 token ratio 溢出时应定位采样、评分或更新问题，不能默默截断 log-ratio 并仍称为原目标。

## 10.9 本仓库的数学接口

[`alignment.py`](../src/fm_tutorial/alignment.py) 接受已评分的 token tensor，而不是文本奖励服务。

```python
advantages, valid_groups = group_advantages(rewards)  # rewards: [G,K]
active_mask = response_mask & valid_groups[:, None, None]
loss = grpo_loss(current_logp, old_logp, advantages,
                 active_mask, clip_epsilon=0.2,
                 ref_log_probs=reference_logp, kl_beta=0.0)
```

`current_logp/old_logp/reference_logp/response_mask` 均为 `[G,K,T]`，这里 $T$ 已表示评分后的 token 步数。
函数也支持展平后的 `[G*K,T]`；advantage 对应 `[G,K]` 或 `[G*K]`。
若从完整 IDs/logits 得到这些分数，必须先按第 08 章移位，再 reshape，mask 也先移位。
`completion_log_probs` 返回序列和，不能直接拿它当逐 token `current_logp`。

`group_advantages` 返回 `[G,K]` advantage 与 `[G]` valid mask。
`grpo_loss` 不知道你的 prompt 分组，因此调用者必须显式组合 valid mask。
全空 mask 返回与 current tensor 相连的零 loss，方便数学诊断；训练循环应跳过没有有效信号的更新。

```bash
python scripts/alignment_check.py --device cpu
```

这只检查确定性的 SFT/DPO/GRPO 数学行为，不加载真实数据，不产生能力评价或训练曲线。
真实 rollout、verifier、checkpoint、分布式同步由外部训练流程负责。
可选用固定版本的 [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)、TRL 或 Open-Instruct 实现整条链路。

## 10.10 同一数学任务上的真实比较

使用 [MATH](https://github.com/hendrycks/math) 官方 train，按题目 ID 冻结 train/dev。
共享第 08 章的 SFT checkpoint：一支继续 SFT，一支 DPO，一支 RLVR。
RLVR 从 train prompt 在线采样，用冻结训练侧数学 verifier 产生奖励；不读取 test 标准答案构造训练。
SFT 示范、DPO pair、RLVR prompt 的源题池一致，明确记录各自可用信号的差异。

评价固定 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的 `gsm8k` 和 `hendrycks_math`，保存 scorer/filter 与任务版本。
`hendrycks_math`、`minerva_math` 和 `hendrycks_math500` 是不同协议/集合，不得只报一个模糊的 MATH 分数。
每条分支采用同样推理 token 限额、解码、few-shot 和工具权限。
greedy 单答与多样本 voting/best-of-K 的结果分别报告，并计入后者额外推理成本。

每轮训练日志至少包括 reward 分量、格式有效率、全对/全错/有效组比例、回答长度、截断率、ratio 与 clip fraction。
记录 rollout tokens、评分 tokens、训练 tokens、wall time 和机器环境。
步数、样本数、token 数和费用不是互换单位。
如未运行完整训练，报告“核心损失已验证，真实对齐训练待执行”，不要把单元测试当结果。

## 10.11 奖励黑客与分布外行为

先检查训练 reward 上升是否伴随官方 test 改善。
如果只有 reward 上升，审计高分输出：是否重复答案、输出多个候选、提前结束或攻击解析器？
格式奖励独立记账，不能把“格式+正确性”总 reward 报成准确率。
verifier 发生更新后，应保留旧版本结果，避免 reward 曲线的上升来自判分尺度改变。

训练中发现漏洞，可以用真实训练侧案例修复 verifier 并重新版本化。
漏洞测试属于程序回归测试，不是新的研究 benchmark；不能用十个自制题证明通用推理能力。
公开 benchmark 也有覆盖与污染限制，因此应谨慎解释迁移范围。
长推理链不自动代表更强推理，应同时观察正确率、成本与失败类型。

## 10.12 交付与思考

完整协议见 [作业 04](../assignments/04-alignment.md)，Agent 轨迹信号见 [第 11 章](11-agent-rl.md)。

1. 一个组全答对为何没有相对 policy 梯度，却仍是良好的能力观测？
2. old 与 rollout 评分不同，哪些版本和采样记录能定位原因？
3. 为什么 current 与 reference 的 sampled KL 在 old batch 上不能直接称无偏？
4. 若所有 batch 组都无效，如何调整训练题分布、采样和验证，而不改变 test？
5. token 平均奖励优化为何可能偏向长回答，如何用受控实验检验？

## 主要一手来源

- [PPO 原论文](https://arxiv.org/abs/1707.06347)：概率比与 clipped surrogate。
- [DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)：GRPO、组相对信号与原始长度归一化。
- [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)：reward、log-prob、mask 与 policy loss 实践。
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)：完整训练器、采样和训练—推理不匹配。
- [Open-Instruct](https://github.com/allenai/open-instruct)：开放后训练配方与 RLVR 入口。
- [公开课程地图](https://github.com/jackmcgradylee/from-token-to-agent)：W10 与 HW4 方向，非本项目实验记录。
