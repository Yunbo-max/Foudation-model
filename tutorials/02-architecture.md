# 02　架构再认识：沿着一个 token 的计算路径读 Transformer

> 前置：[01 范式](01-paradigms.md)；实现：[model.py](../src/fm_tutorial/model.py)；实践：[作业一](../assignments/01-foundations.md)。

## 2.1 场景一：最后一个“它”要从哪里找信息

在“研究者修改了模型，并重新评估了它”中，最后一个词的表示需要结合前面出现的对象与动作。
固定窗口的局部规则容易漏掉远处关联；递归结构可传递状态，但训练与信息路径又受其结构约束。
自注意力让当前位置直接从允许的上下文中汇总信息。
注意力权重是一组可学习计算得到的混合系数，不等于经过验证的语法依存或人类解释。
实际模型也会在多层和多个头之间分工，不能用一个头的热图说明全部行为。

我们先跟踪单个 decoder block，再把它堆成语言模型。
记 batch 为 $B$，序列长度为 $T$，隐藏维度为 $d$，层数为 $L$，query head 数为 $H_q$。
本项目采用 pre-norm、RoPE、SwiGLU 和 GQA 的教学实现；它不是 2017 原始 encoder-decoder Transformer 的逐字复刻。
阅读原论文时要区分原始结构与后来常用的 decoder-only 选择。

## 2.2 输入与输出：ID 到分布

输入 $I\in\mathbb N^{B\times T}$ 经嵌入得到 $X_0\in\mathbb R^{B\times T\times d}$。
每层保持主干隐藏状态的 shape 不变，最后经归一化与词表投影得到 $Z\in\mathbb R^{B\times T\times V}$。
如果嵌入和输出权重共享，参数预算减少；若不共享，需计入两份 $Vd$。
这是实现选择，应查实际参数对象身份，不能只看两个矩阵的 shape 一样。

一个 pre-norm block 可写为：

$$
U=X+\mathrm{Attention}(\mathrm{Norm}_1(X)),\qquad
Y=U+\mathrm{FFN}(\mathrm{Norm}_2(U)).
$$

所有 $X,U,Y$ 都是 `[B,T,d]`，所以 residual 加法合法。
残差保留一条直接路径，让子层学习对状态的增量变换。
Pre-norm 的“先”指每个子层输入先归一化，并不意味着主干 residual 每次都会被覆盖成单位尺度。
学习深度较大时，初始化、残差尺度和优化器仍然重要；仅采用 pre-norm 不保证稳定。

## 2.3 注意力：三个投影与两次矩阵乘法

先考虑单个 head，$Q,K\in\mathbb R^{T\times d_h}$，$V_h\in\mathbb R^{T\times d_h}$。
为避免与词表大小 $V$ 混淆，本节把 value 张量写作 $V_h$。
对应参数投影可把 `[B,T,d]` 变为 head 划分后的 `[B,H,T,d_h]`。
注意力为：

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M\in\mathbb R^{T\times T},
\quad P=\mathrm{softmax}_{\mathrm{key}}(S),
\quad O=PV_h\in\mathbb R^{T\times d_h}.
$$

逐步解释：$Q_i\cdot K_j$ 衡量位置 $i$ 对位置 $j$ 的匹配分数。
mask $M_{ij}$ 指定这条信息路径是否允许，允许为 0，禁止为 $-\infty$。
softmax 沿 key 位置 $j$ 归一化，得到当前位置对可见位置的混合权重。
最后用这些权重汇总 value，而不是汇总 key。
如果 $Q,K$ 的分量近似独立、零均值、单位方差，则点积方差随 $d_h$ 增大，除以 $\sqrt{d_h}$ 有助于控制初始分数尺度。
这是尺度动机，不是训练后所有激活都满足独立分布的声明。

多头把这些汇总结果拼接成 `[B,T,H_q d_h]`，再经输出矩阵 $W_o\in\mathbb R^{d\times d}$ 回到 `[B,T,d]`。
通常 $d=H_qd_h$，本仓库配置必须满足这个整除关系。
`view` 前后的内存布局需要注意；`transpose` 不保证结果连续，不能忽略 stride。

## 2.4 因果性与 padding：两种遮罩分别解决什么

自回归训练要求位置 $i$ 仅看到 $j\le i$：

$$
M_{ij}^{\mathrm{causal}}=\begin{cases}0&j\le i\\-\infty&j>i.\end{cases}
$$

批次中较短序列的 PAD key 也应被遮住，避免真实位置读到人为填充。
合并遮罩时，应同时满足“过去或当前位置”与“有效 key”。
完全没有有效 key 的一行可能让直接 softmax 得到 NaN，需要明确处理策略。
本项目损失忽略 PAD 目标，但这不自动修复错误的 key mask。

PyTorch SDPA 的布尔 `attn_mask=True` 表示允许参与；`MultiheadAttention` 的 `key_padding_mask=True` 表示忽略。
接口迁移时必须按官方文档重新检查方向，不能凭变量名猜测。
在同一次 SDPA 调用里同时传显式 mask 和 `is_causal=True` 的支持以当前 API 为准；简单稳妥做法是构造一个完整的显式 mask。
尤其缓存推理时 query 长度为 1、key 长度大于 1，左上角三角 mask 未必符合“当前 token 可看全部历史”的语义。

因果性回归检查可以改变输入后半部分，比较前半部分 logits 是否不变。
这是程序性质检查，可以用短测试输入；它不代表真实语言建模效果。
真实验证损失必须另外在固定公开语料上测量。

## 2.5 RMSNorm：控制尺度，保留均值

对于某位置 $x\in\mathbb R^d$，RMSNorm 为：

$$
\mathrm{RMSNorm}(x)_i=\frac{x_i}{\sqrt{\frac1d\sum_{j=1}^dx_j^2+\epsilon}}g_i,
\qquad g\in\mathbb R^d.
$$

它把向量按均方根缩放，再逐维应用可学习增益。
LayerNorm 通常还减去均值并按中心化方差缩放；是否包含偏置取决于实现。
因此 RMSNorm 与 LayerNorm 不只是“参数数量不同”，它们的变换也不同。
归一化可以用更高精度累积平方和，降低低精度数值误差，随后转回原 dtype。

在训练中检查归一化前后的 RMS 分布，比仅看平均损失更容易定位激活爆炸。
如果某层 RMS 异常，需同时查初始化、残差和学习率；不要机械地增加 epsilon 掩盖原因。
本项目源码用于理解这些计算，尚未提供专门的融合 RMSNorm GPU kernel。

## 2.6 SwiGLU：逐位置的非线性信息加工

注意力跨位置交换信息，FFN 在每个位置对通道做变换。
SwiGLU 的一个常见写法为：

$$
\mathrm{FFN}(x)=\left(\mathrm{SiLU}(xW_g)\odot xW_u\right)W_d,
\quad \mathrm{SiLU}(a)=a\sigma(a).
$$

$W_g,W_u\in\mathbb R^{d\times d_{ff}}$，$W_d\in\mathbb R^{d_{ff}\times d}$，中间量为 `[B,T,d_ff]`。
门分支经非线性调节另一条分支，乘法是逐元素相乘，不是矩阵乘法。
忽略偏置时参数数为 $3dd_{ff}$；普通两矩阵 FFN 为 $2dd_{ff}$。
比较两者时应调整中间维度使参数或计算预算相近，否则同时改变了容量。

某个通道只在部分上下文激活，是它参与计算的现象；不能据此声称它存着一个可完整提取的“知识条目”。
模型知识分布在嵌入、注意力、FFN 和层间状态中，删改一个位置可能影响多个任务。

## 2.7 RoPE：把位置写进 query/key 的旋转角

若没有任何位置机制，自注意力无法仅从重复 token 区分许多排列关系。
RoPE 将每个 head 的偶数维通道配对，在位置 $p$ 应用二维旋转：

$$
R(p\omega)\begin{bmatrix}a\\b\end{bmatrix}=
\begin{bmatrix}\cos(p\omega)&-\sin(p\omega)\\\sin(p\omega)&\cos(p\omega)\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}.
$$

不同通道对使用不同频率 $\omega_k$，通常依据一个基数按维度递减。
对 query 与 key 应用对应旋转后：

$$
(R(p\omega)q)^\top(R(r\omega)k)=q^\top R((r-p)\omega)k.
$$

旋转矩阵正交，点积中的位置依赖自然转成相对距离 $r-p$。
这解释了相对位置性质，但不保证任意远距离都能有效外推。
训练没见过的长度可能产生角度分布、注意力模式与任务分布变化。
修改 RoPE 基数或缩放是额外建模选择，需要长文本评估，不能只看张量没有越界。
本仓库中 `d_h` 必须能配成通道对；生成时位置偏移也必须和训练、缓存策略一致。

## 2.8 场景二：长文档为何“能输入”却未必“能使用”

把 `context_length` 从 128 改为更大，首先扩大允许的张量尺寸。
但若模型只训练过短段，它可能缺乏把远处信息用于当前预测的经验。
更长上下文还改变计算成本和批次容量，训练 token 顺序也可能变化。
这时性能变化不能全归因于位置编码。

在 WikiText/OWT 的真实长文档上，可比较相同目标位置在短前缀与长前缀条件下的 NLL。
必须保证同一个目标 token、相同参数与 tokenizer；截断前缀而不是改目标文本。
按“目标距离文档开头”“依赖线索位置”等分组观察，避免平均数掩盖局部差异。
如果多数样本长度不足，先报告长度分布，不要为了凑实验人为复制句子当真实长文 benchmark。

## 2.9 GQA：query 更多，key/value 共享

MHA 用 $H_q$ 个 query head 和同样多的 KV head；GQA 令 $H_{kv}<H_q$，每组 query 共享一个 KV head。
要求 $H_q\bmod H_{kv}=0$，组大小为 $g=H_q/H_{kv}$。
形状为：

| 张量 | 形状 |
| --- | --- |
| Q | `[B,H_q,T,d_h]` |
| K、value | `[B,H_kv,T,d_h]` |
| 每个 query 的分数 | `[B,H_q,T,T]` |
| 合并 heads 后输出 | `[B,T,d]` |

重复 KV head 到 query head 数是最容易理解的参考实现；高效 kernel 可以避免物理重复。
GQA 降低 KV 投影参数与持久缓存占用，但不把 query head 数减小。
因此它不等于把整个注意力的二次项按同样比例缩小。
模型质量的变化需要在真实语料上验证；数学形状兼容不能证明质量无损。

## 2.10 场景三：同样参数量，部署时谁更容易放进显存

单个请求、每层长度为 $T$ 的 KV cache 大致占：

$$
M_{KV}=2LTH_{kv}d_hs\quad\text{字节},
$$

其中 2 是 key/value 两份，$s$ 为每元素字节数；$B$ 个相同长度请求再乘 $B$。
这是逻辑数据量估算，没有包含块分配、对齐、页表、workspace 与碎片。
将 $H_{kv}$ 减半会将该式的 KV 项减半，权重与其他激活不跟着减半。
这说明只按参数量比较服务成本会漏掉上下文状态。
本项目基础生成器重算上下文，没有实现持久 KV cache，详见 [05](05-inference.md)。

## 2.11 参数预算与实现导读

忽略偏置和 norm，每层 GQA 注意力参数近似为 $2d^2+2dH_{kv}d_h$。
两份 $d^2$ 来自 Q 与 O，K/V 各为 $d\cdot H_{kv}d_h$。
加上 SwiGLU 后，总数近似：

$$
N\approx N_{\mathrm{embedding/head}}+L(2d^2+2dH_{kv}d_h+3dd_{ff})+N_{\mathrm{norm}}.
$$

要得到实际参数数，请运行 `sum(p.numel() for p in model.parameters())`，并确认共享参数如何计数。
阅读 [ModelConfig](../src/fm_tutorial/model.py) 时先查整除、偶数 head 维度、词表与 context 约束。
阅读 `forward` 时依次标注嵌入、QKV reshape、RoPE、mask、输出投影、FFN 与 logits 的 shape。
最后独立核对 next-token 移位与 PAD 忽略；网络看起来正确不代表监督对齐正确。

## 2.12 真实数据消融：先问一个问题，再改变一个条件

在固定真实语料与 tokenizer 下比较 MHA/GQA；以 `n_kv_heads=n_heads` 作为 MHA。
其余配置、有效目标 token 数、batch 与学习率应保持一致，并记录参数数变化。
另一个独立消融可比较 RMSNorm/LayerNorm，但需要你在分支中实现替换；仓库没有自动化 norm 消融开关。
如果要比较 SwiGLU 与普通 FFN，先按参数或 FLOPs 匹配中间宽度，再报告不同匹配准则。
把验证 NLL、训练稳定性、GPU tokens/s、峰值显存分别记录，不让一项指标代替全部结果。

小模型上的结论只适用于该数据和规模；不得写成“所有模型 GQA 均不损失质量”。
没有 GPU 时完成形状、损失与梯度检查，并把性能栏标为未测量。

## 2.13 检查理解

1. 在 $d=128,H_q=4,H_{kv}=2,T=128$ 时，写出每个 Q/K/value/score 的完整 shape。
2. 推导 RoPE 点积只依赖位置差的等式，并指出它为何不保证长上下文外推。
3. 若 key/value 数减半，哪些参数与状态减少，哪些注意力计算仍存在？
4. 为什么因果 mask 的对角线允许读当前 token，而预测标签仍是下一个 token？
5. 设计一项真实长文对比，避免把“更长输入没有崩溃”当成“长程理解提升”。

## 原始来源与阅读目的

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：QKV、多头与位置机制的原始架构。
- [RMSNorm](https://arxiv.org/abs/1910.07467)：均方根归一化。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)：门控 FFN 的原始研究。
- [RoFormer](https://arxiv.org/abs/2104.09864)：旋转位置编码与相对位置性质。
- [GQA](https://arxiv.org/abs/2305.13245)：分组 query 与 KV 共享。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：当前 mask、dropout、GQA 与后端约束；先查本地 PyTorch 版本。
