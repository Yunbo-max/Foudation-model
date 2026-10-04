# 17a　文本 Diffusion：连续向量、离散转移与从全 MASK 开始生成

> 前置：[17 Diffusion 基础](17-diffusion.md)、[02 架构](02-architecture.md)。
> 实现：[continuous.py](../src/fm_tutorial/diffusion/continuous.py)、[discrete.py](../src/fm_tutorial/diffusion/discrete.py)、[models.py](../src/fm_tutorial/diffusion/models.py)。
> 实践：[演示脚本](../scripts/diffusion_demo.py)、[作业六](../assignments/06-diffusion.md)；延伸：[多模态 Diffusion](17b-diffusion-modalities.md)、[原始来源表](../references/diffusion.md)。

文本 Diffusion 要回答两个独立问题：**在哪个空间加噪，以及如何把去噪结果变成合法 token？**
本章从“今天的天气很 `[MASK]`，适合散步”出发，再走到所有待生成位置都是 MASK 的情形。
例句中的词只是说明位置与条件；真实 tokenizer 通常会把一个词拆成多个 token。
本章的小词表和合成句式演示算法，不作为语言能力 benchmark，也不是下列论文的训练复现。

## 17a.1 同一句文本可以有四种表示

设 batch 大小为 $B$，token 长度为 $L$，词表大小为 $V$，向量维度为 $d$。
先把“latent”说清楚，避免把几条不同路线混为一谈。

| 表示 | Shape 与类型 | 每个位置存什么 | 可以怎样加噪 |
| --- | --- | --- | --- |
| Token ID | `[B,L]`，整数 | 词表索引 | categorical 替换或 MASK |
| Token embedding | `[B,L,d]`，浮点 | 查表得到的词向量 | Gaussian |
| Contextual representation | `[B,L,d]`，浮点 | 编码器结合上下文后的向量 | Gaussian；还需可用的文本解码器 |
| 压缩 latent | 如 `[B,M,d_z]`，$M<L$ | 编码器压缩后的序列表示 | Gaussian；由学习到的解码器恢复文本 |

Token ID 是地址，不是测量值。
若把“猫”的 ID 从 5 改为 5000，文本语义没有发生变化。
因此 `ids.float() + torch.randn_like(...)` 不会得到语义合理的文字噪声；取整也无法修复这个问题。
离散路线对类别做随机转移，连续路线先定义有意义的向量表示。

查表 embedding 与上下文编码也不相同：`embedding(ids)` 中同一个 ID 总是取同一行；
`encoder(ids)` 可以让“银行”在金融语境与河岸语境中得到不同表示。
但一个用于理解的 encoder 并不天然带有恢复原文的 decoder。
若采用 $z_0=E_\phi(x)$、$p_\psi(x\mid z_0)$，应先测干净 latent 的重建：

$$
\mathcal L_{\rm rec}=-\log p_\psi(x\mid E_\phi(x)).
$$

这一步提供生成链路的重建基线。
编码器丢掉的信息，不能指望仅靠降低 latent 去噪 MSE 自动找回。
它不是对所有任务指标的严格数学下界，但能隔离“表示/解码损失”与“Diffusion 误差”。
decoder 若采用 AR，它自己的解码成本也必须算进整条生成链路。

## 17a.2 连续路线：噪声落在向量上

[Diffusion-LM](https://arxiv.org/abs/2205.14217) 将词序列映射为连续向量，逐步去噪后再映射回词。
原论文包含学习 embedding、文本输出分布与 rounding/clamping 的专门设计。
它不等于“先训练一个压缩文本 autoencoder，再在其 bottleneck 上训练 Diffusion”。
后者是另一种表示方案，必须独立说明 encoder、decoder、压缩比与重建质量。

本节先固定 embedding 表 $E\in\mathbb R^{V\times d}$：

$$
z_0[b,\ell,:]=E[x_0[b,\ell]],\qquad
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
$$

这里的 Gaussian 作用于 `[B,L,d]` 的每个浮点分量。
与图像一样，可以预测 $\epsilon$、$z_0$ 或 $v$；本仓库 Gaussian 核心采用噪声预测。
去噪网络要处理跨位置关系：只用逐 token MLP，很难依据右侧“适合散步”调整左侧天气描述。
Transformer 可以用双向 attention 看当前噪声序列，以及另行提供的条件。

下面展示一次训练更新的表示接口，核心 API 与 [17](17-diffusion.md) 相同。
`t` 的数组索引为 `0..T-1`，对应论文离散时间 `1..T`；干净端单独处理。
`epsilon_model` 接收浮点向量与归一化时间，输出同 shape 的噪声估计。

```python
import torch
import torch.nn.functional as F
from fm_tutorial.diffusion.continuous import GaussianDiffusion

torch.manual_seed(7)
vocab_size, length, dim = 8, 6, 16
embedding = torch.nn.Embedding(vocab_size, dim)
ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 4, 5]])
process = GaussianDiffusion(steps=100)

# 教学上先固定表示，使这次更新只检验去噪目标。
z0 = embedding(ids).detach()                 # [B,L,d]
t = torch.randint(process.steps, (ids.shape[0],))
noise = torch.randn_like(z0)
zt = process.q_sample(z0, t, noise)

class TinyVectorDenoiser(torch.nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.position = torch.nn.Parameter(torch.randn(1, length, dim) * 0.02)
        self.time = torch.nn.Sequential(torch.nn.Linear(1, dim), torch.nn.SiLU())
        layer = torch.nn.TransformerEncoderLayer(
            dim, 4, 2 * dim, dropout=0.0, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, 2)
        self.output = torch.nn.Linear(dim, dim)

    def forward(self, zt, time):
        h = zt + self.position + self.time(time[:, None])[:, None, :]
        return self.output(self.encoder(h))

epsilon_model = TinyVectorDenoiser(length, dim)
optimizer = torch.optim.AdamW(epsilon_model.parameters(), lr=1e-3)
predicted_noise = epsilon_model(zt, (t.float() + 1) / process.steps)
loss = F.mse_loss(predicted_noise, noise)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这段代码只完成一次更新，不会训练出可用语言模型。
若要联合学习 embedding，直接取消 `detach()` 并只最小化一个向量 MSE 会引入退化风险：
模型可能通过改变目标空间的尺度，让问题变得容易，而没有改善文本生成。
需要文本重建/词表分类约束、表示尺度控制和真正的 held-out 评价。
本章固定随机表演示张量流动，不能替代 Diffusion-LM 原论文的联合训练目标。

### 从向量回到 token：三个不同的选择

给定去噪结果 $\hat z_0$，最近邻 rounding 为

$$
\hat x_\ell=\arg\min_{v\in\{0,\ldots,V-1\}}
\|\hat z_{0,\ell}-E_v\|_2^2.
$$

若使用 cosine，则比较 $\hat z_{0,\ell}/\|\hat z_{0,\ell}\|$ 与 $E_v/\|E_v\|$。
Euclidean 与 cosine 通常给出不同答案；只有向量范数满足相应约束时，它们的排序才一致。
学习 softmax 输出则是 $p_\psi(x_\ell=v\mid\hat z_0)$，由输出头/decoder 训练得到。
它可以利用上下文和类别偏置，不应被称为“和最近邻完全一样”。

```python
def nearest_ids(z, table):
    # z: [B,L,d]; table: [V,d]。这里只适合小词表教学。
    distance2 = (z.square().sum(-1, keepdim=True)
                 + table.square().sum(-1)
                 - 2 * z @ table.T)
    return distance2.argmin(-1)

def cosine_ids(z, table):
    return (F.normalize(z, dim=-1)
            @ F.normalize(table, dim=-1).T).argmax(-1)

learned_head = torch.nn.Linear(dim, vocab_size)
reconstruction_logits = learned_head(z0)     # [B,L,V]
reconstruction_loss = F.cross_entropy(
    reconstruction_logits.flatten(0, 1), ids.flatten())
assert torch.equal(nearest_ids(z0, embedding.weight), ids)

# sample 内部给 model 的仍是整数步，适配到上面的训练时间单位。
epsilon_model.eval()
def epsilon_at_index(z, index):
    return epsilon_model(z, (index.float() + 1) / process.steps)
z_generated = process.sample(
    epsilon_at_index, shape=z0.shape, sampler="ddim", sampling_steps=20)
generated_ids = nearest_ids(z_generated, embedding.weight)
```

最后的断言只说明干净查表向量可以被最近邻还原，不能说明噪声样本能被正确还原。
`generated_ids` 完成 Gaussian 先验到向量到 token 的链路；单次更新的结果没有语言能力意义。
如果预测位于两个 embedding 中间，向量误差很小也可能翻转为另一个离散词。
采样中间把 $\hat z_0$ 投影到最近 embedding 是 clamping，一种额外解码选择；
过早做硬投影会过早承诺 token，需用实际文本评价比较，而不是只看 MSE。

### 场景：固定来源文本，生成改写或回答

[DiffuSeq](https://arxiv.org/abs/2210.08933) 处理条件 seq2seq：来源 $c$ 与目标 $x$ 都映射到连续 embedding，
forward 只给目标部分加噪，来源作为条件保留，reverse 对目标去噪。
用 shape 表达：来源 `[B,L_c,d]`，目标噪声 `[B,L_x,d]`，联合输入 `[B,L_c+L_x,d]`。
这说明“条件文本 embedding”与“被生成文本的 Diffusion latent”是两种角色。
普通文本到图像模型中的 text encoder 也只是提供条件，不意味着它在生成文本。

连续 infilling 有两种建模选择，应在训练和采样中保持一致：
若条件从来不加噪，就保留其干净表示；若训练给全序列加噪，推理时不能随意把已知位置塞成干净向量。
后者常需在每个时间把已知 token 放到匹配该时间的噪声水平，并说明随机耦合和端点。
简单钳制不是一般条件分布的精确采样证明。

另一种文本控制是属性控制，如“表达积极情绪，且主题为天气”。
连续向量允许计算 $\nabla_z\log p_\phi(c\mid z_t,t)$ 来调整去噪轨迹，
但分类器必须理解对应噪声水平；把只看干净文本的分类器直接用于纯 Gaussian 噪声通常不合适。
实际控制强度还需兼顾流畅度、控制成功率和梯度计算成本。
Diffusion-LM 的控制方法见其论文与[作者实现](https://github.com/XiangLi1999/Diffusion-LM)，本仓库没有执行该大型控制实验。

## 17a.3 离散路线：把“加噪”写成转移矩阵

[D3PM](https://arxiv.org/abs/2107.03006) 在离散状态空间定义 forward Markov chain。
本章采用**行向量约定**：$Q_t[i,j]=q(x_t=j\mid x_{t-1}=i)$，即行是前一状态，列是后一状态。
对于从干净类别出发的累积矩阵，$\bar Q_t[i,j]=q(x_t=j\mid x_0=i)$，行是 clean，列是 noisy。
每一行非负且和为 1；列和不一定为 1。
其他论文可能使用列向量，阅读矩阵乘法时需要连同方向一起转换。

$$
\bar Q_0=I,\qquad \bar Q_t=Q_1Q_2\cdots Q_t,
\qquad q(x_t\mid x_0=i)=\operatorname{Cat}(\bar Q_t[i,:]).
$$

这里离散时间是 `1..T`，`0` 是干净端，区别于上节 Gaussian 数组索引。
序列各位置的 forward 噪声可以独立采样；reverse 网络仍能联合读取整个序列。
“噪声独立”不要求“语言中不同 token 独立”。

一个容易枚举的 uniform transition 为

$$
Q_t=(1-\beta_t)I+\beta_t\mathbf1\mathbf1^\top/V.
$$

意思是以 $1-\beta_t$ 保留当前类别，以 $\beta_t$ 从整个词表均匀重抽。
重抽可能仍抽到原类别，因此真正改变 token 的概率是 $\beta_t(1-1/V)$。
例如 $V=3,\beta=0.3$ 时：

| 前一类别 / 后一类别 | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 0.8 | 0.1 | 0.1 |
| 1 | 0.1 | 0.8 | 0.1 |
| 2 | 0.1 | 0.1 | 0.8 |

uniform 有时会把错误单词伪装成正常可见词；MASK 则直接告诉网络“此处信息丢失”。
absorbing mask 引入特殊类别 $m$：

$$
Q_t=(1-\beta_t)I+\beta_t\mathbf1 e_m^\top.
$$

普通 token 留在原位置或变成 MASK，MASK 行始终只转移到 MASK。
两种过程的末端先验不同：uniform 面向均匀类别分布，absorbing 面向全 MASK。
有限 uniform schedule 需要检查 $\bar Q_T$ 是否真的接近均匀。
本仓库默认 toy schedule 的末端并非精确均匀，而 sampler 从均匀先验开始，存在有限步先验失配。
可在独立实验中用 `beta_end=1.0` 使最后一步完全随机化，并比较 schedule 对学习的影响。

### 精确后验：当前 noisy token 与真实 clean token 都已知

记 $x_0=i,x_t=j,x_{t-1}=k$。由 Bayes 与 Markov 性得到

$$
q(x_{t-1}=k\mid x_t=j,x_0=i)
=\frac{\bar Q_{t-1}[i,k]Q_t[k,j]}{\bar Q_t[i,j]}.
$$

分子是“从 $i$ 到 $k$，再到 $j$”的概率；分母等于分子对所有 $k$ 求和。
不要把 $Q_t[k,j]$ 写成 $Q_t[j,k]$，那会把 forward 方向倒过来。
若分母为零，条件事件不可能发生，不能靠加一个 epsilon 就把它说成合法的真实后验。
uniform 正噪声 toy 可避免这类不相容事件；absorbing 需要显式处理可见 token 的支持集。
在 $t=1$ 时 $\bar Q_0=I$，后验退化成 $x_0$，这是很有用的终点检查。

```python
def posterior_row(q_t, qbar_previous, clean, noisy):
    unnormalized = qbar_previous[clean] * q_t[:, noisy]
    total = unnormalized.sum()
    if total <= 0:
        raise ValueError("impossible conditioning event")
    return unnormalized / total

q = 0.7 * torch.eye(3) + 0.3 * torch.ones(3, 3) / 3
qbar_previous = q                         # t=2, 同样的 Q 用两步
p = posterior_row(q, qbar_previous, clean=0, noisy=1)
# 分子 [0.8*0.1, 0.1*0.8, 0.1*0.1]，总和为 0.17。
assert torch.allclose(p, torch.tensor([8/17, 8/17, 1/17]))
```

### 未知 clean token：本仓库采用归一化后验的混合

推理不知道真实 $x_0$。网络输出当前位置的 clean 类别分布
$r_\theta(i\mid x_t,t)$，它可以读取整个 noisy 序列及条件。
本仓库 `reverse_probs` 明确定义为

$$
p_\theta(x_{t-1}=k\mid x_t=j)
=\sum_i r_\theta(i\mid x_t,t)
\underbrace{\frac{\bar Q_{t-1}[i,k]Q_t[k,j]}{\bar Q_t[i,j]}}_{
q(x_{t-1}=k\mid x_t=j,x_0=i)}.
$$

每个候选 clean 的后验先单独归一化，再按预测 clean 概率求和。
若 $r_\theta$ 是真实 clean 后验，这个等式就是全概率公式。
用 `argmax` 选择一个 clean 类别再计算后验，会丢掉模型的不确定性。
只把概率向量代入一次未归一化分子，也一般不等于上式。

**与 D3PM 论文参数化的区别：** 原论文式 (4) 写作
$p_\theta(k\mid j)\propto\sum_i q(k,j\mid i)\tilde p_\theta(i\mid j)$，是联合概率的加权和再整体归一化。
它的 clean 权重与这里表示真实条件 clean 后验的 $r_\theta$ 含义不同。
两者在适当重参数化权重后可联系，不能把相同权重直接代入并宣称数值必然相同。
本章矩阵后验沿用 D3PM 框架，采样实现是条件 clean 后验混合的教学变体。

```python
from fm_tutorial.diffusion.discrete import CategoricalDiffusion

categorical = CategoricalDiffusion(vocab_size=3, steps=20)
x0 = torch.tensor([[0, 1, 2]])
t = torch.tensor([7])                      # long[B]，1..T
forward_probs = categorical.q_probs(x0, t) # [B,L,V]
xt = categorical.q_sample(x0, t)           # [B,L]
posterior = categorical.posterior_probs(x0, xt, t)

clean_probs = torch.softmax(torch.randn(1, 3, 3), dim=-1)
reverse = categorical.reverse_probs(clean_probs, xt, t)
assert torch.allclose(reverse.sum(-1), torch.ones_like(xt, dtype=torch.float))

# 常数 logits 仅检查采样 shape；输出不会有语言意义。
def uniform_model(ids, time):
    return torch.zeros(*ids.shape, 3, device=ids.device)

samples = categorical.sample(uniform_model, shape=(2, 6), device="cpu")
```

朴素实现为每个位置展开所有 $i,k$，中间量有 $V^2$ 项，矩阵存储还涉及 $TV^2$。
大词表不能直接照搬这个小矩阵枚举。
uniform 的 $I$ 加低秩矩阵结构、absorbing 的稀疏支持允许解析简化；
生产实现应利用结构、稳定的 log 运算和分块，不能据 toy 循环估计大模型的成本。

## 17a.4 Mask Diffusion：forward 是丢掉信息，reverse 是逐步揭示

本节把归一化连续时间写作 $t\in[0,1]$，干净端为 0，全 MASK 端为 1。
选择 token 保留概率 $\alpha(t)=1-t$，对每个非条件位置独立采样

$$
q(x_t\mid x_0)=\operatorname{Cat}((1-t)e_{x_0}+t e_m).
$$

这不是每一步都以概率 $t$ 重新 mask：$t$ 是从干净端到当前时间的**累计** mask 概率。
如果要按 forward 时间递增生成嵌套 mask，可给每个位置抽一次 $u\sim U(0,1)$，令 $u<t$ 时 masked。
独立地为多个时间调用 `mask_tokens` 可用于训练，却不自动构成同一条轨迹。

```python
from fm_tutorial.diffusion.discrete import (
    mask_tokens, masked_loss, sample_masked, MaskedDenoiser)

mask_id, vocab_size = 7, 8
x0 = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 4, 5]])
t = torch.tensor([0.2, 0.9])               # float[B]，累计 mask 概率
condition_mask = torch.zeros_like(x0, dtype=torch.bool)
condition_mask[:, :3] = True               # 条件不加噪，也不作为 masked 目标
xt, masked_positions = mask_tokens(x0, t, mask_id, condition_mask)
model = MaskedDenoiser(vocab_size, max_length=6, dim=32, heads=4, layers=2)
logits = model(xt, t)                      # [B,L,V]，双向 attention
loss = masked_loss(logits, x0, masked_positions)
assert torch.equal(xt[condition_mask], x0[condition_mask])
```

`masked_loss` 在 `masked_positions` 上计算同位置 clean token 的交叉熵。
没有 next-token target shift：位置 3 的输出监督位置 3 的原始 token。
如果复用 AR 的 `logits[:, :-1]` 对齐 `ids[:, 1:]`，会把整个去噪任务错位。
没有 masked 位置的批次应返回可微的零损失，避免空集合求平均导致 NaN。

### 教学去噪目标与连续时间 ELBO 的权重

一个直观训练目标是均匀采样 $t$，对抽到的 masked 位置平均 CE：

$$
\mathcal L_{\rm teach}=\mathbb E_{\{t_b\sim U(0,1),x_{t_b}\}_{b=1}^B}
\left[\frac{\sum_{b=1}^B\sum_\ell M_{b\ell}(t_b)\,[-\log r_\theta(x_{0,b\ell}\mid x_{t_b},t_b)]}
{\max(1,\sum_{b=1}^B\sum_\ell M_{b\ell}(t_b))}\right].
$$

本仓库 `masked_loss` 与演示训练采用这个易观察的去噪目标。
它在整个 batch 的 masked 位置上取平均；masked 位置较多的样本获得更大权重。
先对每个样本按自身 masked 数量平均，再对 batch 平均，是另一种目标。
它能说明“隐藏更多 token 时模型如何恢复”，但不自动是完整的似然下界估计器。
尤其 masked token 的随机数量进入分母，已经改变不同样本/时间的相对权重。

[MDLM](https://arxiv.org/abs/2406.07524) 推导了吸收过程与 SUBS 参数化下的连续时间目标。
其 diffusion 项对应固定位置归一化的加权 masked CE，权重为

$$
w(t)=\frac{-\alpha'(t)}{1-\alpha(t)},\qquad
\alpha(t)=1-t\ \Longrightarrow\ w(t)=\frac1t.
$$

因此可写成 $\mathbb E_t[\sum_\ell M_\ell(t)\operatorname{CE}_\ell/t]$，再按固定有效 token 数归一化。
这是“每个 token 的损失乘 indicator 与时间权重”，不是“先把 masked CE 求平均，然后随意乘 $1/t$”。
低噪声时 $1/t$ 大，但 masked 事件概率为 $t$；期望与方差要一起看，不能只看单个权重。
采样 $t=0$ 会出现除零，数值实现需定义端点处理或截断并说明偏差。

```python
# 仅展示时间权重和固定位置归一化，不声称实现完整 MDLM 似然评估。
per_token_ce = F.cross_entropy(
    logits.transpose(1, 2), x0, reduction="none")  # [B,L]
valid_target = ~condition_mask
weighted_sum = (per_token_ce * masked_positions / t[:, None]).sum()
weighted_diagnostic = weighted_sum / valid_target.sum().clamp_min(1)
```

完整似然评估还需一致的 forward schedule、reverse 参数化、时间积分估计、特殊 token 处理与端点项。
本章的 `weighted_diagnostic` 和 unweighted demo 都不应标为 MDLM perplexity。
原论文采用“clean 分布不给 MASK 概率”和“已揭示位置保持不变”的 SUBS 约束；
本仓库采样也实施这两条机制，但没有因此复现其全部训练、似然评估与工程配方。

## 17a.5 从全 MASK 反向走：揭示概率怎样推出来

考虑 $0\le s<t\le1$，当前位置 $x_t=m$。
forward 在 $t$ 前已被 mask 的概率是 $t$，而在 $(s,t]$ 内首次被 mask 的概率是 $t-s$。
所以当真实 clean token 已知时

$$
q(x_s=x_0\mid x_t=m,x_0)=\frac{t-s}{t},\qquad
q(x_s=m\mid x_t=m,x_0)=\frac{s}{t}.
$$

如果 $x_t\ne m$，它在 forward 中从未被 mask，reverse 直接复制当前 token。
把未知 $x_0$ 替换成模型预测 clean 分布后，masked 位置的 reverse 分布为

$$
p_\theta(x_s=v\mid x_t=m)
=\frac{t-s}{t}r_\theta(v\mid x_t,t),\quad v\ne m;
\qquad p_\theta(x_s=m\mid x_t=m)=\frac{s}{t}.
$$

一般 schedule 的揭示概率是 $(\alpha(s)-\alpha(t))/(1-\alpha(t))$。
因此在线性 schedule 下每一步应使用 `(t-s)/t`，不能总用常数 `1/steps`。
例：20 步时第一步 $1\to0.95$ 揭示概率为 0.05，最后一步 $0.05\to0$ 为 1。
最后一步必须把所有剩余 MASK 揭示，且 clean 类别采样必须排除 MASK。

下面是 `sample_masked` 的核心思想，完整参数检查与条件处理见[实现](../src/fm_tutorial/diffusion/discrete.py)。
这里随机决定位置是否揭示，再从该位置的类别分布抽 token。

```python
@torch.no_grad()
def illustrate_reverse_step(model, xt, t, s, mask_id):
    time = torch.full((xt.shape[0],), t, device=xt.device)
    logits = model(xt, time).clone()
    logits[..., mask_id] = -torch.inf       # clean 状态不允许 MASK
    probs = logits.softmax(-1)
    candidate = torch.multinomial(probs.flatten(0, 1), 1).reshape_as(xt)
    reveal = (torch.rand_like(xt, dtype=torch.float) < (t - s) / t)
    reveal &= xt.eq(mask_id)
    return torch.where(reveal, candidate, xt)
```

这个基线以独立 Bernoulli 决定揭示数量，某一步可能不揭示任何位置。
实际模型近似了 clean 条件分布，因此“揭示概率来自精确后验”不代表整条生成链是精确真实语言采样。
已揭示 token 保持不变也是该 absorbing 参数化的限制：一次错误承诺会成为后续上下文。

### 场景一：填空，左边和右边都是条件

给定“今天的天气很 ___，适合散步”，可把左右可见 token 都设为 immutable 条件。
`condition_ids` 与 `condition_mask` 的 shape 都是完整 `[B,L]`；只在 mask 为 True 的位置读取 ID。
其余占位 ID 不是提示，也不允许作为 clean 目标经另一条输入路径传给网络。

```python
condition_ids = torch.tensor([[0, 1, 2, 0, 4, 5]])
condition_mask = torch.tensor([[True, True, True, False, True, True]])
filled = sample_masked(
    model, shape=(1, 6), mask_id=mask_id, steps=20, device="cpu",
    condition_ids=condition_ids, condition_mask=condition_mask)
assert torch.equal(filled[condition_mask], condition_ids[condition_mask])
```

这段代码演示接口；上面的 `model` 还没有完成训练，其填空不能当作正确语言输出。
完整训练需在数据中覆盖目标位置、来源/目标边界与相应噪声率。
“条件固定”包括 forward 不改写、loss 不监督它、每个 reverse step 不覆盖它这三件事。

### 场景二：所有目标位置都为 MASK

无条件生成调用 `sample_masked(model, (B,L), mask_id, ...)`，初始目标完全由 MASK 构成。
网络仍有位置表示、时间表示与训练得到的分布；有条件生成还可以读取提示。
同一次前向输出多个位置的 clean 分布，但独立抽样不直接提供这些新 token 之间的依赖。
下一轮把已揭示 token 作为上下文，才逐步传递新的选择。
“先预测名词，再围绕它形成动作描述”是可能的采样路径，不是固定语法保证。

高 mask 率时，局部上下文线索少，任务更依赖提示、位置和模型学到的整体先验。
若模型只在少量 mask 的填空上训练，全 MASK 推理会产生明显分布差异。
可分别统计不同噪声率的 held-out loss，比较有提示/无提示，以及一次揭示/多轮揭示。
这些是验证机制的消融，不能从某次失败推广为“某一种架构永远不能做高 mask 生成”。

### 场景三：控制主题或保留任意位置

主题指令可以作为固定条件前缀，也可以用专门条件向量输入网络。
“必须保留某个实体名”适合 immutable 位置约束；“整句必须积极”需要模型理解全局属性。
硬保留一个 token 并不自动保证句法一致、事实正确或全局属性达成。
评估应同时记录条件保持率、属性成功率、文本质量与生成预算。

## 17a.6 网络、特殊 token 与采样策略的边界

`MaskedDenoiser(vocab_size, max_length, dim=32, heads=4, layers=2)` 使用 token、位置和时间表示，
双向 Transformer 读取当前状态，输出 `[B,L,V]`。
其 `t` 是 `[B]` 归一化浮点时间，区别于 `CategoricalDiffusion` 接收的整数步。
网络输出包含 MASK 类，采样时将它的 logit 设为 $-\infty$；训练目标中的真实文本不含 MASK。
时间输入是否可省略是另一个设计/消融问题，本仓库显式提供时间。

双向 attention 不会自动造成标签泄漏；关键是网络只读 corrupted target 与授权条件。
真正的泄漏包括把 $x_0$ 的 embedding 拼接到输入、缓存干净 target hidden state，
或把目标答案标成 condition。训练当然用 $x_0$ 计算 loss，但不能把它提供给预测网络。
用同一目标做反向采样前的初始化，也不再是从全 MASK 的生成实验。

本仓库例子固定长度，没有实现通用变长文本协议。
采用真实 tokenizer 时至少区分 EOS、PAD、MASK 三个角色：

- EOS 是序列内容的终止标记；需要训练如何预测，而不是采样后随意删掉。
- PAD 是批次填充；需从目标 loss 中排除，并在 attention 中避免真实 token 读取 PAD。
- MASK 是 forward 的吸收状态；它不是输出正文，也不等同于 PAD。

已知训练长度形成的 PAD mask 可能泄漏长度信息；推理若未知长度，不能直接照搬训练长度。
可选择先预测长度、固定窗口生成 EOS 后截断、或逐块扩展；每种方案都改变建模与评估协议。
如果未来位置能通过双向 attention 影响 EOS，EOS 后内容怎样参与计算也要明确。
`MaskedDenoiser` 的简化接口没有提供专门的 padding attention mask，因此演示使用等长有效序列。

### 随机揭示与 confidence/top-k 是不同策略

上节 sampler 使用 schedule 决定随机揭示，符合所推导的 absorbing reverse 形式。
另一类常见实现先为所有 masked 位置生成候选，再按置信度保留部分位置、把其他候选继续设为 MASK。
指定每步 top-k 数量、根据置信度选择位置或把已揭示 token 再 mask，都改变了这里的采样核。
它们可以是有效的解码启发式，但不能用 `(t-s)/t` 推导自动为其提供概率正确性。

[LLaDA](https://arxiv.org/abs/2502.09992) 及其[作者采样代码](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py)
包含随机/低置信度 remasking 与按块生成的选择。
其中“候选未被保留，仍保持 MASK”要与“把已经提交的 token 改回 MASK”区分开。
本仓库 `sample_masked` 只实现随机揭示基线，没有实现论文完整的 confidence、CFG、块生成或特殊 token 策略。

## 17a.7 与 AR 比较：并行位置不等于更低实际成本

| 问题 | 标准左到右 AR | 本章全序列 Mask Diffusion |
| --- | --- | --- |
| 训练监督 | 当前前缀预测下一个 token | 当前 corrupted 序列预测同位置 clean token |
| attention | 因果 | 双向 |
| 初始生成状态 | 提示和空 continuation | 提示固定，所有目标 MASK |
| 一次调用输出 | 一个新位置，训练时并行所有位置 | 所有目标位置的分布 |
| 依赖形成 | 已生成前缀逐 token 增长 | 随已揭示位置逐轮增长 |
| 已提交错误 | 常规解码不回改 | 本章 absorbing baseline 也不回改 |
| 右侧条件 | 需 FIM 模板或专门建模 | 可直接固定任意位置，仍需训练匹配 |
| 长度 | EOS 逐步终止 | 需长度/EOS/块协议 |
| 似然指标 | 可求因果条件概率乘积 | 需要对应模型的 ELBO/积分估计协议 |

生成 $L$ 个 token 的简单 AR 通常有 $L$ 次 decode 调用；$K$ 步全序列 Diffusion 通常有 $K$ 次网络调用。
本章 mask sampler 在所有目标已揭示后提前结束，实际调用数可能更少，应以执行记录为准。
这只是 NFE（number of function evaluations）口径，不是 FLOPs、延迟或吞吐的结论。
每次 Diffusion 调用可能重新处理整个 `[B,L,d]`，而 AR 可只计算一个新 token 并重用 KV。
连续文本还需计算向量解码，CFG 增加条件/无条件分支计算，属性梯度控制还需反向传播。
测量应报告序列长度、batch、硬件、精度、实际调用数、峰值内存、tokens/s 和质量。

全序列双向网络中，即使某个输入 token 不变，它的深层表示也可能因其他位置揭示而改变。
因此标准 AR 的“过去 KV 永久有效”假设不能原封不动移植。
这不意味着 Diffusion 永远不能 cache；它意味着需要额外的结构或近似并验证误差。
[Block Diffusion](https://arxiv.org/abs/2503.09573) 在块间采用 AR、块内去噪，支持块结构与 KV 复用；
[Fast-dLLM](https://arxiv.org/abs/2505.22618) 研究双向模型的近似 cache 与置信度并行解码。
具体边界分别查[Block Diffusion 作者仓库](https://github.com/kuleshov-group/bd3lms)与论文，不能将这些能力归给本章小模型。

## 17a.8 实际执行与理解检查

从仓库根目录安装项目后运行：

```bash
python scripts/diffusion_demo.py --mode masked --train-steps 300 --sample-steps 20 --out-dir runs/diffusion-masked
python scripts/diffusion_shapes.py
python -m unittest discover -s tests -p 'test_diffusion_discrete.py' -v
```

第一条使用小型合成 token 模式训练去噪网络；应看实际 loss、mask 终点和固定条件检查。
第二条检验连续文本 embedding 与其他模态张量怎样进入公共 Gaussian 算法。
检查程序性质通过，只表示实现满足相应约束；公开文本 held-out 评价另见[作业六](../assignments/06-diffusion.md)。
若要从合成序列走向真实语料，需固定 tokenizer 与数据 split，处理有效长度，
记录目标 token 数与不同噪声率的训练覆盖，再报告 held-out loss 和明确的采样质量指标。
不要拿 masked CE 的一个平均数直接与 AR perplexity 比较。

1. 给 token ID 全部重新编号，为何 discrete transition 的类别含义仍可保持，而 Gaussian 数值加噪会改变任务？
2. 给定 $Q_t$ 与 $\bar Q_{t-1}$，逐项枚举后验并检查 $t=1$；何时条件事件不可能？
3. 找出“先求后验再混合”与“先混合联合概率再归一化”使用相同权重时不同的例子。
4. 从 $t=0.8$ 到 $s=0.6$ 的随机揭示概率是多少？最后到 $s=0$ 时发生什么？
5. 为什么 uniform timestep 下 masked-position mean CE 不等于 MDLM 的连续时间 NELBO？
6. 写出左、右上下文 infilling 的 condition mask，并核对 forward、loss 与每一步 reverse 都保留它。
7. 对全 MASK 生成，哪些信息实际进入网络，哪些 clean 信息只能用于监督？
8. 比较 AR 与 Diffusion 时，除 NFE 外还需要记录哪些成本，以及什么质量与长度协议？

## 原始来源与阅读目的

- [Diffusion-LM，2022](https://arxiv.org/abs/2205.14217) / [作者仓库](https://github.com/XiangLi1999/Diffusion-LM)：连续词向量、学习输出分布、rounding/clamping 与梯度控制。
- [DiffuSeq，ICLR 2023](https://arxiv.org/abs/2210.08933) / [作者仓库](https://github.com/Shark-NLP/DiffuSeq)：来源条件固定、目标 embedding 的 partial noising 与 seq2seq 训练。
- [D3PM，2021](https://arxiv.org/abs/2107.03006) / [Google Research 实现](https://github.com/google-research/google-research/tree/master/d3pm)：行向量转移、精确后验、uniform/absorbing 等结构与原论文联合概率参数化。
- [MDLM，NeurIPS 2024](https://arxiv.org/abs/2406.07524) / [作者仓库](https://github.com/kuleshov-group/mdlm)：SUBS、连续时间加权 masked loss、似然界与采样。
- [LLaDA，2025](https://arxiv.org/abs/2502.09992) / [作者仓库](https://github.com/ML-GSAI/LLaDA)：规模化 masked language modeling、条件生成与 remasking。
- [Block Diffusion，ICLR 2025](https://arxiv.org/abs/2503.09573) / [作者仓库](https://github.com/kuleshov-group/bd3lms)：块间 AR、块内离散 Diffusion、变长与 cache 的结构边界。
- [Fast-dLLM，2025](https://arxiv.org/abs/2505.22618)：在既有模型上研究近似 KV cache 与置信度并行解码；本仓库未运行其大型实验。

以上链接提供原始机制与完整配方；本章文字、矩阵例子和小代码用于独立学习，不搬运第三方训练体系。
