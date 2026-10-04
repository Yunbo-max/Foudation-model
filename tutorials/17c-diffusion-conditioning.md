# 17c　Diffusion 条件生成：从 cat、add 到 AdaLN、attention 与空间控制

> 前置：[17 概率过程与采样](17-diffusion.md)、[17a 文本表示](17a-diffusion-text.md)、[17b 各模态](17b-diffusion-modalities.md)；实现：[conditioning.py](../src/fm_tutorial/diffusion/conditioning.py)；运行：[结构检查](../scripts/conditioning_check.py)、[条件训练演示](../scripts/conditioning_demo.py)；实践：[作业六](../assignments/06-diffusion.md)；原始来源：[diffusion 来源表](../references/diffusion.md)。

假设用户说“生成一辆红色汽车”。模型怎样知道“汽车”是主体、“红色”是属性，而 diffusion timestep 只是当前噪声程度？如果再要求“红车在左、蓝车在右”，一个全局向量是否足够？如果给出人体姿态图，条件的每个位置又怎样进入图像的对应位置？

本章沿着**条件放在哪里、沿哪个轴传递、怎样影响计算、训练和采样怎样配合**解释这些问题。常见缩写 `adlan` 通常指 `AdaLN`，即 adaptive Layer Normalization。`cat`、`add`、AdaLN、cross-attention、ControlNet 与 CFG 不属于同一个层面的互斥选项；同一模型可以同时使用其中几种。

代码使用小张量检查信息流、shape 和梯度。它们没有训练出文字到图像能力，也不构成生成质量 benchmark。所有 Python 代码块可在安装本项目后的仓库根目录独立运行；实现完整性检查另运行 `python scripts/conditioning_check.py`。

## 17c.1 条件生成究竟改变了什么

无条件模型学习 $p_\theta(x_0)$；条件模型学习 $p_\theta(x_0\mid c)$。训练样本必须提供有意义的配对 $(x_0,c)$：图像和描述、记录和目标类别、视频和首帧等。仅在网络接口里增加一个参数，不能让未训练的网络理解它。

以 epsilon 预测为例，常用训练目标为：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\mathcal L=\mathbb E_{(x_0,c),t,\epsilon}
\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|^2.
$$

常见条件生成中，前向核仍是 $q(x_t\mid x_0)$，不必把文字混进 Gaussian 噪声；条件改变的是去噪预测和反向转移 $p_\theta(x_{t-1}\mid x_t,c)$。当终点充分接近纯噪声时，通常仍从 $\mathcal N(0,I)$ 初始化。条件相关的前向过程或终点先验也可以设计，但需要重新定义对应训练和采样，不能由“条件生成”四个字自动推出。

在均方误差下，理想预测是 $\mathbb E[\epsilon\mid x_t,t,c]$。同一带噪状态可能对应多种干净样本；条件会改变这种可能性的权重，因而改变应该预测的噪声。它不是每一步把一张干净目标图叠上去。

必须区分两类输入：

| 输入 | 例子 | 要回答的问题 | 去掉后的问题 |
| --- | --- | --- | --- |
| 噪声时间 $t$ | 整数步、连续时间、noise level | 当前有多少噪声，使用哪一阶段的去噪规律？ | 网络混淆高噪声结构恢复与低噪声细节恢复 |
| 内容条件 $c$ | 类别、文本、姿态、首帧、已知表格列 | 在众多可能样本中希望生成哪一类、满足什么观测？ | 模型生成边缘分布，而非指定条件下的分布 |

两者可以先编码再相加，比如 $e=e_t+e_c$，但仍有不同语义。CFG 的“无条件”分支通常删除内容条件，**保留同一个噪声时间**。视频还存在真实帧时间 $\tau$：$t$ 是去噪进度，$\tau$ 是事件发生时间，不能混用。

下文记号统一为：数据 token $H\in\mathbb R^{B\times N\times D}$，全局条件 $c\in\mathbb R^{B\times K}$，条件 token $C\in\mathbb R^{B\times M\times K}$，空间条件 $S\in\mathbb R^{B\times K\times h\times w}$。$N$ 可以是图像 patch 数、文本位置数、表格列数或视频时空 token 数。

## 17c.2 先用一张表看清信息进入哪条路径

| 方法 | 条件与核心公式 | 条件进入哪里 | 适合解释的场景 | 主要代价或限制 |
| --- | --- | --- | --- | --- |
| Feature/channel cat | $W[H;c]$ 或 $\mathrm{Conv}([F;S])$ | 特征维或空间通道 | 全局类别；对齐的图像、深度、mask | 维度增加；空间 cat 必须对齐坐标 |
| Add | $H+W_c c$ | 每个位置的特征偏移 | 时间、类别、简单元数据 | 广播条件本身不提供位置绑定 |
| FiLM | $(1+\gamma(c))\odot H+\beta(c)$ | 特征缩放与偏移 | 类别、风格、任务状态 | 全局调制量相同；并非 token 检索 |
| AdaLN / AdaGN | $(1+\gamma(c))\odot\mathrm{Norm}(H)+\beta(c)$ | 归一化后的特征 | Transformer / U-Net 的时间与全局条件 | 归一化轴不同；通常先压成全局向量 |
| AdaLN-Zero | AdaLN 子层再乘 $g(c)$ 加回 residual | 调制量与残差门 | DiT block 的条件与初始化 | 需理解零初始化下的梯度路径 |
| Cross-attention | $\mathrm{softmax}(Q_HK_C^\top/\sqrt d)V_C$ | 每个数据位置对条件 token 的读取 | 多对象、多属性、长文本 | attention 为 $N\times M$；mask 必须正确 |
| Prefix / in-context | $\mathrm{SelfAttn}([C;H])$ | 同一 token 序列 | 少量条件 token；联合交互 | 长度变成 $M+N$；完整 attention 成本增长 |
| 空间 residual / ControlNet | $F+Z(G(F,S,t,C))$ | 多尺度局部特征 | 边缘、姿态、深度、布局 | 额外分支与对齐；学习控制不等于硬约束 |
| SPADE | $\gamma(S)_{kij}\mathrm{Norm}(F)_{kij}+\beta(S)_{kij}$ | 逐空间位置的归一化调制 | 分割布局、局部语义 | 需要保留空间结构；不是全局 AdaLN |
| Guidance | 条件/无条件预测或外部梯度的组合 | 采样时的更新方向 | 调整条件遵循强度 | 额外计算；强度影响质量与多样性 |

这张表比较的是**信息路径**，不是效果排名。原始 DiT 在 class-conditional ImageNet 设置中比较 in-context、cross-attention、AdaLN 和 AdaLN-Zero；其结果不能证明 AdaLN 在长文本空间绑定上普遍优于 cross-attention。参见 [DiT 原论文 §3](https://arxiv.org/html/2212.09748v2)。

## 17c.3 Cat：先问拼的是哪个轴

### 特征拼接：同一位置多看一组输入

若 $H$ 为 `[B,N,D]`，全局 $c$ 为 `[B,K]`，先把条件广播为 `[B,N,K]`，再拼接成 `[B,N,D+K]`：

$$
H'_{bn}=W[H_{bn};c_b]+b.
$$

条件和数据在进入下一层前有自己的通道。下一层可以读取两者，但在**单个线性层**中，cat 并没有神秘的额外表达能力：分块 $W=[W_H,W_c]$ 后，严格有 $W[H;c]=W_HH+W_cc$。在这两个相等结果后施加同一个非线性，等价关系仍成立。分别对两条输入分支做不同非线性再相加、保留不同通道经过多层处理，或改变数据投影与参数共享约束，才可能形成不同架构。

下面直接验证这个恒等式；注意 `cat` 不是“两个张量随便拼”，非拼接维必须一致。

```python
import torch
from torch import nn

torch.manual_seed(0)
B, N, D, K, O = 2, 5, 8, 3, 6
h, c = torch.randn(B, N, D), torch.randn(B, K)
layer = nn.Linear(D + K, O)
cb = c[:, None, :].expand(-1, N, -1)
via_cat = layer(torch.cat((h, cb), dim=-1))
via_sum = torch.nn.functional.linear(h, layer.weight[:, :D], layer.bias)
via_sum += torch.nn.functional.linear(cb, layer.weight[:, D:])
torch.testing.assert_close(via_cat, via_sum)
print(via_cat.shape)  # [2,5,6]：仅验证代数关系
```

空间 channel cat 则是 `[B,C,h,w]` 与 `[B,K,h,w]` 沿通道拼接。边缘图位置 `(i,j)` 与图像特征位置 `(i,j)` 对应，卷积可以读取局部条件。不同分辨率需要按该特征层的坐标和尺度下采样/编码；resize 到同样高宽并不自动保证投影、裁剪和相机坐标正确。

### Token 拼接：多了一些位置，而不是多了一些通道

把投影后的文本 `[B,M,D]` 和图像 `[B,N,D]` 沿序列轴拼接，得到 `[B,M+N,D]`，是另一种 `cat`。它要靠后续 self-attention 交换信息；若下一层只是逐 token 的 MLP，图像 token 看不到新增文本 token。

因此问“cat 好不好”时必须补全：沿 `dim=-1` 拼特征、沿 `dim=1` 拼序列，还是在 NCHW 中沿通道拼空间图？三者的条件对齐方式、成本和归纳偏置不同。

## 17c.4 Add：把条件变成特征空间中的偏移

最直接的 add conditioning 为：

$$
H'_{bn}=H_{bn}+W_c c_b,
\qquad W_c:\mathbb R^K\rightarrow\mathbb R^D.
$$

条件宽度不同就先投影；`[B,D]` 必须扩成 `[B,1,D]` 才能沿 token 广播。对图像 feature 则扩成 `[B,D,1,1]`。原 DiT 中 $e_t+e_c$ 是**两个条件 embedding 的相加**，后面再生成 AdaLN 参数；它不等同于直接做 $H+e_c$。

Add 改变的是每个通道的平移，FiLM 还允许按条件改变该通道的增益；AdaLN 更在调制前加入特定归一化。三者不能因为最后都有加号而混为一个方法。

```python
import torch
from fm_tutorial.diffusion.conditioning import FeatureConcat, AddCondition

B, N, D, K = 2, 5, 8, 3
h, c = torch.randn(B, N, D), torch.randn(B, K)
for module in (FeatureConcat(D, K), AddCondition(D, K)):
    out = module(h, c)
    assert out.shape == h.shape
    out.square().mean().backward()
    print(type(module).__name__, tuple(out.shape))
```

广播 add 可以给所有位置发送“这是汽车类”的信号；位置编码和数据特征仍能使不同位置产生不同响应。但条件偏移自身不告诉网络“左边归红车、右边归蓝车”。把复杂描述压成一个向量后，原 token 对应关系不再可直接检索，模型必须从压缩表示中恢复。

前节的线性等价式允许 $W_H$ 自由学习；本项目 `AddCondition` 固定数据路径系数为恒等，只投影条件，因此不与任意 `FeatureConcat` 模块完全相同。公平比较应写清数据投影与参数预算，而非仅凭 `cat` 或 `add` 名称推断容量。

注入位置也有影响。例如，对某个 token 的**所有通道都相同**的标量偏移，会被随后 LayerNorm 的去均值消去；一般逐通道向量偏移并不会全部消失。分析时应写出 `Norm → add → block` 还是 `add → Norm → block`，不能只写“用了 add”。

## 17c.5 FiLM 与 AdaLN：条件开始改变处理特征的方式

### FiLM：条件预测 scale 与 shift

[FiLM 原论文](https://arxiv.org/abs/1709.07871)提出由条件产生逐特征仿射变换。本章采用便于恒等初始化的写法：

$$
(\gamma,\beta)=f_\theta(c),\qquad
H'=(1+\gamma[:,None,:])\odot H+\beta[:,None,:].
$$

原始的 $\gamma\odot H+\beta$ 与这里的 $(1+\gamma)\odot H+\beta$ 是不同参数约定；零 scale 在前一种写法抹去特征，在后一种写法保留特征。读源码时必须核对是否有 `1 + scale`。

Add 对条件的作用只是偏移；FiLM 可以放大、抑制或翻转某些特征通道。若某通道与车轮纹理相关，条件可以改变其处理强度。这个例子解释计算机制，不声称现实网络的某一个通道必然有单一、人可命名的语义。

### AdaLN：先规范特征，再按条件重新调制

对每个 `[B,N]` 位置，LayerNorm 沿最后的 $D$ 个通道计算：

$$
\mu_{bn}=\frac1D\sum_d H_{bnd},\quad
v_{bn}=\frac1D\sum_d(H_{bnd}-\mu_{bn})^2,\quad
\mathrm{LN}(H)_{bnd}=\frac{H_{bnd}-\mu_{bn}}{\sqrt{v_{bn}+\varepsilon}}.
$$

$$
\mathrm{AdaLN}(H,c)=(1+\gamma(c))\odot\mathrm{LN}(H)+\beta(c).
$$

这里条件改变归一化后的 scale/shift，**不改变 LayerNorm 统计量的计算公式**。常见实现把基础 LN 设为 `elementwise_affine=False`，避免再叠一组固定可学习仿射参数；保留固定 affine 也可以，但要明确两套参数的组合。全局 $\gamma,\beta$ 在各 token 相同，归一化后的数据和后续 attention 仍随位置不同。

AdaGN 将 `LN` 换成 GroupNorm，常用于卷积 U-Net；AdaIN 用 InstanceNorm，常见于风格条件。它们的统计轴分别涉及通道组或单通道空间位置，不能只改名字、仍沿最后一维归一化。RMSNorm 不减均值，若对它作条件调制，也应说明所用底层 norm。

```python
import torch
from fm_tutorial.diffusion.conditioning import FiLM, AdaLN

B, N, D, K = 2, 5, 8, 3
h = torch.randn(B, N, D, requires_grad=True)
c = torch.randn(B, K)
film, adaln = FiLM(D, K), AdaLN(D, K)
y_film, y_adaln = film(h, c), adaln(h, c)
assert y_film.shape == y_adaln.shape == h.shape
(y_film.square().mean() + y_adaln.square().mean()).backward()
assert h.grad is not None and torch.isfinite(h.grad).all()
print(y_film.shape, y_adaln.shape)
```

类别、风格向量、任务 ID 或少量元数据适合先做全局 embedding，再调制多层。长文本也能先 pooled 成全局向量，但这主动引入信息瓶颈；例如“红车在左、蓝车在右”的对象属性绑定通常还需要细粒度条件路径与匹配的训练数据。

## 17c.6 AdaLN-Zero：六组向量与两条残差路径

一个标准 DiT AdaLN-Zero block 包含 attention 和 MLP 两个子层。令全局条件 $e=e_t+e_c$，调制头一次输出六组 `[B,D]` 向量：

$$
(\beta_A,\gamma_A,g_A,\beta_M,\gamma_M,g_M)=f(e).
$$

$$
U=(1+\gamma_A)\odot\mathrm{LN}(H)+\beta_A,
\qquad H_1=H+g_A\odot\mathrm{SelfAttn}(U),
$$

$$
V=(1+\gamma_M)\odot\mathrm{LN}(H_1)+\beta_M,
\qquad H_2=H_1+g_M\odot\mathrm{MLP}(V).
$$

广播轴均为 token 轴。scale/shift 调整子层看到的输入；gate 控制子层输出有多少加回主路径。gate 不是 attention probability，也不是决定忽略哪个文本 token 的 mask。

[原作者 `models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py)将调制头的最后一个线性层 weight 和 bias 都置零，六组向量初始都为零。于是子层仍计算 `Attention(LN(H))` 和 `MLP(LN(H))`，但残差乘零，**每个 block 初始严格为恒等映射**。主干 attention/MLP 参数并没有全变成零。

### 为什么零 gate 之后还能学

简化为 $Y=H+gF(H)$，假设有非零的上游损失梯度 $G=\partial\mathcal L/\partial Y$，则初始化 $g=0$ 时：

$$
\frac{\partial\mathcal L}{\partial g}=\sum_n G_n\odot F(H_n),
\qquad
\frac{\partial\mathcal L}{\partial\theta_F}
=g\,G^\top\frac{\partial F}{\partial\theta_F}=0.
$$

因此 gate 对应的调制头输出行可以先收到梯度；该子层的 scale/shift 和 attention/MLP 参数在这个初始时刻被 gate 阻断。最后调制线性层的 weight 全零，还会阻断梯度到它之前的条件变换。gate 更新后，其他路径逐渐开始学习。梯度是否实际非零仍取决于输入、损失与上游梯度，不能说“任何损失都保证更新”。

```python
import torch
from fm_tutorial.diffusion.conditioning import AdaLNZeroBlock

torch.manual_seed(7)
h = torch.randn(2, 5, 8, requires_grad=True)
c = torch.randn(2, 3)
block = AdaLNZeroBlock(d_model=8, cond_dim=3, heads=2)
out = block(h, c)
torch.testing.assert_close(out, h, rtol=0, atol=0)
# 用显式非零上游梯度检查初始参数梯度，不以零损失测试学习。
(out * torch.randn_like(out)).sum().backward()
active = [name for name, p in block.named_parameters()
          if p.grad is not None and p.grad.abs().sum().item() > 0]
assert active
print("initially active parameters:", active)
```

还有一个必须分开的事实：完整原始 DiT 的 **FinalLayer 输出投影也零初始化**，其最终 AdaLN 调制头也置零。因此完整模型初始输出 epsilon 为零；输出投影为零又使第一次 backward 的梯度不能传到这些 block。上面展示的是孤立 block 收到上游梯度后的行为，不能拿它断言完整模型第一步所有 gate 都更新。

AdaLN-Zero 解决的是条件注入和初始残差幅度的联合设计，不提供几何硬约束，也不天然等于更强的文字对齐。比较初始化策略时应保持训练预算、主干、数据与采样设置一致。

## 17c.7 Cross-attention：每个数据位置可以读不同的条件 token

考虑描述“左边一辆红车，右边一辆蓝车”。与 pooled 文本不同，文本 encoder 输出 $C=[c_1,\ldots,c_M]$，保留不同 token 的上下文表示。图像 patch 的当前位置特征 $H$ 做 query，文本做 key/value：

$$
Q=HW_Q,\quad K=CW_K,\quad V=CW_V,
\quad A=\mathrm{softmax}_M(QK^\top/\sqrt{d_h}),
\quad H'=H+(AV)W_O.
$$

多头分数 shape 是 `[B,heads,N,M]`；softmax 沿条件 token 轴。不同图像位置、不同头、不同噪声阶段可以读取不同的条件组合，读取权重依赖当前数据特征。它建立了**动态 token 对齐的路径**；是否正确绑定对象、数量和方位，需要配对数据、位置表示和评价验证，不由 attention 公式保证。

Cross-attention 与 self-attention 的分数轴不同：前者为 `N×M`，后者为 `N×N`。常见 block 两者都有，先让图像位置交换信息，再读取文本。参见 [LDM 论文](https://arxiv.org/abs/2112.10752)与[作者 `attention.py`](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py)中的 `CrossAttention`、`BasicTransformerBlock`。

```python
import torch
from fm_tutorial.diffusion.conditioning import CrossAttentionCondition

torch.manual_seed(0)
h, text = torch.randn(2, 5, 8), torch.randn(2, 4, 6)
# 本模块约定 True = 忽略；这不是所有 attention API 的统一约定。
padding_mask = torch.tensor([[False, False, True, True],
                             [False, False, False, True]])
cross = CrossAttentionCondition(8, 6, heads=2).eval()
out = cross(h, text, padding_mask=padding_mask)
changed = text.clone()
changed[padding_mask] = 1000 * torch.randn_like(changed[padding_mask])
torch.testing.assert_close(out, cross(h, changed, padding_mask=padding_mask))
assert out.shape == h.shape
print(out.shape)
```

这里验证 padding 不影响结果，不是验证模型理解了语句。实际文本编码器还要收到其自己的 padding mask，才能避免有效 token 表示先被 padding 污染。条件序列至少保留一个有效 token；把整行 key 全 mask 会产生无定义的读取，应改为有效的 null token，而不是期望 softmax 自动变成零。

Cross-attention 自身并不提供文本顺序编码：若同时置换其 K/V token，且不改变内容与 mask，读取结果不变。文本的顺序关系通常已经写入文本 encoder 的上下文表示；图像位置来自二维位置编码或卷积结构。不能把“有 attention”当作“已经有所有坐标信息”。

## 17c.8 Prefix、in-context 与 joint attention 怎样区分

将条件先投影到 $D$，形成 $Z_0=[C_0;H_0]$，完整非因果 self-attention 在每一层得到 $Z_{\ell+1}=[C_{\ell+1};H_{\ell+1}]$。数据 token 读取条件，条件 token 也可以读取数据并更新；最终只取数据部分预测 epsilon。

原 DiT 的 **in-context conditioning** 是将 timestep 与 class embedding 作为两个额外 token 放入序列，最后删除它们。这是 token 序列维的拼接，与第 17c.3 节的逐位置 feature cat 不同；条件只有两个全局 token，也不能直接等同于长文本 conditioning。

```python
import torch
from fm_tutorial.diffusion.conditioning import PrefixCondition

h, condition_tokens = torch.randn(2, 5, 8), torch.randn(2, 3, 6)
prefix = PrefixCondition(d_model=8, cond_dim=6, heads=2)
out = prefix(h, condition_tokens)
assert out.shape == h.shape
out.square().mean().backward()
print(out.shape)  # 内部长度 3+5，输出保留 5 个数据 token
```

这个教学组件只展示一个拼接 self-attention block，返回数据 token。完整多层 in-context 模型应把更新后的**整个联合序列**传到下一层；连续调用只返回 `H` 的组件、每次重新插入原始 `C`，会得到另一种架构。输入需要数据位置、条件位置或模态/type 信息；本小组件不实现完整文本 encoder 和二维位置编码。

“Prefix”在这里指序列前面的可见条件 token，不能与自回归 LLM 的 prefix tuning/KV 注入自动划等号。扩散去噪通常没有 next-token 的因果限制；若把 LLM causal mask 原样搬来，token 信息流会随拼接顺序改变。

Joint attention 是更宽泛的联合读取机制。比如 [SD3 / MMDiT 原论文](https://arxiv.org/html/2403.03206v1)保留图像和文本的独立参数路径，在 attention 中联合交互，再分别处理两种流。它并不等于“先投影到相同宽度，然后所有 token 共用同一个普通 block”的简单 prefix。比较时要写清 Q/K/V 是否分流、哪些 token 更新、哪些层合流及返回哪些状态。

## 17c.9 空间条件：为什么姿态图不能随便 pooled 成一个向量

类别“人”只告诉模型生成什么；骨架图还告诉它手腕、肩膀、膝盖的位置。把整张姿态图平均池化后，很多不同姿态可能得到相近摘要。保留 `[B,K,h,w]` 结构，使特征层能够读取对应位置的控制信号。

最直接的办法是 channel cat；也可以先用条件 encoder $E_S$ 提取多尺度特征，再相加、cross-attend 或调制。对深度、边缘、分割、稀疏姿态，编码器和预处理不同：深度要定义单位及无效区，分割 ID 不能当作有距离意义的连续颜色值，姿态热图要定义关节顺序。

### 零初始化空间 residual：从“先不扰动”开始学习

令 $G$ 为可训练条件分支，$Z$ 为 weight/bias 都置零的输出卷积：

$$
F'=F+Z(G(F,S,t,C)).
$$

初始 $Z(G)=0$，主路径输出不改变；有上游梯度时，输出 zero-conv 可以先收到梯度，前面的条件分支初始梯度则被零 weight 阻断。这与 AdaLN-Zero gate 的学习次序有相似处，但一个是残差输出投影，一个是逐通道乘法门，不能视为相同结构。

```python
import torch
from fm_tutorial.diffusion.conditioning import ZeroSpatialResidual

feature, hint = torch.randn(2, 8, 6, 6), torch.randn(2, 3, 6, 6)
adapter = ZeroSpatialResidual(channels=8, hint_channels=3)
out = adapter(feature, hint)
torch.testing.assert_close(out, feature, rtol=0, atol=0)
(out * torch.randn_like(out)).sum().backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0
           for p in adapter.parameters())
print(out.shape)
```

[ControlNet 原论文](https://arxiv.org/abs/2302.05543)将预训练主干与可训练控制分支连接，用零初始化卷积逐步接入条件；[作者 `cldm.py`](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py)可追踪 hint 编码、多尺度 zero-conv 输出及主干接收控制的路径。上面的 `ZeroSpatialResidual` 只是展示这一基本计算，不是复制完整 ControlNet，也没有预训练主干、冻结策略或多尺度网络。

“Adapter”泛指额外条件模块；有的输出多尺度空间特征，有的把图像参考转成 attention token。应查看具体模块的 shape 与注入点，不能把所有 adapter 都解释为一个 1×1 zero-conv。

### IP-Adapter：参考图像也可以走独立的 attention 条件路径

[IP-Adapter 原论文](https://arxiv.org/abs/2308.06721)把参考图经图像 encoder 与投影变成条件 token，同时保留文本条件。其关键是解耦的 cross-attention，概念式可写为：

$$
R=\mathrm{Attn}(Q_H,K_{text},V_{text})
+\lambda_{img}\,\mathrm{Attn}(Q_H,K_{img},V_{img}).
$$

两次 attention 分别在文本 token 与图像 token 上做 softmax，后面才组合输出。这不等价于把两组 K/V 拼在一起做**一次** softmax：拼接时两种模态在同一个归一化分母中竞争，解耦时各有自己的分母与图像支路强度。源码可能先把输入 token 放在一个张量中，但计算时又切开；应追踪实际 attention，而不是看到 `cat` 就结束分析。

从[作者 `attention_processor.py`](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py)检查独立图像 K/V 投影、两次读取及加权合并。参考图传递的是视觉内容/外观线索；它与逐位置对齐的姿态、深度控制不同，也不自动保证复制人物身份或精确布局。本项目没有实现或加载完整 IP-Adapter。

### SPADE：scale/shift 也可以随空间位置变化

全局 AdaLN/AdaGN 通常输出 `[B,C]` scale/shift 后广播。SPADE 从布局图产生 `[B,C,h,w]` 的调制量：

$$
F'_{bcij}=\gamma(S)_{bcij}\,\mathrm{Norm}(F)_{bcij}+\beta(S)_{bcij}.
$$

同一通道在“道路”和“天空”位置可以用不同调制量；它把局部语义写入归一化后的每个位置。参见 [SPADE 原论文](https://arxiv.org/abs/1903.07291)与[作者仓库](https://github.com/NVlabs/SPADE)。SPADE 是空间自适应归一化方法，原论文研究语义图像合成，并非 diffusion 专属。

条件位置对齐只是一条可学习控制路径。ControlNet、spatial cat、SPADE 都不保证每个骨架点精确命中、每个深度值严格满足或生成几何完全一致；如果任务需要严格观测保持，还要定义采样约束与验收规则。

## 17c.10 Conditioning 与 guidance：网络读条件，采样调整方向

**Conditioning** 是网络内部接入 $c$；**guidance** 是采样时调整预测或 score。一个模型可以有很完整的 cross-attention conditioning，却使用 guidance scale 1；也可以用外部分类器指导一个无条件去噪器。

### Classifier guidance：梯度来自带噪分类器

由 Bayes 公式：

$$
\nabla_{x_t}\log p_t(x_t\mid c)
=\nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(c\mid x_t).
$$

Classifier guidance 训练一个读取 $(x_t,t)$ 的分类器，再把 $\nabla_{x_t}\log p_\phi(c\mid x_t,t)$ 加入 score/采样均值。普通干净图像分类器不自动适用于高噪声状态；还需要对输入求梯度，不能把类别概率标量直接加到 epsilon。对应 epsilon 参数化为 $\epsilon=-\sigma_t s$ 时，score 的正向分类梯度在 epsilon 中表现为减去 $\sigma_t$ 倍该梯度。

参见 [Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233)与[作者 `gaussian_diffusion.py`](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)中的 `condition_mean`、`condition_score`。具体系数还取决于采样器的更新式，不能把 DDPM 的均值修正原样贴入任意 ODE。

### CFG：训练 null 分支，采样组合两份预测

[Classifier-Free Guidance 原论文](https://arxiv.org/abs/2207.12598)联合学习条件和无条件预测。常见实现训练时以一定概率把**内容条件**换成有效的 null 条件；类别可预留 null class ID，文本可采用该模型训练过的空文本/空条件表示。不是随机将噪声时间也清零，也不是仅在采样时临时放入一个从未训练的零向量。

本章及代码使用如下 scale 约定：

$$
\epsilon_u=\epsilon_\theta(x_t,t,\varnothing),\quad
\epsilon_c=\epsilon_\theta(x_t,t,c),\quad
\epsilon_{\mathrm{cfg}}=\epsilon_u+w(\epsilon_c-\epsilon_u).
$$

此时 $w=0$ 是无条件预测，$w=1$ 是原始条件预测，$w>1$ 是沿条件差异方向外推。有的论文写 $(1+\lambda)\epsilon_c-\lambda\epsilon_u$，对应 $w=1+\lambda$；有的工程接口把 0 当作“关闭额外 guidance”，必须查看实现，不能仅比较数字。

```python
import torch
from fm_tutorial.diffusion.conditioning import classifier_free_guidance

u, c = torch.randn(2, 4), torch.randn(2, 4)
torch.testing.assert_close(classifier_free_guidance(u, c, scale=0), u)
torch.testing.assert_close(classifier_free_guidance(u, c, scale=1), c)
torch.testing.assert_close(classifier_free_guidance(u, c, scale=3), u + 3 * (c-u))
print("CFG conventions checked")
```

理想 score 下，固定噪声时刻的组合满足：

$$
s_w=(1-w)\nabla\log p_t(x_t)+w\nabla\log p_t(x_t\mid c)
=\nabla\log\!\left[p_t(x_t)^{1-w}p_t(x_t\mid c)^w\right].
$$

这是该噪声层上一个**未归一化乘积形式**的 score。跨时间的这些分布不必构成同一前向过程的相容边缘，因此该恒等式并不是“最终一定精确采到 $p_0(x_0\mid c)^w p_0(x_0)^{1-w}$”的证明。实际网络误差、有限步采样和过大的 scale 还会带来细节失真、饱和或多样性损失，应在真实条件任务中评价。

一般每步需要条件和无条件两份预测；可分两次 forward，也可把相同的 `xt,t` 与两套条件在 batch 维拼接，一次调用处理两倍 batch。后者减少调用开销，并没有把两份计算变成一份。在包含方差/其他输出通道的模型中，应按原参数化决定哪些预测接受 guidance。

### 把条件模块接到一次真实训练与 DDIM 采样

[conditioning_demo.py](../scripts/conditioning_demo.py)真的优化 epsilon loss，再调用本项目 DDIM；数据明确是两类合成二维 Gaussian 点，中心为 `[-1,0]` 与 `[1,0]`，没有文字 encoder、图像 codec 或真实数据 benchmark。

```bash
python scripts/conditioning_check.py
python scripts/conditioning_demo.py --method adaln --train-steps 300 --sample-steps 20 --condition-dropout 0.1 --guidance-scale 1.5 --out-dir runs/conditioning-adaln
# 可改为 --method cat、add 或 film；每次独立训练并保存运行记录。
```

按代码追踪四步：

1. `xt: [B,2]` 与三个时间特征拼成 `[B,5]`，投影到 `[B,1,32]`。数组时间 `t=0..T-1` 映射为 `(t+1)/T`，训练和采样采用同一约定。
2. 标签 0/1 经 `Embedding(3,32)` 得到内容条件；标签 2 是可学习 null 条件。`cat/add/film/adaln` 在同一模块接口读取 `[B,1,32]` 和 `[B,32]`，输出头恢复 `[B,2]` epsilon。
3. 训练按 `condition_dropout` 把部分标签改为 2，**原始干净点、目标噪声与实际 timestep 都保留**。同一网络因而学习 $\epsilon_\theta(x_t,t,y)$ 与 $\epsilon_\theta(x_t,t,\varnothing)$。
4. DDIM 每步将相同 `xt,t` 扩成双 batch，按 `[null,condition]` 顺序预测并拆开，再用本节 CFG 公式。两种类别采样使用相同的初始噪声数组，使条件改变对应样本的生成路径。

这与原 DiT 的“时间 embedding + class embedding → AdaLN”还有一个区别：小演示把时间放在数据特征路径，class 才作为调制条件；两种方式都明确提供了噪声时间，不能说 `c` 就是 timestep。演示的 token 数为 1，不适合评价空间 attention。

查看 `training.csv` 的实际训练 loss、`samples.npy` 的样本与 `report.json` 的 null 训练计数、类别均值、参数量和成对初始噪声标记。单次末 batch loss 含随机波动；均值移动只解释条件路径产生的行为，不是 held-out 质量证明。不同方法参数数量也不完全相同，不能把 300 步结果当作架构排名。

这个短训练演示显式使用 100 步线性 beta 日程，从 `1e-4` 到 `0.12`，终点 $\bar\alpha_T\approx0.0019$；标准 Gaussian 初始化仍是近似终点先验。它与核心模块默认的 cosine 日程不同：当 $\bar\alpha_T$ 极小时，$\hat x_0=(x_T-\sqrt{1-\bar\alpha_T}\hat\epsilon)/\sqrt{\bar\alpha_T}$ 会大幅放大短训练网络的 epsilon 误差。这里选择温和日程是教学配置，不代表一般任务的最佳日程。改变 `--diffusion-steps` 还会改变终点信噪比，极少步 CLI 检查只验证程序链路。脚本记录完整日程和实际训练分支计数；若一次运行没有看到 null 或条件训练样本，会明确报错。

## 17c.11 四类实际任务怎样选择和组合条件

### 图像：对象语义、属性绑定与几何布局是不同的信息

“生成汽车类别”可以使用 label embedding 加时间 embedding，再经 AdaLN-Zero。“一辆红车”可以由 pooled 文本提供全局条件，也可以读取文本 token。“红车在左、蓝车在右”更需要保留对象、属性和方位的对应关系；cross-attention / joint attention 提供读取路径，训练分布和空间表示决定能否学到绑定。

“按此边缘/姿态/深度生成”则需要条件的坐标结构，可用 spatial cat、多尺度控制分支或空间调制。实际组合可以是：**时间 + 全局类别 → AdaLN；文本 token → cross-attention；姿态 → ControlNet；采样 → CFG**。这些路径分别服务不同信息，不要求四选一。

### 表格：目标类别与已知列不能混成一种条件

合成 $p(\mathrm{row}\mid y)$ 时，外部类别 $y$ 可以进入全局 add/FiLM/AdaLN；例如按类别生成记录，训练要有对应行与标签。若 $y$ 是该行的目标列，可以选择只生成其余列，或在输出中按协议固定 $y$，避免宣称一个软条件自动保证目标列取值。

补全 $p(x_{\mathrm{unknown}}\mid x_{\mathrm{known}},m)$ 时，条件包括**已知值、列 ID、哪些列已知的 mask**。金额、年龄、职业列有各自类型；数值标准化后输入，类别用合适 embedding，missing 不等于数值 0。列 token conditioning 需要列位置/类型信息；若把列次序打乱又不提供 ID，模型不知每个值属于哪列。

固定已知数值可使用观测一致的采样覆盖规则；类别列要遵守离散转移或固定 token 规则。全局 row 标签适合广播调制；已知列值更像带 schema 的细粒度观测。两者都不自动保证跨行外键、金额总额或其它硬逻辑约束，参见 [17b 表格](17b-diffusion-modalities.md)。

### 文本补全与 Gaussian inpainting 的 mask 不同

在 absorbing-mask 文本 diffusion 中，可以将已知 prompt token 标记为不参与遮掩/更新，未生成部分保持 MASK，使用模型对应的离散 loss 与解掩码规则。硬保持的是 token ID；有些模型采用不同上下文加噪策略，应遵循其训练协议。

在 Gaussian 图像 inpainting 中，若每步直接把干净观测贴到仍带噪的状态上，会产生噪声层级不匹配。一个基本观测覆盖思路是：在当前时刻构造
$x_t^{obs}=\sqrt{\bar\alpha_t}x_0^{obs}+\sqrt{1-\bar\alpha_t}\epsilon^{obs}$，然后用 $m\odot x_t^{obs}+(1-m)\odot x_t^{gen}$ 覆盖已知区；更新到下一时刻后，应使用**下一时刻**的观测噪声层级。最后到干净端才恢复精确观测值。

这个思路的边缘噪声匹配并不自动给出完整精确条件 posterior：观测噪声怎样跨步耦合、是否重采样及使用哪个 sampler 都是算法设计。mask 的空间/列/token 语义应分别定义；不能把 Gaussian 覆盖式用于 categorical ID，也不能把离散 MASK 当作一张图像的零像素。

### 视频与 4D：首帧条件还需要时间和相机结构

视频首帧/参考图可以编码为空间特征或条件 token；文本提供事件描述，真实帧时间 $\tau$ 表示运动位置，diffusion timestep $t$ 表示噪声强度。逐帧只接相同的全局条件，无法充分提供帧间身份和运动对应；去噪网络还要有时间 attention、时空卷积或相应耦合。

4D 在这里是随真实时间变化的 3D 场景。相机内外参、视角、真实时间、已有几何或观测帧都可能成为条件；它们可以用 embedding 调制、ray/point token attention 或空间控制接入。但向网络加一个 camera vector 不自动建立多视角几何一致性：坐标约定、投影、点/面对应、时间耦合及渲染/重建监督仍需定义，参见 [17b 视频与 4D](17b-diffusion-modalities.md)。

## 17c.12 阅读源码与验收：先验证路径，再比较生成质量

按以下顺序读一个条件 diffusion 模型：

1. 找到配对数据与条件 encoder：`c` 是类别、pooled 向量、token 还是空间图？训练/推理是否用相同表示和归一化？
2. 找到每一层注入点：哪个轴 cat，add 在 norm 前还是后，AdaLN 生成几组参数，attention 哪边做 Q/K/V？
3. 核对时间、padding、观测 mask、null 条件：各 mask 的 `True` 意义是否相同，是否出现整行全 mask？
4. 核对初始化及梯度：零的是 residual gate、输出卷积还是完整输出 head？第一步哪些参数实际收到梯度？
5. 核对采样参数化：网络预测 epsilon、score、$v$ 或 velocity；CFG 插入的位置、scale 约定、batch 拼接顺序是否对应？

运行 `python scripts/conditioning_check.py` 可以检查小组件与条件/无条件的调用链。它证明 shape、代数和梯度实现，没有训练大型条件模型。比较 cat/add/AdaLN/cross-attention 时，应固定数据 split、codec、主干/预算、优化器、条件 encoder、采样器与 CFG；记录实际参数、吞吐、显存及真实任务指标，不能用随机张量输出大小给方法排榜。

理解检查：为什么 feature cat 后接单个线性层可拆成两项相加？为什么 AdaLN 的 scale/shift 初始为零不意味着输出为零？为什么孤立 AdaLN-Zero block 和完整零输出头 DiT 的第一步梯度不同？为什么 cross-attention 需要正确的文本与图像位置表示？为什么 $w=1$ 已经是条件预测？为什么 Gaussian inpainting 应按当前噪声层覆盖观测，而离散文本 prompt 可以按其协议固定 token？

下一步做 [作业六](../assignments/06-diffusion.md)：先用模块验证一种条件路径，再在真实配对数据与独立评价上判断条件是否被使用。原论文入口与作者代码定位见 [diffusion 来源表](../references/diffusion.md)。
