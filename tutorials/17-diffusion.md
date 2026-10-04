# 17　Diffusion 基础：从一份带噪观测到可采样的生成模型

> 前置：[00 概率与张量](00-prerequisites.md)、[02 架构](02-architecture.md)；后续：[17a 文本扩散](17a-diffusion-text.md)、[17b 图像、表格、视频与 4D](17b-diffusion-modalities.md)。
> 原创实现：[continuous.py](../src/fm_tutorial/diffusion/continuous.py)、[models.py](../src/fm_tutorial/diffusion/models.py)；实验：[diffusion_demo.py](../scripts/diffusion_demo.py)；交付：[作业六](../assignments/06-diffusion.md)；一手来源：[阅读索引](../references/diffusion.md)。

## 17.1 场景一：知道怎样弄脏，不等于知道怎样恢复

设有一批二维位置：某些点靠近左侧簇，另一些靠近右侧簇。
给每个点加随机扰动很容易；给一份落在两个簇之间的带噪观测，原位置却不唯一。
图像也一样：高噪声下，一块灰色区域可能来自天空、墙面或衣服。
生成模型需要学习数据中这些可能性怎样分布，然后从随机初态构造新样本。
它不是保存每张图片曾加过的噪声，再按原路逐项减掉。

本章沿着一条计算链解释：如何定义噪声，如何在任意噪声等级构造训练输入，网络到底预测什么，预测怎样变成一次更新，更新如何组成完整采样。
二维簇、缺损图像和轨迹补全是贯穿本章的教学场景。
它们帮助判断公式与代码是否对应；合成演示不提供真实图像或文本生成能力的证据。

先把三件事分开：

| 层次 | 要回答的问题 | 二维教学例子 |
| --- | --- | --- |
| 表示 | 数据在哪个空间中？ | 一个样本是 `[2]` 浮点向量 |
| 概率过程 | 什么叫加噪、逆转和采样？ | Gaussian transition 与 schedule |
| 去噪网络 | 怎样从当前状态预测更新所需的量？ | 输入位置与时间的 MLP |

把 MLP 换成 U-Net 或 Transformer，不会自动改变前向噪声公式。
把像素换成文本 latent，会改变表示、解码器与数据分布，不能只改一个 shape 就完成语言生成。
后两章分别沿着这些变化展开。

## 17.2 Gaussian 中的均值、方差与那个 $I$

把一个样本展平成 $x\in\mathbb R^D$，多元正态写作 $\mathcal N(\mu,\Sigma)$。
均值 $\mu$ 指定分布中心；协方差矩阵 $\Sigma$ 的对角元是各坐标方差，非对角元表示共同变化。
标准正态 $\epsilon\sim\mathcal N(0,I)$ 中，$I$ 是 $D\times D$ 单位矩阵。
它表示各维方差为 1、协方差为 0；对于联合 Gaussian，这还意味着各维独立。
这里的 $I$ 不表示“每个数据样本相同”，也不需要在代码中创建大单位矩阵。

若 $z=\mu+\sigma\epsilon$，那么 $z\sim\mathcal N(\mu,\sigma^2I)$。
因此方差为 $\beta$ 的噪声，要乘 $\sqrt\beta$，不能乘 $\beta$。
若误用 $\beta\epsilon$，实际方差会是 $\beta^2$，整条 schedule 都改变了。

```python
import torch

x0 = torch.tensor([[2.0, -1.0], [-2.0, 1.0]])  # [B=2,D=2]
beta = 0.1
epsilon = torch.randn_like(x0)                # 每个坐标独立 N(0,1)
noisy = (1 - beta) ** 0.5 * x0 + beta ** 0.5 * epsilon
```

对图像 `[B,C,H,W]` 调用 `randn_like`，意味着每个像素通道获得独立基础噪声。
数据的空间相关性由真实样本分布与网络学习，不是由 $I$ 提供。
若真实数据各维尺度差异很大，例如“年龄”约几十而“资产”约百万，相同 Gaussian 噪声的相对影响便很不一样。
连续特征应先依据训练集统计量标准化；类别变量另见 [17a](17a-diffusion-text.md) 与 [17b](17b-diffusion-modalities.md)，不能把类别 ID 的数值大小当作物理距离。

## 17.3 前向 Markov 链：只看当前状态的加噪规则

从真实样本 $x_0\sim p_{\mathrm{data}}$ 出发，定义数学时间 $t=1,\ldots,T$：

$$
q(x_t\mid x_{t-1})
=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I),
\qquad 0<\beta_t<1.
$$

写 $\alpha_t=1-\beta_t$，单步采样就是
$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}z_t$。
每一步使用新的独立 $z_t\sim\mathcal N(0,I)$。
Markov 指条件于 $x_{t-1}$ 后，生成下一状态不再需要更早的历史：

$$
q(x_t\mid x_{0:t-1})=q(x_t\mid x_{t-1}),
\qquad q(x_{1:T}\mid x_0)=\prod_{t=1}^Tq(x_t\mid x_{t-1}).
$$

“只看当前状态”是概率条件独立性，不是说当前状态没有历史信息。
$x_{t-1}$ 已经保留了此前加噪与原样本的一部分影响。
系数 $\sqrt{\alpha_t}$ 同时削弱原信号；如果只反复做 $x\leftarrow x+\sigma z$，得到的是另一种过程。
这组 Gaussian 链是 [DDPM 的定义，式 2–4](https://arxiv.org/pdf/2006.11239)；下面的展开用来自己核对它的边缘分布。

## 17.4 为什么训练时可以一跳到任意时间

展开前两步，有

$$
x_2=\sqrt{\alpha_2\alpha_1}x_0
+\sqrt{\alpha_2\beta_1}z_1+\sqrt{\beta_2}z_2.
$$

两个独立 Gaussian 的线性组合仍是 Gaussian，噪声方差为
$\alpha_2\beta_1+\beta_2=1-\alpha_2\alpha_1$。
定义累计信号系数 $\bar\alpha_t=\prod_{s=1}^t\alpha_s$，并令干净端 $\bar\alpha_0=1$。
归纳得到

$$
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

这里 $\epsilon$ 是前 $t$ 次噪声合成后的标准正态变量，不是最后一次的 $z_t$。
闭式与逐步加噪在“固定 $x_0$、固定 $t$ 时的分布”上一样。
用不同的独立 $\epsilon$ 直接画出各个时间状态，不会得到同一条 Markov 轨迹。
要研究相邻状态相关性，应真正逐步运行；要训练某一时间的去噪器，边缘采样已经足够。

```python
# 逐步轨迹：每一步从前一个状态继续。
x = x0.clone()
trajectory = [x.clone()]
for beta_t in torch.linspace(1e-4, 0.02, 100):
    x = (1 - beta_t).sqrt() * x + beta_t.sqrt() * torch.randn_like(x)
    trajectory.append(x.clone())

# 训练输入：一次公式即可得到所选噪声等级的状态。
from fm_tutorial.diffusion.continuous import GaussianDiffusion

diffusion = GaussianDiffusion(steps=100, schedule="cosine")
t = torch.tensor([10, 80], dtype=torch.long)
noise = torch.randn_like(x0)
xt = diffusion.q_sample(x0, t, noise=noise)
```

上面两段故意使用不同 schedule，分别说明“逐步链”和“库接口”；不要逐元素比较它们。
若验证分布等价，应固定同一 schedule、同一 $x_0$，重复采样并比较均值与方差。

## 17.5 数学时间和数组索引：干净端没有占用一格

本仓库连续过程用数组索引 `i=0..T-1`，对应数学时间 `t=i+1`。
所以 `alpha_bar[0]` 已经是第一次加噪后的累计系数，并不等于 1。
公式中的 $x_0$ 则始终是干净数据；不要因变量名 `t=0` 就以为 `q_sample` 返回干净输入。

| 意义 | 数学记号 | 连续代码 |
| --- | --- | --- |
| 干净状态 | $x_0,\bar\alpha_0=1$ | 独立输入 `x0`，DDIM 的 `previous_t=-1` |
| 第一次加噪 | $x_1,\bar\alpha_1$ | `t=0`，`alpha_bar[0]` |
| 最后一次加噪 | $x_T,\bar\alpha_T$ | `t=steps-1` |
| 网络的归一化时间 | $t/T$ | `(t.float()+1)/steps` |

`t` 通常是 `[B]` 整数张量，每个样本可以处于不同等级。
取出一个 `[B]` 系数后，需要把它扩成 `[B,1,...,1]`，才能逐样本广播到全部特征。
图像不能直接拿 `[B]` 乘 `[B,C,H,W]`；广播从末尾维度对齐，偶然相等的尺寸还可能掩盖错误。

```python
def extract(values, t, x):
    # values: [T]；t: [B]；x: [B,...]
    selected = values.to(device=x.device, dtype=x.dtype)[t]
    return selected.reshape(x.shape[0], *([1] * (x.ndim - 1)))

ab = extract(diffusion.alpha_bar, t, x0)
xt_by_formula = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
torch.testing.assert_close(xt, xt_by_formula)
```

离散文本的实现采用另一套约定：`t=1..T`，且 `Q_bar[0]=I`；见 [17a](17a-diffusion-text.md)。
移植代码时先写下时间映射，再移植公式。

## 17.6 场景二：100 次加噪之后，为什么仍能看出原图

看累计系数，而不是只看步数。若每步 $\beta_t$ 都很小，$T$ 很大也可能保留大量信号。
条件均值是 $\sqrt{\bar\alpha_T}x_0$，条件方差是 $1-\bar\alpha_T$。
以单维信号方差约为 1 的已标准化数据为例，信噪比为

$$
\operatorname{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t},
\qquad \log\operatorname{SNR}(t)=\log\bar\alpha_t-\log(1-\bar\alpha_t).
$$

把常见的线性端点 `1e-4..0.02` 原封不动缩到 100 步，乘积约为 $0.364$。
此时信号幅度仍为 $0.603$，SNR 约为 $0.571$；从标准正态初始化逆向采样，就有明显的终端分布不匹配。
这是本例的系数计算，不是任何模型质量指标。

```python
betas = torch.linspace(1e-4, 0.02, 100)
alpha_bar = (1 - betas).cumprod(dim=0)
print(float(alpha_bar[-1]), float(alpha_bar[-1].sqrt()))
```

本教程短步数演示默认使用 cosine schedule。
一种定义从连续曲线 $F(u)=\cos^2\!\left(\frac{u+s}{1+s}\frac\pi2\right)$ 开始，
令 $\bar\alpha_t=F(t/T)/F(0)$，再由
$\beta_t=1-\bar\alpha_t/\bar\alpha_{t-1}$ 得到离散增量。
实际代码把 $\beta_t$ 上限裁到小于 1，避免 $\alpha_t=0$ 造成除零。
因此终端累计系数很小但仍为正；schedule 应检查单调性、端点和数值范围。
cosine 形式可对照 [Improved DDPM，§3.2](https://arxiv.org/html/2102.09672v1)。

SNR 也说明噪声强度与数据尺度不能分开讨论。
没有归一化的资产金额，即便 $\bar\alpha_t$ 很小也可能留下明显信号。
不应仅因为“最后一张看起来像噪声”，就认为先验匹配已经得到证明。

## 17.7 逆向到底需要学习哪个分布

加噪时知道 $x_0$，生成时没有真实 $x_0$，只有当前状态 $x_t$。
目标是构造

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t),
\qquad p(x_T)=\mathcal N(0,I).
$$

真实逆条件 $q(x_{t-1}\mid x_t)$ 隐含了整个数据分布。
对两个簇之间的观测，它可以对应不同的干净来源；一份当前观测无法唯一确定原图与原噪声。
DDPM 用时间条件 Gaussian 近似各个小逆步：
$p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I)$。
小前向步有助于这个近似；“前向是 Gaussian”本身并不保证混合数据的有限步逆条件精确 Gaussian。

训练中容易计算的另一个分布是 $q(x_{t-1}\mid x_t,x_0)$。
它多条件了已知训练样本 $x_0$，拥有闭式解，用来构造监督与变分目标。
必须区分“已知干净样本的解析后验”和“生成时网络预测的逆条件”。
作者实现中可沿 [posterior 与 model mean 的不同函数](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py) 检查这一区别。

## 17.8 自己完成一次 Gaussian 后验配方

令 $y=x_{t-1}$。由 Bayes，有

$$
q(y\mid x_t,x_0)\propto q(x_t\mid y)q(y\mid x_0).
$$

忽略不含 $y$ 的项，负对数为

$$
\frac{\|x_t-\sqrt{\alpha_t}y\|^2}{2\beta_t}
+\frac{\|y-\sqrt{\bar\alpha_{t-1}}x_0\|^2}{2(1-\bar\alpha_{t-1})}.
$$

展开两个平方，$y^\top y$ 的系数给出精度，也就是方差的倒数：

$$
\Lambda_t=\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}}
=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}.
$$

一次项的向量为
$h_t=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0$。
配方 $\Lambda_t\|y-\Lambda_t^{-1}h_t\|^2/2$ 后读出

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
\qquad
\tilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

两项系数是概率推导得到的，未必加起来等于 1，不要把它强行理解成普通凸平均。
例如 $\alpha_1=0.8,\alpha_2=0.9$ 时，$\bar\alpha_2=0.72$，第二步后验方差为 $0.07143$。
$x_0$ 与 $x_2$ 的系数约为 $0.31944$ 与 $0.67763$；若标量 $x_0=2,x_2=1$，后验均值约为 $1.31651$。
可以用自己完成的平方展开核对，而不是记住这三个小数。

在 $t=1$，$1-\bar\alpha_0=0$，前面的精度表达式不能直接除以它。
应取后验公式的端点：$\tilde\beta_1=0$、$\tilde\mu_1=x_0$。
已知 $x_0$ 时，最后一个后验就是集中在该点的退化分布。
这不是说生成网络能无误恢复未知的真实样本。

## 17.9 预测噪声，怎样变成预测逆向均值

设网络输出 $\epsilon_\theta(x_t,t)$，与 $x_t$ 同 shape。
由闭式加噪反解一个干净估计：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

将 $\hat x_0$ 代入上一节的 $\tilde\mu_t$，整理后得到

$$
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left[x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right].
$$

所以噪声预测网络并不另造一套采样概率，它是在参数化逆向均值。
教学实现的中间逆步固定 $\sigma_t^2=\tilde\beta_t$，最后一步返回干净估计。
学习方差是另一项建模选择，需要另写输出头与目标。
`predict_x0` 是代数转换，`posterior` 是解析系数计算，二者本身都不训练网络。

```python
eps_hat = noise  # 只在检查公式时使用已知 oracle 噪声
x0_hat = diffusion.predict_x0(xt, t, eps_hat)
mean, variance = diffusion.posterior(x0_hat, xt, t)
torch.testing.assert_close(x0_hat, x0)
```

上面的 oracle 可以验证逆公式，不能放进真实生成器。
高噪声处 $\sqrt{\bar\alpha_t}$ 很小，预测误差会被 $\hat x_0$ 的除法放大。
随意把 latent 或标准化表格的 $\hat x_0$ 裁到 `[-1,1]`，会改变模型更新。
只有表示确实有这个范围，并且训练与采样约定一致时才讨论这种裁剪。

## 17.10 一次训练更新为什么只抽一个时间

用训练数据 $x_0$、随机时间 $t$ 和新采的 $\epsilon$ 构造输入，最常见教学损失为

$$
L_{\mathrm{simple}}=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2\right].
$$

一个 batch 的不同样本随机处于不同噪声等级。
长期训练覆盖全部时间；每次更新不必先执行 $T$ 次前向加噪，再依次训练 $T$ 个网络。
本项目共享一套参数，通过时间输入区分任务。
训练随机抽时间，与生成必须依次更新当前状态，是两种不同的计算组织。

```python
from fm_tutorial.diffusion.models import TimeMLP

torch.manual_seed(7)
net = TimeMLP(input_dim=2, hidden_dim=64)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
diffusion = GaussianDiffusion(steps=100, schedule="cosine")

for update in range(200):
    # 两簇位置仅用于机制演示；不是公开研究 benchmark。
    side = torch.randint(0, 2, (64, 1)).float() * 2 - 1
    x0 = torch.cat([2 * side, torch.zeros_like(side)], dim=1)
    x0 = x0 + 0.2 * torch.randn_like(x0)
    t = torch.randint(0, diffusion.steps, (x0.shape[0],))
    epsilon = torch.randn_like(x0)
    xt = diffusion.q_sample(x0, t, noise=epsilon)
    eps_hat = net(xt, (t.float() + 1) / diffusion.steps)
    loss = (eps_hat - epsilon).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

`TimeMLP` 接收归一化浮点时间；`GaussianDiffusion` 的采样回调接收原始整数索引。
后面会显式写一个适配函数，保证训练与采样使用相同时间表示。
把训练的 `(t+1)/T` 改成采样的 `t/T`，虽然 shape 一样，也会产生时间错位。

## 17.11 有噪声标签，为什么不等于记住那份真实噪声

平方损失的总体最优预测是条件均值：

$$
\epsilon_*(x_t,t)=\mathbb E[\epsilon\mid x_t,t].
$$

给定 $x_t$，可能有多个 $x_0$ 与 $\epsilon$ 组合产生它；模型只看到当前观测和时间。
每个训练样本提供一份随机标签，很多这样的训练对共同估计这个条件均值。
不存在仅凭观测就能逐次恢复所有真实随机数的一般保证。
同理，$x_0$ 的 MSE 回归得到 $\mathbb E[x_0\mid x_t,t]$，不等于抽一份完整的后验样本。

这解释了二维双簇的一个现象：高噪声位置的干净估计可以落在簇之间。
生成依靠多次条件更新和采样随机性，逐步形成最终分布。
不能拿某一个中间 $\hat x_0$ 的模糊程度，直接判定最终样本质量。
同样，低训练 MSE 只证明某种平均预测误差较低；仍需独立检查采样后的分布与真实数据指标。

## 17.12 MSE 与 ELBO：联系存在，但权重不能丢了再说相等

对负对数似然应用变分界，可写为

$$
\mathbb E[-\log p_\theta(x_0)]\le
\mathbb E_q\left[
D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))
+\sum_{t=2}^TD_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t))
-\log p_\theta(x_0\mid x_1)\right].
$$

固定方差的两个 Gaussian，其依赖均值参数的 KL 项为
$\|\tilde\mu_t-\mu_\theta\|^2/(2\sigma_t^2)$。
代入噪声参数化，得到时间权重

$$
\lambda_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)},
\qquad L_{t-1}=\lambda_t\mathbb E\|\epsilon-\epsilon_\theta\|^2+C_t.
$$

`L_simple` 去掉了这组非均匀权重，因而一般不是原 ELBO 的数值。
固定 schedule 的先验 KL 不训练均值网络；最后的重建项还依赖观测似然与离散化约定。
本仓库采样器只返回末步均值，没有实现这项观测 likelihood；不能把零后验方差直接代入 Gaussian log density。
不能把“噪声 MSE”直接报告为 bits/dim，或者仅由它推出模型 likelihood。
具体目标区别可对照 [DDPM 式 5、12、14](https://arxiv.org/pdf/2006.11239)。

若均匀抽 $t$ 估计一项有限和，需要考虑 $T$ 这一常数；优化均值型损失时常把它吸收进尺度。
若按非均匀分布 $r(t)$ 抽时间，又想估计原目标，则应按 $1/r(t)$ 修正采样权重。
否则不仅减少方差，也改变了各噪声等级的训练重要性。
`mean()` 还平均全部特征维度；比较不同分辨率或长度时，应明确是每样本总和还是每元素平均。

## 17.13 完整 DDPM 采样：最后一步不再添加噪声

从 $x_T\sim\mathcal N(0,I)$ 开始，每一步先预测，再更新当前状态：

$$
x_{t-1}=\mu_\theta(x_t,t)+\sqrt{\tilde\beta_t}z,
\qquad z\sim\mathcal N(0,I)\ (t>1),\quad z=0\ (t=1).
$$

下面是与本仓库数组时间一致的完整循环：

```python
@torch.no_grad()
def epsilon_model(x, t):
    return net(x, (t.float() + 1) / diffusion.steps)

@torch.no_grad()
def ddpm_loop(shape):
    x = torch.randn(shape)
    for i in range(diffusion.steps - 1, -1, -1):
        t = torch.full((shape[0],), i, dtype=torch.long)
        eps_hat = epsilon_model(x, t)
        x0_hat = diffusion.predict_x0(x, t, eps_hat)
        mean, variance = diffusion.posterior(x0_hat, x, t)
        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = mean + variance.sqrt() * z
    return x

# 库调用组织相同的步骤；p_sample 负责一次更新。
net.eval()
samples = diffusion.sample(epsilon_model, (32, 2), device="cpu", sampler="ddpm")
one_t = torch.full((32,), 99, dtype=torch.long)
one_step = diffusion.p_sample(epsilon_model, torch.randn(32, 2), one_t)
```

`no_grad` 控制梯度记录，`eval` 控制网络的训练态行为，两者作用不同。
最后一个 mean 是模型估计的干净输出；不添加新噪声，并不保证没有模型误差。
若混合 batch 中存在不同时间，应逐样本把 `t=0` 的噪声项置零，不能只检查 batch 的第一个元素。
使用 GPU 时，数据、时间、网络和 schedule 系数需放到对应设备；上面的显式循环只展示 CPU 情况。

## 17.14 场景三：只允许 20 次网络调用，能直接跳过 80 步吗

DDPM 的相邻步系数针对 $t\to t-1$ 推导。
把 `for` 改成每隔 5 步运行一次，而仍套原相邻系数，不能得到正确的 $t\to t-5$ 更新。
需要使用所选起点与终点的累计系数。
DDIM 给出可复用噪声预测器的另一族更新；训练边缘分布保持一致，采样轨迹可以不同。
对应推导见 [DDIM §3–4 与附录 C.1](https://arxiv.org/html/2010.02502v4)。

对任意选定的数学时间 $s<t$，记 $a_t=\bar\alpha_t,a_s=\bar\alpha_s$。
先计算 $\hat x_0$，再设

$$
\sigma_{t\to s}=\eta\sqrt{\frac{1-a_s}{1-a_t}}
\sqrt{1-\frac{a_t}{a_s}},
\qquad
x_s=\sqrt{a_s}\hat x_0+
\sqrt{1-a_s-\sigma_{t\to s}^2}\epsilon_\theta(x_t,t)+\sigma_{t\to s}z.
$$

$\eta=0$ 时，固定初始噪声和模型后，后续路径确定；初始随机性仍使输出形成分布。
在完整相邻时间表上，$\eta=1$ 对应采用后验方差的 DDPM 更新。
跳步时它对应重设时间表的过程，不能声称与原来逐步链逐样本相同。
本教程把 $\eta$ 限于 `[0,1]`，确保这组系数的非负性；扩大范围要重新检查根号。

```python
samples_ddim = diffusion.sample(
    epsilon_model, (32, 2), device="cpu",
    sampler="ddim", sampling_steps=20, eta=0.0,
)
# 可读的一步：数组索引 99 跳到 94，对应数学时间 100 -> 95。
t = torch.full((32,), 99, dtype=torch.long)
previous_t = torch.full_like(t, 94)
x94 = diffusion.ddim_step(epsilon_model, torch.randn(32, 2), t, previous_t, eta=0.0)
```

干净终点取数学 $s=0$，于是 $a_s=1$：$\sigma=0$，方向项也是 0，最后直接得到 $\hat x_0$。
代码用 `previous_t=-1` 明确代表这个端点，不能读取 `alpha_bar[-1]`，那会误读最后一格。
减少网络调用改变了离散误差；采样步数、schedule、模型和评测协议应一起记录。
[作者代码的 `compute_alpha` 与 generalized update](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py) 适合对照端点处理。

## 17.15 $\epsilon$、$x_0$、$v$：预测对象可以换，系数与损失要一起换

写 $a=\sqrt{\bar\alpha_t}$、$b=\sqrt{1-\bar\alpha_t}$，因此 $a^2+b^2=1$。
加噪公式是 $x_t=ax_0+b\epsilon$。
另一种参数化定义 $v=a\epsilon-bx_0$，相当于对 $(x_0,\epsilon)$ 做二维正交变换：

$$
\begin{bmatrix}x_t\\v\end{bmatrix}
=\begin{bmatrix}a&b\\-b&a\end{bmatrix}
\begin{bmatrix}x_0\\\epsilon\end{bmatrix},
\quad
x_0=ax_t-bv,\qquad \epsilon=bx_t+av.
$$

给定 $x_t,t$，不同网络输出之间的转换为：

| 网络输出 | 干净估计 | 噪声估计 |
| --- | --- | --- |
| $\hat\epsilon$ | $(x_t-b\hat\epsilon)/a$ | $\hat\epsilon$ |
| $\hat x_0$ | $\hat x_0$ | $(x_t-a\hat x_0)/b$ |
| $\hat v$ | $ax_t-b\hat v$ | $bx_t+a\hat v$ |

高噪声处 $a$ 小，$\epsilon\to x_0$ 转换敏感；低噪声处 $b$ 小，$x_0\to\epsilon$ 转换敏感。
$v$ 的形式避免这些转换中的显式小分母，但它不凭空消除学习误差和数值误差。
相同的未加权 MSE 在三个输出空间中意味着不同的时间权重，因此不能只替换 target 而宣称训练目标完全相同。
$v$ 参数化可对照 [Progressive Distillation](https://arxiv.org/abs/2202.00512)；本仓库的 Gaussian 网络接口采用 $\epsilon$ 预测，$v$ 表用于读其他实现。

## 17.16 Score 是当前分布的梯度，不是“图像评分”

连续密度的 score 定义为

$$
s_*(x,t)=\nabla_x\log p_t(x),
\qquad p_t(x)=\int q_t(x\mid x_0)p_{\mathrm{data}}(x_0)\,dx_0.
$$

梯度对当前样本坐标 $x$ 求导，不是对参数 $\theta$ 或时间求导。
它与 $x$ 同维度，指向局部 log density 上升的方向；不是一个标量“质量分数”。
对一维 $\mathcal N(\mu,\sigma^2)$，score 为 $-(x-\mu)/\sigma^2$。
它在中心为 0；离中心越远，向中心拉回的幅度越大。
对双簇混合分布，两个簇会共同影响这个向量场，不能用“最近训练样本的方向”替代它。

条件 Gaussian 核的 score 容易算：

$$
\nabla_x\log q_t(x\mid x_0)
=-\frac{x-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

但生成需要的是边缘 $p_t$ 的 score，并不知道 $x_0$。
对边缘积分求导，在适当可微与可积条件下可得

$$
\nabla_x\log p_t(x)
=\mathbb E[\nabla_x\log q_t(x\mid x_0)\mid x,t]
=-\frac{\mathbb E[\epsilon\mid x,t]}{\sqrt{1-\bar\alpha_t}}.
$$

这就连接了噪声 MSE 与 denoising score matching：
$s_\theta(x,t)=-\epsilon_\theta(x,t)/\sqrt{1-\bar\alpha_t}$。
训练标签是一份条件核 score；总体最优预测是它的条件期望，才成为边缘 score。
作者对这些概念的说明见 [Yang Song 的原作者文章](https://yang-song.net/blog/2021/score/)。
不要把训练时可计算的条件 score 直接当成生成时已知的精确边缘 score。

若直接训练 score 网络，可以使用
$\mathbb E[\lambda(t)\|s_\theta(x_t,t)+\epsilon/\sigma(t)\|^2]$。
取 $\lambda(t)=\sigma(t)^2$ 并设 $s_\theta=-\epsilon_\theta/\sigma(t)$，就回到未加权 noise MSE。
其他 $\lambda(t)$ 会改变各噪声等级的贡献；这也是为什么“网络输出可互相转换”不表示任意 loss 配方都相同。

## 17.17 从有限步 DDPM 到连续时间 VP SDE

令连续时间 $t\in[0,1]$，小步方差约为 $\beta(t)\,dt$。
利用 $\sqrt{1-\beta(t)dt}\approx1-\beta(t)dt/2$，前向增量变成

$$
dx=-\tfrac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dW_t.
$$

$W_t$ 是 Wiener 过程，其长度 $dt>0$ 的增量方差为 $dt$；数值步应乘 $\sqrt{dt}$。
这是 variance preserving（VP）SDE。
令 $B(t)=\int_0^t\beta(u)du$，解的条件均值与标准差为

$$
m(t)x_0=e^{-B(t)/2}x_0,\qquad
\sigma(t)=\sqrt{1-e^{-B(t)}},
\qquad x_t=m(t)x_0+\sigma(t)\epsilon.
$$

若 $\beta(t)=\beta_{\min}+t(\beta_{\max}-\beta_{\min})$，则
$B(t)=\beta_{\min}t+(\beta_{\max}-\beta_{\min})t^2/2$。
当数据 covariance 初始为 $I$ 时，它在前向过程保持为 $I$；一般初始 covariance 则趋近 $I$。
“VP”不表示每个固定样本的条件方差恒定，后者从 0 增大到接近 1。

```python
from fm_tutorial.diffusion.continuous import VPSDE

sde = VPSDE()
x0 = torch.randn(8, 2)
t = torch.rand(8) * 0.999 + 0.001  # 连续时间，避开零噪声端点
epsilon = torch.randn_like(x0)
mean, std = sde.marginal_stats(x0, t)
xt = sde.marginal(x0, t, noise=epsilon)
score_target = sde.score_from_epsilon(epsilon, t)
```

`std` 是按 batch 广播后的标准差；`score_target` 仍是一份训练条件标签。
同样训练 `TimeMLP(xt,t)` 预测 $\epsilon$，采样时再把输出除以负标准差即可。
SDE 的连续 `t` 已在 `[0,1]`，不使用整数时间的 `(t+1)/T` 适配。

```python
# VP 用自己的连续时间模型，不复用上面 Gaussian schedule 的 checkpoint。
net_sde = TimeMLP(input_dim=2, hidden_dim=64)
optimizer_sde = torch.optim.Adam(net_sde.parameters(), lr=1e-3)
for update in range(200):
    side = torch.randint(0, 2, (64, 1)).float() * 2 - 1
    x0_batch = torch.cat([2 * side, torch.zeros_like(side)], dim=1)
    x0_batch = x0_batch + 0.2 * torch.randn_like(x0_batch)
    t_batch = torch.rand(64) * 0.999 + 0.001
    eps = torch.randn_like(x0_batch)
    noisy_batch = sde.marginal(x0_batch, t_batch, noise=eps)
    loss = (net_sde(noisy_batch, t_batch) - eps).square().mean()
    optimizer_sde.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_sde.step()
net_sde.eval()
```

## 17.18 VP 与 VE：两种噪声几何

一般前向 SDE 写为 $dx=f(x,t)dt+g(t)dW_t$。
本节只讨论扩散系数与位置无关、各向同性的情况；位置相关的矩阵扩散需要额外项。
VE（variance exploding）的一种写法为

$$
dx=\sqrt{\frac{d\sigma^2(t)}{dt}}\,dW_t,
\qquad x_t\mid x_0\sim\mathcal N(x_0,[\sigma^2(t)-\sigma^2(0)]I).
$$

VE 不衰减条件均值，噪声 variance 则持续增加。
常见扰动训练也直接写 $x_t=x_0+\sigma(t)\epsilon$，这对应从已经有 $\sigma(0)$ 微量噪声的端点开始；两种边界约定要区分。
大终端噪声相对于数据尺度占优时，才可以用宽 Gaussian 近似终端分布。

| 项目 | VP | VE |
| --- | --- | --- |
| 前向 drift | $-\beta(t)x/2$ | 0 |
| 信号均值 | 逐渐衰减 | 保留 $x_0$ |
| 噪声尺度 | 趋近单位尺度 | 随 schedule 增长 |
| 终端先验近似 | $\mathcal N(0,I)$ | 大方差 Gaussian |

这不是“某个更适合所有模态”的排名。
不同尺度会改变网络输入、score 幅度、求解误差与训练权重。
原始统一框架的 VP/VE 推导见 [Score SDE §3 与附录 B](https://arxiv.org/html/2011.13456v2)。
本仓库可执行连续 SDE 教学组件是 VP，VE 表格用于概念比较。

## 17.19 Reverse SDE：负时间增量是公式的一部分

已知精确边缘 score 时，逆时间 SDE 为

$$
dx=[f(x,t)-g(t)^2s_*(x,t)]dt+g(t)d\bar W_t,
\qquad t:1\to0.
$$

这里 $dt<0$，逆 Wiener 增量的方差是 $|dt|$。
若代码用正步长 $h>0$ 表示从 $t$ 走到 $t-h$，Euler–Maruyama 更新应是

$$
x_{t-h}=x_t+[-f(x_t,t)+g(t)^2s_\theta(x_t,t)]h
+g(t)\sqrt h\,z.
$$

先写原公式、再令 $dt=-h$，可以避免把 score 方向反掉。
对 VP，$f=-\beta x/2$、$g^2=\beta$，正步长 drift 为
$\beta x/2+\beta s_\theta$。
直觉上 score 往高密度方向推，但另一个 drift 也在工作；不能只保留 score 就叫 reverse VP。

```python
# 概念步：dt 是负数。库 reverse_step 接口采用这个约定。
@torch.no_grad()
def score_model(x, t):
    return sde.score_from_epsilon(net_sde(x, t), t)

dt = -0.01
t_now = torch.full((8,), 0.8)
xt = sde.marginal(x0, t_now)
x_previous = sde.reverse_step(score_model, xt, t_now, dt)
```

在训练完成后才能把 `score_model` 当作数据 score 的近似。
数学逆转结论需要正确终端分布、精确 score 与适当正则性；实际采样另外受到先验近似、网络误差与离散误差影响。
可以沿 [作者 `SDE.reverse` 实现](https://github.com/yang-song/score_sde/blob/main/sde_lib.py) 对照 drift 符号。

## 17.20 Probability-flow ODE：去掉随机项，score 系数还要减半

与前向 SDE 分享边缘分布的 probability-flow ODE 是

$$
\frac{dx}{dt}=f(x,t)-\tfrac12g(t)^2s_*(x,t).
$$

生成同样从终端向初始端积分。
在正步长 $h$ 的逆向更新中，使用 $-f+g^2s_\theta/2$，且不添加 $z$。
仅从 reverse SDE 删除随机项而保留完整 $g^2s$，不是 probability-flow ODE。
这个 $1/2$ 可以通过 Fokker–Planck 方程核对：

$$
\partial_t p=-\nabla\!\cdot(fp)+\tfrac12g^2\Delta p
=-\nabla\!\cdot\left[\left(f-\tfrac12g^2\nabla\log p\right)p\right].
$$

因为 $p\nabla\log p=\nabla p$，ODE 的连续性方程与 SDE 的密度演化相同。
“边缘分布相同”依赖精确 score 和正确积分，不表示两种过程具有相同路径或同一初态的相同输出。
有限步网络近似下，更不能把二者质量视为自动相等。
来源为 [Score SDE 式 13](https://arxiv.org/html/2011.13456v2)，这里用密度方程说明系数的作用。

```python
# score_model 必须来自对应 VP 连续时间训练，不能复用错误的整数时间标签。
samples_sde = sde.sample(score_model, (16, 2), device="cpu", steps=100)
samples_ode = sde.sample(
    score_model, (16, 2), device="cpu", steps=100,
    probability_flow=True, t_min=0.001,
)
```

本仓库用固定步长教学积分，不是自适应高阶求解器，也不实现精确 likelihood 计算。
DDIM 的确定性更新与某些 ODE 离散化有联系；不能因此把任意 DDIM 时间表、参数化和 ODE 数值方法当作同一算法。

## 17.21 Langevin、predictor–corrector 与零噪声端点

在固定噪声时间 $t$，Langevin 更新可以写作
$x\leftarrow x+\delta s_\theta(x,t)+\sqrt{2\delta}z$。
它试图在该等级的 $p_t$ 上探索；有限步、近似 score 和离散步长不保证精确平稳分布。
沿多个噪声等级运行，是 annealed Langevin 的基本思路。
predictor–corrector 则交替做“走向下一时间的 predictor”与“在当前等级修正的 corrector”。
这些不是把训练多做一次，而是增加采样中的网络调用。
概念及原始算法入口见 [作者 score-based 说明](https://yang-song.net/blog/2021/score/) 和 [score_sde 仓库](https://github.com/yang-song/score_sde)。
本教程不将它们包装成未经执行的质量提升选项。

接近 $t=0$ 时，$\sigma(t)$ 很小，$-\epsilon_\theta/\sigma(t)$ 容易放大误差。
数据集中在低维结构附近时，score 场还可能变化很快，使粗步积分表现出数值困难。
通常显式停在 $t_{\min}>0$，必要时再做符合模型参数化的 denoising 端点处理。
`VPSDE.sample` 返回 `t_min` 的积分状态，不把它宣称为严格 $t=0$ 的解。
减小步长可以改善离散误差，但不能修复错误 score；增加 clamp 可以避免 NaN，却可能改变动力学。
检查有限值、范数与不同步长稳定性，比只看循环执行完更有信息。

## 17.22 场景四：同一份初始噪声，怎样指定“生成左侧簇”

条件生成学习 $\epsilon_\theta(x_t,t,c)$，其中 $c$ 可以是类别、文本表示或其他观测。
条件输入不等于 $x_0$：类别“左簇”并没有给出每个点的完整真实位置。
训练时随机将一部分条件替换为空条件 $\varnothing$，使同一个网络也学习无条件预测。
这为 classifier-free guidance（CFG）准备两份输出。

本章采用常见 scale 约定：

$$
\epsilon_{\mathrm{cfg}}=
\epsilon_\theta(x_t,t,\varnothing)+
w\,[\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)].
$$

$w=0$ 是无条件，$w=1$ 是普通条件，$w>1$ 是向条件方向外推。
原论文也写 $(1+\gamma)\epsilon_c-\gamma\epsilon_u$，两者对应 $w=1+\gamma$。
所以报告 `guidance=3` 之前，要说明参数 convention。
条件丢弃与组合公式见 [Classifier-Free Diffusion Guidance](https://arxiv.org/pdf/2207.12598)。

```python
# 示意：conditional_net 需另行实现并用条件丢弃训练，本仓库 TimeMLP 无条件头。
def guided_epsilon(conditional_net, x, t, condition, empty_condition, scale):
    eps_u = conditional_net(x, t, empty_condition)
    eps_c = conditional_net(x, t, condition)
    return eps_u + scale * (eps_c - eps_u)
```

CFG 改变采样向量场，不是新增监督标签，也不是仅改变随机数温度。
较大 guidance 可能改变多样性与输出尺度；应连同条件遵循度、覆盖度和成本共同评估。
理想 score 下的组合能对应当前时间的重加权 score，但不应直接断言最终样本精确来自某个简单的加幂数据分布。
每步两次条件评估的逻辑可以合并 batch，但网络计算量并未因此自动减半。

## 17.23 场景五：补图时，已知的左半边怎样保持

设 mask $M$ 的 1 表示已知坐标，观测为 $y=M\odot x_0$。
在噪声时间 $t$，已知区域同样应处在相应噪声等级：

$$
x_t^{\mathrm{known}}=\sqrt{\bar\alpha_t}y+
\sqrt{1-\bar\alpha_t}\epsilon_{\mathrm{known}},
\qquad
x_t\leftarrow M\odot x_t^{\mathrm{known}}+(1-M)\odot x_t.
$$

每个高噪声步骤直接注入干净 $y$，会让一半输入的 SNR 与时间标签不相符。
若定义一条 Markov 已知轨迹，噪声应遵循它的相邻条件关系；使用固定噪声的确定性已知路径又是另一种选择。
简单逐步投影只满足当前坐标约束，不自动证明从精确条件分布 $p(x_0\mid y)$ 采样。
尤其未知区域与已知区域的边缘连续性，需要模型、条件建模或更严格的逆问题算法支持。

最清楚的教学方法是训练条件去噪器：始终提供观测 $y$ 和 mask $M$，只把待生成区域的 target 计入相应损失，并在采样中遵循同一输入约定。
最后干净端再次应用 $x_0\leftarrow M\odot y+(1-M)\odot x_0$，精确保存观测。
传感器观测若有测量噪声，则约束应使用观测 likelihood，而不是假定每个已知数值绝对正确。
原始 score 框架的条件逆问题入口可见 [Score SDE §5](https://arxiv.org/html/2011.13456v2)。

轨迹补全也遵循同一原则：已知位置、未知位置和时间轴需明确。
把“视频帧时间”与“扩散噪声时间”混用，会在 shape 完全正确时仍产生错误条件。
连续 Gaussian 投影不应照搬到离散 mask 文本；后者可以令已知 token 根本不被前向腐蚀，详见 [17a](17a-diffusion-text.md)。

## 17.24 Flow matching 与 rectified flow：相关的路径，不同的监督目标

现在改用新时间 $\tau\in[0,1]$，方向从噪声到数据；为避免与本章 $x_0$ 混淆，端点记作 $z\sim p_{\mathrm{noise}}$ 与 $y\sim p_{\mathrm{data}}$。
一个简单条件插值是

$$
x_\tau=(1-\tau)z+\tau y,\qquad
\frac{dx_\tau}{d\tau}=y-z.
$$

以可采样的配对 $(z,y)$ 训练速度网络：

$$
L_{\mathrm{FM}}=\mathbb E_{\tau,z,y}
\|u_\theta(x_\tau,\tau)-(y-z)\|^2,
\qquad \frac{dx}{d\tau}=u_\theta(x,\tau).
$$

总体最优网络同样预测条件期望 $\mathbb E[y-z\mid x_\tau,\tau]$。
单个训练配对的路径是直线，不表示边缘速度场中的每条生成轨迹都直。
初始配对方式也会影响学习到的运输；独立配对不等于求得全局最优运输。
flow matching 的一般框架可使用不同 probability path，包括与 diffusion 相关的路径，见 [Flow Matching §3–4](https://arxiv.org/html/2210.02747v2)。

rectified flow 使用这种直线插值的速度回归，并可用模型生成的配对再训练以改善路径，见 [Flow Straight and Fast](https://arxiv.org/html/2209.03003v1)。
它不是在 DDPM 里把噪声项设为 0，也不是把网络输出变量改名为 `velocity`。
diffusion 的 $v=a\epsilon-bx_0$ 参数化与这里的 $y-z$ 速度 target 不同。
时间方向、路径系数、训练权重与采样 ODE 必须一起核对。

```python
# 独立教学目标示意；不是 GaussianDiffusion 的 noise loss。
z = torch.randn_like(x0)
tau = torch.rand(x0.shape[0])
r = tau.reshape(-1, *([1] * (x0.ndim - 1)))
x_tau = (1 - r) * z + r * x0
velocity_target = x0 - z
velocity_hat = net(x_tau, tau)
flow_loss = (velocity_hat - velocity_target).square().mean()
```

可以复用网络模块，但不能将一份 noise checkpoint 直接当作上述 velocity checkpoint。
本仓库演示主线运行 DDPM/DDIM/VP，flow 代码片段用于明确目标边界，未提供 flow 训练性能结果。

## 17.25 去噪器架构：为什么扩散模型不等于 U-Net

概率过程只要求去噪器能读 $(x_t,t,c)$ 并输出匹配参数化的张量。
二维位置用 MLP 足以展示这些操作；图像可用 U-Net 的多尺度路径或 patch Transformer；文本连续表示可用双向序列网络。
架构选择决定哪些坐标之间能交换信息，schedule 决定噪声如何变化。
两个问题相互影响但不可互相替代。

`TimeMLP` 把浮点时间嵌入后与特征一起计算，输出 `[B,D]` 噪声预测。
对图像、视频或点云使用向量去噪器时，可以先展平再恢复形状，以检查 shared process 接口。
这种展平 MLP 不具有图像多尺度先验，也未利用视频局部时间关系；shape 检查不证明生产架构适用。
具体表示、位置结构、条件注入与源码路线见 [17b](17b-diffusion-modalities.md)。

与自回归 Transformer 比较时，先比较因果结构：AR 按 token 因果预测，连续 diffusion 通常在同一噪声等级同时读整个可见状态。
同时预测全部坐标，并不意味着整个生成过程只调用一次网络。
扩散采样的串行轴是噪声等级；AR 的串行轴通常是新 token 位置。
真实延迟取决于长度、网络大小、采样预算、缓存与硬件，不能从这句话推出谁普遍更快。

## 17.26 从源码与演示开始，怎样做一次有证据的检查

先阅读 `GaussianDiffusion` 的 schedule 与 `q_sample`，再按公式找 `predict_x0`、`posterior`、`p_sample`、`ddim_step` 和 `sample`。
之后读 `VPSDE` 的 `marginal_stats`、`score_from_epsilon`、`reverse_step`，确认负 `dt` 与 ODE 的半系数。
最后看网络如何接收时间，避免把概率接口和网络输入习惯混为一谈。

从仓库根目录执行：

```bash
python scripts/diffusion_demo.py --help
python -m unittest discover -s tests -p 'test_diffusion_continuous.py' -v
```

按 [作业六](../assignments/06-diffusion.md) 跑有界 CPU 演示，记录 seed、更新次数、schedule、训练与采样步数、参数化及输出统计量。
合成点上的 loss 下降与采样有限性可以检查机制；真实图片质量、文本语义与科学轨迹准确性需要另外的公开数据、固定 split 和合适指标。
阅读 [原始来源表](../references/diffusion.md) 时，先对照作者的公式和更新函数，再读大型框架的优化实现。
本章的短实现没有声称复现原论文训练规模或原论文数值。

出现问题时，可按下面的对应关系定位：

| 观察到的现象 | 先核对什么 |
| --- | --- |
| 最后状态仍明显带信号 | $\bar\alpha_T$、数据尺度与终端先验 |
| 训练稳定但采样异常 | 时间适配、预测参数化与训练 schedule |
| $t=0$ 处 NaN | 数组时间语义、零标准差除法与后验端点 |
| DDIM 跳步之后越走越噪 | 所选 `previous_t` 的累计系数、干净端 `-1` |
| SDE 样本远离数据 | 逆时间符号、score 的负号、噪声的 $\sqrt{|dt|}$ |
| ODE 与 SDE drift 相同 | probability-flow 的 $1/2$ 系数 |
| 已知区域每步突兀跳变 | 当前时间的噪声约束、mask 与观测尺度 |

## 17.27 检查理解

1. 在图像 `[B,3,32,32]` 中，$I$ 的数学维度是什么？为什么 `randn_like` 不必创建它？
2. 从两次独立加噪推导 $1-\bar\alpha_2$。闭式得到各时间边缘，为什么不自动构成同一条轨迹？
3. 数组索引 0 对应哪个数学时间？DDIM 的 `previous_t=-1` 为什么不能用 Python 的负索引读取？
4. 用平方展开推导 $\tilde\mu_t,\tilde\beta_t$，并解释 $t=1$ 的退化端点。
5. 两个不同 $x_0$ 能产生相同 $x_t$ 时，noise MSE 的最优预测为什么是条件均值？
6. 写出 noise MSE 在 ELBO 中的时间权重；若改成非均匀时间采样，怎样保持原目标？
7. 从 $\epsilon_\theta$ 写出一次 DDPM 更新，并指出最后一步哪些随机量必须为零。
8. 从数学时间 80 跳到 20 的 DDIM 用哪两个 $\bar\alpha$？$s=0$ 时为什么输出恰为 $\hat x_0$？
9. 给定 $x_t$ 与 $\hat v$，写出 $\hat x_0,\hat\epsilon$；这里的 $v$ 和 flow matching 的速度有什么不同？
10. 区分 $\nabla_x\log q_t(x\mid x_0)$ 与 $\nabla_x\log p_t(x)$，推导二者的条件期望关系。
11. 分别用负 `dt` 与正 $h$ 写 reverse VP 更新，解释 ODE 中为何出现 $1/2$。
12. CFG 中 $w=1$ 表示什么？若某库定义为 $(1+\gamma)\epsilon_c-\gamma\epsilon_u$，参数怎样换算？
13. 为什么在高噪声补图输入中反复注入干净观测会错配时间？最终保持已知像素与精确条件采样是否是同一保证？
14. 设计一个真实数据实验，把“算法运行正确”“预测损失下降”和“生成质量提升”作为三个可分别验证的命题。
