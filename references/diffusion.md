# Diffusion：论文与作者源码导读

核对日期：2026-10-04。配套：[17 基础算法](../tutorials/17-diffusion.md)、[17A 文本](../tutorials/17a-diffusion-text.md)、[17B 各模态](../tutorials/17b-diffusion-modalities.md)。本页选取能解释算法发展的代表性工作，运行大型项目时另行锁定 commit、权重 revision、数据与环境；这里不宣称穷尽所有最新模型。

## 先读算法，再读大系统

| 要解决的理解问题 | 原始论文 | 作者源码入口 | 对照本仓库 |
|---|---|---|---|
| Gaussian 前向、闭式采样、后验与 noise prediction | [DDPM](https://arxiv.org/abs/2006.11239) | [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) | [continuous.py](../src/fm_tutorial/diffusion/continuous.py)：`GaussianDiffusion` |
| 怎样用同一个模型改变采样步数 | [DDIM](https://arxiv.org/abs/2010.02502) | [functions/denoising.py](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py) | `ddim_step`、`sample(..., sampler="ddim")` |
| 噪声日程怎样控制信息衰减 | [Improved DDPM](https://arxiv.org/abs/2102.09672) | [openai/improved-diffusion](https://github.com/openai/improved-diffusion) | cosine schedule；没有实现论文中的 learned variance |
| score、Langevin、reverse SDE 与 ODE 怎样连接 | [Score-SDE](https://arxiv.org/abs/2011.13456)；[NCSN](https://arxiv.org/abs/1907.05600) | [sde_lib.py](https://github.com/yang-song/score_sde/blob/main/sde_lib.py)、[sampling.py](https://github.com/yang-song/score_sde/blob/main/sampling.py) | `VPSDE`：VP marginal、reverse Euler-Maruyama 与 Euler ODE；VE 与 predictor-corrector 在教程讲解 |
| 条件分支与无条件分支怎样组合 | [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) 是相关的 **classifier-guidance** 代码，需区分两者 | 17C 的 CFG 函数、条件 dropout 与二维实际训练演示；不把两种 guidance 混成同一算法 |
| velocity field 怎样训练 | [Flow Matching](https://arxiv.org/abs/2210.02747)；[Rectified Flow](https://arxiv.org/abs/2209.03003) | [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)、[gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) | 与 DDPM/score 比较；本仓库没有实现完整 flow matching runner |

建议沿官方代码中的 `q_sample / posterior / p_sample` 或同义函数定位，不从大型配置系统开始。注意 TensorFlow、JAX 与 PyTorch 版本差异；本仓库是独立写的小型 PyTorch 实现，没有复制这些实现。

## 文本：连续与离散是两条路线

| 工作 | 主要学习点 | 原始入口 |
|---|---|---|
| Diffusion-LM | Gaussian word vectors、解码回 token、连续中间变量上的控制 | [论文](https://arxiv.org/abs/2205.14217)、[XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) |
| DiffuSeq | 条件 seq2seq，源序列与目标序列的不同处理 | [论文](https://arxiv.org/abs/2210.08933)、[Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) |
| D3PM | 转移矩阵、categorical 后验、absorbing 状态与辅助 CE | [论文](https://arxiv.org/abs/2107.03006)、[Google Research d3pm](https://github.com/google-research/google-research/tree/master/d3pm) |
| MDLM | absorbing mask、连续时间目标、Rao-Blackwellization 与采样 | [论文](https://arxiv.org/abs/2406.07524)、[kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) |
| LLaDA | 大规模 masked language modeling、条件生成、推理时的解掩码策略 | [论文](https://arxiv.org/abs/2502.09992)、[ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) |

[discrete.py](../src/fm_tutorial/diffusion/discrete.py) 中的 dense uniform categorical kernel 用于小词表数学检验。`reverse_probs` 实现的是以预测的 **clean-token 条件分布** 混合各自已归一化的后验；这与原 D3PM Eq.4 的 joint-mixture 参数化有区别。`masked_loss` 是 unweighted masked-token CE 教学损失，不等于 MDLM 的完整加权 likelihood bound。`sample_masked` 用线性 masking 日程的随机解掩码，未复刻 LLaDA 的全部推理策略。

## 图片、表格、视频与动态 3D

| 模态/工作 | 为什么要读 | 原始入口 |
|---|---|---|
| 图像 LDM | 把图像压缩与 latent 去噪分开，学习 cross-attention 条件 | [论文](https://arxiv.org/abs/2112.10752)、[CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) |
| 图像 DiT | patch、time embedding、AdaLN 条件与 Transformer 去噪 | [论文](https://arxiv.org/abs/2212.09748)、[facebookresearch/DiT](https://github.com/facebookresearch/DiT) |
| 表格 TabDDPM | Gaussian 数值列与 multinomial 类别列在同一个任务中建模 | [论文](https://arxiv.org/abs/2209.15421)、[yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm) |
| 表格 TabDiff | 混合类型的连续时间过程、列的噪声日程与 Transformer | [论文](https://arxiv.org/abs/2410.20626)、[MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff) |
| Video Diffusion Models | 时空网络怎样联合生成多帧 | [论文](https://arxiv.org/abs/2204.03458)、[作者项目](https://video-diffusion.github.io/) |
| Wan2.1 | 3D VAE、视频 Transformer、flow matching 工业实现 | [技术报告](https://arxiv.org/abs/2503.20314)、[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| SV4D | 从输入视频生成多视角、多时刻画面，再优化动态 3D 表示 | [论文](https://arxiv.org/abs/2407.17470)、[Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) |
| 4DiM | 显式以相机与时间控制画面；输出空间与几何表示的区别 | [论文](https://arxiv.org/abs/2407.07860)、[作者项目](https://4d-diffusion.github.io/) |
| 4DGen | 动态 3D 表示的优化与时空约束、diffusion prior 的角色 | [论文](https://arxiv.org/abs/2312.17225)、[VITA-Group/4DGen](https://github.com/VITA-Group/4DGen) |
| 静态点云 diffusion | 逐点噪声目标与无序集合、等变性；用于区分跨样本 ID 与跨帧轨迹对应 | [论文](https://arxiv.org/abs/2103.01458)、[luost26/diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud) |

阅读大模型时检查实际训练目标：使用 Transformer、latent 或视频数据并不能确定模型使用 DDPM。Wan2.1 的 flow matching 配方不能直接替换成 DDPM epsilon loss；SV4D/4DGen 的动态几何流程也不能缩写成“对 `[B,F,N,3]` 随机张量加噪就复现了”。[形状脚本](../scripts/diffusion_shapes.py) 仅展示共享概率算法怎样作用于不同表示。

## 条件生成专题来源

配套 [17C](../tutorials/17c-diffusion-conditioning.md) 与 [conditioning.py](../src/fm_tutorial/diffusion/conditioning.py)。

| 方法 | 原始来源与源码 | 阅读重点 |
|---|---|---|
| DiT 的四类 conditioning 与 adaLN-Zero | [论文](https://arxiv.org/abs/2212.09748)、[models.py](https://github.com/facebookresearch/DiT/blob/main/models.py) | `c=t+y` 与 adaptive normalization 是不同操作；六路调制、残差门与单独输出层初始化 |
| Text cross-attention | [LDM](https://arxiv.org/abs/2112.10752)、[attention.py](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py) | noisy data 提供 Q，文本提供 K/V；保留条件 token 与 padding 语义 |
| FiLM | [论文](https://arxiv.org/abs/1709.07871) | 条件生成逐特征的仿射变换；FiLM 本身不是 diffusion 专属算法 |
| 空间自适应归一化 SPADE | [论文](https://arxiv.org/abs/1903.07291)、[NVlabs/SPADE](https://github.com/NVlabs/SPADE) | scale/shift 保留空间位置；原方法是语义图像合成，不是 DDPM sampler |
| ControlNet | [论文](https://arxiv.org/abs/2302.05543)、[lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | 冻结主干、可训练分支、多层空间 residual 与 zero convolution |
| 图像提示 IP-Adapter | [论文](https://arxiv.org/abs/2308.06721)、[tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) | text/image 分开的 cross-attention；不是所有 reference-image conditioning 都叫 ControlNet |
| Joint attention / MMDiT | [SD3 论文](https://arxiv.org/abs/2403.03206) | 双模态不同参数、联合 attention；不要与单向 text K/V 混同 |
| Classifier guidance | [论文](https://arxiv.org/abs/2105.05233)、[openai/guided-diffusion](https://github.com/openai/guided-diffusion) | 学 noisy classifier，使用关于 xt 的 log-likelihood 梯度 |
| CFG | [论文](https://arxiv.org/abs/2207.12598) | 学 conditional/null 两个分支，再按参数约定组合；与网络条件注入可以同时使用 |

本仓库新实现是独立的小模块。`ZeroSpatialResidual` 仅展示 zero-conv 控制残差结构，没有复制完整 ControlNet 分支；`PrefixCondition` 没有实现完整 MMDiT；简单 CFG 组合不等于离散 logits 上的任意线性外推。

## 源码核对清单（运行前）

1. 数据张量的每一轴是什么？时间轴是 diffusion time 还是物理时间？
2. 网络输出是 epsilon、x0、v、score、velocity 还是 clean-token logits？
3. 前向日程、训练采样时间分布、loss 权重与推理日程怎样对应？
4. 离散转移按行还是按列存储？哪个方向表示 clean→noisy？
5. 条件 dropout、guidance 参数约定、codec scaling、终点处理在哪里？
6. checkpoint 对应哪个数据、训练配置、模型与评测版本？

数学测试、短训练、公开 benchmark、完整论文复现分别记录。参考模型的原始结果仍属于原作者；本教程只报告 [validation.md](../docs/validation.md) 中真实执行过的范围。
