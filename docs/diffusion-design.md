# Diffusion 专题设计

用户要求在已有中文教程中增加 diffusion：细讲 DDPM、score-based 与核心算法，用代码贯通连续 latent 文本、离散文本、图片、表格、视频及 4D 数据。保留原有 16 周课程与代码。

## 结构

- `tutorials/17-diffusion.md`：专题入口与基础算法；概率、Gaussian、Markov、加噪闭式、训练、DDPM/DDIM、score、VP/VE SDE、reverse SDE、probability-flow ODE、CFG、flow matching 边界。
- `tutorials/17a-diffusion-text.md`：连续 embedding/latent 的编码与解码，D3PM 转移矩阵与精确后验，absorbing mask、训练权重、采样与 AR 对比。
- `tutorials/17b-diffusion-modalities.md`：图像 pixel/latent、U-Net/DiT，混合类型表格，视频时空 latent，动态几何 4D；场景、shape、源码入口、评测和失败边界。
- `src/fm_tutorial/diffusion/`：原创 PyTorch 算法，CPU 可运行；连续 Gaussian/VP、离散 categorical/mask、微型去噪网络。
- `scripts/diffusion_demo.py` 与 `scripts/diffusion_shapes.py`：真实执行的算法教学演示；合成输入只证明机制，不作为 benchmark 或论文复现。
- `assignments/06-diffusion.md` 与 `references/diffusion.md`：实验路线与逐项原始来源。

## 约定与验收

连续 Gaussian 的数组索引 `0..T-1` 对应论文时间 `1..T`；干净端 alpha_bar=1 单独处理。离散 D3PM 使用 `1..T` 与 `Q_bar[0]=I`。float32 教学计算，测试包含终点、后验归一化、精确参考枚举、梯度、采样有限性、mask 固定条件与噪声强度。没有 GPU 训练或大型 pretrained 权重执行时不报告该类能力。

依赖沿用 Python>=3.10、PyTorch>=2.3 与 NumPy，不引入大型框架。原始论文与作者仓库用链接引用，教程文字与小实现原创。保持既有文件；README、覆盖表、验证记录及 CI 只补充本专题。
