# 实践六：用代码理解 Diffusion

这是本项目新增专题的学习实践，不是清华公开课的官方新增 HW。先读 [17 基础算法](../tutorials/17-diffusion.md)，再读 [文本](../tutorials/17a-diffusion-text.md) 与 [各模态](../tutorials/17b-diffusion-modalities.md)。所有命令从仓库根目录运行，环境沿用 README 的 `pip install -e .`。

## 第一层：直接检查算法

```bash
python -m unittest discover -s tests -p 'test_diffusion*.py' -v
python scripts/diffusion_shapes.py
```

按顺序解释测试为什么成立：

- 已知 x0 与实际 noise，能否从 xt 重建 x0？这只是 oracle-noise 检查；生成时没有 oracle。
- 最后一跳为什么没有随机噪声？DDIM 的 clean endpoint 为什么用 alpha_bar=1？
- VPSDE 的 reverse update 为什么与正向 drift 的时间方向不同？ODE 的 score 系数为什么减半？
- 用三类、两步的枚举计算离散后验，能否与矩阵实现对上？改变行/列方向应破坏哪个测试？
- 全部 MASK 时能否采样出词？已知 prompt 是否保持？去噪网络能否读取右侧上下文？

`diffusion_shapes.py` 使用明确标注的合成张量，覆盖连续文本 embedding、图像 latent、数值列、视频 latent、对齐的动态点轨迹，以及类别列与 MASK 文本。shape 和重建误差不证明这些模态的生成质量。

## 第二层：实际训练与采样

```bash
# 同一个 Gaussian denoiser，先理解 DDPM，再比较 DDIM 的跳步
python scripts/diffusion_demo.py --mode gaussian --sampler ddpm --train-steps 300 --diffusion-steps 100 --out-dir runs/diffusion-ddpm
python scripts/diffusion_demo.py --mode gaussian --sampler ddim --train-steps 300 --diffusion-steps 100 --sample-steps 20 --out-dir runs/diffusion-ddim

# VP 的 epsilon 预测可换算成 score；比较两个数值解法
python scripts/diffusion_demo.py --mode score --sampler sde --train-steps 300 --sample-steps 100 --out-dir runs/diffusion-sde
python scripts/diffusion_demo.py --mode score --sampler ode --train-steps 300 --sample-steps 100 --out-dir runs/diffusion-ode

# 随机解掩码；前两个词固定为 the small
python scripts/diffusion_demo.py --mode masked --train-steps 300 --sample-steps 20 --out-dir runs/diffusion-masked

# 小词表 categorical 转移，训练 clean-token 辅助 CE
python scripts/diffusion_demo.py --mode categorical --train-steps 300 --diffusion-steps 100 --out-dir runs/diffusion-categorical
```

连续演示默认在二维八分量 Gaussian mixture 上训练；文本演示默认用八句固定长度的教学文本。每个目录保存 `training.csv`、`samples.npy`、`report.json`、`denoiser.pt`，文本还保存 `samples.txt`。`first_batch_loss` 与 `last_batch_loss` 来自不同随机 batch，只描述实际运行，不能当作 held-out 改善。短训练可能生成很差的样本，也应保留。

这里分别训练相同种子的小模型来方便入门。正式比较采样器时，应锁定**同一个 checkpoint 和同一组初始噪声**，再改变 sampler/NFE；不能把两个不同训练的模型当作严格 sampler 消融。`denoiser.pt` 保存 state_dict 与配置记录；恢复模型后，用 `GaussianDiffusion.sample` 直接比较。记录训练成本和推理成本，不能仅比较步数。

## 第三层：换成公开数据

### 先理解条件注入

读 [17C 条件生成](../tutorials/17c-diffusion-conditioning.md)，执行：

```bash
python scripts/conditioning_check.py
python scripts/conditioning_demo.py --method cat --train-steps 300 --out-dir runs/conditioning-cat
python scripts/conditioning_demo.py --method add --train-steps 300 --out-dir runs/conditioning-add
python scripts/conditioning_demo.py --method film --train-steps 300 --out-dir runs/conditioning-film
python scripts/conditioning_demo.py --method adaln --train-steps 300 --out-dir runs/conditioning-adaln
```

先解释特征/通道拼接与 token 拼接的轴，再验证投影相加、FiLM、AdaLN 和零残差的计算。
生成演示实际训练两类二维 Gaussian 点（中心分别为 `[-1,0]` 与 `[1,0]`），以条件 dropout 学习 null 分支，再在同一份初始 noise 上分别采样两个类别。它显示条件路径怎样接进 epsilon 去噪与 CFG，不是新 benchmark。
四个模块的参数量和初始化不同；这些独立训练只做机制演示，不能按损失或样本均值排序谁更优。公平消融还需匹配数据流、容量、训练预算、初始噪声、采样与评估。
`guidance_scale=0/1/>1` 分别代表 null/普通条件/强化引导，`condition_dropout` 只丢弃语义条件，保留 diffusion timestep 与训练目标。详细接口见脚本 `--help`；输出包含模型、实际 batch 损失、采样点与 null 分支训练次数。

连续演示可接入你已完成预处理的浮点数组 `[N,...]`：

```bash
python scripts/diffusion_demo.py --mode gaussian --sampler ddim --data-npy data/your_train_array.npy --train-steps 1000 --sample-steps 20 --out-dir runs/diffusion-array
```

脚本保留非 batch 轴，例如输入 `[N,C,H,W]` 会生成 `[samples,C,H,W]`。这个 flatten MLP 只适合小表示；实际图像/视频模型应使用 U-Net/DiT/时空网络。脚本不会训练 VAE、自动归一化或处理类别列，数组必须已按训练集拟合的转换正确编码。记录来源文件、数据许可、官方 split、预处理版本与哈希；另用独立验证/测试集评估。

| 实践路线 | 数据与作者实现 | 最低有效评估 |
|---|---|---|
| 小图像 | 从 DDPM 作者实验中的 CIFAR-10 路线开始，或用 MNIST 检查基础链路 | 固定预处理、独立测试集、样本图；正式指标锁定 feature extractor 与样本量 |
| 连续文本 | Diffusion-LM / DiffuSeq 的官方数据与代码 | 先测 codec/rounding，再测生成质量、条件任务与多样性；不要用 Gaussian MSE 当 perplexity |
| 离散文本 | MDLM / LLaDA 的公开配方，完整 tokenization 与 masking 规则 | 记录加权目标、评估 token、NFE、长度、prompt 与 sampling；unweighted demo CE 不能当 likelihood bound |
| 混合表格 | TabDDPM / TabDiff 官方配置中的 Adult 等数据集 | 同一 split 的 downstream utility、单列与联合分布、规则违反、隐私风险分别测 |
| 视频 | Video Diffusion / Wan 的官方推理与验证数据协议 | 外观、运动、时序一致性、条件遵循、实际运行成本；固定 frame/fps/resize |
| 动态 3D（4D） | SV4D / 4DGen 的官方输入与动态几何流程 | held-out camera/time、重投影、几何与时序一致性；视频指标不能替代几何评估 |

原始链接与源码入口见 [diffusion.md](../references/diffusion.md)。不要同时从零训练六种大模型；先用本地算法演示建立理解，再选择一条真实数据路线。预算来自实际参数、吞吐、显存与可用硬件，不预设 GPU 数量或固定八小时可完成。

## 交付

提交一份能把公式、代码函数、张量 shape 与实验日志对应起来的报告。至少包括一次失败及其原因、一项 oracle 算法检查、一项真实数据评估和 sampler 成本记录；未执行的项目标注为计划。理解到位的标准是能解释为什么当前方法失败，并指出需要改日程、目标、网络、表示、采样器或评估中的哪一个。
