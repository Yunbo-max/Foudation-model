# 00　预备知识：把语言模型看成一个可检查的计算过程

> 主线入口：[01 学习范式](01-paradigms.md)；实践入口：[作业一](../assignments/01-foundations.md)。

## 0.1 我们究竟要学会什么

想象你要做一个能续写百科段落的系统。
第一版读取一段真实文章，输出下一词的概率；第二版能遵从指令；第三版调用搜索与程序，并依据结果修改行动。
三个版本可能共用 Transformer，却使用不同的训练目标、数据和评估器。
因此，理解 foundation model 不能只背网络结构，也不能只会调用一个聊天接口。
你需要贯通“文本怎样变成数组、数组怎样得到概率、概率怎样变成梯度、梯度怎样产生可测量的能力”。

本教材按唐杰 2026 课程公开地图的主题组织，正文与代码为原创讲解。
地图入口是个人复现项目 [from-token-to-agent](https://github.com/jackmcgradylee/from-token-to-agent) 中的课程照片与说明，不是清华官方完整仓库。
[唐杰、杜晋华等人的课程教材](https://github.com/dujh22/AML-LLM) 是基础主题的教学来源；[2024 助教资料归档](https://dujh22.github.io/Dujinhua_wiki/aml2024/) 对应较早课程。
三者应分开看：课程照片说明主题，教材解释基础，个人仓库记录作者自己的实现与实验。
个人仓库中的成绩、硬件预算与进度不能当成本教材结果，也不能据此宣布官方验收要求。

学习完成后，你应能做出以下判断：

| 观察到的现象 | 能解释的机制 | 能做的检查 |
| --- | --- | --- |
| 训练损失骤降 | 更好的预测，也可能是泄漏或移位错误 | 查切分、去重和标签位置 |
| 更长文本显存爆炸 | 注意力中间量或 KV cache 增长 | 查张量形状与峰值显存 |
| 回答通顺但事实错 | 似然目标与事实验证目标有区别 | 固定题集与外部答案 |
| 模型扩大后效果不变 | 数据、优化或计算预算不足 | 查 tokens、学习率与验证曲线 |

## 0.2 最小数学词典：维度比符号更重要

本书用 $B$ 表示 batch 中的序列数，$T$ 表示每条序列的位置数，$V$ 表示词表大小，$d$ 表示隐藏维度。
输入 ID 是 $I\in\{0,\ldots,V-1\}^{B\times T}$，它们是整数索引，不是带距离含义的实数。
嵌入矩阵 $E\in\mathbb R^{V\times d}$ 将索引映射成 $X=E[I]\in\mathbb R^{B\times T\times d}$。
ID 100 与 ID 101 不意味着两个词“更接近”；词的可学习关系在向量及后续变换里。

矩阵乘法的基本规则是：

$$
(m\times k)(k\times n)=(m\times n),\qquad
(AB)_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj}.
$$

例如将每个位置的 $d$ 维状态投影到 $V$ 个候选 token，需要 $W\in\mathbb R^{d\times V}$。
于是 $Z=XW\in\mathbb R^{B\times T\times V}$，最后一维才是词表。
如果把 softmax 写在 $T$ 维，就变成比较位置，而不是比较候选 token。
张量广播只是在满足规则的维度上共享运算，不能替你保证语义正确。
建议在每段核心公式旁写下 shape，尤其注意 `transpose`、`reshape` 与 head 维度。

## 0.3 从分数到概率：softmax 与交叉熵

网络输出的是任意实数分数 $z\in\mathbb R^V$，称作 logits。
softmax 将它变成分布：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}},\qquad \sum_i p_i=1.
$$

逐步解释：每个候选先指数化，保证非负；再除以总和，保证归一化。
给所有分数加同一个常数不会改变分布，这让我们可以先减最大值，避免指数溢出。
目标 token 为 $y$ 时，交叉熵是 $\ell=-\log p_y$，使用自然对数时单位为 nat/token。
正确 token 的概率越大，损失越小；错误候选的相对分数也会通过分母影响损失。

不要先计算很小的概率再取对数；使用 `log_softmax` 或框架交叉熵进行稳定计算。
梯度的一个重要结论是：

$$
\frac{\partial \ell}{\partial z_i}=p_i-\mathbf 1[i=y].
$$

对于正确候选，梯度为负，梯度下降会提高它的分数。
对于其他候选，梯度为正，梯度下降会降低它们的分数。
这解释了为什么只提供一个目标 ID，也能更新整个词表分布。
平均损失的指数 $\mathrm{PPL}=e^{L}$ 叫困惑度，但它只能在相同 tokenization、评估文本和计算口径下直接比较。

## 0.4 场景一：为什么“输入等于标签”不一定作弊

真实文本经过 tokenizer 后得到 $x_0,x_1,\ldots,x_{T-1}$。
模型在位置 $t$ 看见 $x_0\ldots x_t$，预测 $x_{t+1}$。
本仓库 [model.py](../src/fm_tutorial/model.py) 的 API 接收同形状的 `input_ids` 与 `labels`，在内部完成移位：

$$
Z_{:,0:T-1,:}\quad\text{对齐}\quad Y_{:,1:T}.
$$

这里 `labels=input_ids` 表示“从同一条原始 token 序列构造下一 token 监督”。
因果 mask 保证当前位置不能读未来，移位保证它不把当前位置自身作为预测目标。
若调用者先移位一次，模型又移位一次，就会训练“跳过一个 token”的任务。
若完全不移位，模型可通过当前位置的嵌入学会复制，损失低却没有续写能力。

PAD 是凑齐批次长度的占位，不应进入损失；本仓库 PAD ID 为 257。
EOS ID 为 256，表示文档结束；它通常是需要预测的真实训练目标。
“不计 PAD 损失”与“注意力不能读 PAD key”是两个不同要求，后者由注意力 mask 负责。
训练样本若拼接多个文档，还要明确 EOS 是否允许后文接触前文；这是数据策略，不会因写一个 EOS 自动隔离。

## 0.5 梯度、优化器与训练步骤

设参数全部记作 $\theta\in\mathbb R^N$，一个 batch 的标量损失为 $L(\theta)$。
链式法则沿网络计算图反向传播，得到与参数同形状的 $g=\nabla_\theta L$。
基本梯度下降是 $\theta_{s+1}=\theta_s-\eta_sg_s$，学习率 $\eta_s$ 控制步长。
梯度不是正确答案，它表示当前位置附近让损失下降的方向。
训练中模型、数据和随机性都在变化，单步下降不能保证未见文本表现改善。

Adam 使用梯度的一阶、二阶历史平均调节各参数的更新尺度：

$$
m_s=\beta_1m_{s-1}+(1-\beta_1)g_s,
\quad v_s=\beta_2v_{s-1}+(1-\beta_2)g_s^2,
$$

$$
\hat m_s=\frac{m_s}{1-\beta_1^s},\quad
\hat v_s=\frac{v_s}{1-\beta_2^s},\quad
\theta\leftarrow\theta-\eta_s\frac{\hat m_s}{\sqrt{\hat v_s}+\epsilon}.
$$

平方、除法均按元素进行；$\epsilon$ 防止除零。
AdamW 将权重衰减从梯度适应项里解耦，更新含 $-\eta_s\lambda\theta$。
不同参数组是否衰减、优化器状态的精度、学习率 schedule 都属于可复现配置。
不要仅记录“使用 AdamW”，而遗漏 beta、epsilon 与衰减策略。

一次训练迭代通常依次执行取数据、前向、损失、清零梯度、反向、必要的裁剪和参数更新。
梯度累积会把几个 microbatch 的梯度合成一次更新；相应损失缩放和有效 token 数必须一致。
验证时使用 `eval()` 和禁用梯度；这会改变 dropout 行为并降低不必要的内存开销。
两者不能互相替代：禁用梯度不自动关闭 dropout，`eval()` 也不自动禁用梯度。

## 0.6 场景二：两个漂亮曲线，只有一个是可信实验

甲把原始文章切成小块后随机分训练集和验证集，同一文章的相邻句可能进入两边。
乙先对规范化后的文档去重，再按文档稳定分割，之后各自 tokenize 与切块。
即使甲的验证损失更低，也可能只是模型读过了相近内容。
本仓库 [data.py](../src/fm_tutorial/data.py) 采用规范化后全局精确去重；单源输入做稳定哈希切分，提供独立验证源时保留上游边界并清理跨 split 重复。
精确去重仍不消除改写、镜像与近似重复；应把它写成已完成的最低检查，而不是“没有泄漏”的万能证明。

Tokenizer 也是从数据估计出来的。
如果在验证文本上统计 BPE 合并，你已经把验证分布的信息用于模型开发。
本项目仅在 `train.jsonl` 学习词表，再用同一个 tokenizer 编码验证文本。
训练词表、模型 checkpoint 与评估配置之间要保持 ID 身份一致。
换 tokenizer 后复用旧嵌入，即使词表大小恰好相同，也通常不再表示相同的 token。

## 0.7 真实数据上的第一条实验路径

主线采用作者真实公开发布的 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。
它是论文作者用 GPT-3.5/GPT-4 生成的故事数据；这一已记录的合成来源与捏造训练样本、实验结果是不同概念。
数据卡记录来源与许可；下载器记录 resolved revision、split、字段与选择规则。
起步取 1000 条训练与 100 条验证只是链路开发，不代表完整语料，也不保证代表性。
手写 BPE 在较大词表与文本量下重算 pair 可能很慢，先用 512 的预算确认接口。

从仓库根目录按 README 安装项目后，先查看真实 CLI 契约：

```bash
python scripts/download_tinystories.py --help
python scripts/prepare_data.py --help
python scripts/train_tokenizer.py --help
python scripts/train.py --help
```

安装项目和可选数据依赖后，执行公开数据链路：

```bash
pip install -e '.[data]'
python scripts/download_tinystories.py --out-dir data/raw --max-train 1000 --max-validation 100
python scripts/prepare_data.py --input data/raw/train.jsonl --validation-input data/raw/validation.jsonl --out-dir data/clean --min-chars 80
python scripts/train_tokenizer.py --data-dir data/clean --out-dir data/processed --vocab-size 512
```

`--validation-input` 保留官方 train/validation 分区，清理跨区精确重复；验证不参与词表学习。
也可把授权取得的真实语料转换为每行一个 `{"text": "..."}` 的 JSONL。
若只有一个输入源，准备器的 Python 契约如下，可直接在安装后的环境中调用：

```python
from fm_tutorial.data import prepare_jsonl
prepare_jsonl("data/raw/corpus.jsonl", "data/clean", min_chars=80,
              val_fraction=0.1, text_field="text")
```

这条 Python 路径只生成清理后的 JSONL；选择该路径时，还需执行词表训练与编码，再启动模型训练：

```bash
python scripts/train_tokenizer.py --data-dir data/clean --out-dir data/processed --vocab-size 512
```

`train_tokenizer.py` 产生 `tokenizer.json`、`train.bin` 和 `val.bin`；二进制 ID 使用 uint32，每文档追加 EOS。
扩展可选 [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) 的 `wikitext-2-raw-v1` 真实百科文本；标题/空行与文档边界的转换需明示。
重新哈希切分后的 PPL 不能称为官方 WikiText test 成绩，正式对比需额外保留官方 test。
随后运行小规模真实数据训练：

```bash
python scripts/train.py --config configs/cpu.json --data-dir data/processed --out-dir runs/cpu --steps 100 --device cpu
```

100 步仅是你准备运行的起步预算，不是已经完成的训练结果，也不保证模型学到可用能力。
看损失是否有限、checkpoint 是否可读取、固定验证批次是否可复现。
脚本默认验证只覆盖固定有限切片，报告其目标数与可能重叠，不冒称完整官方语料 benchmark。
[cpu.json](../configs/cpu.json) 用于开发链路；[100m.json](../configs/100m.json) 用于约 0.1B 规模的 GPU 实验，具体参数以实际配置与计数为准。
CPU 跑通不能替代 GPU 的吞吐、显存与长期稳定性验证。

## 0.8 场景三：为什么重启后“同一个实验”变了

只保存权重会遗漏优化器动量、随机数状态和已经消费的数据位置。
续训时，旧权重配上新动量会改变更新；换 batch 采样序列又会改变梯度。
[train.py](../scripts/train.py) 的 checkpoint 记录配置、模型、优化器、token 计数与 RNG，阅读其恢复逻辑再设计续训检查。
不同硬件和算子可能仍有数值差异；确定性是一组明确条件，不是一个 seed 的全部作用。

每个实验至少保存：代码版本、数据版本、tokenizer 哈希、配置、seed、实际 token 数、参数数、运行环境、时间与验证口径。
失败和跳过的实验也应保留原因，例如 OOM、没有 GPU、数据未授权或依赖缺失。
未测量的表格单元应写“未运行”，不要用理论估算代替实测并省略标识。

## 0.9 检查理解

1. 已知 logits 为 `[B,T,V]`，为什么交叉熵之前通常要选择 `[:, :-1, :]`？
2. 同一个模型在两种词表下 PPL 不同，能直接判断谁更好吗？给出一个受 token 长度影响的原因。
3. 一个 batch 有不同长度的文档，怎样区分注意力 mask、PAD 损失 mask 与 EOS？
4. 给出一次续训所需的四类状态，并解释只存权重为什么不够。
5. 在真实 WikiText 文本上列出三条最长文档、空行比例与精确重复数；这些是数据检查，不是能力 benchmark。

能回答前四题后，进入 [01](01-paradigms.md)；第 5 题的真实统计应随 [作业一](../assignments/01-foundations.md) 提交。

## 原始来源与阅读目的

- [课程公开地图与个人扩展](https://github.com/jackmcgradylee/from-token-to-agent)：确认主题结构和来源性质，不搬运其实验结论。
- [AML-LLM](https://github.com/dujh22/AML-LLM)：深度学习、Transformer 与预训练教学背景。
- [CS336 assignment1](https://github.com/stanford-cs336/assignment1-basics)：官方从零建模作业入口。
- [PyTorch 自动求导入门](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：图、梯度与反向传播 API。
- [AdamW 原论文](https://arxiv.org/abs/1711.05101)：解耦权重衰减的动机。
- [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)：随机性与确定性的边界。
