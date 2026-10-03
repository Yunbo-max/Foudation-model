# 作业一　Foundations：真实数据上的 tokenizer、Transformer 与预训练

> 配套教材：[00](../tutorials/00-prerequisites.md)–[03](../tutorials/03-scaling.md)；官方扩展：[CS336 basics](https://github.com/stanford-cs336/assignment1-basics)。

## A. 任务和证据边界

本作业要求你走通真实语料→去重/切分→词表→训练→验证→生成→消融的链路。
目标不是截图证明“模型会说话”，而是让另一人能追溯数据、复现训练配置、核对指标。
主线使用作者真实公开发布的 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。
它本来就是 GPT-3.5/GPT-4 生成的简化故事语料；“合成来源”与“自己捏造实验数据或结果”是不同概念。
必须在数据卡明示来源，也不能把简化英语故事的效果推广成通用知识或中文能力。
可用真实 [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) 或 [CS336 OWT sample](https://huggingface.co/datasets/stanford-cs336/owt-sample) 做领域扩展。

本仓库提供原创教学 tokenizer、模型和单设备训练器。
约 0.1B 完整预训练需要你实际运行；仓库中有配置不等于存在已训练权重和成绩。
本作业的硬件与预算建议由本教材制定，不是唐杰课程官方要求。
第三方作业与源码从官方链接取得，按各自许可使用，不把整包第三方代码混入本项目。

## B. 第一阶段：建立可追溯的小规模真实链路

安装按根 README 执行，数据下载需要可选依赖：

```bash
pip install -e '.[data]'
python scripts/download_tinystories.py --help
python scripts/prepare_data.py --help
python scripts/train_tokenizer.py --help
```

起步先取公开训练集的 1000 条、验证集的 100 条：

```bash
python scripts/download_tinystories.py --out-dir data/raw --max-train 1000 --max-validation 100
python scripts/prepare_data.py --input data/raw/train.jsonl --validation-input data/raw/validation.jsonl --out-dir data/clean --min-chars 80
python scripts/train_tokenizer.py --data-dir data/clean --out-dir data/processed --vocab-size 512
python scripts/train.py --config configs/cpu.json --data-dir data/processed --out-dir runs/cpu --steps 100 --device cpu
python scripts/generate.py --checkpoint runs/cpu/last.pt --tokenizer data/processed/tokenizer.json --prompt 'Once upon a time' --max-new-tokens 64 --temperature 0.8 --seed 42
```

这些命令是你将运行的实验步骤，本教材不预填损失、吞吐或输出。
选择前若干条不是随机代表性采样；下载器将此选择规则和 resolved revision 记录在 `source_metadata.json`。
TinyStories 官方 train/validation 边界通过 `--validation-input` 保留，同时清理跨 split 的精确重复。
若你只传一个真实 JSONL，准备器采用稳定哈希划分；这是新的教学 split，不能冒称官方验证成绩。
手写 BPE 每轮重算 pair，规模扩大时可能很慢；小链路不是秒级运行保证。

提交第一次运行的环境、源码版本、完整命令和真实日志。
若下载失败，保留错误与未完成状态；不要用手写几段故事替代研究训练集。
已有合法数据时可自备 JSONL，但需补来源、版本、文档边界与许可说明。

## C. 数据与词表验收

检查规范化前后的数量，统计空文档、过短文档、精确重复及跨 split 重复清理。
列出训练与验证文档长度分位数、总 UTF-8 字节与 token 数。
说明 `min_chars=80` 以字符计数，不是 token 阈值，也不是普遍最佳阈值。
实际处理过程由 [data.py](../src/fm_tutorial/data.py) 和 [prepare_data.py](../scripts/prepare_data.py) 定义。

词表必须只用 train 学习：

| 产物 | 应检查什么 | 为什么 |
| --- | --- | --- |
| `tokenizer.json` | 实际词表、特殊 ID、merge 次序、哈希 | 编码身份与 checkpoint 绑定 |
| `train.bin` | little-endian uint32、无 PAD、文档 EOS | 模型输入与 loss 口径 |
| `val.bin` | 使用相同 tokenizer、独立真实 split | 避免验证信息参与词表学习 |
| `metadata.json` | 文档数、token 数、源与文件哈希 | 把二进制追溯回文本 |

在真实数据中选取若干含标点、换行与非 ASCII 的文档，检查 `decode(encode(text))`。
另用中文、emoji 和空字符串做回归检查，说明它们仅验证 UTF-8 与 API，不构成 benchmark。
ByteBPE 从字节开始，EOS=256、PAD=257；字节层面的覆盖不代表多语言学习能力。
词表训练预算不等于实际词表大小；若没有更多合并 pair，应记录实际值。

## D. 模型数学与接口验收

画出一个 batch 从 `[B,T]` 到 `[B,T,V]` 的形状表。
至少解释嵌入、QKV、RoPE、GQA 共享、causal/key padding mask、RMSNorm、SwiGLU 与词表头。
查 [ModelConfig](../src/fm_tutorial/model.py) 的整除与偶数 head 维度限制。
两个默认配置使用共享 embedding/head，实际参数数应按参数对象计数。

模型内部进行 next-token 移位，因此输入 `labels=input_ids`，不要在外部再移位一次。
用手算或独立交叉熵核对一个短输入的监督位置与 loss 分母。
改变未来 token，应不影响过去位置 logits；PAD 目标不贡献损失，真实位置不能注意到 PAD key。
这些是 correctness evidence，与公开语料上的 NLL 分开报告。

```bash
python -m unittest discover -s tests -v
```

若 torch 不可用而数学测试被 skip，记录 skip 数并安装所需环境后重跑；不能把 skip 宣称成完整验证。
本作业不要求复制 CS336 的 staff 解答，官方测试应在它自己的独立目录与版本中运行。

## E. 训练与验证：记录的数必须代表你真正算过的东西

[train.py](../scripts/train.py) 将模板的 nominal vocab 改为实际 tokenizer 大小，并写 `effective_config.json`。
它随机抽取训练 token 窗口，允许跨 EOS 接触前一文档；报告这个 packing 策略。
每条 context 为 $T$ 的窗口产生 $T-1$ 个监督目标，`tokens_seen` 记录这种有效目标数。
验证器用固定位置的有限切片，`validation_targets` 说明评估量；短文件的切片可能重叠。
默认 `val_loss` 是固定有限切片指标，不能叫完整官方语料 PPL。

如需全文评估，应扩展一个独立评估器：

1. 保持官方 split、tokenizer 和文档处理规则。
2. 规定上下文窗口与 stride，确保每个目标只计一次，避免重叠目标重复加权。
3. 累加所有有效目标的 NLL 和计数，最后取比值，不平均不同大小 batch 的均值。
4. 明示文档开头、EOS、PAD 与截断的处理。
5. 报告结果适用的是哪个数据版本、tokenization 和上下文条件。

这项全文评估扩展尚未由基础训练脚本自动完成。
提交时可以只报告切片指标，但必须写准确名称和局限。
生成 samples 应使用固定真实 prompt 池，保存原始输出、seed 和实际停止长度，避免只挑好看样例。

## F. 续训验收：保持 schedule 与数据身份

CPU 配置的 schedule 总 horizon 是 200 步；`--steps` 是累计目标，不是新增步数。
从第 100 步继续到第 200 步：

```bash
python scripts/train.py --config configs/cpu.json --data-dir data/processed --out-dir runs/cpu --steps 200 --device cpu --resume runs/cpu/last.pt
```

保持模型配置、超参数、文件哈希、tokenizer 与设备类型一致。
恢复逻辑包含参数、优化器、AMP scaler、RNG 与 token 计数；源码中仍需核对当前格式。
比较“一次训练 200 步”和“100 步+恢复至 200 步”，使用独立 out-dir 与相同真实输入。
CPU 的严格一致性与 CUDA 上的数值确定性条件不同；报告实际比较结果与容限。
只有权重能加载，不足以证明训练精确续接。

## G. 约 0.1B 阶段：先用实测决定预算

可扩大真实数据为 `--max-train 20000 --max-validation 2000`，再逐步决定更大规模。
这只是一个子集建议；20k 文档并不等于充分训练 100M 模型的数据预算。
训练器会循环随机抽窗口，所以消费 token 数可超过独特语料 token 数。
报告重复读取程度并监测过拟合，不把重复消费当成等量新信息。

[100m.json](../configs/100m.json) 的关键设置是 $d=768,L=12,H_q=12,H_{kv}=4,d_{ff}=2688,T=512$。
实际词表 8192 时共享头参数约为 99,502,848；词表 2048 时约为 94,784,256。
以运行打印的参数计数为最终依据，避免称所有词表下都是同一个 100M 模型。
先运行短 GPU pilot：

```bash
python scripts/train.py --config configs/100m.json --data-dir data/processed --out-dir runs/100m-pilot --steps 100 --device cuda
```

100 步仅检测可行性；完整预算根据实测 tokens/s、显存、有效数据量与可用时间决定。
修改配置中的 `max_steps` 后另开实验目录；不要改 schedule 再冒称原实验精确续训。
GPU 显存需求随 batch、dtype、激活与注意力实现变化，先测 OOM 边界，不给硬性“某显存必能训练”的保证。
混合精度并不降低全部 optimizer 状态；记录实际 dtype 与 AMP 行为。

## H. 至少一项有解释力的真实消融

可选择词表预算、MHA/GQA、context 长度或 norm 替换中的一项。
基础代码提供 `n_kv_heads` 与 context 配置；norm 替换需要自己实现，不存在自动 ablation CLI。
以同一真实训练/验证数据、tokenizer（词表实验除外）、相同有效 token 数、seed 集合与评估口径对比。
分别报告参数数、NLL、tokens/s、峰值显存与失败率。
词表实验应补充 BPB 与 token 长度；GQA 实验应说明 cache 尚未实现，所以只能理论估算 cache 项。

先写假设，例如“减小 KV heads 是否在该规模改变验证 NLL，同时降低 KV 投影参数”。
再给控制变量、实际运行表与解释；若观察不支持假设，也是一份有效报告。
不把一次短跑的微小差异写成确定结论，不把没有 GPU 的配置差异写成实测加速。

## I. 最终交付

提交数据卡、完整命令、配置、tokenizer/数据哈希、checkpoint 元数据、原始训练日志、验证协议和消融报告。
研究产物可以保存在你自己的实验目录；不必把大权重与受限制语料上传公开仓库。
报告需分清：工程检查完成、CPU 真实数据短训练完成、GPU pilot 完成、完整预训练完成。
每一项按实际证据标记，未运行项注明缺硬件、数据或预算。

验收者应能回答三件事：模型读了什么、损失到底怎么算、比较中哪些条件变化了。
能复述公式却没有真实记录，或有漂亮结果却无法追溯输入，都不构成完整交付。

## J. 延伸与自检

在独立目录阅读 [CS336 assignment1](https://github.com/stanford-cs336/assignment1-basics) 的 handout、`tests/adapters.py` 与官方数据说明。
先固定访问时 commit；官方目录和测试会演进，本作业不是其认证提交。
可自行实现更高效 BPE、完整语料评估、更多初始化或优化器消融，但每项都要记录预算。
最后回答：若模型在 TinyStories 验证集表现好，为什么还不能直接宣称它能回答百科问题？
