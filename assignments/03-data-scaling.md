# 作业 03：真实数据管道与受约束 Scaling 外推

本作业对应公开课程地图 HW3：Raw Pipeline、Data Card、Scaling Extrapolation。
目标是把真实文本转成可追溯训练载荷，获得真实测量，再检验一个预先声明的外推假设。
所有提交数字必须来自本次运行日志，或明确标为引用的外部测量；没有运行的字段写“未测”。

## A. 三个需要分开的实验问题

**问题一：清洗有没有改善学习？** 固定模型与训练曝光量，仅改变数据处理。
这检验数据策略，不能同时换 tokenizer、增加参数和延长训练。

**问题二：多训练一些 token 会怎样？** 固定参数量，改变有效监督 token 曝光量。
如果始终从同一小语料反复采样，这回答训练时长问题，不证明增加独立数据的收益。

**问题三：固定预算如何选择模型和数据？** 需要多种参数量、数据预算和接近等算力的实验。
它需要更丰富的测量设计，当前单轴拟合工具不能直接回答。
一个漂亮的五点曲线不能自动识别所有参数、不可约损失或最优模型规模。

相关概念见 [数据章](../tutorials/06-data.md)、[合成数据章](../tutorials/07-synthetic-data.md) 与 [Scaling 章](../tutorials/03-scaling.md)。

## B. 数据来源与研究评价必须先固定

推荐的自然文本主线是 [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext)。
固定 Hub revision、配置 `wikitext-103-raw-v1`、官方 train/validation/test 与文档恢复规则。
WikiText 的原始记录可能是一行；按行过滤与按文章过滤是不同实验。
保留标题和段落时应明确如何把行恢复成文档，并对每个输出文档保留源范围。

大型数据研究扩展采用 [DCLM](https://github.com/mlfoundations/dclm) 官方数据池、训练配方和评价套件。
阅读 [The Pile](https://github.com/EleutherAI/the-pile) 作为多源数据混合案例，逐组件核对数据使用条件。
不能把代码仓库许可证当作所有数据组件的统一授权。

仓库另提供 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的真实上游下载入口，适合小规模链路实践。
TinyStories 本身是公开发表的模型生成故事数据，**不是自然网页语料**。
对它进行真实训练可得到真实测量，但结论范围限于这个合成故事分布。
不得把它的局部验证损失称为通用语言能力 benchmark，也不另造故事评分测试来证明能力。

[LuckVd/aml-llm-lab](https://github.com/LuckVd/aml-llm-lab) 仅作教育参考。
其 [`presets/README.md`](https://github.com/LuckVd/aml-llm-lab/blob/master/presets/README.md) 明确所有 scaling/system 数字由函数生成。
**本作业禁止把这些 SYNTHETIC presets 当作真实训练、GPU 性能或 Scaling 外推证据。**
拟合自己预设的函数再恢复参数，最多验证拟合程序，不能证实自然训练规律。

## C. 数据卡的必填字段

不要只提交一句“用了公开数据”。数据卡至少包括：

| 字段 | 要回答的问题 |
| --- | --- |
| 来源与版本 | 数据集 ID、URL、revision、取得日期、文件摘要是什么？ |
| 使用条件 | 数据许可与各组件限制是什么？ |
| 文档单位 | 一条记录是一行、一页、一篇文章还是一段对话？ |
| 预处理 | Unicode、空白、提取、语言与长度规则是什么？ |
| 留存与偏差 | 每个来源/长度区间丢弃了多少？哪些群体或语言可能被误杀？ |
| 去重 | 精确/近重复阈值、代表文档选择、跨 split 处理是什么？ |
| 污染检查 | 对哪些已有 benchmark、哪些字段、采用什么匹配规则？ |
| tokenization | tokenizer 训练 split、词表摘要、特殊 ID、EOS/packing 规则是什么？ |
| 划分 | 原始 split 是否保留？processed val 是否缩小？ |
| 预算 | 独立 token、曝光 token、优化步、训练机器与耗时是什么？ |

缺少近重复或污染实现时明确写出缺口，不把精确去重叫完整无污染保证。
本仓库不保存所有原始元字段到训练 JSONL，应另存来源 manifest。

## D. 可运行的小规模链路

先安装本仓库与可选真实数据下载依赖：

```bash
python -m pip install -e ".[data]"
python scripts/download_tinystories.py --out-dir data/raw \
  --max-train 200 --max-validation 50
python scripts/prepare_data.py --input data/raw/train.jsonl \
  --validation-input data/raw/validation.jsonl --out-dir data/clean
python scripts/train_tokenizer.py --data-dir data/clean \
  --out-dir data/processed --vocab-size 512
python scripts/train.py --config configs/cpu.json --data-dir data/processed \
  --out-dir runs/data-pilot --steps 20 --device cpu
```

这些数字是人为设置的**运行预算**，不是预期结果或 benchmark 规模。
下载脚本保留上游 train/validation，并记录解析后的 revision；取前若干文档可能有顺序偏差，报告选择策略。
BPE 为教学实现，逐轮重算 pair 计数，大规模 tokenizer 学习成本可能很高。
CPU 配置是链路配置，不能冒充约 0.1B 正式预训练。
真实参数量、实际词表和 supervision tokens 从运行日志读取，不从配置名称猜测。

WikiText 或你已取得的其他真实 JSONL，也可使用后面三条命令，替换输入来源。
不要把不同数据覆盖在同一目录里；每个条件单独命名并保存 manifest 摘要。
已有 `last.pt` 的 run 需显式 resume 或换输出目录，避免混合日志。

当前 `--validation-input` 会删除与 train 重复的 validation 记录，processed val 是处理后的开发集。
最终官方评价保持原始评价集，并明确采用训练侧去污染策略。
`train.py` 固定验证切片的 `val_loss` 是有限切片 NLL；切片可能重叠，不是完整官方测试指标。

## E. 清洗的受控比较

定义三个条件：基础规范化、规范化+精确去重、再增加一个明确的过滤规则。
从同一真实原始池出发，所有条件共用冻结 tokenizer 和官方评价集。
记录 raw、accepted、unique、tokenized 的数量，并计算分来源留存率。

过滤条件使语料量减少时，有两种合法比较：

1. 固定独立语料预算，比较被选择文本的质量；所有条件有足够独立 token。
2. 固定训练曝光预算，允许不同重复次数，但同时报告独立 token 数和重复率。

二者不同，不要把第二类实验中更多重复曝光解释为增加独立数据。
采用相同模型结构、优化设置、有效 batch、种子集合、训练曝光和评价配置。
训练随机性较大时使用多个独立种子，并报告每个种子的原始测量。
若没有资源做重复运行，应明确结果没有跨种子不确定性估计。

## F. 先定义横轴，再设计 Scaling sweep

当前 [`scaling.py`](../src/fm_tutorial/scaling.py) 拟合单轴局部模型：

$$\log L=a+b\log X,\qquad L=e^aX^b.$$

$L$ 是一致单位的 held-out loss，$X$ 是 `tokens` 或 `params`。
它没有不可约项 $E$，不是 $E+A/N^\alpha+B/D^\beta$ 的联合 Chinchilla 拟合器。
它不能给 compute-optimal allocation、置信区间或因果结论。

| sweep | 横轴 | 必须固定 | 当前实现允许的结论 |
| --- | --- | --- | --- |
| token sweep | 实际监督曝光 tokens | 实际参数量、评价/数据协议 | 此区间的局部训练趋势 |
| parameter sweep | 实际参数量 | 实际监督曝光 tokens、评价/数据协议 | 此区间的局部规模趋势 |
| joint/factorial sweep | 参数量与 token 预算 | 可比较配方与评价 | 需外部联合模型，当前工具拒绝混杂 |

至少准备三个不同的 train 横轴值和一个未经拟合的 holdout 测量。
这是工具的最低可运行条件，不是识别可靠 scaling law 的充分样本量。
最好预先保留较大的规模点检验外推，而不是看到曲线后挑选最好预测的一点。
若 holdout 参与选函数、调参数或反复修方案，应再保留第二层最终测试。

## G. Token 数与训练 schedule 的具体约定

本仓库无 PAD 的长度 $T$ 输入窗口监督 $T-1$ 个目标。
batch 为 $B$、gradient accumulation 为 $A$、更新 $S$ 次时：

$$D_{\mathrm{exposure}}=SBA(T-1).$$

此处 $D$ 可含反复采样的同一 token，和训练文件中的独立 token 数不同。
最终 CSV 的 `tokens` 应使用日志 `tokens_seen`，并在 manifest 解释其为曝光量。
外部训练器可能计数输入 token 或所有 token，不能混进同一 CSV。

`train.py --steps` 是累计更新目标，学习率 schedule 的终点由配置 `training.max_steps` 固定。
从同一个长 schedule 取早期 checkpoint，是训练轨迹研究；早期点并未走完独立的衰减 schedule。
若研究各 token 预算下的终端模型，需要为每个预算设定对应 `max_steps` 与一致的相对 warmup/decay 规则。
不要把中断运行和各自充分训练的模型混合比较，却不记录差别。
同轨迹 checkpoint 强相关，不能当成多个独立随机种子。

## H. CSV 与拟合命令

文件只填写真实日志导出的数值，格式如下；这里不提供任何假测量行：

```csv
params,tokens,loss,split
```

`params` 来自实际参数计数；`tokens` 来自实际曝光计数；`loss` 来自同一 frozen evaluation。
`split` 每行写 `train` 或 `holdout`；这里的 split 指拟合测量点划分，和语料 train/val 是两层不同划分。
每个数值行另有 run ID、配置/数据/代码摘要、种子与日志路径的 provenance 表。

```bash
python scripts/fit_scaling.py --csv measurements.csv --axis tokens \
  --output runs/scaling-fit.json
```

拟合 `tokens` 时要求 `params` 恒定；拟合 `params` 时要求 `tokens` 恒定。
没有显式 split 时，程序保留最大横轴的约 20% 行作为外推 holdout；正式作业应预先显式指定。
对混杂轴、非正数、NaN/Inf 或不足的不同 train 横轴，程序应拒绝拟合。
拒绝输入是有效的科学边界，不应通过改列名绕过。

## I. 评价外推而不是只评价训练拟合

报告 train log-RMSE、holdout log-RMSE、每个 holdout 的真实 loss 与预测 loss。
画真实测量点、train/holdout 标识、拟合曲线及训练范围；横轴注明 units。
不在没测过的区间画一条曲线就声称“已验证”。

残差有系统弯曲时，可能存在不可约项、优化不足、数据重复或方法改变。
小误差也可能来自 holdout 离训练范围很近，不能推广到大几个数量级。
指数接近零不必强行修成负数；先检查学习是否饱和、评价切片与训练是否真正变化。
对跨 tokenizer 的 loss 不直接比较 token perplexity；本作业主线冻结 tokenizer。

进一步的联合研究可参考 [Chinchilla 原论文](https://arxiv.org/abs/2203.15556) 和 [CS336 Assignment 3](https://github.com/stanford-cs336/assignment3-scaling)。
采用多种 $N$ 与 $D$ 的 factorial/iso-compute 设计，记录优化配方与算力测量，再评估可识别性。
不可把别人论文的拟合系数当作本项目训练出的规律。
CS336 的课程训练服务属于其作业环境，本仓库不保证外部读者能调用，也不把服务结果当本地执行。

## J. 交付清单与验收

- 数据卡、原始来源 manifest、预处理版本、逐阶段统计与重复/污染审计。
- 冻结 tokenizer、训练/验证摘要和数据配置，不提交无权公开的完整原始文本。
- 实际 run logs、参数计数、tokens_seen、seed、环境、checkpoint 摘要。
- 真实 measurements CSV、拟合测量点 split、held-out 预测和残差图。
- 同模型/同预算的清洗比较；如仅完成链路实践，明确没有完整下游研究评价。
- 使用 DCLM 时固定 eval commit 与 CORE/EXTENDED `v1`/`v2`，不得混合比较。

验收重点是可追溯、控制变量与外推边界，不是要求一定出现正提升。
完整失败日志、原因分析与“假设未获支持”也是合法交付。
若尚未训练，不提交装满估计数字的表；提交协议、可运行命令和待测项。

思考：更多重复曝光是否等价于更大的数据集？固定 token 预算为何仍不能完全固定训练 FLOPs？
同一个模型更低的验证 loss，在什么条件下才能支持跨任务泛化结论？
若单轴工具拒绝你的 CSV，这暴露的是程序问题，还是实验中同时改变了两种变量？
