# 验证记录：执行过什么，证据支持什么

记录日期：2026-10-03。精确代码版本以本文件所在的 Git commit 为准。
这里的结果来自实际运行；小输入回归测试仅检查程序与数学行为，不是模型能力 benchmark。

## 1. 环境与检查范围

| 项目 | 实际环境 |
| --- | --- |
| Python | 3.12.14，Linux x86_64 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.3 |
| Hugging Face Hub | 1.33.0 |
| HTTPX | 0.28.1 |
| CUDA | 不可用；没有执行 GPU 实验 |
| CPU 线程设置 | `OMP_NUM_THREADS=1`，`MKL_NUM_THREADS=1` |

完成 editable package 构建与安装；核心依赖和下载依赖分别安装后验证。
这不是一次所有操作系统、Python/PyTorch 版本的兼容矩阵测试。

实际检查包括：

- 全部 71 项 unittest，无失败、无跳过。
- `alignment_check.py` 的 CPU 数学诊断，正常退出；不是对齐训练或 benchmark。
- 所有脚本的 `--help`、Python 编译、Markdown 本地文件链接与 Git 空白检查。
- 全部 17 章、5 份作业的独立数学、接口及完成范围审阅；完整 Agent 示例执行成功。
- 原始文本下载、清洗、BPE、真实训练、断点恢复、生成、测量 CSV 与拟合入口。

下载器已声明直接使用的 Hub 与 HTTPX 依赖。代码仓库的来源导读分别核对官方目录、README、维护状态与许可；本地链接检查器不检查外部网页的未来可用性。

## 2. 真实上游数据与可追踪性

数据集：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。
固定 revision：`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
选用作者 README 指定的原始文本版 `TinyStories-train.txt` / `TinyStories-valid.txt`，没有换用 GPT4V2。
这个数据集由作者生成并真实公开发布；其合成来源不应被当作自然网页分布，也不是本项目编造的数据。
数据卡在该 revision 的许可字段为 `cdla-sharing-1.0`，与本仓库代码的 Apache-2.0 分开。

| split | 实际选择 | 读取到的响应体字节 | 原始文件完整大小 |
| --- | --- | ---: | ---: |
| train | 原始训练文件前 256 篇完整故事 | 262,144 | 1,924,281,556 |
| validation | 原始验证文件前 64 篇完整故事 | 65,536 | 19,447,282 |

两次 HTTP 响应为 206，文件按需流式读取后显式关闭；没有下载整个 1.9 GB 训练文件。
上述字节数是客户端消费的响应体，不代表完整网络流量。
段落内部换行保留，文档首尾空白删除；原始文本与 Parquet 的换行表示不同，不能混用哈希。

| 文件/产物 | SHA256 |
| --- | --- |
| 原始 train JSONL | `7a79662bcbcf8ed91f68824a9b401267311398f5f44f765162608037890e5030` |
| 原始 validation JSONL | `24d7f3525ee7db639f0d81c66bf63c5f369d5939a7e6dad83631db732f1a2521` |
| 清洗后 train JSONL | `bf82a36a63ea09e5099a3c61a8be28d000fb9e696fedb47ae7e489cf807e431d` |
| 清洗后 val JSONL | `b6910121b5cf5d8c53d3892cc29cc6a8a08c1a052565834bc69c2817c525d931` |
| tokenizer JSON | `4c2db51f8de1a67cad62f50c971cf12da1e1cf7299afb20cd59e0ed21ac3bd97` |
| train token binary | `e3264c72b0c4206eff3a9a929256fa2a425e05a1bf59f9ce5844fde9aee399a1` |
| val token binary | `d41d922212e66466fc4a356ffd2f1cde084fdb5089cddda8752f6db9cd9d505a` |

清洗参数为 `min_chars=80`，NFC、换行规范化、精确文档去重，保留上游分区。
本次 320 篇均保留，短文本过滤、重复删除和跨分区重复删除均为 0。
仅在训练文本学习 BPE；实际词表为 512，train 87,602 tokens、val 18,271 tokens，每文档包含一个 EOS。
前缀选择很小且可能有偏，不代表全量数据或正式 benchmark。

下载器最初的 Parquet 流式方案在本环境遇到一次进程退出异常，也存在正常退出的重复运行。
最终方案移除该原生扫描路径，直接读取固定版本的原始文本；最终 256/64 链路及另一次 8/4 下载都正常退出。
没有据一次正常退出宣称修复了 Arrow 内部竞争问题。

## 3. 实际训练与恢复

使用 [cpu.json](../configs/cpu.json)，只覆盖 tokenizer 的实际词表为 512；模型有 **434,816 个参数**。
context 128、batch 4、梯度累积 1、seed 42、FP32、学习率调度总 horizon 200。
依次运行到累计 5、10、15、20、25 个 optimizer updates，每次都保存 checkpoint，后续从该 checkpoint 恢复。
所有进程正常退出。`tokens_seen` 为处理过的训练目标计数，允许采样重复，不是独立文档或唯一 token 数。

| 累计更新 | 训练目标计数 | 最后一批 train loss | 固定切片 val loss |
| ---: | ---: | ---: | ---: |
| 5 | 2,540 | 6.2491321564 | 6.2171034813 |
| 10 | 5,080 | 6.1448278427 | 6.1096127033 |
| 15 | 7,620 | 6.0037546158 | 5.9797145128 |
| 20 | 10,160 | 5.9123263359 | 5.8668426275 |
| 25 | 12,700 | 5.8049392700 | 5.7671439648 |

loss 单位为 nat/token，val 每次统计固定的 2,032 个目标，覆盖有限切片。
这些测量证明当前输入、训练和恢复链路能工作；不能称为完整 TinyStories 测试成绩、能力提升或充分预训练。
25 步后执行最大 16 个新 token 的采样并正常退出。输出仍含无意义字节和乱码，没有可用语言质量结论。
恢复的逐位一致性另由含 dropout 的 CPU 回归测试检查，实际训练的正常恢复与该测试是不同证据。

### 复现本次链路

安装见 [README](../README.md)。以下从仓库根目录运行：

```bash
python scripts/download_tinystories.py --out-dir data/raw-final --max-train 256 --max-validation 64 --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
python scripts/prepare_data.py --input data/raw-final/train.jsonl --validation-input data/raw-final/validation.jsonl --out-dir data/clean-final --min-chars 80
python scripts/train_tokenizer.py --data-dir data/clean-final --out-dir data/processed-final --vocab-size 512
python scripts/train.py --config configs/cpu.json --data-dir data/processed-final --out-dir runs/final-chain --steps 5 --device cpu
for target_step in 10 15 20 25; do
  python scripts/train.py --config configs/cpu.json --data-dir data/processed-final --out-dir runs/final-chain --steps "$target_step" --device cpu --resume runs/final-chain/last.pt
done
python scripts/generate.py --checkpoint runs/final-chain/last.pt --tokenizer data/processed-final/tokenizer.json --prompt 'Once upon a time' --max-new-tokens 16 --temperature 0.8
```

`--steps` 是累计 optimizer updates，不能被解释为“在旧 checkpoint 上再加这些步”。
保留下载 metadata、清洗报告、token metadata、effective config、metrics 和 checkpoint；这些运行产物默认不提交到 Git。

## 4. CSV 拟合入口的实际检查

从上述实际 val 测量生成 CSV，固定参数数目，前四行标 `train`，25 步标 `holdout`：

```csv
params,tokens,loss,split
434816,2540,6.217103481292725,train
434816,5080,6.109612703323364,train
434816,7620,5.979714512825012,train
434816,10160,5.86684262752533,train
434816,12700,5.767143964767456,holdout
```

该 CSV 的 SHA256 为 `bd4fe7df5a087b194e1bc1755258134cfedceb9bde3377b5a428ef90a6084c76`（本次文件采用 CSV writer 的 CRLF）。

```bash
python scripts/fit_scaling.py --csv runs/final-chain/measurements.csv --axis tokens --output runs/final-chain/scaling-fit.json
```

入口正常退出，局部描述性拟合指数约为 -0.04094，holdout 预测 5.84363，实际 5.76714。
这是同一次极短训练轨迹上的相关 checkpoint，包含 warmup；只验证真实测量能通过 CSV→拟合→未参与拟合的点预测的接口。
它不是充分的 Scaling Law 实验，没有估计不可约项 E、可信区间或 compute-optimal 配比；不能外推为 Chinchilla 定律。
正式实验需按 [作业三](../assignments/03-data-scaling.md) 做预算、配置、独立运行与评估控制。

## 5. 独立审阅发现并修复的回归

- 长序列 completion log-prob、DPO margin 等低精度计算先提升到 FP32，保留 FP64 和梯度连接，防止有限输入发生中间溢出。
- Attention score 的 FP32 casts 不会阻止 autocast 重写 matmul；score/mask/softmax 显式禁用 autocast，value 运算保留混合精度。
- 全 PAD 的零损失在归约前提升精度，避免 `finite_logits.sum() * 0` 先溢出为 NaN。
- 深度嵌套工具输出采用有界、安全序列化，避免递归错误逃出 Agent 循环；不调用不可信对象的递归 repr。
- REINFORCE 公式补齐与本章折扣目标一致的外层折扣因子；GitHub 数学格式与丢失的 TeX 转义已修正。

数值与程序修复均有先失败、修复后通过的回归输入。CPU autocast 检查验证精度语义，不冒充 GPU 混合精度吞吐验证。

## 6. 未执行范围

没有执行约 100M 模型的正式预训练、多 GPU、原创 Triton forward/backward、生产推理压测、完整 SFT/DPO/RLVR 对照训练、真实长程 Agent RL 或公开能力 benchmark。
约 100M 配置的参数计数已核对：8192 词表时 99,502,848，2048 词表时 94,784,256，512 词表时 93,604,608；参数计数不代表训练完成。
对应章节和作业给出原理、源码入口与真实任务的实验协议，实际执行后才能增加测量结果。
GitHub Actions 配置随项目提交，远端运行状态以 GitHub 页面为准，本地通过不自动等于云端 CI 已通过。
