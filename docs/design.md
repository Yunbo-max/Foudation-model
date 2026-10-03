# Foundation Model Tutorial：设计与边界

本项目应用户要求创建并提交至 Yunbo-max/Foudation-model，2026-10-03。

## 学习目标

原创中文教材，沿唐杰 2026 课程公开地图讲清现代语言模型的完整链路；参考课程团队教材、个人实践和官方开源项目。不是清华官方课程镜像。每章用具体场景连接基础、公式、实现、实验与误区。所有公开链接进入来源表，新闻只证明报道，不承担算法事实。

## 范围

16 章主线、基础预备、五项作业指南、源码导读、可运行 CPU/GPU 基础代码。大规模训练、GPU Triton 性能、真实长程 Agent RL 需要外部硬件与数据；准确区分本地验证和需用户运行的实验。禁止虚构训练曲线、模型能力、性能提升或数据。第三方代码仅链接，原创实现不整包复制无许可证仓库。

## 文件与接口

- `tutorials/00-prerequisites.md`、`tutorials/01-*.md` 至 `tutorials/16-*.md`：教材。
- `assignments/01-foundations.md` 至 `assignments/05-agent.md`：真实数据与基准上的交付指南。
- `references/sources.md`、`references/open-training.md`：来源性质、许可证和源码阅读入口。
- `src/fm_tutorial/tokenizer.py`：`ByteBPETokenizer.train(texts, vocab_size)`；`.encode(text, add_eos=False)`、`.decode(ids)`、`.save(path)`、`.load(path)`；`.vocab_size`；字节 0..255，EOS=256，PAD=257，合并从258开始。
- `src/fm_tutorial/data.py`：`prepare_jsonl(input_path, output_dir, min_chars=80, val_fraction=0.1, text_field='text', validation_input=None)`；输出 `train.jsonl`、`val.jsonl` 和数据报告。规范化后去重；未传官方验证输入时使用稳定哈希分割，传入时保留上游 split 并删除验证集中的交叉重复。读取真实文本，不内置伪训练集。
- `scripts/prepare_data.py`、`scripts/train_tokenizer.py`：命令行准备真实 JSONL、训练 tokenizer 后输出 `train.bin`、`val.bin` uint32 tokens（每文档 EOS）及 tokenizer.json。
- `src/fm_tutorial/model.py`：`ModelConfig(vocab_size, context_length=128, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=352, dropout=0.0)`；`TransformerLM(config)`；`forward(input_ids, labels=None)` 返回 `{'logits': Tensor, 'loss': Tensor|None}`。labels 与 input_ids 同尺寸，模型内部完成 next-token 移位；PAD 不进入损失。
- `scripts/train.py`：`--config configs/cpu.json --data-dir data/processed --out-dir runs/cpu --steps N --device cpu|cuda`；checkpoint 记录模型配置、state、optimizer、tokens 与 RNG；CPU configuration 与约100M configuration 明确区别；验证集固定且不用于 tokenizer 学习。
- `src/fm_tutorial/alignment.py`：PyTorch completion mask、sequence log-prob、SFT、DPO 和 group-relative PPO/GRPO 核心损失，数值稳定且完成区间一致。
- `src/fm_tutorial/scaling.py`、`scripts/fit_scaling.py`：仅消费真实测量 CSV，held-out 预测、有限数据可识别性边界。
- `src/fm_tutorial/agent.py`：有界轨迹、外部 verifier 与 self-judge 区分；接受 callable policy/tool/verifier，行为日志可审计。不宣称部署过真实长程 agent。
- `tests/`：标准 unittest；torch 不可用时明确 skip 数值测试，CI 安装 torch 后全跑。回归测试小输入仅验证数学和程序正确性，不充当研究 benchmark。

## 验证

先固定行为测试再实现。重点检查 tokenizer UTF-8 roundtrip、切分去重、因果遮罩、next-token shift、梯度、DPO 方向、GRPO 无效组、agent 终止、Markdown 内链和命令。记录环境与测试结果。推送后核对远端树及关键文件，不强制更新分支。
