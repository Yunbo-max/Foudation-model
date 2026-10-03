# 作业二　Systems：GPU kernel、并行训练与 serving 的可复现实验

> 配套教材：[04 系统](../tutorials/04-systems.md)、[05 推理](../tutorials/05-inference.md)；官方扩展：[CS336 systems](https://github.com/stanford-cs336/assignment2-systems)。

## A. 工作范围：先确认哪些部分已有实现

本项目显式 attention、单设备训练与重算上下文生成构成清楚的 baseline。
已有可选 CUDA 混合精度与梯度累积；没有原创 Triton kernel、GPU backward、DDP/FSDP 或生产 serving 实现。
本作业要求你在独立扩展中完成这些任务，不能把学习指南当成已完成的系统成果。
没有 GPU 时可完成数学、shape 与数值参考检查，GPU 性能、多卡与服务成绩应标为未运行。
硬件型号、卡数与时间预算由实际资源决定，本教材建议不代表唐杰官方硬件要求。

系统实验也需真实工作负载。
主要输入采用作业一的真实 TinyStories、WikiText 或 CS336 OWT 文档，保留版本与 tokenizer。
随机张量可用于 kernel 边界/梯度检查，但必须标成微基准或回归输入，不用它宣称真实任务表现。
不把 Triton 文档表格中的 TFLOPS、vLLM 论文加速比例复制为本次结果。

## B. 先建立 baseline 账本

保存环境信息：GPU 型号与数量、驱动、CUDA/ROCm、PyTorch、Triton、Python、设备互联及代码 commit。
记录模型实际参数数、batch/context/head 形状、dtype、mask、checkpoint 与数据哈希。
测 forward、backward、optimizer、数据读取与完整 step，区分各段时间。
先用 profiler 确认瓶颈，避免优化一个只占总时间很小的算子。

建议保留一个尚未填数的测量表，待实际执行后填写：

| 实现 | 输入来自 | forward ms | backward ms | step ms | 显存峰值 | 数值误差 |
| --- | --- | --- | --- | --- | --- | --- |
| 本项目显式参考 | 固定真实训练 batch | 待测 | 待测 | 待测 | 待测 | 参考 |
| PyTorch SDPA | 相同 batch 与权重 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 自写/官方教学 Triton 扩展 | 相同激活与 mask | 待测 | 待测 | 待测 | 待测 | 待测 |

“待测”说明这是实验设计，没有性能结论。
将复用官方教学 kernel 与自行原创 kernel 在作者/许可信息上分开记录。
官方示例能运行，也不代表你已完成全部支持形状或生产部署。

## C. 正确性：融合前先定义数学契约

输入 Q/K/value 必须明确 `[B,H,T,d_h]` 或 GQA 的不同 head 数。
规定 causal、key padding、all-PAD、dropout 与输出 dtype 行为。
检查位置机制是在 kernel 外还是内部；不要重复应用 RoPE。
用显式 PyTorch 作为参考，分别核对输出与 `dQ/dK/dV`。
使用相对/绝对容限并报告最大误差、误差分布与失败形状。

至少包含：短/长长度、非整 tile 长度、不同 head dimension、非连续输入、重复 token、PAD key 和 causal 边界。
全掩码行应采用明确行为，不能让偶发 NaN 隐藏到 batch 均值。
前向相同不保证反向正确；特别检查 softmax 行内归一化导数和 KV 梯度累积。
若输出梯度规模很小，也要测较大幅度，避免宽容限掩盖绝对错误。

本项目 tests 的小张量用于 correctness，真实 corpus NLL 另作 end-to-end 检查。
替换 attention 后，在相同 checkpoint 与真实验证 batch 上核对 loss 与 logits。
不要一边改变训练权重，一边以误差归因 kernel。

## D. 官方 kernel 路径：GPU backward 是独立交付

阅读 [FlashAttention 原论文](https://arxiv.org/abs/2205.14135) 的 IO-aware 思路，再读 [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
先解释 online softmax 的 $m,\ell,u$ 三个状态及跨块重缩放。
然后在独立目录固定官方版本，遵照其设备与依赖要求运行测试。
可进一步完成 [CS336 assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) 的官方 handout 与 adapter 接口。

访问时两套 CS336 README 的课程年份不同，必须分别固定版本，不能假设 `main` 永远不变。
在官方目录中查看：

```bash
uv run pytest --help
```

实际测试命令、支持 GPU 与评分要求以该版本 handout 为准。
不要把这里的 `fm_tutorial` 函数名直接填成官方 adapter，先逐一核对参数与返回语义。
提交原创或明确注明来源的 forward、backward 扩展代码，包含 tile/warp 参数和适用形状。
本项目没有自动运行这些外部 GPU 测试；你需保存原始输出与失败日志。

## E. 正确 GPU 计时：真实 batch 的 forward/backward 示例

以下代码在你已拥有 CUDA、真实训练二进制与本地可信 checkpoint 时运行。
它只测固定真实 batch 的 forward+backward，不含 optimizer、数据搬运与服务排队，因此不能命名成端到端训练吞吐。
相同 batch 重复用于系统测量，不作为泛化成绩，也不把它写成新的训练数据集。

```python
from pathlib import Path
import numpy as np
import torch
from fm_tutorial.model import ModelConfig, TransformerLM

assert torch.cuda.is_available(), "本测量需要真实 CUDA GPU"
ckpt = torch.load("runs/cpu/last.pt", map_location="cpu", weights_only=True)
cfg = ModelConfig(**ckpt["model_config"])
model = TransformerLM(cfg).cuda()
model.load_state_dict(ckpt["model_state"])
model.eval()  # 固定 dropout 行为；仍允许梯度计算
tokens = np.memmap("data/processed/train.bin", dtype="<u4", mode="r")
assert len(tokens) >= 4 * cfg.context_length
batch = np.asarray(tokens[:4 * cfg.context_length], dtype=np.int64).reshape(4, -1)
x = torch.tensor(batch, dtype=torch.long, device="cuda")

def forward_backward():
    model.zero_grad(set_to_none=True)
    model(x, labels=x)["loss"].backward()

for _ in range(10):
    forward_backward()
torch.cuda.synchronize()
samples_ms = []
torch.cuda.reset_peak_memory_stats()
for _ in range(30):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    forward_backward()
    end.record()
    end.synchronize()
    samples_ms.append(start.elapsed_time(end))
print({"median_forward_backward_ms": float(np.median(samples_ms)),
       "p95_ms": float(np.quantile(samples_ms, 0.95)),
       "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
```

脚本不更新 optimizer；加入 optimizer 后是另一计时口径。
输入二进制须与 checkpoint 中的数据哈希匹配，运行前核对，示例没有替你执行完整指纹检查。
示例用 FP32，不代表 AMP 性能；比较不同 dtype 时同时记录质量与误差容限。
峰值包含已加载模型与本测量激活，reserved 与 allocated 应分别理解。
使用 PyTorch benchmark/profiler 时遵照官方同步规则，编译与预热耗时单独报告。

## F. 多卡训练：先数值一致，再扩展性能

单卡与两卡先使用同一全局真实 batch，分片后聚合梯度，比较一步更新。
各卡有效 token 数一致时可平均本地梯度；不一致时按全局有效目标数加权。
固定学习率、优化器、模型初始化与全局数据顺序，容限反映实际浮点归约差异。
确认 sampler 不重复消费同一批样本，并把 epoch/seed 处理写清。

本项目训练脚本没有 `--distributed` 或 `--nproc`，不能把 `torchrun` 加在前面就自动得到正确 DDP。
先在独立扩展实现 process group、rank/device、数据分片、DDP 包装、累计同步与 rank-aware checkpoint。
如采用 FSDP，按当前官方接口处理权重与 optimizer 状态保存，不能复用单卡 state_dict 逻辑而不检查。
断点恢复还要恢复每 rank 的 RNG 与数据位置；只验证 rank 0 不够。

强扩展固定总 batch/总 token，比较 $S_p=t_1/t_p$ 与 $E_p=S_p/p$。
弱扩展固定每卡 batch，报告扩大后的全局 batch 与优化行为变化。
两种实验分表，报告卡间互联、通信比例与显存，避免把全局 batch 增大当成免费提速。
没有多卡时可交一份预算与通信模型，性能项标未运行。

## G. Serving：本项目基线与优化引擎分开

先用 [generate.py](../scripts/generate.py) 验证模型/tokenizer 身份、EOS、PAD 与最大输出长度。
它每步重算上下文，超过窗口会截断，没有 KV cache；只能称教学生成基线。
自行加入 cache 时，在窗口内比较 cached/uncached 最后 logits 与 greedy 序列，并核对 RoPE offset。
随后检查不同 batch 长度、EOS 提前结束、请求取消、超限策略和缓存回收。

可另用 [vLLM 官方项目](https://github.com/vllm-project/vllm) 部署官方支持格式的开放模型。
安装/启动/benchmark 参数以固定版本的官方文档与 `vllm --help` 为准，保留模型 revision 和 license。
本项目 checkpoint 不是直接支持的 Hugging Face 模型格式；若用不同模型服务，明确是独立系统实验。
如果要公平比较同一模型，必须实现正确格式适配并验证权重、tokenizer、chat template 与 logits。

请求池来自真实公开验证文档，保存文档 ID、prompt 选择与截断规则。
固定 prompt 长度组、并发、到达模式、输出上限、温度、seed 和 EOS。
请求文本不带评估答案；服务性能回放不等于任务质量评估。
若支持 prefix cache，分别测冷/热缓存并声明命中规则，不把热缓存成绩当无缓存基线。

## H. 延迟、吞吐与成本交付

至少记录每请求 TTFT、ITL/TPOT、E2E latency、实际 input/output tokens、成功/错误/超时状态。
汇总 p50/p95/p99 时写样本量；样本过少的极端分位数不稳定。
总吞吐按完整计时区间计算，包含加载与预热是否计入必须明示。
按并发扫描，画延迟与吞吐关系；没有测量，不画推测的漂亮曲线。

计费使用实际资源价格与日期，或只报告 GPU 秒。
单位请求成本包括失败、重试、verifier、工具、CPU/存储与闲置资源；需要区分设备预算与实付费用。
若使用质量评估，先固定任务成功与延迟限制，再算 goodput。
相同 tokens/s 但错误率很高的系统，不应凭吞吐被判成更经济。

## I. 最终验收与未完成项

提交环境与版本清单、真实输入索引、原始时间记录、profiler trace、数值误差、kernel/并行/serving 配置、费用口径与复现命令。
将 GPU forward、GPU backward、多卡、服务四项分别标记实际完成状态。
复用官方 kernel 要标其来源；自己只改 tile 参数不能称从零实现全部算法。
对不支持的形状、OOM、后端回退和吞吐下降保留记录。
只有 CPU 公式验证时，仍可完成理论部分，但整份系统作业不能写“已完成 GPU 性能验证”。

自检：一个 kernel 快了两倍但完整 step 只快 5%，应该如何用占比与时间线解释？
自检：四卡吞吐增大但 GPU 秒/请求上升，若 deadline 更紧，是否仍可能是合理方案？
自检：优化服务换了 checkpoint、tokenizer 或采样策略，为什么不能只归因调度算法？
