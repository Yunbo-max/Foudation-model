# 作业 05 · 可验证 Agent：从执行器到公开任务评测

本作业的目标是建立一个有工具、预算、日志和外部验收的 Agent 系统，并在既有公开任务协议上记录实际表现。
先做控制流检查，再进行真实基准运行；有训练资源时才扩展为 Agent RL。
本仓库已提供有界执行器与行为测试，未提供训练完成的长程 Agent 或公开基准成绩。
完成状态必须分成“接口验证”“基准运行”“训练收益”，不要合并成一个勾选项。

前置阅读：[Agent RL](../tutorials/11-agent-rl.md)、[系统基础](../tutorials/12-agent-foundations.md)、[记忆](../tutorials/13-memory.md) 和 [评估与安全](../tutorials/15-evaluation-safety.md)。

## A. 明确任务和验收

选择以下一种公开任务路径，并在开始前冻结版本和 task ID 集合。

| 路径 | 官方入口 | 必须准备 | 最终判定 |
| --- | --- | --- | --- |
| 多环境交互 | [AgentBench](https://github.com/THUDM/AgentBench/blob/main/README.md) | 指定任务服务器与配置 | 官方任务结果 |
| 网页 | [BrowserGym](https://github.com/ServiceNow/BrowserGym/blob/main/README.md) + [WebArena](https://github.com/web-arena-x/webarena) | 浏览器、网站服务、账号与重置 | 官方状态验收 |
| 代码修复 | [SWE-bench](https://github.com/SWE-bench/SWE-bench/blob/main/README.md) | 指定子集、Docker 镜像、补丁 | 官方容器测试 |
| 终端任务 | [Terminal-Bench](https://github.com/harbor-framework/terminal-bench/blob/main/README.md) | 固定数据标签、Harbor 环境 | 官方任务测试 |
| 视觉控制扩展 | [MineRL](https://github.com/minerllabs/minerl/blob/dev/README.md) | 固定环境与动作空间 | 该任务官方协议 |

不要自己编写一个简单问答环境，然后把它命名为 Agent benchmark。
自编小输入用于检查执行器；能力证据使用公开任务、官方验收和实际日志。
若机器不能启动所选环境，提交完整配置、明确“未运行”，并保留已完成的接口证据。
任选子集时写清选择方法；调试样本和最终报告样本分开，避免按成功与否选择任务。

## B. 检查本地执行器

在仓库根目录运行：

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

执行器 API 为：

```text
run_agent(policy, tools, verifier, *, max_steps=8,
          time_budget_seconds=30.0, self_judge=None, log_path=None)
```

`policy(history)` 返回结构化工具调用或最终答案。
工具以关键字参数执行；`verifier(answer, history)` 必须返回严格的 Python `bool`。
`{"success": True}` 不符合 verifier 接口；模型自评也不能使验证失败变为成功。
回调接收历史和答案副本，不能通过修改传入对象改写已保存的验收依据。
`log_path` 为一次轨迹写 JSONL，已有文件会被覆盖；每次运行使用唯一文件名。

可保存以下代码为自己的接口练习脚本并执行：

```python
from fm_tutorial.agent import run_agent

def policy(history):
    if not history:
        return {"type": "tool", "tool": "multiply",
                "arguments": {"a": 6, "b": 7}}
    return {"type": "final", "answer": history[-1]["observation"]}

result = run_agent(
    policy,
    {"multiply": lambda a, b: a * b},
    lambda answer, history: answer == 42,
    max_steps=3,
    time_budget_seconds=5.0,
    self_judge=lambda answer, history: {"success": True},
    log_path="runs/agent/interface-001.jsonl",
)
print(result["status"], result["answer"], result["steps"])
```

这段脚本证明真实 callable 调用与日志传递，不是训练样本或任务能力测试。
随后故意更换最终答案、调用不存在的工具、触发工具异常并耗尽预算，核对日志与终止原因。
每项检查记录预期行为和实际状态，不需要为这些小输入报告“成功率”。
时间预算在回调前后检查，不能强制中断阻塞工具；真实工具应有网络或进程超时。

## C. 写一个官方环境适配器

官方环境的 reset/step、观察和奖励可能不符合本仓库 callable 接口。
先写清映射，再实现适配器；不要因为接口较简单而删除原任务难点。

1. `reset(task_id)` 必须恢复任务初始状态，并保存环境版本与初始观察。
2. 将观察转换成策略输入，保存未压缩原始观察或可回查引用。
3. 将模型动作解析为官方动作，校验参数、权限和副作用范围。
4. 执行官方 step 后保存观察、奖励、官方终止字段与工具错误。
5. 将官方验收结果作为最终任务结果，保持 `self_judge` 单独记录。

某些环境在每步返回奖励，某些需要任务结束后独立评分；如实遵循官方协议。
如果本仓库执行器的 final 动作与环境 terminated 语义不同，适配器需要明确区别。
保存解析失败和非法动作；它们应按预先固定规则计入预算，不能静默删除。
沙箱、工具授权、资源约束与环境重置由实际环境适配层负责。

## D. 先验证环境，再测试策略

用官方 reference/oracle 或 gold 补丁确认安装和任务验收可用。
这一运行证明评测环境能工作，不是你的 Agent 取得了相应成绩。
环境失败时先排查，不把安装问题写成模型失败，也不把 gold 成功写成 Agent 成功。

SWE-bench 当前 README 提供 v5 CLI，也保留旧模块入口。
在独立的 SWE-bench checkout 中，按其版本安装后可先阅读：

```bash
swebench --help
swebench eval --help
```

README 中 gold 单实例检查是：

```bash
swebench eval verified --gold -i sympy__sympy-20590 \
    --run-id validate-gold --task-repo ./swe-bench-tasks
```

这里 `./swe-bench-tasks` 需按官方 README 获取；镜像、空间和平台要求也应按该版本准备。
提交自己的预测补丁时使用新的 run ID，避免该基准缓存复用旧补丁结果。
实际任务评测输出和 Agent 轨迹必须一起保存。

Terminal-Bench 当前仓库提供持续基准及发布标签，使用 Harbor。
选择并记录固定发布，避免正式实验使用随时间变化的 `latest`。
旧 Terminal-Bench 1 的入口位于 [terminal-bench-1](https://github.com/harbor-framework/terminal-bench-1)，复现旧实验不要混用新任务数据。
BrowserGym 和 AgentBench 则按各自任务配置及 README 启动；官方依赖变化时锁定相应 commit。

## E. 建立一个预算一致的对照

先固定生成模型与工具，运行无记忆、无反思的基础策略。
再只改变一个部件，例如失败摘要、带来源记忆或计划状态。
保持相同任务、总 token、工具次数、重试机会与环境重置规则。
如果方法需要更多预算，额外报告同预算对照，说明收益与资源变化。

每任务至少保存：task ID、尝试 ID、策略版本、模型版本、起止时间、预算、轨迹位置、官方结果、终止原因。
汇总表包含实际任务数、成功数、成功率、token、工具次数、耗时和环境故障数。
同一任务多次尝试按任务分组；使用配对比较，不把所有重试当成独立新任务。
只运行子集时，标题和结论都写明子集规模与选择规则。

给失败轨迹分类：观察缺失、目标定位、规划、非法动作、工具失败、验证失败、预算耗尽。
抽查每类代表轨迹，分析原因时引用动作和观测证据。
没有实际日志的指标写“未测”，不填示例数字、预置性能或第三方分数。

## F. 可选：从轨迹到训练

完成公开任务基线后，再决定是否进行参数训练。
训练任务与报告任务必须隔离，成功轨迹需经独立验收并保留来源。
SFT 仅对模型生成动作或完成区间计算损失，工具观察不能误作模型目标。
筛选成功轨迹后 SFT 属于模仿学习，不直接等于 on-policy Agent RL。

真正 RL 还需要多轨迹采样、策略版本与行为 log-prob、任务回报、掩码、优化器和训练后验收。
组内奖励无方差、环境故障和策略滞后需要显式处理。
本仓库 [alignment.py](../src/fm_tutorial/alignment.py) 提供损失教学部件，没有把这些组件连接成完整训练系统。
官方进阶入口：[open-instruct](https://github.com/allenai/open-instruct)、[NeMo RL](https://github.com/NVIDIA-NeMo/RL)。

## G. 交付物与验收标准

交付 `manifest`、原创适配器、固定策略配置、原始轨迹、官方结果文件和分析报告。
报告开头明确哪些阶段已运行、哪些尚未运行；训练扩展另附数据卡和真实训练日志。
README 式安装说明需说明硬件、依赖、任务重置与官方评分命令。
验收重点是可复现、预算公平、外部验证、来源隔离和失败诊断，而不是分数必须达到某个虚构目标。
没有 GPU 仍可完成执行器理解与环境集成；没有基准环境资源则不能宣称取得该基准成绩。

返回 [课程导航](../README.md)；源码入口见 [开放训练导读](../references/open-training.md)。
