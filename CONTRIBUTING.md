# 如何完善教材和实践

先阅读 [设计边界](docs/design.md)、[覆盖表](docs/coverage.md) 和 [来源表](references/sources.md)。新增教程应讲清基础概念、场景、数学与代码的连接，并给出一手资料。新增实验应使用已有公开 benchmark 或有明确来源的真实数据，不添加伪造训练曲线或把手造题当基准。

修改核心数值代码时，先固定能暴露问题的行为测试，再修改实现。检查 completion target mask、reference detach、old-policy 概率、验证集隔离与 checkpoint 恢复等语义。文档和代码变更必须保持命令与函数一致。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_links.py
python -m compileall -q src scripts
```

测试中的短文本、矩阵和控制流输入是程序诊断，不是学习效果实验。报告应区分结构正确性、实测训练行为和正式 benchmark 能力。修改 GPU kernel、分布式或 serving 时，记录实际硬件、版本、计时同步与数据规模。

提交真实测量 CSV 时同时提供来源、代码 commit、数据 revision、配置、单位和评估过程；在没有实验结果时给出可运行命令和待验证问题，不写“已提升”。第三方代码复用需对应许可和署名，无许可仓库只链接。
