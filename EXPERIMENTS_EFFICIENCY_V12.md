# V12：统一加速路径与效率复测

大、中、小图采用同一条路径：CPU 估计特征坐标和关系统计，GPU 分块执行完整骨干与传播，GPU 加速读出。CPU 固定 8 线程，节点块 32,768、边块 1,048,576、四跳传播、seed 42。全部节点、边、注意力头与 35 个 ridge 候选均保留。

## 同机完整适配

Intel Core i9-14900HX、RTX 4060 Laptop 8 GB、32 GB RAM。统一路径的两种方法均使用同一 PyTorch 2.5.1+cu121、NumPy 2.2.6、SciPy 1.15.3 环境。图推理采用 FP32，禁用 TF32；读出保持原实现精度：EvoGFM ridge 为 FP64，GraphAny gelss 为 FP32。

适配时间包含原始特征预处理、关系统计、完整骨干/传播、全部读出拟合与验证选型、最终预测；数据和检查点加载另列。每次独立进程执行，不复用跨方法表征。

| 数据集 | 节点数 | EvoGFM 准确率 | GraphAny 准确率 | EvoGFM 秒 | GraphAny 秒 | EvoGFM/GraphAny 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 79.60% | 79.40% | 0.92 | 2.07 | 0.44× |
| ogbn-arxiv | 169,343 | 67.88% | 58.18% | 18.38 | 5.77 | 3.19× |
| ogbn-products | 2,449,029 | 74.13% | 61.04% | 78.98 | 71.93 | 1.10× |

全训练标签下，EvoGFM 在 Cora 更快，在 arxiv 和 products 更慢，三者准确率差分别为 +0.20、+9.71、+13.09 个百分点。不能据此声称所有规模都比 GraphAny 快。

完整适配主表沿用公开验证池选型：EvoGFM 分别使用 500、29,799、39,323 个额外验证标签；GraphAny 固定公开推理路径使用 0 个验证标签。下面单独控制可用标签总数。

## 相同标签总预算：arxiv

从官方训练池按 seed 42 每类最多取 25 个标签，共 996 个。EvoGFM 将其中 200 个留作选型，796 个拟合；GraphAny 将同一组 996 个全部用于拟合。两者均不使用官方验证池标签，测试节点仍为全部官方测试节点。保持上面的统一加速路径。

| 方法 | 拟合标签 | 选型标签 | 总标签 | 准确率 | 完整适配秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EvoGFM | 796 | 200 | 996 | 50.07% | 1.85 |
| GraphAny | 996 | 0 | 996 | 46.99% | 5.63 |

EvoGFM 约快 3.05 倍，准确率高 3.08 个百分点。这是新增的固定总标签预算协议，应单独列为对照，不能当作原论文相同协议的一行。

## 与本地加速前实现比较

前测使用提交 423fa0c 的模块快照、CPU 四线程、PyTorch 2.6.0+cpu；后测为统一 CUDA 八线程路径。此表是整套运行配置的前后变化，包含设备、线程与运行库变化，不能将全部收益归因于某一代码改动。两者使用同一本地源检查点。用户附件的检查点文件与本地文件身份未核实为相同，因此没有用附件时间作为严格前后倍数的分母。

| 数据集 | 加速前秒 | 统一路径秒 | 观察到的加速 | 加速前准确率 | 统一路径准确率 | 全图预测改变节点数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 1.43 | 0.92 | 1.55× | 79.60% | 79.60% | 0 |
| ogbn-arxiv | 33.04 | 18.38 | 1.80× | 67.88% | 67.88% | 0 |
| ogbn-products | 838.25 | 78.98 | 10.61× | 74.13% | 74.13% | 2 |

## 统一路径的分阶段成本

| 数据集 | 预处理秒 | 关系统计秒 | 骨干/传播秒 | 读出与选型秒 | 含加载总秒 | 峰值 RSS GiB | 峰值 CUDA 分配 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cora | 0.345 | 0.004 | 0.191 | 0.352 | 0.939 | 1.139 | 0.161 |
| ogbn-arxiv | 0.156 | 0.023 | 0.587 | 17.582 | 18.430 | 1.872 | 0.427 |
| ogbn-products | 0.563 | 0.332 | 23.087 | 54.792 | 113.702 | 14.271 | 4.279 |

## 代码改变

1. 七个读出特征家族仍各拟合五个正则候选。先只预测验证节点，确定家族和正则后才对全图预测一次，保持家族顺序和 0.1 个百分点替换门槛。
2. 将完整边聚合改为 CSR 稀疏乘法，四跳传播复用同一稀疏矩阵。
3. 固定随机二次特征、RBF 特征和大 ridge 系统可在 CUDA 上计算，ridge 系统保持双精度。最小系统维度不超过 512 时使用 CPU；这条固定调度规则适用于所有图。
4. GPU 骨干按节点块计算特征，按边块计算每个头的分数，对全图执行分段 softmax 和 CSR 聚合。图边及特征坐标保留在 CPU，全部头、层和边均参与。
5. 标签传播与误差修正保留 20 步和全部候选，在 CUDA CSR 上执行。
6. GraphAny 保留作者模型、熵归一化函数、Wisconsin 权重与 CPU gelss；均值传播改为等价 CSR，并在统一路径中使用 CUDA。这是 Windows 移植实现的测量，不是原生 DGL 运行器的速度。

## 持续更新成本

按已有 NC 协议顺序接触 Cora、Citeseer，复用匹配源曲率。每图四次 GGN 更新、完整线搜索，并在更新后重评两张完整图。这里记录已有持续学习方法的成本，没有与 GraphAny 建立持续学习速度优劣结论。

一次性准备及初始评价 3.48 秒；Cora 更新及两图重评 3.65 秒；Citeseer 更新及两图重评 6.15 秒；总计 13.28 秒。
Cora 的四步线搜索均保留原参数；Citeseer 第一步接受更新，之后保持。更新后两图验证准确率各下降 0.4 个百分点，本轮成本测量不构成新的保留或积累增益证据。

## 原图分类统计对照

本轮未重跑图分类，保留用户附件的既有对照。

| 数据集 | EvoGFM | 简单图统计 SVM |
| --- | ---: | ---: |
| DD | 81.75 | 81.66 |
| PROTEINS | 75.38 | 76.24 |
| MUTAG | 84.91 | 86.43 |

## 复现

所有命令从仓库根目录执行。需提供源检查点和原始数据；大图可通过 data/locations.json 复用本机原始目录。CPU 默认路径仍可运行。CUDA 路径需要带 CUDA 的 PyTorch，本轮使用现有 2.5.1+cu121 环境；项目支持 PyTorch 2.5–2.6。

```bash
python -m pip install -e '.[benchmarks]'

python evaluate.py --checkpoint runs/reference/best.pt --nc cora ogbn-arxiv ogbn-products --lp --node-readout advanced --threads 8 --inference-device cuda --kernel-device cuda --split test --output runs/unified_cuda

git clone https://github.com/DeepGraphLearning/GraphAny.git external/GraphAny
git -C external/GraphAny checkout 02c810e9977e89bda66f43ad4e206937af37cf26

python benchmark_efficiency.py --dataset ogbn-arxiv --threads 8 --inference-device cuda --kernel-device cuda --output runs/efficiency/arxiv_evogfm
python benchmark_efficiency.py --dataset ogbn-arxiv --method graphany --device cuda --threads 8 --inference-device cuda --kernel-device cuda --graphany-root external/GraphAny --output runs/efficiency/arxiv_graphany

# Equal total label pool: 25/class, with five/class held out by EvoGFM.
python benchmark_efficiency.py --dataset ogbn-arxiv --shots 25 --holdout-shots 5 --threads 8 --inference-device cuda --kernel-device cuda --output runs/efficiency/budget_evogfm
python benchmark_efficiency.py --dataset ogbn-arxiv --method graphany --device cuda --shots 25 --holdout-shots 5 --threads 8 --inference-device cuda --kernel-device cuda --graphany-root external/GraphAny --output runs/efficiency/budget_graphany
```

本轮为固定单 seed 的开发复测，计时为单次实测，未估计波动区间。PyTorch 2.6 CPU 环境：41 项通过、1 项 CUDA 测试跳过。PyTorch 2.5 CUDA 环境：42 项全部通过。ruff 与差异格式检查通过。

[全部 30 项完成测量及开发路径](results_efficiency_v12.json) 包含 CPU 4/8/16 线程、GPU 读出、全 GPU 状态计算和统一路径；保留全 GPU 状态计算时 arxiv 的小幅下降。products 纯 CPU 后测在标签传播阶段停止，未生成最终结果，没有填入结果表。反演在这些静态适配测量中均未在线更新；持续更新成本单独列出。
