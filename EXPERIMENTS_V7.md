# EvoGFM V7：扩大领域、任务层级与标注稀缺性

同一个 Pubmed、Photo、Texas 源模型（epoch 21），本机 CPU，seed 42。新增五张节点分类图、三套整图分类数据；节点分类累计 16 张图，整图分类累计 5 套。完整读出沿用 V6 的候选族与验证选择规则。

## 1. 新增节点分类

| 新图 | GraphAny 四版本最好公开值 | EvoGFM | 差值 |
| :--- | ---: | ---: | ---: |
| wisconsin | 71.77 | 76.47 | +4.70 |
| cornell | 66.49 | 81.08 | +14.59 |
| chameleon | 62.59 | 69.74 | +7.15 |
| squirrel | 49.70 | 51.87 | +2.17 |
| coauthor-cs | 90.85 | 91.47 | +0.62 |

指标均为准确率（%）。GraphAny 数值取自 [ICLR 2025 原论文 Table 5](https://arxiv.org/html/2405.20445#A4.T5)，每张图选择四个公开训练版本中的最高均值。完整原表保留全部其他模型；这五个点均高于对应 GraphAny 均值。Coauthor-CS 仍低于原表 GCN 的 91.83%。公开均值和我们的单 seed 结果，训练来源和平均方式不同。

Wisconsin/Cornell/Chameleon/Squirrel 使用官方 Geom-GCN split 0；后两者使用原始图，以对应 GraphAny 表。Coauthor-CS 有 18,333 节点、81,894 条无向边、6,805 维输入；按 GraphAny 的 seed-42 分层划分，共 300 个训练节点、9,016 个验证节点、9,017 个测试节点。缓存复用原始数据，不复用其他项目的预测或模型权重。

## 2. 同时少标签和缺边

| 图 | 训练标签数 | 原读出 | 完整读出 | 差值 |
| :--- | ---: | ---: | ---: | ---: |
| chameleon | 25 | 29.82 | 29.82 | +0.00 |
| squirrel | 25 | 24.30 | 22.86 | -1.44 |
| coauthor-cs | 72 | 75.96 | 82.40 | +6.44 |

每类最多 5 个训练节点，只观察 20% 的边；额外使用官方验证标签选择读出。Coauthor-CS 的小类训练池不足 5 个，实际共有 72 个训练标签。两种读出使用同一个源模型和相同标签/边子集。Coauthor-CS 增益为 6.44 点；Chameleon 没有提升，Squirrel 下降 1.44 点。这是自设困难条件下的读出对照。

## 3. 从节点任务迁移到整图分类

| 数据集 | 图数 | 图统计对照 | EvoGFM | GraphCL 公开值 |
| :--- | ---: | ---: | ---: | ---: |
| MUTAG | 188 | 86.67 ± 6.88 | 87.72 ± 5.96 | 86.80 |
| DD | 1178 | 78.86 ± 2.42 | 77.59 ± 3.05 | 78.62 |
| IMDB-BINARY | 1000 | 69.60 ± 4.20 | 70.50 ± 4.98 | 71.14 |
| PROTEINS | 1113 | 77.26 ± 3.09 | 77.18 ± 2.61 | 74.39 |
| NCI1 | 4110 | 72.00 ± 1.98 | 78.39 ± 2.03 | 77.87 |

MUTAG 为 188 张分子图，DD 为 1,178 张蛋白质图（最大 5,748 节点），IMDB-BINARY 为 1,000 张社交图。MUTAG、DD、PROTEINS、NCI1 使用原始节点标签的 one-hot，不用连续属性或边标签。IMDB 没有原始节点属性，以节点度的 64 维 one-hot 为输入，度超过 63 时合并到最后一维；图标签只训练目标 SVM。

源模型全部冻结。外十折测量，内五折选择 SVM；标准化只拟合对应训练折。统计对照选 7 个 C，完整读出选 3 个块权重 × 7 个 C。± 为外折标准差。[GraphCL 原表](https://arxiv.org/html/2010.13902#S5.T4) 的预训练、划分与统计方式不同。MUTAG 高于 GraphCL，低于 InfoGraph 89.01%；DD、IMDB 仍低于 GraphCL。PROTEINS/NCI1 完整标签值沿用 V6。

## 4. 每类只标 5 / 20 张图

| 数据集 | 5-shot 图统计 | 5-shot EvoGFM | 20-shot 图统计 | 20-shot EvoGFM | 完整标签 EvoGFM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MUTAG | 81.84 | 80.20 | 80.88 | 80.94 | 87.72 |
| DD | 67.07 | 69.36 | 73.86 | 73.69 | 77.59 |
| IMDB-BINARY | 56.80 | 58.00 | 60.20 | 59.90 | 70.50 |
| PROTEINS | 64.79 | 63.62 | 70.62 | 69.19 | 77.18 |
| NCI1 | 52.85 | 53.41 | 57.23 | 57.01 | 78.39 |

这里的 shot 是整张图的类别标注数量。每个外层训练折内，以 seed 42+fold 按类别抽取支持集；其余训练标签不使用。内五折只使用支持集，外层测试折从不参与选择。两种读出使用相同支持图，五套数据均跑完十个外折。完整标签列与低标注列使用相同外层划分。

5-shot 的 DD 和 IMDB 上完整读出优于统计对照；PROTEINS、MUTAG 出现下降，NCI1 接近随机水平。这说明整图的极低标注迁移仍是薄弱环节；该任务补充了完整标签 SVM 评估没有覆盖的困难条件。

## 5. 六领域连续适应

Cora → Citeseer → Wisconsin → Cornell → Actor → Amazon-ratings；Minesweeper、Tolokers 保留为未来领域。static 不更新，cumulative 保留系数和曲率，reset 每个领域从源状态重启。NC 沿用固定原读出；LP 为原生 masked-edge AUC。这些是 validation 测量，与公开 NC/NCN/HeaRT 测试分别报告。

| 任务 / 对照 | 最终均分 | 遗忘百分点 | 未来 Minesweeper 变化 | 未来 Tolokers 变化 |
| :--- | ---: | ---: | ---: | ---: |
| nc static | 62.697 | 0.000 | +0.000 | +0.000 |
| nc cumulative | 62.749 | 0.370 | -0.105 | -0.259 |
| nc reset | 63.307 | 0.450 | +0.295 | -0.210 |
| lp static | 84.233 | 0.000 | +0.000 | +0.000 |
| lp cumulative | 84.559 | 0.029 | -0.060 | -0.141 |
| lp reset | 84.381 | 0.171 | -0.021 | -0.067 |

遗忘按前五个领域的学习后最高分减最终分平均。LP cumulative 最终均分高于 static/reset，遗忘低于 reset；原两张未来领域增益均为负。NC cumulative 最终均分略高于 static、低于 reset。稳定的知识积累仍未得到支持。


进一步复用同一批已保存的七个前缀状态，增加从未参与逆映射更新的 Chameleon、Squirrel。重放过程不更新任何参数；原两张未来图的轨迹与主运行逐项一致。下表为最终前缀相对初始前缀的变化（百分点）。

| 任务 / 对照 | Minesweeper | Tolokers | Chameleon | Squirrel |
| :--- | ---: | ---: | ---: | ---: |
| nc static | +0.000 | +0.000 | +0.000 | +0.000 |
| nc cumulative | -0.105 | -0.259 | +0.686 | +0.721 |
| nc reset | +0.295 | -0.210 | +2.606 | +1.382 |
| lp static | +0.000 | +0.000 | +0.000 | +0.000 |
| lp cumulative | -0.060 | -0.141 | -0.044 | -0.020 |
| lp reset | -0.021 | -0.067 | -0.019 | -0.001 |

NC 在每个前缀用未来图的官方训练标签重新拟合固定 ridge 评价头，验证标签测量分数；这些标签不更新共享逆映射。LP 只测量原生 held-out 边的 AUC。Chameleon/Squirrel 的 NC 有正收益，但低于 reset，因此还没有证明累积状态带来的独立优势。所有未来图和全部前缀一并保留。


## 6. 官方难负例预测

Cora 官方难负例测试：MRR 16.97%，Hits@10 35.48%。配置先在官方 validation 上选定，再测试；每条正例配 500 条官方难负例。 Uniform/mixed 的验证 MRR 分别为 16.32/14.61%，因此选 uniform。测试 MRR 高于 NCN 的 14.66%，低于 GAE 的 18.32%；没有测试验证落选的 mixed 配置。

Pubmed 是源预训练图，本轮未把它作为未见目标的链路预测实验。Citeseer 的既有官方 HeaRT 结果保留。

## 计算修复与运行

高维节点读出仍求解同一个中心化 ridge 系统，仅把预测的 float64 运算分为 1,024 行一块，以减少 Coauthor-CS 内存占用。DD 的 float32 特征分解有不收敛，IMDB 两张图有 float32 非有限谱；这两种情况均用 float64 重解原对称系统，没有丢弃图。完整数据和全部外折保留。

```bash
python prepare.py --datasets wisconsin cornell chameleon squirrel coauthor-cs
python evaluate.py --checkpoint runs/reference/best.pt --nc wisconsin cornell chameleon squirrel coauthor-cs --lp --node-readout advanced --output runs/v7_nodes
# 验证配置后加 --split test；困难条件同时加 --shots 5 --edge-rate 0.2。
python classify_graphs.py --checkpoint runs/reference/best.pt --dataset IMDB-BINARY --balanced-readout --output runs/imdb_v7
python classify_graphs.py --checkpoint runs/reference/best.pt --dataset IMDB-BINARY --balanced-readout --shots 5 --embedding-cache runs/imdb_v7/embeddings.pt --output runs/imdb-binary_5shot_v7
python continual.py --checkpoint runs/reference/best.pt --stream cora citeseer wisconsin cornell actor amazon-ratings --future minesweeper tolokers --task both --nc-metric auto --output runs/continual_v7
python evaluate_future.py --checkpoint runs/reference/best.pt --states runs/continual_v7 --datasets minesweeper tolokers chameleon squirrel --output runs/future_v7
```

新训练生成的模型通常在 `runs/source/best.pt`；可替换上述 checkpoint。原始逐折结果、支持图索引、所选参数和完整连续轨迹见 [results_v7.json](results_v7.json)。各节点图的配置在访问测试标签前保存到 `runs/expanded_v7/FROZEN_*.json`。历轮最好与完整公开基线表见 [BEST_TABLES.md](BEST_TABLES.md)。
