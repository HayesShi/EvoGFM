# 扩大知识积累实验

2026-11-08；seed 42；本轮验证和测试全部完成。

## 1. 做了什么

- 将未来目标扩到 20 个数据条目，覆盖引用、网页、社交、商品、音乐、航空和工业网络。Tolokers 与 tolokers-2 来自同一物理图，合计为 19 张物理图。
- 对每个未来目标测每类最多 1、5、20 个标签，以及全部官方训练标签。各预算的支持集固定且嵌套；某类训练节点不足预算时全部使用。
- 比较六种方案：原六图流的正序与倒序、完整图 NC 更新、混合领域流、同配图流，以及增加源训练 NC 权重的基座。进入源训练或更新流的图不作为该方案的未来目标。
- 所有方案保持反演主线：冻结两层 Transformer，在线更新 32 个反演系数，8 个偏置固定，每张学习图做 4 次更新尝试。
- 每个前缀在未来图上重新拟合一个 ridge 评价头，alpha 固定为 0.1。未来图的标签不更新共享系数；其未标注特征和边用于正常的图表示计算。

源图仍为 Pubmed、Photo、Texas。主基座源训练 60 个 epoch，按源验证选中 epoch 21；替代基座将 NC 损失权重从 0.2 增到 1.0，选中 epoch 26。本轮复用这两个现成检查点。

原流正序：Cora → Citeseer → Wisconsin → Cornell → Actor → Amazon-ratings。

原流倒序：Amazon-ratings → Actor → Cornell → Wisconsin → Citeseer → Cora。

混合流：Cornell → Wisconsin → Actor → Amazon-ratings → Roman-empire → Computers。

同配流：Cora → Citeseer → Computers → Coauthor-CS。

## 2. 统一方案怎么选

用六种方案共同可作未来目标、且验证节点不少于 100 的多类分类图选择：Air-EU、Air-US、BlogCatalog、Chameleon、DBLP、LastFM-Asia、Squirrel、Wiki、WikiCS。

每图每预算等权，比较 1/5/20-shot 的最终前缀平均验证准确率，不挑中间最好前缀。AP、AUC 不混入该准确率均值。选择在本轮测试分数被读取前固定；原正序也固定为顺序敏感性对照。全部 20 个目标、四档预算均完成测试。

| 方案 | 最终验证 Acc 均值 | 相对初始 | 相对 reset |
| --- | ---: | ---: | ---: |
| 原基座、原流倒序 | 49.004 | +1.206 | +0.691 |
| 原基座、原流正序 | 48.987 | +1.189 | +0.365 |
| 原基座、混合流、完整图更新 | 48.635 | +0.837 | +1.513 |
| 源 NC 权重 1.0、原流正序、完整图更新 | 48.400 | +0.261 | +0.158 |
| 原基座、同配流、完整图更新 | 47.807 | +0.008 | −0.509 |
| 原基座、原流正序、完整图更新 | 47.465 | −0.333 | −0.861 |

倒序与正序的整组成绩几乎持平。混合流相对 reset 的收益更大，但最终绝对成绩更低，因此主结果沿用原基座和原倒序流。替代基座在完整图更新下优于原基座的完整图更新，但仍没有超过原来的掩边更新方案；不能把不同更新视图之间的差距单独归因于源损失权重。

## 3. 验证信号对应的测试结果

以下六项对应本轮测试前已经列出的验证信号，均使用同一倒序最终状态。数值为测试准确率；差值为百分点。

初始：未学六张图。reset：每次从源状态重新开始，最终只保留最后一张学习图的适应。累积：接着之前的状态依次学完六张图。

| 未来目标 | 标签预算 | 实际标签总数 | 初始 | reset | 累积 | Δ初始 | Δreset |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Computers | 每类最多 1 | 10 | 32.423 | 32.586 | 38.636 | +6.213 | +6.051 |
| BlogCatalog | 每类最多 1 | 6 | 44.287 | 45.666 | 49.448 | +5.162 | +3.783 |
| BlogCatalog | 每类最多 5 | 30 | 66.391 | 66.706 | 68.716 | +2.325 | +2.009 |
| DBLP | 全部官方训练标签 | 80 | 70.968 | 71.501 | 73.395 | +2.427 | +1.894 |
| LastFM-Asia | 每类最多 20 | 208 | 73.183 | 73.678 | 74.890 | +1.707 | +1.211 |
| Chameleon | 每类最多 20 | 100 | 42.105 | 41.447 | 43.421 | +1.316 | +1.974 |

Computers 和 BlogCatalog 的 1-shot 测试集分别有 6,776 和 2,538 个节点。

正序时，这六项相对初始也全部提高；相对 reset，Computers 1-shot 为 +0.192 点，BlogCatalog 1-shot 为 +0.670 点，LastFM-Asia 20-shot 为 −0.083 点。收益大小仍受学习顺序影响。

## 4. 所有任务的整体情况

下表只使用统一倒序方案的测试结果。“同时胜出”要求累积同时高于初始与 reset；每个数据条目在该预算内计一次。

| 指标 | 标签/类 | 任务数 | 同时胜出数 | 相对初始平均变化 |
| --- | --- | ---: | ---: | ---: |
| Accuracy | 1 | 13 | 8 | −0.405 |
| Accuracy | 5 | 13 | 6 | +0.538 |
| Accuracy | 20 | 13 | 5 | +0.770 |
| Accuracy | 全部 | 13 | 5 | +0.802 |
| AUC | 1 | 4 | 0 | −0.502 |
| AUC | 5 | 4 | 1 | −0.620 |
| AUC | 20 | 4 | 2 | +0.149 |
| AUC | 全部 | 4 | 2 | +0.129 |
| AP | 1 | 3 | 2 | +0.107 |
| AP | 5 | 3 | 1 | −0.432 |
| AP | 20 | 3 | 1 | −0.748 |
| AP | 全部 | 3 | 1 | −0.028 |

最突出的个例集中在 1-shot，但不能得出“标签越少一定涨得越多”。例如 Computers 全训练标签时下降 0.221 点，BlogCatalog 全训练标签时下降 0.039 点。

明显下降也保留：Air-Brazil 1-shot 为 −19.231 点，Air-EU 1-shot 为 −3.750 点，Chameleon 5-shot 为 −3.070 点，Tolokers 5-shot AUC 为 −1.823 点，Squirrel 1-shot 为 −1.633 点。Air-Brazil 的验证/测试只有 25/26 个节点；它在验证集出现的大涨没有保住，不能作为主打证据。

原来的工业目标在全训练标签下仍然没有普遍改善：artnet-exp AP −0.141，city-reviews AP −0.075，tolokers-2 AP +0.131 点。本轮固定 alpha=0.1，之前报告按验证集选 alpha，两轮头协议不同，不能覆盖旧表数值。

20-shot 与全部训练标签在部分图上是同一组节点，不是独立重复实验。AP 表中的 tolokers-2 与 AUC 表中的 Tolokers 是同一物理图的不同特征/划分版本。

## 5. 找到的规律和仍未解决的部分

- 更有说服力的故事是：已有跨域经验可以明显帮助部分少标签新图。Computers、BlogCatalog 的提升同时超过初始和 reset，并在正反两种顺序下出现。
- 不同目标需要不同标签量才能受益：BlogCatalog 在 1/5-shot 明显提升，LastFM-Asia 在 20-shot 更清楚。新图完全没有参与共享系数更新。
- 仅靠同配率不能判断是否有效。训练标签可见边上的同配率，Computers 约 0.782，BlogCatalog 约 0.352，两者均受益；这只是描述性对照，不是因果机制证明。
- 知识保留没有因此自动变强。复用同一状态的固定旧头回测：倒序累积 62.500%，reset 62.811%，仍低 0.311 点；正序累积相对 reset 高 0.654 点。
- 本轮是新增的少标签知识积累协议，不能直接加到别人的静态 GFM 原论文主表。它补强积累证据，公开主任务的原表比较仍见 BEST_TABLES.md。
- 所有结果为单 seed 42，当前没有多 seed 的波动或显著性结论。

## 6. 文件与复跑

- [全部候选、预算和正负结果表](ACCUMULATION_TABLES.md)
- [全部 11,712 条测量、配置、支持节点、源模型信息和各前缀系数](results_accumulation.json)
- [先前固定头保留及机制实验](EXPERIMENTS_345.md)

已有源检查点和六图状态时，可直接运行：

```bash
# 默认评估验证集的所有前缀；预算是每类标签上限，0 为全部训练标签。
python explore_accumulation.py --states runs/round345/forward runs/round345/reverse4 --datasets computers blogcatalog lastfm-asia chameleon dblp --output runs/accumulation_validation

# 固定状态后测测试集，仅测初始和最终前缀。
python explore_accumulation.py --states runs/round345/forward runs/round345/reverse4 --datasets computers blogcatalog lastfm-asia chameleon dblp --split test --final-only --output runs/accumulation_test

# 完整实验目录齐备后重新导出所有结果表。
python collect_accumulation.py
```

`--nc-view full` 可用于 `continual.py` 比较完整图 NC 更新；默认仍为 `masked`。宽特征的 CPU 传播改为精确稀疏计算，减少边数×特征维度的中间内存；两跳传播的前向与梯度一致性测试通过。代码共 38 项测试通过，Black 与 Ruff 检查通过。
