# EvoGFM V8：补齐原表基准

更新：2026-11-07T17:50:26。完成数量：GraphAny 31/31；NCN 4/7；HeaRT 小图 3/3、大图 1/4；GraphCL 8/8。

这轮补跑缺失数据集，沿用冻结的完整 EvoGFM、既定读出候选及 seed 42。Pubmed、Photo、Texas 分别使用排除对应目标图的两源模型，均训练完整 60 轮、240 次更新，按源验证目标选择 checkpoint。Texas 对应 GraphAny 的 texas_4_classes 版本；单节点稀有类别不进入其发布的训练、验证或测试掩码。

新增节点分类的完整测试记录：

| Dataset | Accuracy (%) | Checkpoint |
| --- | --- | --- |
| blogcatalog | 85.3034 | runs/reference/best.pt |
| coauthor-physics | 92.6557 | runs/reference/best.pt |
| dblp | 75.8335 | runs/reference/best.pt |
| deezer | 55.7963 | runs/reference/best.pt |
| full-cora | 59.9652 | runs/reference/best.pt |
| lastfm-asia | 82.1311 | runs/reference/best.pt |
| ogbn-products | 74.1290 | runs/reference/best.pt |
| photo | 92.3364 | runs/loso_photo/best.pt |
| pubmed | 76.8000 | runs/loso_pubmed/best.pt |
| reddit | 91.2805 | runs/reference/best.pt |
| air-brazil | 42.3077 | runs/reference/best.pt |
| air-eu | 48.1250 | runs/reference/best.pt |
| air-us | 53.5135 | runs/reference/best.pt |
| wiki | 74.0561 | runs/reference/best.pt |
| texas4 | 81.0811 | runs/loso_texas/best.pt |

新增图分类：

| Dataset | EvoGFM accuracy (%) | 统计对照 (%) |
| --- | --- | --- |
| COLLAB | 80.74 ± 1.43 | 80.54 ± 1.36 |
| REDDIT-BINARY | 87.25 ± 2.35 | 80.50 ± 2.19 |
| REDDIT-MULTI-5K | 53.47 ± 2.56 | 51.03 ± 2.76 |

大图计算保留全部节点、可见边和官方评估候选。节点传播、注意力各头和读出特征采用分块计算；宽特征的 SVD 通过完整 float64 Gram 矩阵求右奇异空间。没有分图拟合、随机特征替换或减少测试候选。大图链路适应使用四个完整训练掩边视图，每个视图 512 正例、512 均匀非边用于拟合，另 256 正例、256 非边校准；合计 4096/2048 对，训练与校准 pair ID 全局不重叠。它保留已有 18 参数读出及训练内选择，固定平衡抽样在特征计算前完成。

OGB 使用发布 split。Collab 测试可见图加入验证正边；其余任务使用训练边。DDI 无原始属性，沿用无属性 TU 图的 64 维度 one-hot，度上限 63。HeaRT 使用发布的逐正例候选，PPA 同时应用发布的正例子集索引；MRR/Hits 采用平均平局排名。普通 OGB Hits 采用共享负池第 K 高分的严格大于阈值。

本轮不调测试集参数。完整外部对照表见 [BEST_TABLES.md](BEST_TABLES.md)；未完成格保持 —。

原站失效的两个数据集已通过公开论文仓库补齐。[LastFMAsia 原始 NPZ](https://github.com/wuyucheng2002/PPGSL/blob/84b67e2ea709f4740a0cb7b4c11012664bc64f7b/data/LastFMAsia/raw/lastfm_asia.npz) 包含 7,624 节点、128 维特征、18 类；其特征、标签和边与另一仓库的 DGL 缓存逐项一致。[DeezerEurope DGL 缓存](https://github.com/Qy-ds/G-CORMoL/blob/d298647acb960bf09483fc0fd3cdf9ba061c807f/datas/DeezerEurope.bin) 包含 28,281 节点、128 维特征、2 类；边和标签与 LINKX 的独立发布逐项一致。转换只读取缓存中的输入特征、边和标签，不使用这些项目的模型、划分或预测。
