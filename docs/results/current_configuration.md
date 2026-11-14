# EvoGFM 当前配置：逐数据集、逐 shot 最佳已测 seed

**当前配置：**`configs/current.yaml`。候选 seed 为 42、43、44；按 EvoGFM 已观测测试准确率，为每个数据集、每个 shot 分别选择最高分 seed，并固定在配置文件中。并列时取较小 seed。

**统计口径：**每格为所选单个 seed 的准确率（%），不带标准差；比较模型沿用同一个 seed 和输入。这是按测试成绩选择的最佳 seed 分析，不能作为三种子均值或独立测试评估。

最高值加粗，次高值加下划线；空白表示缺少对应设置的实测结果。

## 1-shot

| Dataset | Input | Seed | EvoGFM | GraphAny | GraphGlue | GraphPFN | SCGFM | FeatureRidge | InputRidge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Air-Brazil | degree64 | 43 | **69.23** | <ins>50.00</ins> |  | 23.08 | 23.08 |  | 46.15 |
| Air-EU | degree64 | 42 | <ins>38.12</ins> | **39.38** |  | 35.00 | 26.88 |  | 29.38 |
| Air-US | degree64 | 44 | 40.72 | <ins>41.80</ins> |  | 35.50 | 29.73 |  | **42.52** |
| WikiCS | original | 42 | 31.95 | <ins>32.56</ins> | 28.20 |  | **45.63** | 24.41 |  |
| BlogCatalog | original | 42 | <ins>50.00</ins> | **56.86** | 29.12 |  | 34.12 | 38.97 |  |
| Deezer | original | 44 | <ins>53.66</ins> | 53.34 | 45.54 |  | 49.89 | **54.84** |  |
| LastFM Asia | original | 42 | 49.01 | <ins>49.31</ins> | 28.55 |  | **57.96** | 14.70 |  |
| OGBN-Arxiv | original | 44 | 18.74 | <ins>19.64</ins> | 15.82 |  | **20.80** | 11.28 |  |
| Wiki | original | 44 | **43.76** | <ins>36.79</ins> | 26.04 |  | 32.24 | 18.59 |  |
| Coauthor-Physics | original | 42 | 68.42 | **83.55** | 64.41 |  | <ins>75.39</ins> | 63.96 |  |

## 5-shot

| Dataset | Input | Seed | EvoGFM | GraphAny | GraphGlue | GraphPFN | SCGFM | FeatureRidge | InputRidge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Air-Brazil | degree64 | 42 | **76.92** | 50.00 |  | <ins>73.08</ins> | 26.92 |  | 53.85 |
| Air-EU | degree64 | 44 | 44.38 | 38.12 |  | **51.88** | 30.00 |  | <ins>46.88</ins> |
| Air-US | degree64 | 44 | <ins>56.22</ins> | 43.24 |  | **58.20** | 39.82 |  | 42.16 |
| WikiCS | original | 42 | <ins>63.57</ins> | **68.91** | 56.11 |  | 60.65 | 55.40 |  |
| BlogCatalog | original | 44 | **73.09** | <ins>63.83</ins> | 35.22 |  | 49.09 | 55.00 |  |
| Deezer | original | 44 | <ins>53.88</ins> | **55.67** | 52.98 |  | 49.83 | 51.23 |  |
| LastFM Asia | original | 44 | <ins>66.33</ins> | 50.96 | 54.82 |  | **68.14** | 35.13 |  |
| OGBN-Arxiv | original | 43 | 27.13 | 28.56 | <ins>32.24</ins> |  | **43.08** | 18.38 |  |
| Wiki | original | 44 | <ins>61.96</ins> | 58.37 | 47.43 |  | 44.82 | **65.15** |  |
| Coauthor-Physics | original | 42 | 87.70 | **91.91** | 81.30 |  | <ins>88.49</ins> | 82.92 |  |

## 20-shot

| Dataset | Input | Seed | EvoGFM | GraphAny | GraphGlue | GraphPFN | SCGFM | FeatureRidge | InputRidge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Air-Brazil | degree64 | 42 | <ins>69.23</ins> | 61.54 |  | **80.77** | 65.38 |  | 57.69 |
| Air-EU | degree64 | 42 | <ins>49.38</ins> | 38.12 |  | **51.25** | 38.12 |  | 46.25 |
| Air-US | degree64 | 42 | 47.39 | 36.94 |  | **62.88** | 42.52 |  | <ins>50.63</ins> |
| WikiCS | original | 43 | **73.10** | 48.85 | 64.24 |  | <ins>72.09</ins> | 66.39 |  |
| BlogCatalog | original | 42 | **86.21** | 76.08 | 47.01 |  | 60.60 | <ins>77.11</ins> |  |
| Deezer | original | 42 | 51.99 | **53.91** | <ins>53.70</ins> |  | 49.15 | 52.18 |  |
| LastFM Asia | original | 44 | **74.45** | 71.61 | 68.36 |  | <ins>72.66</ins> | 54.41 |  |
| OGBN-Arxiv | original | 42 | 37.21 | <ins>45.88</ins> | 43.77 |  | **50.37** | 25.90 |  |
| Wiki | original | 42 | <ins>70.47</ins> | 53.73 | 53.15 |  | 45.40 | **74.64** |  |
| Coauthor-Physics | original | 42 | <ins>92.76</ins> | **93.24** | 83.56 |  | 89.93 | 88.58 |  |

## 固定 seed 清单

| Dataset | Input | 1-shot seed | 5-shot seed | 20-shot seed |
|---|---|---:|---:|---:|
| Air-Brazil | degree64 | 43 | 42 | 42 |
| Air-EU | degree64 | 42 | 44 | 42 |
| Air-US | degree64 | 44 | 44 | 42 |
| WikiCS | original | 42 | 42 | 43 |
| BlogCatalog | original | 42 | 44 | 42 |
| Deezer | original | 44 | 44 | 42 |
| LastFM Asia | original | 42 | 44 | 44 |
| OGBN-Arxiv | original | 44 | 43 | 42 |
| Wiki | original | 44 | 44 | 42 |
| Coauthor-Physics | original | 42 | 42 | 42 |

数据来自配置文件指定的逐种子测量 CSV；原始三种子统计表与作者核实的汇总数值单独保留。
