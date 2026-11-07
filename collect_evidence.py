"""Export all measured group3/4/5 results, without selecting winning prefixes."""

import argparse
import json
from pathlib import Path
from evogfm.data import save_json

CONTROLS = ["static", "reset", "cumulative", "reset-curvature"]
ORDERS = ["forward", "reverse4"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
            "",
        ]
    )


def collect(args):
    root = Path(args.runs)
    ppa_path = root / "ppa/results.json"
    ppa_results = (
        read(ppa_path) if ppa_path.exists() else dict(rows=[], status="running")
    )
    streams = {}
    for order in ORDERS:
        path = root / order
        streams[order] = dict(
            fixed=read(path / "retention/retention.json"),
            nc=read(path / "nc_validation.json"),
            lp=read(path / "lp_validation.json"),
            graphland=read(path / "graphland_future/results.json"),
            old_nc=read(path / "old_future/nc_validation.json"),
            old_lp=read(path / "old_future/lp_validation.json"),
        )
        if len(streams[order]["graphland"]["rows"]) != 84:
            raise ValueError("Incomplete future trajectories")
    mechanism = dict(
        nc={
            variant: read(root / folder / f"test_{ablation}.json")
            for variant, folder, ablation in [
                ("full", "nc_full", "none"),
                ("zero-inverse", "nc_zero", "zero-inverse"),
                ("propagation", "nc_propagation", "propagation"),
            ]
        },
        regression=read(root / "regression/results.json"),
        mutag=read(root / "mutag/results.json"),
        episode=read(root / "mutag/episode.json"),
        ppa=ppa_results,
        ppa_full=read(Path("runs/v8/heart_ogbl-ppa/results.json")),
    )
    complete = len(mechanism["ppa"]["rows"]) == 4
    if not complete and not args.partial:
        raise ValueError("PPA candidates not fully scored")
    legacy = {
        task: read(Path("runs/continual_v7") / f"{task}_validation.json")
        for task in ("nc", "lp")
    }
    save_json(
        Path("results_345.json"),
        dict(
            complete=complete,
            configuration=read(root / "configuration.json"),
            streams=streams,
            mechanism=mechanism,
            legacy_v7=legacy,
        ),
    )
    report = [
        "# 第 3、4、5 组实验结果",
        "",
        f'2026-11-07；seed 42；源检查点 epoch {mechanism["nc"]["full"]["source_epoch"]}。',
        (
            "状态：全部完成。"
            if complete
            else "状态：第 3、4 组完成；第 5 组 PPA 消融仍在运行，未填入未完成成绩。"
        ),
        "",
        "两种学习顺序、四个控制、全部前缀均报告。保留及旧未来图使用验证集；新增 GraphLand 未来图报告验证和测试；机制表使用公开测试集。",
        "旧 V7 对小图实际只做 1 次更新；本轮对每张图恰好做 4 次，训练批次不足时循环使用。",
        "因此两条持续流已全部重跑，旧 V7 单独保留，没有混入新表。",
        "",
        "## 3. 知识保留",
        "",
        "正序：Cora → Citeseer → Wisconsin → Cornell → Actor → Amazon-ratings；倒序完全反向。",
        "NC 的固定头在该图首次学完时拟合，此后保留权重、中心和类别映射；重拟合头另列。",
        "LP 全程使用同一源头，直接测共享系数变化后的表现。遗忘和 BWT 的单位均为百分点。",
        "系数与曲率的数值数组为 13,120 字节/任务（NC、LP 各一份，不含边界和元数据）；固定旧 NC 头合计 6.12 MiB/顺序（四个控制），由评估器保存。",
        "",
    ]
    nc_rows, lp_rows = [], []
    for order, data in streams.items():
        for kind in CONTROLS:
            fixed = data["fixed"]["summary"][kind]
            nc = data["nc"]["summary"][kind]
            lp = data["lp"]["summary"][kind]
            nc_rows.append(
                [
                    order,
                    kind,
                    f'{100*fixed["final_average"]:.3f}',
                    f'{100*fixed["old_domains_final_average"]:.3f}',
                    f'{100*nc["final_average"]:.3f}',
                    f'{100*fixed["forgetting"]:.3f}',
                    f'{100*fixed["backward_transfer"]:+.3f}',
                ]
            )
            lp_rows.append(
                [
                    order,
                    kind,
                    f'{100*lp["final_average"]:.3f}',
                    f'{100*sum(lp["matrix"][-1][:-1])/5:.3f}',
                    f'{100*lp["forgetting"]:.3f}',
                    f'{100*lp["backward_transfer"]:+.3f}',
                ]
            )
    order_deltas = [
        100
        * (
            streams[order]["fixed"]["summary"]["cumulative"]["final_average"]
            - streams[order]["fixed"]["summary"]["reset"]["final_average"]
        )
        for order in ORDERS
    ]
    novel_gains, family_gains = [], []
    for data in streams.values():
        for name in ["artnet-exp", "city-reviews", "tolokers-2"]:
            last = {
                r["control"]: r["test_percent"]
                for r in data["graphland"]["rows"]
                if r["dataset"] == name and r["prefix"] == 6
            }
            (family_gains if name == "tolokers-2" else novel_gains).append(
                last["cumulative"] - last["static"]
            )
    zero_wins = sum(
        z["accuracy_percent"] > f["accuracy_percent"]
        for z, f in zip(
            mechanism["nc"]["zero-inverse"]["rows"], mechanism["nc"]["full"]["rows"]
        )
    )
    graph_values = [r["test_percent"] for r in mechanism["mutag"]["rows"]]
    regression_values = {
        r["control"]: r["test_percent"] for r in mechanism["regression"]["rows"]
    }
    if complete:
        ppa_test = {
            r["variant"]: r for r in mechanism["ppa"]["rows"] if r["split"] == "test"
        }
        ppa_full = next(
            r for r in mechanism["ppa_full"]["rows"] if r["split"] == "test"
        )
        ppa_note = "- PPA HeaRT：" + "；".join(
            f"完整方法相对 {kind}，MRR {ppa_full['mrr_percent']-ppa_test[kind]['mrr_percent']:+.3f}、Hits@20 {ppa_full['hits20_percent']-ppa_test[kind]['hits20_percent']:+.3f} 个百分点"
            for kind in ["zero-inverse", "propagation"]
        )
    else:
        ppa_note = "- PPA 消融正在运行，结论待完整测试评分。"
    report += [
        table(
            [
                "顺序",
                "控制",
                "NC固定头六域均值",
                "旧五域均值",
                "NC重拟合六域均值",
                "固定头遗忘",
                "固定头BWT",
            ],
            nc_rows,
        ),
        table(["顺序", "控制", "LP六域AUC", "旧五域AUC", "遗忘", "BWT"], lp_rows),
        "static 的零遗忘与其新域成绩一起看；cumulative 的优势须同时对照 reset。",
        "",
        "## 4. 知识积累",
        "",
        "未来图只拟合评价头，不更新共享系数；各前缀使用相同训练标签与五个 ridge 正则，按验证 AP 选择。",
        "下表为最终前缀的测试 AP。static 等于初始状态；完整验证/测试曲线见矩阵文件。tolokers-2 是已知图族扩展。",
        "",
    ]
    future_rows = []
    for order, data in streams.items():
        rows = data["graphland"]["rows"]
        for name in ["artnet-exp", "city-reviews", "tolokers-2"]:
            last = {
                r["control"]: r["test_percent"]
                for r in rows
                if r["dataset"] == name and r["prefix"] == 6
            }
            future_rows.append(
                [
                    order,
                    name,
                    *[f"{last[k]:.3f}" for k in CONTROLS],
                    f'{last["cumulative"]-last["static"]:+.3f}',
                    f'{last["cumulative"]-last["reset"]:+.3f}',
                ]
            )
    report += [
        table(["顺序", "未来图", *CONTROLS, "累积−初始", "累积−reset"], future_rows),
        "旧 Minesweeper、Tolokers、Chameleon、Squirrel 的 NC/LP 全部曲线同时保留，指标分别列出。",
        "",
        "## 5. 机制消融",
        "",
        "各变体重新拟合目标读出，使用相同候选家族和验证/支持集选择规则。所有结果为测试值，越高越好。",
        "",
    ]
    ablation_rows = []
    for kind in ["full", "zero-inverse", "propagation", "statistics"]:
        if kind == "statistics":
            cells = ["—"] * 4
        else:
            nc = {
                r["dataset"]: r["accuracy_percent"]
                for r in mechanism["nc"][kind]["rows"]
            }
            ppa = next(
                (
                    r
                    for r in (
                        mechanism["ppa_full"]["rows"]
                        if kind == "full"
                        else mechanism["ppa"]["rows"]
                    )
                    if r["split"] == "test" and (kind == "full" or r["variant"] == kind)
                ),
                None,
            )
            regression = next(
                r for r in mechanism["regression"]["rows"] if r["control"] == kind
            )
            cells = [
                f'{nc["cora"]:.3f}',
                f'{nc["roman-empire"]:.3f}',
                f'{ppa["mrr_percent"]:.3f}' if ppa else "运行中",
                f'{regression["test_percent"]:.3f}',
            ]
        graph = next(r for r in mechanism["mutag"]["rows"] if r["variant"] == kind)
        ablation_rows.append([kind, *cells, f'{graph["test_percent"]:.3f}'])
    report += [
        table(
            [
                "变体",
                "Cora Acc",
                "Roman Acc",
                "PPA HeaRT MRR",
                "city-roads-M 100×R²",
                "MUTAG AUROC",
            ],
            ablation_rows,
        ),
        "PPA 使用每 split 全部 100,000 个发布正例 × 500 个难负例；两项消融只共用无标签结构计算，各自拟合校准头。",
        "完整 PPA 数值复用同检查点、同完整候选的 V8 测量。MUTAG 使用 GP2F 作者采样代码：169 张测试图、19 张支持图，支持集内部选 SVM。",
        "这是一个 seed42 episode，不能当作多 episode 平均。新工业回归保留源状态；未新增回归持续状态。",
        "",
        "## 这些结果说明什么",
        "",
        f"- 保留：固定头 NC 的 cumulative 相对 reset，正序 {order_deltas[0]:+.3f}、倒序 {order_deltas[1]:+.3f} 个百分点。",
        f"- 积累：两张新领域图 × 两顺序，最终前缀有 {sum(v>0 for v in novel_gains)}/4 项高于初始状态；已知图族 tolokers-2 的变化为 {family_gains[0]:+.3f}、{family_gains[1]:+.3f} 个百分点。",
        f"- 机制：NC 两图中去反演有 {zero_wins}/2 项高于完整方法；MUTAG 四变体极差 {max(graph_values)-min(graph_values):.3f} 个百分点。道路回归完整方法相对传播为 {regression_values['full']-regression_values['propagation']:+.3f}，相对去反演为 {regression_values['full']-regression_values['zero-inverse']:+.3f}。",
        ppa_note,
        "",
        "## 原始测量与复跑",
        "",
        "- [全部测量、配置和 episode ID](results_345.json)",
        "- [所有回测矩阵与未来曲线](EVIDENCE_MATRICES.md)",
        "- `python run_evidence.py --groups 3 4 5 --threads 3`",
        "",
    ]
    Path("EXPERIMENTS_345.md").write_text("\n".join(report), encoding="utf-8")
    appendix = [
        "# 完整回测矩阵与未来曲线",
        "",
        "所有数值乘 100；未到达的固定头任务记为 —，不填零。",
        "",
    ]
    for order, data in streams.items():
        stream = data["nc"]["settings"]["stream"]
        for kind in CONTROLS:
            for label, matrix in [
                ("NC fixed", data["fixed"]["summary"][kind]["matrix"]),
                ("NC refit", data["nc"]["summary"][kind]["matrix"]),
                ("LP source head", data["lp"]["summary"][kind]["matrix"]),
            ]:
                appendix += [
                    f"## {order} / {kind} / {label}",
                    "",
                    table(
                        ["前缀", *stream],
                        [
                            [
                                prefix,
                                *[
                                    "—" if value is None else f"{100*value:.3f}"
                                    for value in row
                                ],
                            ]
                            for prefix, row in enumerate(matrix)
                        ],
                    ),
                ]
        for name in ["artnet-exp", "city-reviews", "tolokers-2"]:
            for split in ["validation", "test"]:
                lookup = {
                    (r["control"], r["prefix"]): r[f"{split}_percent"]
                    for r in data["graphland"]["rows"]
                    if r["dataset"] == name
                }
                appendix += [
                    f"## {order} / {name} / {split} AP",
                    "",
                    table(
                        ["控制", *map(str, range(7))],
                        [
                            [kind, *[f"{lookup[kind,p]:.3f}" for p in range(7)]]
                            for kind in CONTROLS
                        ],
                    ),
                ]
        for task in ("nc", "lp"):
            for name in ["minesweeper", "tolokers", "chameleon", "squirrel"]:
                old = data[f"old_{task}"]
                metric = old["trajectories"]["static"][0]["scores"][name]["metric"]
                appendix += [
                    f"## {order} / {name} / {task} validation {metric}",
                    "",
                    table(
                        ["控制", *map(str, range(7))],
                        [
                            [
                                kind,
                                *[
                                    f"{100*v:.3f}"
                                    for v in old["summary"][kind]["future_curves"][name]
                                ],
                            ]
                            for kind in CONTROLS
                        ],
                    ),
                ]
    Path("EVIDENCE_MATRICES.md").write_text("\n".join(appendix), encoding="utf-8")
    print(
        "Wrote EXPERIMENTS_345.md, EVIDENCE_MATRICES.md, results_345.json", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/round345")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Show completed measurements with unfinished PPA cells marked",
    )
    collect(parser.parse_args())
