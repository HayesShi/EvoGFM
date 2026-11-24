"""Render one readable report from the A experiment indexes."""

import argparse
import json
from pathlib import Path
import re
from statistics import fmean, stdev


DATASETS = {
    "bitcoin-alpha": "Bitcoin-Alpha",
    "bitcoin-otc": "Bitcoin-OTC",
    "uci": "UCI-Message",
    "wikipedia": "Wikipedia",
    "reddit-body": "Reddit-body",
    "reddit-title": "Reddit-title",
    "superuser": "SuperUser",
    "wiki-talk": "Wiki-Talk",
    "stack-overflow": "Stack-Overflow",
}
LARGE = ("superuser", "wiki-talk", "stack-overflow")
METHODS = {
    "ours": "本文",
    "head_fused": "Head-Fused",
    "neutronrt_adapt": "NeutronRT-adapt",
    "full_refresh": "FullGraph-Refresh",
    "demand_only": "Demand-Only",
    "shared_lru": "Shared-LRU",
    "ours_rebuild_moments": "重扫图矩",
    "ours_no_scores": "去除分数保留",
    "ours_no_projections": "去除投影保留",
    "ours_feature_layer_one": "仅共享特征分支与第一层",
}
SYSTEM_GROUPS = {"2/3/7/8 formal system replay", "2/3/7/8 system replay"}
SYSTEM_METHODS = (
    "ours",
    "head_fused",
    "neutronrt_adapt",
    "full_refresh",
    "demand_only",
    "shared_lru",
)
REQUEST_MODES = {
    "uniform": "均匀",
    "fixed-hotspot": "固定热点",
    "rotating-hotspot": "轮换热点",
    "identical": "完全重复",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(label, path):
    path = Path(path)
    target = f"/{path.as_posix()}" if path.drive else path.as_posix()
    return f"[{label}](<{target}>)"


def table(headers, rows):
    def row(values):
        return "| " + " | ".join(str(value) for value in values) + " |"

    return "\n".join(
        [row(headers), row(["---"] * len(headers)), *(row(values) for values in rows)]
    )


def fmt(value, decimals=2):
    return "—" if value is None else f"{value:,.{decimals}f}"


def scaled(value, divisor):
    return None if value is None else value / divisor


def mean_values(samples, field):
    values = [sample[field] for sample in samples if sample.get(field) is not None]
    return fmean(values) if values else None


def total_time(samples):
    values = [sample["total_s"] for sample in samples]
    if len(values) > 1:
        return f"{fmt(fmean(values))} ± {fmt(stdev(values))}"
    return fmt(values[0]) if values else "—"


def signature(row):
    return (
        row["temporal_history_backend"],
        row["unified_device_projection"],
        row["deterministic_cuda_algorithms"],
        row.get("implementation_revision", ""),
    )


def repeat_id(row):
    return re.search(r"(?:^|[-/])r(\d+)(?=[-/])", row["case"])[1]


def reference_label(check):
    return Path(check["reference"]).name if check else "—"


def is_verified(check):
    return check is not None and check["valid"] and check["disagreement"] == 0


def methods_for(rows, dataset, condition):
    return {
        row["method"]: row
        for row in rows
        if row["dataset"] == dataset and row["condition"] == condition
    }


def quality_tables(quality):
    def metrics(method):
        return (
            f'{fmt(100 * method["auc"])}% / {fmt(method["mrr"], 4)}' if method else "—"
        )

    rows = []
    for cohort in quality["cohorts"]:
        methods = cohort["methods"]
        first = next(iter(methods.values()), {})
        candidates = next(
            (
                m["test_candidate_pairs"]
                for m in methods.values()
                if m.get("test_candidate_pairs")
            ),
            None,
        )
        decoupled = metrics(methods.get("decoupled"))
        if "decoupled" in cohort["currently_blocked"] and "decoupled" not in methods:
            decoupled = "本机内存受限"
        rows.append(
            (
                DATASETS[cohort["dataset"]],
                fmt(first.get("snapshots"), 0),
                fmt(candidates, 0),
                metrics(methods.get("ours")),
                metrics(methods.get("undergs")),
                decoupled,
            )
        )
    native = [
        (DATASETS[dataset], metrics(result["ours"]), metrics(result["decoupled"]))
        for dataset, result in quality["native_1000_gpu_replays"].items()
    ]
    return (
        table(
            (
                "数据集",
                "测试快照",
                "候选对",
                "本文 AUC / MRR",
                "UnderGS",
                "DecoupledDGNN",
            ),
            rows,
        ),
        table(("数据集", "本文 AUC / MRR", "DecoupledDGNN"), native),
    )


def main_system_tables(rows, checks):
    comparison = [row for row in rows if row["group"] in SYSTEM_GROUPS]
    quality_rows, runtime_rows, resource_rows = [], [], []
    for dataset in DATASETS:
        anchors = {
            (signature(row), repeat_id(row)): row
            for row in comparison
            if row["dataset"] == dataset
            and row["method"] == "neutronrt_adapt"
            and is_verified(checks.get(row["case"]))
        }
        for method in SYSTEM_METHODS:
            observed = [
                row
                for row in comparison
                if row["dataset"] == dataset and row["method"] == method
            ]
            by_config = sorted({signature(row) for row in observed}) or [None]
            for config in by_config:
                subset = [row for row in observed if signature(row) == config]
                verified = [
                    row for row in subset if is_verified(checks.get(row["case"]))
                ]
                title = METHODS[method]
                if len(by_config) > 1:
                    history, projection, deterministic, revision = config
                    title += f" ({history}, 统一投影 {int(projection)}, 确定性 {int(deterministic)}, 代码 {revision or '—'})"
                references = [
                    checks[row["case"]] for row in subset if row["case"] in checks
                ]
                comparisons = [item for item in references if not item["reference_row"]]
                values = [
                    item["actual_quality"]
                    for item in references
                    if "actual_quality" in item
                ]
                references_used = (
                    ", ".join(sorted({reference_label(item) for item in comparisons}))
                    or "—"
                )
                quality_rows.append(
                    (
                        DATASETS[dataset],
                        title,
                        references_used,
                        (
                            f"{sum(is_verified(item) for item in comparisons)}/{len(comparisons)}"
                            if comparisons
                            else "—"
                        ),
                        fmt(scaled(mean_values(values, "auc"), 0.01)),
                        fmt(mean_values(values, "mrr"), 4),
                        (
                            f'{max(item["max_abs"] for item in comparisons):.3g}'
                            if comparisons
                            else "—"
                        ),
                        (
                            sum(
                                item["probability_violation_count"]
                                for item in comparisons
                            )
                            if comparisons
                            else "—"
                        ),
                        (
                            fmt(
                                max(item["disagreement"] for item in comparisons) * 100,
                                4,
                            )
                            if comparisons
                            else "—"
                        ),
                    )
                )
                matched = [
                    (anchors[config, repeat_id(row)]["total_s"], row["total_s"])
                    for row in verified
                    if (config, repeat_id(row)) in anchors
                ]
                speedup = (
                    f"{fmt(sum(a for a, _ in matched) / sum(b for _, b in matched))}× ({len(matched)})"
                    if matched
                    else "—"
                )
                runtime_rows.append(
                    (
                        DATASETS[dataset],
                        title,
                        f"{len(subset)}/5",
                        len(verified),
                        total_time(verified),
                        fmt(mean_values(verified, "update_throughput"), 0),
                        fmt(mean_values(verified, "candidate_throughput"), 0),
                        speedup,
                    )
                )
                if verified:
                    resource_rows.append(
                        (
                            DATASETS[dataset],
                            title,
                            len(verified),
                            *(
                                fmt(max(row[field] for row in verified) / 2**20, 1)
                                for field in (
                                    "peak_rss_bytes",
                                    "peak_gpu_allocated_bytes",
                                    "peak_gpu_reserved_bytes",
                                    "peak_shared_cache_bytes",
                                )
                            ),
                        )
                    )
    return quality_rows, runtime_rows, resource_rows


def experiment_rows(rows, group, fields):
    selected = [row for row in rows if row["group"] == group]
    selected.sort(
        key=lambda row: (
            list(DATASETS).index(row["dataset"]),
            row["condition"],
            row["method"],
            row["case"],
        )
    )
    return [tuple(access(row) for access in fields) for row in selected]


def experiment_sections(root, index):
    rows = load(index / "case_index.json")
    checks = load(index / "fidelity_index.json")
    quality = load(index / "quality_index.json")
    indexed = load(index / "status.json")
    live = indexed["live"]
    active = live.get("active") or {}
    complete = live["completed_cases"]
    total = live["queue_size"]
    timestamp = live["timestamp"].replace("T", " ")
    sections = [
        [
            f"数据更新时间 {timestamp}。调度器已完成 **{complete:,}/{total:,}** 个作业，登记过 {live['failed_cases']} 次失败。"
            + (f"当时正在运行 `{active['id']}`。" if active else ""),
            "`—` 表示该指标尚无可用值。参数扫表列出已完成条件，下面各节给出完整设计。"
            "“有偏差”和“待核验”行保留实测时间，不能用于声称结果等价的加速。",
            " · ".join(
                (
                    link("实时队列", root / "status.json"),
                    link("逐项作业与配置", root / "queue"),
                    link("逐项原始指标", index / "case_index.csv"),
                    link("概率核验记录", index / "fidelity_index.json"),
                    link("预测质量来源", index / "quality_index.json"),
                )
            ),
        ]
    ]
    strict, native = quality_tables(quality)
    fidelity, runtime, resources = main_system_tables(rows, checks)
    sections.append(
        [
            "统一候选的离散快照测试。三张 Bitcoin/UCI 小图沿用已锁定的时间快照切分，扩展图按完整时间分桶及事件分位边界切分。每个方法均在验证集选定模型后测试。",
            strict,
            "**DecoupledDGNN 原生协议。** 下表使用原生 1,000 负例和时间切分。",
            native,
            "Wiki-Talk、Stack-Overflow 的 DecoupledDGNN 作者程序需要的两份双精度缓冲分别至少为 3.14 GB 和 7.16 GB，本机 WSL 内存配额不足。"
            + link(
                "资源测算", root / "quality/decoupled/large_resource_requirements.json"
            ),
        ]
    )
    sections.append(
        [
            "概率使用 `atol=rtol=1e-6`，排序并列时按节点 ID 处理。主回放优先与 FullGraph-Refresh 比较。偏差数为各次运行之和，"
            "最大绝对差与排名位置偏差率取各次运行的最大值。参照自身不计入通过率。",
            details(
                "九图逐对核验",
                table(
                    (
                        "数据集",
                        "方法",
                        "参照",
                        "通过/有参照",
                        "AUC (%)",
                        "MRR",
                        "最大绝对差",
                        "超容差数",
                        "排名偏差 (%)",
                    ),
                    fidelity,
                ),
            ),
            "SuperUser 的 100% 覆盖复测发现，单个 FP32 舍入单位足以让冻结树模型跨过切分阈值。"
            + link(
                "浮点偏差复现",
                root
                / "diagnostics/superuser_100percent_unified_capture_full/readout_boundary_diagnosis.json",
            ),
        ]
    )
    overview = []
    for name in DATASETS.values():
        times = []
        for method in SYSTEM_METHODS:
            collected = [
                row for row in runtime if row[0] == name and row[1] == METHODS[method]
            ]
            times.append(collected[0][4] if len(collected) == 1 else "见明细")
        overview.append((name, *times))
    sections.append(
        [
            "六种执行策略各计划独立运行五次。时间包含完整初始化、更新、图特征、候选评分和最终读出，训练单列。下表对核验通过的运行报均值 ± 样本标准差。"
            "一轮只有单值。每图的主回放与下面的短工作负载采用不同的请求轨迹。",
            table(
                ("数据集（秒）", *(METHODS[method] for method in SYSTEM_METHODS)),
                overview,
            ),
            details(
                "九图六方法的完整回放时间与吞吐",
                table(
                    (
                        "数据集",
                        "方法",
                        "已跑/计划",
                        "通过数",
                        "总时间（s）",
                        "更新/s",
                        "候选对/s",
                        "相对 NeutronRT-adapt 提速（配对数）",
                    ),
                    runtime,
                ),
            ),
        ]
    )
    parts = ["以下 P50 为无人工等待的服务时间。"]
    sections.append(parts)

    def identity(row):
        return DATASETS[row["dataset"]]

    def condition(row):
        return row["condition"]

    def method_name(row):
        return METHODS.get(row["method"], row["method"])

    def validity(row):
        check = checks.get(row["case"])
        if check is None:
            return "待核验"
        return "通过" if is_verified(check) else "有偏差"

    def metric(field, decimals=2, divisor=1, section=None):
        def cell(row):
            value = (row.get(section) or {}).get(field) if section else row.get(field)
            return fmt(scaled(value, divisor), decimals)

        return cell

    def cohort(title, group, headers, fields):
        measured = experiment_rows(
            rows, group, (identity, condition, method_name, *fields, validity)
        )
        content = table(("数据集", "条件", "方法", *headers, "核验"), measured)
        parts.append(
            details(f"{title}（{len(measured)} 条结果）", content)
            if measured
            else f"{title}待出结果。"
        )

    cohort(
        "查询覆盖率",
        "4 coverage",
        ("端点覆盖 (%)", "候选对", "查询 P50（ms）", "依赖边访问"),
        (
            metric("mean_active_node_coverage", 2, 0.01),
            metric("candidates", 0),
            metric("query_p50_ms"),
            metric("backbone_edges", 0),
        ),
    )
    parts.append(
        "`e4_*_n` 的 `n` 表示每个版本的请求数。"
        + "、".join(f"`{name}` 为{label}" for name, label in REQUEST_MODES.items())
        + "。"
    )
    overlap = (
        "候选对",
        "总时间（s）",
        "查询 P50（ms）",
        "隐藏表示命中端点",
        "共享分数命中",
        "基础点积",
    )
    overlap_fields = (
        metric("candidates", 0),
        metric("total_s"),
        metric("query_p50_ms"),
        metric("hidden_hit_endpoints", 0, section="request_reuse"),
        metric("shared_score_hits", 0, section="request_reuse"),
        metric("base_dot_products", 0),
    )
    cohort("请求重叠与工作量", "4 overlap", overlap, overlap_fields)
    parts.append("SuperUser 另有将计数移出服务计时的配对回放。")
    cohort("请求重叠的独立计时", "4 overlap service timing", overlap, overlap_fields)

    matchups = []
    for dataset in ("wiki-talk", "stack-overflow"):
        for method in ("ours", "head_fused", "neutronrt_adapt"):
            row = methods_for(
                [row for row in rows if row["group"] == "4 overlap"],
                dataset,
                "e4_uniform_1",
            ).get(method)
            if row is not None:
                matchups.append(
                    (
                        DATASETS[dataset],
                        METHODS[method],
                        fmt(row["total_s"]),
                        fmt(row["query_p50_ms"]),
                        validity(row),
                    )
                )
    if matchups:
        parts[0:0] = [
            "**大图短回放。** 全图与完整历史下的 16 个版本，共 4,096 个候选对。",
            table(
                ("数据集", "方法", "总时间（s）", "查询 P50（ms）", "核验"), matchups
            ),
        ]

    parts = ["吞吐在整个已排空的测试段计算。"]
    sections.append(parts)
    cohort(
        "六档到达率",
        "5 paced tail latency",
        (
            "到达对/s",
            "处理对/s",
            "查询 P50/P95/P99（ms）",
            "更新 P99（ms）",
            "最大积压",
        ),
        (
            metric("candidate_pairs_per_second", 0, section="arrival"),
            metric("candidate_throughput", 0, section="arrival"),
            lambda row: " / ".join(fmt(row[f"query_p{q}_ms"], 1) for q in (50, 95, 99)),
            metric("update_p99_ms", 1, section="arrival"),
            metric("max_waiting_requests", 0, section="arrival"),
        ),
    )

    parts = ["提交耗时计拓扑与时序状态摄入和全局状态更新。"]
    sections.append(parts)
    single_edits = [
        row
        for row in rows
        if row["group"] == "6 effective update batch sizes"
        and row["condition"] == "e6_insert_1"
    ]
    edit_summary = []
    for dataset in LARGE:
        method_rows = methods_for(single_edits, dataset, "e6_insert_1")
        if all(
            method in method_rows
            and is_verified(checks.get(method_rows[method]["case"]))
            for method in ("ours", "ours_rebuild_moments")
        ):
            ours = method_rows["ours"]["commit_p50_ms"]
            rebuild = method_rows["ours_rebuild_moments"]["commit_p50_ms"]
            edit_summary.append(
                (DATASETS[dataset], fmt(ours), fmt(rebuild), f"{fmt(rebuild / ours)}×")
            )
    if edit_summary:
        parts.append(
            table(
                ("单边插入", "本文 P50（ms）", "重扫图矩 P50（ms）", "提速"),
                edit_summary,
            )
        )
    cohort(
        "提交延迟与有效更新吞吐",
        "6 effective update batch sizes",
        (
            "变动/提交",
            "提交数",
            "提交 P50（ms）",
            "提交 P99（ms）",
            "有效更新/s",
            "总时间（s）",
        ),
        (
            metric("effective_updates_per_commit", 0),
            metric("edit_commits", 0),
            metric("commit_p50_ms"),
            metric("commit_p99_ms"),
            metric("effective_update_throughput"),
            metric("total_s"),
        ),
    )

    parts = ["工作计数与无专门计数的计时分列。"]
    sections.append(parts)
    cohort(
        "算子工作量与消融",
        "7 exact work and ablation",
        ("基础点积", "依赖边访问", "重算分数", "参考耗时（s，有计数）"),
        (
            metric("base_dot_products", 0),
            metric("dependency_edges_consumed", 0, section="work_profile"),
            metric("recomputed_base_scores", 0, section="work_profile"),
            metric("total_s"),
        ),
    )
    cohort(
        "未加工作计数的消融计时",
        "7 ablation latency without work instrumentation",
        ("总时间（s）", "查询 P50（ms）"),
        (metric("total_s"), metric("query_p50_ms")),
    )
    cohort(
        "各图规模及分阶段耗时",
        "7 scale and stage profile",
        (
            "初始活动边",
            "有效变动",
            "总时间（s）",
            "图矩差分（s）",
            "图矩重扫（s）",
            "谱状态（s）",
            "图特征（s）",
        ),
        (
            metric("initial_edges", 0),
            metric("effective_updates", 0),
            metric("total_s"),
            metric("moment_delta", section="scoped_stages"),
            metric("moment_rescan", section="scoped_stages"),
            metric("spectral_state", section="scoped_stages"),
            metric("graph_features_s", section="stages"),
        ),
    )

    parts = [
        "主回放的内存指标取已通过运行中的最大值。",
        details(
            "九图峰值资源",
            table(
                (
                    "数据集",
                    "方法",
                    "通过数",
                    "峰值 RSS（MiB）",
                    "GPU 分配（MiB）",
                    "GPU 预留（MiB）",
                    "新增缓存（MiB）",
                ),
                resources,
            ),
        ),
    ]
    sections.append(parts)
    resource_fields = (
        metric("total_s"),
        metric("peak_rss_bytes", 1, 2**20),
        metric("peak_gpu_allocated_bytes", 1, 2**20),
        metric("peak_shared_cache_bytes", 1, 2**20),
        metric("recomputed_base_scores", 0, section="work_profile"),
    )
    budget_headers = (
        "总时间（s）",
        "峰值 RSS（MiB）",
        "GPU 分配（MiB）",
        "新增缓存（MiB）",
        "重算分数",
    )
    cohort("预算扫描与重算次数", "8 memory and budget", budget_headers, resource_fields)
    cohort(
        "SuperUser 显存分项测量",
        "8 detailed device scratch profile",
        budget_headers,
        resource_fields,
    )
    cohort(
        "无工作计数的预算计时",
        "8 measured latency and memory without work instrumentation",
        budget_headers,
        resource_fields,
    )
    parts.append(
        "图、模型、共享中间对象与 CUDA 临时区分项。"
        + link("逐作业资源字段", index / "case_index.csv")
    )

    parts = ["`common_projection` 表示两方采用相同的固定形状 CUDA 投影及确定性执行。"]
    sections.append(parts)
    depth = [row for row in rows if row["group"] == "9 trained depth comparison"]
    depth_rows = []
    for dataset in LARGE:
        conditions = (
            "e9_depth_2",
            "e9_depth_3",
            "e9_depth_2_common_projection",
            "e9_depth_3_common_projection",
        )
        for setting in conditions:
            for method in ("ours", "head_fused"):
                selected = [
                    row
                    for row in depth
                    if row["dataset"] == dataset
                    and row["condition"] == setting
                    and row["method"] == method
                ]
                checked = [
                    row for row in selected if is_verified(checks.get(row["case"]))
                ]
                if selected or setting in conditions[:2]:
                    depth_rows.append(
                        (
                            DATASETS[dataset],
                            setting,
                            METHODS[method],
                            f"{len(checked)}/{len(selected)}",
                            total_time(checked),
                            fmt(mean_values(checked, "base_dot_products"), 0),
                            fmt(
                                scaled(
                                    max(
                                        (
                                            row["peak_gpu_allocated_bytes"]
                                            for row in checked
                                        ),
                                        default=None,
                                    ),
                                    2**20,
                                ),
                                1,
                            ),
                        )
                    )
    parts.append(
        table(
            (
                "数据集",
                "深度设置",
                "方法",
                "核验通过/已跑",
                "总时间（s）",
                "基础点积",
                "GPU 峰值（MiB）",
            ),
            depth_rows,
        )
    )
    cohort(
        "图 7 Wiki-Talk 敏感性",
        "10 workload sensitivity",
        ("图版本", "请求", "候选对", "总时间（s）", "候选对/s", "查询 P99（ms）"),
        (
            metric("topology_versions", 0),
            metric("requests", 0),
            metric("candidates", 0),
            metric("total_s"),
            metric("candidate_throughput", 0),
            metric("query_p99_ms"),
        ),
    )
    return ["\n\n".join(section) for section in sections]


def details(title, content):
    return f"<details>\n<summary>{title}</summary>\n\n{content}\n\n</details>"


def write_report(root, output, report):
    original = report.read_text(encoding="utf-8")
    sections = experiment_sections(root, output)
    blocks = [
        f"<!-- a-results:{index} -->\n{content}\n<!-- /a-results -->"
        for index, content in enumerate(sections)
    ]
    managed = re.compile(r"<!-- a-results:(\d+) -->\n.*?\n<!-- /a-results -->", re.S)
    existing = [int(index) for index in managed.findall(original)]
    if existing == list(range(len(blocks))):
        updated = managed.sub(lambda match: blocks[int(match[1])], original)
    elif not existing:
        headings = re.split(r"(?m)(?=^## (?:[1-9]\. |先跑哪几组))", original)
        if len(headings) != len(blocks) + 1:
            raise ValueError("Experiment report headings changed")
        title, body = headings[0].split("\n\n", 1)
        headings[0] = f"{title}\n\n{blocks[0]}\n\n{body}"
        for index, block in enumerate(blocks[1:], start=1):
            headings[index] = f"{headings[index].rstrip()}\n\n{block}\n\n"
        updated = "".join(headings)
    else:
        raise ValueError("Experiment report result blocks are incomplete")
    if updated == original:
        return report
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(write_report(args.root, args.output, args.report))


if __name__ == "__main__":
    main()
