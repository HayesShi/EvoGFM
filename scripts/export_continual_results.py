"""Render fixed continual-study seeds from archived paired measurements."""

import pandas as pd

try:
    from .table_formatting import rank_values
except ImportError:
    from table_formatting import rank_values


TARGET_NAMES = {
    "twitch-en": "Twitch EN",
    "twitch-de": "Twitch DE",
    "twitch-fr": "Twitch FR",
    "github-developer": "GitHub",
}
ORDER_NAMES = {
    "stream_random": "Random",
    "stream_domain_grouped": "Domain-grouped",
    "stream_spectral_grouped": "Spectral-grouped",
    "stream_interleaved": "Domain-interleaved",
}
ARM_NAMES = {"nc": "Node only", "lp": "Link only", "joint": "Joint"}


def markdown_continual(config, root):
    if not config.get("continual"):
        return ""
    fixed = config["continual"]
    arrivals = pd.read_csv(root / config["sources"]["arrival_candidates"])
    history = pd.read_csv(root / config["sources"]["arrival_history"])
    candidates = pd.read_csv(root / config["sources"]["continual_candidates"])
    retention = pd.read_csv(root / config["sources"]["retention"])
    lines = [
        "## 持续学习：固定旧记录中的 seed",
        "",
        "候选为已有的 42、43、44，本轮没有重训。以下为单个完整 seed 的配对结果，单位为 pp，保留负值。",
        "",
        "### Table 5：学习顺序",
        "",
        "每种顺序按 20-shot 总体迁移增益选择一个 seed；1/5/20-shot 和历史保持均沿用该完整运行。",
        "",
        "| Order | Seed | 1-shot Δ | 5-shot Δ | 20-shot Δ | 历史保持 Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, seed in fixed["arrival_order"]["seeds"].items():
        part = arrivals[(arrivals.kind == kind) & (arrivals.seed == seed)]
        scores = [
            float(part[part.shots == shot].iloc[0].gain_pp) for shot in [1, 5, 20]
        ]
        old = history[(history.kind == kind) & (history.seed == seed)].iloc[0]
        lines.append(
            f"| {ORDER_NAMES[kind]} | {seed} | "
            + " | ".join(f"{value:+.2f}" for value in scores)
            + f" | {old.final_anchor_gain_pp:+.2f} |"
        )
    lines += [
        "",
        "### Table 6：不同监督方式",
        "",
        "每张接收图按 Joint 的 20-shot NC 增益选一个 seed；三种监督方式、NC/LP、A2/A6 全部沿用该 seed。",
        "",
        "| Target | Seed | Supervision | NC Acc Δ | Hard LP AUC Δ |",
        "|---|---:|---|---:|---:|",
    ]
    selected = {}
    for graph, seed in fixed["supervision"]["seeds"].items():
        part = candidates[
            (candidates.graph == graph) & (candidates.seed == seed)
        ].set_index("arm")
        selected[graph] = (seed, part)
        arms = ["nc", "lp", "joint"]
        nc = rank_values([f"{part.loc[arm, 'nc_gain_pp']:+.2f}" for arm in arms])
        lp = rank_values([f"{part.loc[arm, 'hard_lp_gain_pp']:+.2f}" for arm in arms])
        for index, arm in enumerate(arms):
            lines.append(
                f"| {TARGET_NAMES[graph]} | {seed} | {ARM_NAMES[arm]} | {nc[index]} | {lp[index]} |"
            )
    lines += [
        "",
        "### Table 6(b)：配对联合监督分析",
        "",
        "| Target | Seed | Joint − Node: NC | Joint − Node: Hard LP | Joint 两项均为正 |",
        "|---|---:|---:|---:|---|",
    ]
    for graph, (seed, part) in selected.items():
        joint, node = part.loc["joint"], part.loc["nc"]
        both = "是" if joint.nc_gain_pp > 0 and joint.hard_lp_gain_pp > 0 else "否"
        lines.append(
            f"| {TARGET_NAMES[graph]} | {seed} | {joint.nc_gain_pp-node.nc_gain_pp:+.2f} | {joint.hard_lp_gain_pp-node.hard_lp_gain_pp:+.2f} | {both} |"
        )
    lines += [
        "",
        "### Table A2：联合监督下的 LP 读出",
        "",
        "| Target | Seed | Source head fixed | Initial target head fixed | Target head refitted |",
        "|---|---:|---:|---:|---:|",
    ]
    for graph, (seed, part) in selected.items():
        row = part.loc["joint"]
        values = rank_values(
            [
                f"{row[key]:+.2f}"
                for key in [
                    "source_lp_gain_pp",
                    "initial_target_lp_gain_pp",
                    "hard_lp_gain_pp",
                ]
            ]
        )
        lines.append(f"| {TARGET_NAMES[graph]} | {seed} | " + " | ".join(values) + " |")
    lines += [
        "",
        "### Table A6：联合监督下的类别均衡变化",
        "",
        "| Target | Seed | Accuracy Δ | Balanced Accuracy Δ |",
        "|---|---:|---:|---:|",
    ]
    for graph, (seed, part) in selected.items():
        row = part.loc["joint"]
        lines.append(
            f"| {TARGET_NAMES[graph]} | {seed} | {row.nc_gain_pp:+.2f} | {row.balanced_gain_pp:+.2f} |"
        )
    seed = fixed["retention"]["seed"]
    lines += [
        "",
        "### Table 7(a)：历史保持",
        "",
        f"按 Random 顺序的 EvoGFM 历史保持增益选 seed {seed}，两种方法沿用同一个 seed。准确率为 %，变化为 pp。",
        "",
        "| Method | Seed | Prefix 4 | Prefix 15 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ["EvoGFM", "GraphKeeper"]:
        part = retention[(retention.seed == seed) & (retention.method == method)]
        start = part[part.prefix == 4].accuracy_percent.mean()
        end = part[part.prefix == 15].accuracy_percent.mean()
        lines.append(
            f"| {method} | {seed} | {start:.2f} | {end:.2f} | {end-start:+.2f} |"
        )
    lines += [
        "",
        "Table 7(b)/(c) 的完整流 AA/AF/BWT/FWT 及三种子统计不由此预览重算；原始多 seed 表保留。",
        "",
    ]
    return "\n".join(lines)
