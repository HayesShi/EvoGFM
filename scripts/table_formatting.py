"""Emphasize comparable results without changing their displayed values."""

from decimal import Decimal
import re


def plain(value):
    return re.sub(r"\*\*|</?(?:ins|u)>", "", value).strip()


def rank_values(values):
    """Bold the largest mean and underline the second distinct displayed mean."""
    values = [plain(value) for value in values]
    means = []
    for value in values:
        match = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)(?:\s*±\s*(?:\d+(?:\.\d+)?|待重算))?%?",
            value,
        )
        means.append(Decimal(match[1]) if match else None)
    ranks = sorted({mean for mean in means if mean is not None}, reverse=True)
    if sum(mean is not None for mean in means) < 2:
        return values
    return [
        (
            f"**{value}**"
            if mean == ranks[0]
            else f"<ins>{value}</ins>" if len(ranks) > 1 and mean == ranks[1] else value
        )
        for value, mean in zip(values, means)
    ]


def format_experiment_tables(markdown):
    """Rank within each dataset, metric and experimental comparison group."""
    lines = markdown.splitlines()
    heading = ""
    index = 0
    while index < len(lines):
        if lines[index].startswith("#"):
            match = re.search(r"\bTable (\d+|A\d+)(\([a-z]\))?\.", lines[index])
            heading = match[1] + (match[2] or "") if match else ""
        if not lines[index].startswith("|"):
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines[index:end]
        ]
        data = rows[2:]

        def rank(cells):
            marked = rank_values([data[row][col] for row, col in cells])
            for (row, col), value in zip(cells, marked):
                data[row][col] = value

        if heading in {"1", "5", "7(a)", "7(d)", "A8(a)", "A8(b)"}:
            columns = (
                range(2, len(rows[0]))
                if heading == "1"
                else [4] if heading == "7(a)" else range(1, len(rows[0]))
            )
            for col in columns:
                rank([(row, col) for row in range(len(data))])
        elif heading == "6":
            for target in dict.fromkeys(row[0] for row in data):
                for col in [2, 3]:
                    rank(
                        [
                            (row, col)
                            for row in range(len(data))
                            if data[row][0] == target
                        ]
                    )
        elif heading == "A1":
            for row in data:
                parts = [plain(value).split(" / ") for value in row[2:4]]
                for metric in [0, 1]:
                    marked = rank_values([part[metric] for part in parts])
                    for part, value in zip(parts, marked):
                        part[metric] = value
                row[2:4] = [" / ".join(part) for part in parts]
        else:
            starts = {
                "2(a)": 1,
                "2(b)": 1,
                "2(c)": 1,
                "3(a)": 2,
                "3(b)": 2,
                "4(a)": 2,
                "4(b)": 2,
                "A2": 1,
                "A3": 1,
                "A3(b)": 1,
                "A4": 1,
                "A11": 3,
            }
            if heading in starts:
                stop = (
                    4
                    if heading in {"3(a)", "3(b)"}
                    else 3 if heading == "A3" else len(rows[0])
                )
                for row in range(len(data)):
                    rank([(row, col) for col in range(starts[heading], stop)])
        rows[2:] = data
        lines[index:end] = ["| " + " | ".join(row) + " |" for row in rows]
        index = end
    return "\n".join(lines) + "\n"
