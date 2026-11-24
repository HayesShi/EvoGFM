"""Resume large public SNAP releases using verified HTTP byte ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from urllib.request import Request, urlopen

from experiments.db_a_native_trace import write_json


SOURCES = {
    "reddit-body.tsv": ("soc-redditHyperlinks-body.tsv", 318_931_394),
    "reddit-title.tsv": ("soc-redditHyperlinks-title.tsv", 368_581_209),
    "stack-overflow.txt.gz": ("sx-stackoverflow.txt.gz", 529_869_668),
}


def fetch(path, url, first, last, total):
    length = last - first + 1
    if path.is_file() and path.stat().st_size == length:
        return length
    temp = path.with_suffix(".tmp")
    for attempt in range(5):
        try:
            span = length if attempt == 0 else 512 * 1024
            with temp.open("wb") as handle:
                for start in range(first, last + 1, span):
                    end = min(last, start + span - 1)
                    request = Request(
                        url,
                        headers={
                            "Range": f"bytes={start}-{end}",
                            "Accept-Encoding": "identity",
                        },
                    )
                    with urlopen(request, timeout=45) as response:
                        expected = f"bytes {start}-{end}/{total}"
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range") != expected
                        ):
                            raise ValueError(
                                f"Unexpected HTTP range; required {expected}"
                            )
                        remaining = end - start + 1
                        while remaining:
                            block = response.read(min(256 * 1024, remaining))
                            if not block:
                                raise IOError("Truncated response")
                            handle.write(block)
                            remaining -= len(block)
            temp.replace(path)
            return length
        except (OSError, ValueError) as error:
            if attempt == 4:
                raise RuntimeError(f"Range {first}-{last} failed for {url}") from error
            time.sleep(min(30, 2**attempt))


def assemble_parts(folder, name, target):
    """Assemble an individually finished release while later releases download."""
    _, expected = SOURCES[name]
    prefix = folder / f"{name}.prefix"
    combined = prefix.stat().st_size if prefix.is_file() else 0
    sources = [prefix] if prefix.is_file() else []
    for part in sorted((folder / f"{name}.parts").glob("*.part")):
        if int(part.stem) != combined:
            raise ValueError(f"Missing data range at {combined} for {name}")
        combined += part.stat().st_size
        sources.append(part)
    if combined != expected:
        raise ValueError(f"The release still requires {expected-combined} bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".merge")
    with partial.open("wb") as out:
        for part in sources:
            with part.open("rb") as source:
                while data := source.read(1024 * 1024):
                    out.write(data)
    if partial.stat().st_size != expected:
        raise ValueError("Assembled size does not match the official release")
    partial.replace(target)


def run(folder, workers, names):
    chunk_size = 4 * 2**20
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    requests = {}
    plans = []
    for name in names:
        filename, size = SOURCES[name]
        target = folder / name
        if target.exists() and target.stat().st_size == size:
            continue
        parts = folder / f"{name}.parts"
        parts.mkdir(exist_ok=True)
        # Existing partial-download bytes are safe to reuse only on the same
        # verified URL. The named prefix is a stopped curl on that URL.
        prefix = folder / f"{name}.prefix"
        resume = prefix.stat().st_size if prefix.exists() else 0
        intervals = [
            (first, min(size, first + chunk_size) - 1)
            for first in range(resume, size, chunk_size)
        ]
        rows = [
            (
                parts / f"{first:012}.part",
                f"https://snap.stanford.edu/data/{filename}",
                first,
                last,
                size,
            )
            for first, last in intervals
        ]
        results[name] = {
            "completed_bytes": resume,
            "total_bytes": size,
            "parts": len(rows),
        }
        requests[name] = rows, target, prefix
        plans.extend((index, name, row) for index, row in enumerate(rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        work = {pool.submit(fetch, *row): name for _, name, row in sorted(plans)}
        for task in as_completed(work):
            name = work[task]
            try:
                results[name]["completed_bytes"] += task.result()
            except RuntimeError as error:
                results[name].setdefault("errors", []).append(str(error))
            write_json(folder / "download_status.json", results)
    for name, (rows, target, prefix) in requests.items():
        if results[name].get("errors"):
            continue
        merged = target.with_suffix(target.suffix + ".merge")
        with merged.open("wb") as output:
            for part in ([prefix] if prefix.is_file() else []) + [
                row[0] for row in rows
            ]:
                with part.open("rb") as input_file:
                    while block := input_file.read(1024 * 1024):
                        output.write(block)
        if merged.stat().st_size != SOURCES[name][1]:
            raise ValueError(f"Exact release length check failed for {name}")
        merged.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--files", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    run(args.directory, args.workers, args.files)


if __name__ == "__main__":
    main()
