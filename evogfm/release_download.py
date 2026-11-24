"""Resumable public release downloads with checked HTTP byte ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
import urllib.request


def download_ranges(url, destination, workers=12, block_bytes=1024 * 1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=30
    ) as response:
        total = int(response.headers["Content-Length"])
        ranges = response.headers.get("Accept-Ranges", "") == "bytes"
    if destination.exists() and destination.stat().st_size == total:
        return destination
    temporary = destination.with_name(destination.name + ".download")
    if not ranges or total <= block_bytes:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, 128 * 1024)
        if temporary.stat().st_size != total:
            raise OSError("incomplete public release")
        temporary.replace(destination)
        return destination
    parts = destination.with_name(destination.name + ".parts")
    parts.mkdir(exist_ok=True)
    count = (total + block_bytes - 1) // block_bytes

    def fetch(index):
        start, stop = index * block_bytes, min(total, (index + 1) * block_bytes) - 1
        path = parts / f"{index:06d}.part"
        if path.exists() and path.stat().st_size == stop - start + 1:
            return index
        error = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{stop}"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{stop}/{total}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise OSError(f"server did not honor byte range {index}")
                    with path.open("wb") as handle:
                        shutil.copyfileobj(response, handle, 128 * 1024)
                if path.stat().st_size != stop - start + 1:
                    raise OSError(f"truncated byte range {index}")
                return index
            except OSError as exc:
                error = exc
                time.sleep(min(attempt + 1, 3))
        raise OSError(f"public release range {index} failed: {error}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, future in enumerate(
            as_completed([executor.submit(fetch, i) for i in range(count)]), 1
        ):
            future.result()
            if completed % workers == 0 or completed == count:
                print(
                    json.dumps(
                        dict(
                            stage="download",
                            file=destination.name,
                            complete_blocks=completed,
                            total_blocks=count,
                            total_bytes=total,
                        )
                    ),
                    flush=True,
                )
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:06d}.part").open("rb") as stream:
                shutil.copyfileobj(stream, output)
    if temporary.stat().st_size != total:
        raise OSError("assembled public release has wrong length")
    temporary.replace(destination)
    for index in range(count):
        (parts / f"{index:06d}.part").unlink()
    parts.rmdir()
    return destination
