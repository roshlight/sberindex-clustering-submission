"""Fetch selected public ZIP members by bounded HTTP Range; replay from cached ranges."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import httpx
from collect_round5_sme import clean_headers, digest


class RangeReader(io.RawIOBase):
    def __init__(self, client, directory, source, offline=False):
        self.client, self.directory, self.source = client, directory, source
        self.length = int(source["headers"]["content-length"])
        self.position, self.read_bytes = 0, 0
        self.offline = offline
        directory.mkdir(parents=True, exist_ok=True)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (
            self.position if whence == 1 else self.length if whence == 2 else 0
        )
        if not 0 <= self.position <= self.length:
            raise ValueError("Seek outside archive")
        return self.position

    def read(self, size=-1):
        size = min(self.length - self.position, size if size >= 0 else self.length)
        if not size:
            return b""
        if size > 8_000_000 or self.read_bytes + size > 30_000_000:
            raise ValueError("Bounded sample download exceeded; full archive not requested")
        start, end = self.position, self.position + size - 1
        path = self.directory / f"range_{start}_{end}.bin"
        meta = path.with_suffix(".json")
        if path.exists():
            data = path.read_bytes()
            record = json.loads(meta.read_text(encoding="utf-8"))
            if digest(data) != record["sha256"] or len(data) != size:
                raise ValueError("Cached range changed")
        else:
            if self.offline:
                raise FileNotFoundError(path)
            with self.client.stream(
                "GET",
                self.source["url"],
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            ) as response:
                if response.status_code != 206 or response.headers.get("content-range") != (
                    f"bytes {start}-{end}/{self.length}"
                ):
                    raise ValueError("Server did not honor exact Range; body not downloaded")
                data = b""
                for part in response.iter_bytes():
                    data += part
                    if len(data) > size:
                        raise ValueError("Range body exceeded requested size")
                if len(data) != size:
                    raise ValueError("Truncated range")
                path.write_bytes(data)
                record = dict(
                    url=self.source["url"],
                    request_range=f"bytes={start}-{end}",
                    status=response.status_code,
                    headers=clean_headers(dict(response.headers)),
                    bytes=len(data),
                    sha256=digest(data),
                )
                meta.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.position += size
        self.read_bytes += size
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    inventory = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in ["archive_2023_12", "archive_2024_06", "archive_2024_07", "archive_2024_12"]:
            source = json.loads((args.raw / (name + ".http.json")).read_text(encoding="utf-8"))
            directory = args.raw / name
            stream = RangeReader(client, directory, source, args.offline)
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                rows = [
                    dict(
                        name=m.filename,
                        bytes=m.file_size,
                        compressed_bytes=m.compress_size,
                        crc32=f"{m.CRC:08x}",
                        offset=m.header_offset,
                        timestamp=list(m.date_time),
                    )
                    for m in members
                ]
                # A deterministic small member is a schema/semantics sample, never representative.
                chosen = min(
                    (m for m in members if m.filename.lower().endswith(".xml")),
                    key=lambda m: (m.compress_size, m.filename),
                )
                if chosen.file_size > 50_000_000:
                    raise ValueError("Uncompressed member exceeds sample cap")
                content = archive.read(chosen)  # zipfile checks the entire chosen member CRC.
                sample = directory / "sample.xml"
                if args.offline:
                    if sample.read_bytes() != content:
                        raise ValueError("Member replay differs")
                else:
                    sample.write_bytes(content)
                    (directory / "member_inventory.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )
                result = dict(
                    archive=name,
                    url=source["url"],
                    archive_bytes=stream.length,
                    archive_sha256_server_declared=source["headers"].get("x-amz-meta-sha256"),
                    whole_archive_hash_verified=False,
                    range_bytes_read=stream.read_bytes,
                    archive_member_count=len(members),
                    selected_member=chosen.filename,
                    selected_member_bytes=len(content),
                    sample_sha256=digest(content),
                    sample_crc32_verified=True,
                    selection="smallest compressed XML member",
                    representative_sample=False,
                )
                inventory.append(result)
                print(json.dumps(result))
    output = args.raw / ("archive_sample_replay.json" if args.offline else "archive_samples.json")
    output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
