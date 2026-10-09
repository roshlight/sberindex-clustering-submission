"""Import the published consumption file from a downloaded official ZIP archive."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED = "9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61"
URL = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
args = parser.parse_args()
with zipfile.ZipFile(args.archive) as archive:
    matches = [
        item for item in archive.infolist() if Path(item.filename).name == "consumption.parquet"
    ]
    if len(matches) != 1 or matches[0].file_size > 20_000_000:
        raise ValueError("Expected one bounded consumption.parquet member")
    content = archive.read(matches[0])
digest = hashlib.sha256(content).hexdigest()
if digest != EXPECTED:
    raise ValueError("Source file changed since the audited snapshot; review before use")
args.output_dir.mkdir(parents=True, exist_ok=True)
target = args.output_dir / "consumption.parquet"
if target.exists() and target.read_bytes() != content:
    raise FileExistsError("Refusing to replace a different consumption.parquet")
if not target.exists():
    with target.open("xb") as handle:
        handle.write(content)
print(
    json.dumps(
        {"file": str(target), "sha256": digest, "source": URL, "data_license": "CC BY-SA 4.0"}
    )
)
