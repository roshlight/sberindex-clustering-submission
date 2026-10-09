"""Read Rosstat BDPMO indicator 8112027 by exact source OKTMO codes.

One municipality per request binds exported numeric cells to a code explicitly
selected in the official query form. No external name matching or code padding
is performed. Raw responses, requests and SHA-256 hashes are cached together.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import ssl
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

INDICATOR = "8112027"
YEARS = ("2023", "2024")
POPULATION_TYPES = {"11": "Все население", "12": "Городское население"}
REGION_URL = re.compile(r"https://rosstat\.gov\.ru/dbscripts/munst/munst[0-9]{2}/?$")
DIMENSIONS = ("Pokazateli", "munr", "tippos", "oktmo", "mest", "god", "period")
ALLOWED_HOSTS = frozenset({"rosstat.gov.ru"})


def _validate_url(value: str | httpx.URL):
    url = httpx.URL(value)
    if (
        url.scheme != "https"
        or url.host not in ALLOWED_HOSTS
        or url.port not in {None, 443}
        or url.userinfo
    ):
        raise ValueError("Only the official Rosstat HTTPS origin is allowed")


def _normalize_label(value: str) -> str:
    return " ".join(html.unescape(value).split())


def parse_options(document: str) -> dict[str, list[dict[str, str]]]:
    result = {}
    for name in DIMENSIONS:
        codes = re.findall(r"p_" + name + r'\[([0-9]+)\]="([^"\r\n]+)"', document)
        selected = re.search(
            r'<SELECT\s+NAME="' + name + r'"[^>]*>(.*?)</SELECT>',
            document,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not selected or not codes:
            raise ValueError(f"Missing official dimension: {name}")
        labels = re.findall(r"<OPTION\b[^>]*>([^<]*)", selected.group(1), flags=re.IGNORECASE)
        codes = sorted((int(index), code) for index, code in codes)
        if [index for index, _ in codes] != list(range(len(labels))):
            raise ValueError(f"Code/label index mismatch: {name}")
        entries = [
            {"code": code, "label": _normalize_label(label)}
            for (_, code), label in zip(codes, labels, strict=True)
        ]
        by_code = {}
        for entry in entries:
            if entry["code"] in by_code and by_code[entry["code"]] != entry["label"]:
                raise ValueError(f"Conflicting duplicate source dimension code: {name}")
            by_code[entry["code"]] = entry["label"]
        # Duplicate identical UI choices do not identify a second territory.
        # Their full original occurrence order remains in the raw HTML cache.
        result[name] = [{"code": code, "label": label} for code, label in by_code.items()]
    if {row["code"] for row in result["Pokazateli"]} != {INDICATOR}:
        raise ValueError("Response is not the requested population indicator")
    for name, required in (
        ("god", set(YEARS)),
        ("mest", set(POPULATION_TYPES)),
        ("period", {"208"}),
    ):
        if not required <= {row["code"] for row in result[name]}:
            raise ValueError(f"Requested values unavailable: {name}")
    if not any(
        "на 1 января" in row["label"].casefold() for row in result["period"] if row["code"] == "208"
    ):
        raise ValueError("Period 208 is not explicitly January 1")
    if not any("человек" in row["label"] for row in result["Pokazateli"]):
        raise ValueError("Population units not confirmed as persons")
    return result


def population_payload(options: dict, oktmo: str) -> dict[str, str]:
    if oktmo not in {item["code"] for item in options["oktmo"]}:
        raise ValueError("OKTMO must exactly match a code returned by the official form")
    values = {
        "Pokazateli": [INDICATOR],
        "munr": [item["code"] for item in options["munr"]],
        "tippos": [item["code"] for item in options["tippos"]],
        "oktmo": [oktmo],
        "mest": list(POPULATION_TYPES),
        "god": list(YEARS),
        "period": ["208"],
    }
    return {
        "Qry": ";".join(name + ":" + ",".join(values[name]) for name in DIMENSIONS) + ";",
        "QryGm": "Pokazateli_z:1;munr_z:2;tippos_z:3;oktmo_z:4;period_z:5;god_s:1;mest_b:1;",
        "QryFootNotes": ";",
        "YearsList": "2023;2024;",
        "tbl": "1",
        "Format": "CSV",
        "YearFrom": "2023",
        "YearTo": "2024",
    }


def is_indicator_title(value: str, indicator_label: str | None = None) -> bool:
    if indicator_label is not None:
        label = _normalize_label(indicator_label)
        return value == label or value.startswith(label + ",")
    return "Оценка численности населения" in value


def year_header(cells: list[str]) -> list[str]:
    years = cells[1:]
    while years and not years[-1]:
        years.pop()
    if not years or any(not year for year in years):
        raise ValueError("Blank interior year header would make cell attribution ambiguous")
    if not set(years) <= set(YEARS) or len(years) != len(set(years)):
        raise ValueError("Unexpected or duplicate years in CSV header")
    return years


def source_population_types(options: dict) -> dict[str, str]:
    """Select exact mest captions; code meanings remain 11 total and 12 urban."""
    if "mest" not in options:
        return dict(POPULATION_TYPES)  # Direct-parser callers with the standard layout.
    labels = {
        row["code"]: row["label"] for row in options["mest"] if row["code"] in POPULATION_TYPES
    }
    if (
        set(labels) != set(POPULATION_TYPES)
        or any(not label for label in labels.values())
        or len(set(labels.values())) != len(labels)
    ):
        raise ValueError("Requested population-type captions are missing or ambiguous")
    return labels


def csv_context_rows(document: str, indicator_label: str | None = None):
    """Read physical rows, joining only pre-header continuation of a title.

    Moscow's official tippos caption contains an unescaped newline. Preserve
    physical row numbers for numeric cells and retain the entire logical title;
    this helper never joins rows after the explicit year header.
    """
    pending = None
    rows = csv.reader(io.StringIO(document.replace("\r\r\n", "\n")), delimiter=";")
    for number, row in enumerate(rows, 1):
        cells = [_normalize_label(value) for value in row]
        if not any(cells):
            continue
        if is_indicator_title(cells[0], indicator_label):
            if pending is not None:
                raise ValueError("Indicator title has no explicit year header")
            if len(cells) != 1:
                raise ValueError("Indicator title has unexpected CSV fields")
            pending = (number, cells[0])
            continue
        if pending is not None:
            if cells[0] == "" and any(cells[1:]):
                yield pending[0], [pending[1]]
                pending = None
            elif len(cells) == 1:
                pending = (pending[0], _normalize_label(pending[1] + " " + cells[0]))
                continue
            else:
                raise ValueError("Indicator title has no explicit year header")
        yield number, cells
    if pending is not None:
        raise ValueError("Indicator title has no explicit year header")


def parse_population_csv(
    document: str,
    *,
    require_complete: bool = True,
    indicator_label: str | None = None,
    population_types: dict[str, str] | None = None,
) -> list[dict]:
    """Parse four requested cells, rejecting unknown layouts/duplicates.

    Rosstat can split the selected years into separate tables when another
    source dimension changes. Each table carries its own explicit year header.

    Missing numeric cells remain null. A source dash is retained as unresolved
    rather than borrowing a zero-marker definition from a different product.
    """
    if "<html" in document.lower():
        raise ValueError("HTML returned instead of CSV")
    lines = [_normalize_label(line) for line in document.splitlines() if line.strip()]
    if not any(is_indicator_title(line, indicator_label) and "человек" in line for line in lines):
        raise ValueError("CSV indicator/units not confirmed")
    if not any("на 1 января" in line.casefold() for line in lines):
        raise ValueError("CSV reference date not confirmed")
    population_types = source_population_types(
        {"mest": [{"code": code, "label": label} for code, label in population_types.items()]}
        if population_types is not None
        else {}
    )
    header = None
    section = None
    records = []
    population_by_label = {value: key for key, value in population_types.items()}
    for number, cells in csv_context_rows(document, indicator_label):
        if cells and is_indicator_title(cells[0], indicator_label):
            section = cells[0]
            header = None
            continue
        nonempty_years = [cell for cell in cells[1:] if cell]
        if cells and cells[0] == "" and nonempty_years:
            header = year_header(cells)
            continue
        if cells and cells[0] in population_by_label:
            if header is None or len(cells) < 1 + len(header):
                raise ValueError("Population row has no valid year header")
            if any(cells[1 + len(header) :]):
                raise ValueError("Unexpected extra cells in population row")
            for year, raw in zip(header, cells[1 : 1 + len(header)], strict=True):
                if re.fullmatch(r"[0-9]+", raw):
                    value, status = int(raw), "observed"
                elif raw in {"", "-", "…", "..."}:
                    value, status = None, "missing_or_unresolved_source_marker"
                else:
                    raise ValueError(f"Unrecognized population value: {raw!r}")
                records.append(
                    {
                        "reference_date": f"{year}-01-01",
                        "population_type": population_by_label[cells[0]],
                        "population_type_label": cells[0],
                        "value": value,
                        "value_raw": raw,
                        "value_status": status,
                        "unit": "persons",
                        "source_csv_row": number,
                        "source_section": section,
                    }
                )
            continue
        if any(cells):
            raise ValueError("Unknown nonempty CSV row; cell attribution requires review")
    keys = [(row["reference_date"], row["population_type"]) for row in records]
    expected = {(f"{year}-01-01", code) for year in YEARS for code in POPULATION_TYPES}
    if (
        len(keys) != len(set(keys))
        or not set(keys) <= expected
        or (require_complete and set(keys) != expected)
    ):
        raise ValueError("CSV does not contain exactly the requested municipality's four cells")
    by_key = {(row["reference_date"], row["population_type"]): row["value"] for row in records}
    for year in YEARS:
        total, urban = (by_key.get((f"{year}-01-01", code)) for code in POPULATION_TYPES)
        if total is not None and urban is not None and urban > total:
            raise ValueError("Urban population exceeds total population")
    return records


class BdpmoClient:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        interval: float = 0.5,
        timeout: float = 30,
        ca_bundle: str | Path | None = None,
        cache_only: bool = False,
    ):
        if interval < 0:
            raise ValueError("Request interval cannot be negative")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.last_request = 0.0
        self.cache_only = cache_only
        self.network_request_count = 0
        self.cache_hit_count = 0
        context = ssl.create_default_context()
        if ca_bundle is not None:
            context.load_verify_locations(cafile=str(ca_bundle))
        self.client = httpx.Client(verify=context, timeout=timeout, follow_redirects=False)
        self._options = {}

    def close(self):
        self.client.close()

    def request(self, method: str, url: str, payload: dict | None = None, *, refresh=False):
        _validate_url(url)
        request = {"method": method, "url": url, "payload": payload}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        raw_path, meta_path = self.cache_dir / f"{key}.raw", self.cache_dir / f"{key}.json"
        if raw_path.exists() and meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != metadata["raw_sha256"]:
                raise ValueError("Cached response hash mismatch")
            if metadata["status_code"] == 200 and not refresh:
                _validate_url(metadata["final_url"])
                self.cache_hit_count += 1
                return raw, metadata
        if self.cache_only:
            raise ValueError(f"Cache-only mode has no accepted saved response for request {key}")
        next_request = self.client.build_request(method, url, data=payload)
        redirects = []
        for _ in range(6):
            _validate_url(next_request.url)
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            response = self.client.send(next_request, follow_redirects=False)
            self.network_request_count += 1
            self.last_request = time.monotonic()
            if response.next_request is None:
                break
            redirects.append(
                {
                    "url": str(response.url),
                    "status": response.status_code,
                    "location": str(response.next_request.url),
                }
            )
            next_request = response.next_request
        else:
            raise ValueError("Too many Rosstat redirects")
        raw = response.content
        metadata = {
            **request,
            "request_key": key,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "redirects": redirects,
        }
        if raw_path.exists() and meta_path.exists():
            history = self.cache_dir / "history" / key
            history.mkdir(parents=True, exist_ok=True)
            version = hashlib.sha256(meta_path.read_bytes()).hexdigest()
            (history / f"{version}.raw").write_bytes(raw_path.read_bytes())
            (history / f"{version}.json").write_bytes(meta_path.read_bytes())
        raw_path.write_bytes(raw)
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        response.raise_for_status()
        _validate_url(response.url)
        return raw, metadata

    def options(self, region_url: str, *, refresh=False):
        if not REGION_URL.fullmatch(region_url):
            raise ValueError("Expected an official BDPMO regional URL")
        region_url = region_url.rstrip("/") + "/"
        if region_url not in self._options or refresh:
            endpoint = region_url + "DBInet.cgi"
            document, metadata = self.request("POST", endpoint, {"pl": INDICATOR}, refresh=refresh)
            self._options[region_url] = parse_options(document.decode("cp1251")), metadata
        return self._options[region_url]

    def population(self, region_url: str, oktmo: str) -> list[dict]:
        options, option_metadata = self.options(region_url)
        payload = population_payload(options, oktmo)
        raw, metadata = self.request("POST", region_url.rstrip("/") + "/DBInet.cgi", payload)
        records = parse_population_csv(
            raw.decode("cp1251"),
            indicator_label=options["Pokazateli"][0]["label"],
            population_types=source_population_types(options),
        )
        source_name = next(item["label"] for item in options["oktmo"] if item["code"] == oktmo)
        for record in records:
            record.update(
                {
                    "source_oktmo": oktmo,
                    "source_oktmo_name": source_name,
                    "source_region_url": region_url,
                    "indicator_id": INDICATOR,
                    "source_url": metadata["url"],
                    "source_sha256": metadata["raw_sha256"],
                    "request_key": metadata["request_key"],
                    "options_request_key": option_metadata["request_key"],
                    "retrieved_at": metadata["retrieved_at"],
                    "published_at": None,
                    "territory_id": None,
                    "join_status": "source_oktmo_unlinked_to_sber",
                }
            )
        return records
