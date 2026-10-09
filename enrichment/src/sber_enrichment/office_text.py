"""Bounded Office text previews for source discovery, not numeric data ingestion.

DOCX scope is word/document.xml only, including table paragraphs. XLSX scope is
the first 12 physical rows of each sheet, or 80 rows for explicitly named legend
and methodology sheets. Cell coordinates and sheet names are retained. Formulas
are never evaluated and cached formula results are not treated as observations.
No files are extracted to disk and no relationship URL is fetched.
"""

import io
import re
import stat
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 2048
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250
MAX_SHEETS = 64
HEADER_ROWS = 12
LEGEND_ROWS = 80
MAX_COLUMNS = 32
MAX_CELL_CHARS = 4000
MAX_SHARED_STRINGS = 200_000

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
LEGEND_NAME = re.compile(
    r"предисл|метод|описан|легенд|условн|примеч|инструк|read.?me|notes|legend|metadata|definition",
    re.IGNORECASE,
)


class _Preview:
    def __init__(self, max_chars):
        self.max_chars = max_chars
        self.parts = []
        self.length = 0
        self.truncated = False

    def add(self, text):
        """Append within the requested character limit; return whether it fitted."""
        part = ("\n" if self.parts else "") + text
        remaining = self.max_chars - self.length
        if len(part) > remaining:
            self.parts.append(part[:remaining])
            self.length += max(0, remaining)
            self.truncated = True
            return False
        self.parts.append(part)
        self.length += len(part)
        return True

    def result(self, kind):
        return {
            "text": "".join(self.parts),
            "kind": kind,
            "links": [],
            "truncated": self.truncated,
        }


def _check_archive(archive):
    members = archive.infolist()
    if len(members) > MAX_MEMBERS:
        raise ValueError("Office ZIP has too many members")
    seen = set()
    total = 0
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or "\x00" in name
            or ":" in name
            or path.is_absolute()
            or ".." in path.parts
            or name.rstrip("/") != path.as_posix()
        ):
            raise ValueError("Unsafe Office ZIP member path")
        key = name.casefold().rstrip("/")
        if key in seen:
            raise ValueError("Duplicate Office ZIP member")
        seen.add(key)
        if member.flag_bits & 1:
            raise ValueError("Encrypted Office ZIP members are unsupported")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("Office ZIP symlink members are unsupported")
        if member.file_size > MAX_MEMBER_BYTES:
            raise ValueError("Office ZIP member exceeds uncompressed limit")
        if member.file_size / max(1, member.compress_size) > MAX_COMPRESSION_RATIO:
            raise ValueError("Office ZIP compression ratio exceeds limit")
        total += member.file_size
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Office ZIP total uncompressed size exceeds limit")


def _xml(archive, name):
    """Read an already size-checked member and reject DTD/entity declarations."""
    with archive.open(name) as stream:
        content = stream.read(MAX_MEMBER_BYTES + 1)
    if len(content) > MAX_MEMBER_BYTES:
        raise ValueError("Office XML exceeds uncompressed limit")
    # Removing NUL also catches declaration bytes in UTF-16/UTF-32 XML.
    markup = content.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in markup or b"<!ENTITY" in markup:
        raise ValueError("DTD/entity declarations are not accepted in Office XML")
    return ET.fromstring(content)


def _docx(archive, preview):
    document = _xml(archive, "word/document.xml")
    if document.tag != f"{{{WORD_NS}}}document":
        raise ValueError("DOCX document has an unsupported XML namespace or root")
    if not preview.add(
        "[DOCX main document text only; headers, footnotes and comments are outside this preview]"
    ):
        return preview.result("docx_main_text")
    for number, paragraph in enumerate(document.iter(f"{{{WORD_NS}}}p"), start=1):
        tokens = []
        for item in paragraph.iter():
            if item.tag == f"{{{WORD_NS}}}t":
                tokens.append(item.text or "")
            elif item.tag == f"{{{WORD_NS}}}tab":
                tokens.append("\t")
            elif item.tag in {f"{{{WORD_NS}}}br", f"{{{WORD_NS}}}cr"}:
                tokens.append("\n")
        text = "".join(tokens).strip()
        if text and not preview.add(f"[paragraph {number}] {text}"):
            break
    return preview.result("docx_main_text")


def _worksheet_target(target):
    if not isinstance(target, str) or "\\" in target or ":" in target or "\x00" in target:
        raise ValueError("Unsafe worksheet relationship target")
    relative = target.lstrip("/")
    if ".." in PurePosixPath(relative).parts or not relative:
        raise ValueError("Unsafe worksheet relationship target")
    relative = relative if relative.startswith("xl/") else "xl/" + relative
    if not relative.startswith("xl/worksheets/") or not relative.endswith(".xml"):
        raise ValueError("Unexpected worksheet relationship target")
    return relative


def _shared_strings(archive, preview):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    strings = []
    root = _xml(archive, "xl/sharedStrings.xml")
    if root.tag != f"{{{SHEET_NS}}}sst":
        raise ValueError("Unsupported shared-string XML namespace or root")
    for item in root:
        if len(strings) >= MAX_SHARED_STRINGS:
            preview.truncated = True
            break
        text = "".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t"))
        # Retain a per-entry flag: unused long data cells need not truncate a header preview.
        strings.append((text[:MAX_CELL_CHARS], len(text) > MAX_CELL_CHARS))
    return strings


def _cell_text(cell, shared, preview):
    if cell.find(f"{{{SHEET_NS}}}f") is not None:
        preview.truncated = True
        return "[formula omitted; cached result not used]"
    kind = cell.attrib.get("t")
    node = cell.find(f"{{{SHEET_NS}}}v")
    text = node.text or "" if node is not None else ""
    if kind == "s":
        if not re.fullmatch(r"\d+", text):
            raise ValueError("Invalid shared-string index")
        index = int(text)
        if index >= len(shared):
            if index >= MAX_SHARED_STRINGS:
                preview.truncated = True
                return "[shared string beyond preview limit]"
            raise ValueError("Shared-string index does not exist")
        text, cut = shared[index]
        preview.truncated |= cut
    elif kind == "inlineStr":
        text = "".join(item.text or "" for item in cell.iter(f"{{{SHEET_NS}}}t"))
    elif kind == "e":
        text = f"[Excel error: {text}]"
    if len(text) > MAX_CELL_CHARS:
        preview.truncated = True
        text = text[:MAX_CELL_CHARS]
    return text


def _column_number(reference):
    match = re.fullmatch(r"([A-Z]+)([1-9]\d*)", reference)
    if match is None:
        raise ValueError("Invalid XLSX cell reference")
    column = 0
    for character in match[1]:
        column = 26 * column + ord(character) - ord("A") + 1
    return column, int(match[2])


def _xlsx(archive, preview):
    workbook = _xml(archive, "xl/workbook.xml")
    if workbook.tag != f"{{{SHEET_NS}}}workbook":
        raise ValueError("XLSX workbook has an unsupported XML namespace or root")
    relationships = _xml(archive, "xl/_rels/workbook.xml.rels")
    rels = {}
    for rel in relationships:
        identifier = rel.attrib.get("Id")
        if identifier in rels:
            raise ValueError("Duplicate workbook relationship ID")
        rels[identifier] = rel.attrib
    sheets = workbook.findall(f"{{{SHEET_NS}}}sheets/{{{SHEET_NS}}}sheet")
    preview.truncated |= len(sheets) > MAX_SHEETS
    shared = _shared_strings(archive, preview)
    if not preview.add(
        f"[XLSX header/legend preview only: first {HEADER_ROWS} rows per sheet, "
        f"{LEGEND_ROWS} for named legend/methodology sheets; at most {MAX_SHEETS} sheets and "
        f"{MAX_COLUMNS} columns. Raw cell values, no date formatting or formula evaluation. "
        "This is not a complete statistical table.]"
    ):
        return preview.result("xlsx_header_preview")
    for sheet in sheets[:MAX_SHEETS]:
        name = sheet.attrib.get("name", "")
        relation = rels.get(sheet.attrib.get(f"{{{REL_NS}}}id"))
        if relation is None or relation.get("TargetMode", "").lower() == "external":
            raise ValueError("Missing or external worksheet relationship")
        root = _xml(archive, _worksheet_target(relation.get("Target")))
        if root.tag != f"{{{SHEET_NS}}}worksheet":
            raise ValueError("Unsupported worksheet XML namespace or root")
        row_limit = LEGEND_ROWS if LEGEND_NAME.search(name) else HEADER_ROWS
        if not preview.add(f"[sheet={name!r}; rows 1–{row_limit} preview]"):
            break
        previous_row = 0
        for row in root.findall(f"{{{SHEET_NS}}}sheetData/{{{SHEET_NS}}}row"):
            number = int(row.attrib.get("r", "0"))
            if number <= previous_row:
                raise ValueError("XLSX row references are missing, duplicated or unordered")
            previous_row = number
            if number > row_limit:
                preview.truncated = True
                break
            values = []
            seen_cells = set()
            for cell in row:
                reference = cell.attrib.get("r", "")
                column, cell_row = _column_number(reference)
                if cell_row != number or reference in seen_cells:
                    raise ValueError("XLSX cell references are duplicated or contradict row")
                seen_cells.add(reference)
                if column > MAX_COLUMNS:
                    preview.truncated = True
                    continue
                text = _cell_text(cell, shared, preview)
                if text:
                    values.append(f"{reference}={text}")
            if values and not preview.add(f"[sheet={name!r}; row={number}] " + " | ".join(values)):
                return preview.result("xlsx_header_preview")
    return preview.result("xlsx_header_preview")


def parse_office(body: bytes, extension: str, max_chars: int = 100_000) -> dict:
    """Return text/kind/links/truncated; reject unsafe or malformed packages.

    ``extension`` accepts docx/xlsx with or without a leading dot. Unsupported
    formats and archive/XML violations raise ValueError. ZIP metadata is checked
    before decompression; XML members, member count, ratio, total expanded bytes,
    output characters, sheet/row/column scope and cell length are all bounded.
    ``truncated`` reports preview omissions, not an estimate of data coverage.
    """
    if not isinstance(body, bytes) or len(body) > MAX_ARCHIVE_BYTES:
        raise ValueError("Office input must be bytes within the compressed-size limit")
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or not 0 < max_chars <= 1_000_000
    ):
        raise ValueError("max_chars must be an integer from 1 through 1000000")
    if not isinstance(extension, str) or extension.lower().lstrip(".") not in {"docx", "xlsx"}:
        raise ValueError("Only DOCX and XLSX previews are supported")
    try:
        with ZipFile(io.BytesIO(body)) as archive:
            _check_archive(archive)
            preview = _Preview(max_chars)
            if extension.lower().lstrip(".") == "docx":
                return _docx(archive, preview)
            return _xlsx(archive, preview)
    except (BadZipFile, ET.ParseError, KeyError, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"Invalid Office package: {type(error).__name__}") from error
