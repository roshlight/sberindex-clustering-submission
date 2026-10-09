"""Saved-source format regressions: Kirov mest captions and Moscow line break."""

import hashlib
import json

import httpx
import pytest

from sber_enrichment.bdpmo import BdpmoClient, parse_population_csv, source_population_types
from sber_enrichment.population_bulk import AttributionReview, fetch_group, parse_batch_csv

TITLE = "Оценка численности населения на 1 января текущего года, человек"
OPTIONS = {
    "Pokazateli": [{"code": "8112027", "label": TITLE}],
    "munr": [{"code": "33000000", "label": "Муниципальные образования"}],
    "tippos": [{"code": "5", "label": "Муниципальный округ"}],
    "oktmo": [
        {"code": "33516000", "label": "Кикнурский муниципальный округ"},
        {"code": "33501000", "label": "Другое точное название"},
    ],
    "mest": [{"code": "11", "label": "Все"}, {"code": "12", "label": "Городское"}],
    "period": [{"code": "208", "label": "на 1 января"}],
}


class SavedResponse:
    def __init__(self, document):
        self.raw = document.encode("cp1251")

    def request(self, method, url, payload):
        return self.raw, {
            "url": url,
            "raw_sha256": hashlib.sha256(self.raw).hexdigest(),
            "request_key": "saved-fixture",
            "retrieved_at": "2026-09-15T00:00:00+00:00",
        }


def test_kirov_exact_mest_codes_recover_saved_values_preserving_source_caption_and_row():
    document = (
        f"{TITLE}, Кикнурский муниципальный округ, на 1 января\n"
        ";2023;2024;\nВсе;6535;6384;\nГородское;4045;3978;\n"
    )
    rows = fetch_group(
        SavedResponse(document),
        "https://rosstat.gov.ru/dbscripts/munst/munst33/",
        OPTIONS,
        {"request_key": "official-form"},
        ["33516000"],
    )["33516000"]
    assert [(r["population_type"], r["value"]) for r in rows] == [
        ("11", 6535),
        ("11", 6384),
        ("12", 4045),
        ("12", 3978),
    ]
    assert rows[0]["population_type_label"] == "Все"
    assert rows[-1]["source_csv_row"] == 4
    with pytest.raises(ValueError, match="Unknown nonempty"):
        parse_population_csv(document)  # Captions cannot be guessed without the form.
    with pytest.raises(ValueError, match="Unknown nonempty"):
        parse_population_csv(
            document.replace("Городское;", "Похожее название;"),
            population_types=source_population_types(OPTIONS),
        )


def test_batch_uses_the_same_exact_mest_mapping():
    document = f"{TITLE}\n;2023;2024;\n" + "".join(
        f"{row['label']};;;\nВсе;100;90;\nГородское;50;40;\n" for row in OPTIONS["oktmo"]
    )
    rows = parse_batch_csv(document, OPTIONS, [row["code"] for row in OPTIONS["oktmo"]])
    assert rows["33501000"][-1]["source_csv_row"] == 8
    assert rows["33501000"][-1]["population_type_label"] == "Городское"


@pytest.mark.parametrize("foreign", [False, True])
def test_moscow_unescaped_title_newline_keeps_exact_attribution_and_physical_row(foreign):
    options = {
        **OPTIONS,
        "oktmo": [{"code": "45301000", "label": "Богородское"}],
        "mest": [
            {"code": "11", "label": "Все население"},
            {"code": "12", "label": "Городское население"},
        ],
    }
    name = "Вешняки" if foreign else "Богородское"
    caption = (
        f"{TITLE}, Муниципальные образования города Москвы, Внутригородская территория\n"
        "(внутригородское муниципальное образование) города федерального значения, "
        f"{name}, на 1 января"
    )
    document = (
        f"\r\n{caption}\r\n;2023;2024;\r\n"
        "Все население;108657;107571;\r\nГородское население;108657;107571;\r\n"
    )

    def fetch():
        return fetch_group(
            SavedResponse(document),
            "https://rosstat.gov.ru/dbscripts/munst/munst45/",
            options,
            {"request_key": "source-options"},
            ["45301000"],
        )

    if foreign:
        with pytest.raises(AttributionReview):
            fetch()
    else:
        rows = fetch()["45301000"]
        assert rows[0]["value"] == 108657
        assert rows[1]["value"] == 107571
        assert rows[0]["source_csv_row"] == 5
        assert rows[0]["source_section"] == caption.replace("\n", " ")
        with pytest.raises(ValueError, match="Unknown nonempty"):
            parse_population_csv(document + "Unknown row inside numeric table\n")


def test_multiline_batch_preserves_original_numeric_row():
    options = {
        **OPTIONS,
        "mest": [
            {"code": "11", "label": "Все население"},
            {"code": "12", "label": "Городское население"},
        ],
    }
    document = f"{TITLE}, Внутригородская территория\n(официальное продолжение)\n;2023;2024;\n"
    document += "".join(
        f"{r['label']};;;\nВсе население;100;90;\nГородское население;50;40;\n"
        for r in options["oktmo"]
    )
    rows = parse_batch_csv(document, options, [r["code"] for r in options["oktmo"]])
    assert rows["33501000"][-1]["source_csv_row"] == 9
    assert rows["33501000"][-1]["source_section"].endswith("(официальное продолжение)")


def test_cache_only_never_sends_on_cache_miss_and_reads_verified_hit(tmp_path):
    client = BdpmoClient(tmp_path, interval=0, cache_only=True)

    def forbidden(request):
        pytest.fail("Cache-only replay attempted network access")

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(forbidden))
    url = "https://rosstat.gov.ru/dbscripts/munst/munst33/DBInet.cgi"
    payload = {"pl": "8112027"}
    with pytest.raises(ValueError, match="Cache-only mode"):
        client.request("POST", url, payload)
    request = {"method": "POST", "url": url, "payload": payload}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    raw = b"saved"
    (tmp_path / f"{key}.raw").write_bytes(raw)
    (tmp_path / f"{key}.json").write_text(
        json.dumps(
            {
                **request,
                "status_code": 200,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "final_url": url,
            }
        )
    )
    assert client.request("POST", url, payload)[0] == raw
    assert client.network_request_count == 0
    assert client.cache_hit_count == 1
    with pytest.raises(ValueError, match="Cache-only mode"):
        client.request("POST", url, payload, refresh=True)
    client.close()
