import json
from pathlib import Path

import httpx
import pytest

from sber_enrichment.bdpmo import (
    BdpmoClient,
    parse_options,
    parse_population_csv,
    population_payload,
)


def _options_html():
    values = {
        "Pokazateli": [("8112027", "Оценка численности населения, человек")],
        "munr": [("79700000", "Городские округа")],
        "tippos": [("7", "Городской округ")],
        "oktmo": [("79701000", "Город Майкоп")],
        "mest": [("11", "Все население"), ("12", "Городское население")],
        "god": [("2023", "2023"), ("2024", "2024")],
        "period": [("208", "на 1 января")],
    }
    document = ""
    for name, items in values.items():
        document += "".join(f'p_{name}[{i}]="{code}";' for i, (code, _) in enumerate(items))
        document += (
            f'<SELECT NAME="{name}">'
            + "".join(f"<OPTION>{label}" for _, label in items)
            + "</SELECT>"
        )
    return document


CSV = """Оценка численности населения на 1 января текущего года, человек, Город Майкоп
;2023;2024;
Все население;163766;161898;
Городское население;139687;137965;
"""


def test_options_bind_exact_code_and_named_measurement_period():
    options = parse_options(_options_html())
    payload = population_payload(options, "79701000")
    assert "oktmo:79701000;" in payload["Qry"]
    assert "god:2023,2024;" in payload["Qry"]
    assert "mest:11,12;" in payload["Qry"]
    with pytest.raises(ValueError, match="exactly match"):
        population_payload(options, "79701000000")


def test_option_dimensions_cannot_silently_shift_codes():
    document = _options_html().replace("p_oktmo[0]", "p_oktmo[1]")
    with pytest.raises(ValueError, match="index mismatch"):
        parse_options(document)


def test_identical_repeated_ui_code_is_distinct_from_conflicting_label():
    document = (
        _options_html()
        .replace('p_oktmo[0]="79701000";', 'p_oktmo[0]="79701000";p_oktmo[1]="79701000";')
        .replace(
            "<OPTION>Город Майкоп</SELECT>", "<OPTION>Город Майкоп<OPTION>Город Майкоп</SELECT>"
        )
    )
    assert parse_options(document)["oktmo"] == [{"code": "79701000", "label": "Город Майкоп"}]
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        parse_options(document.replace("<OPTION>Город Майкоп</SELECT>", "<OPTION>Other</SELECT>"))


def test_csv_recovers_year_and_population_type_without_external_names():
    records = parse_population_csv(CSV)
    assert len(records) == 4
    values = {(row["reference_date"], row["population_type"]): row["value"] for row in records}
    assert values["2023-01-01", "11"] == 163766
    assert values["2024-01-01", "12"] == 137965


def test_missing_marker_is_not_converted_to_zero():
    records = parse_population_csv(CSV.replace("137965", "-"))
    assert records[-1]["value"] is None
    assert records[-1]["value_raw"] == "-"


def test_separate_year_tables_keep_explicit_dates_and_source_context():
    document = """Оценка численности населения на 1 января, человек, Муниципальные образования
;2024;
Все население;554248;
Городское население;530311;
Оценка численности населения на 1 января, человек, Городские округа
;2023;
Все население;549846;
Городское население;526851;
"""
    rows = parse_population_csv(document)
    assert len(rows) == 4
    assert rows[0]["reference_date"] == "2024-01-01"
    assert rows[0]["value"] == 554248
    assert rows[2]["reference_date"] == "2023-01-01"
    assert rows[2]["value"] == 549846
    assert rows[0]["source_section"] != rows[2]["source_section"]


@pytest.mark.parametrize(
    "document",
    [
        CSV + "Городское население;1;1;\n",
        CSV.replace("139687", "999999"),
        CSV.replace(";2023;2024;", ";2022;2023;"),
        "<html>temporarily unavailable</html>",
    ],
)
def test_bad_tables_fail_without_inventing_cells(document):
    with pytest.raises(ValueError):
        parse_population_csv(document)


def test_cache_hash_tampering_is_rejected_and_origin_is_restricted(tmp_path):
    import hashlib

    client = BdpmoClient(tmp_path, interval=0)
    try:
        url = "https://rosstat.gov.ru/dbscripts/munst/munst79/"
        request = {"method": "GET", "url": url, "payload": None}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        (tmp_path / f"{key}.raw").write_bytes(b"modified")
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"raw_sha256": "wrong", "status_code": 200})
        )
        with pytest.raises(ValueError, match="hash mismatch"):
            client.request("GET", url)
        with pytest.raises(ValueError, match="official"):
            client.request("GET", "https://example.org/")
    finally:
        client.close()


def test_ca_bundle_augments_default_context_and_cli_forwards_it(tmp_path, monkeypatch):
    import importlib.util
    from unittest.mock import Mock

    from sber_enrichment import bdpmo

    context = Mock()
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(bdpmo.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(bdpmo.httpx, "Client", factory)
    pem = tmp_path / "public-ca.pem"
    instance = BdpmoClient(tmp_path / "cache", ca_bundle=pem)
    context.load_verify_locations.assert_called_once_with(cafile=str(pem))
    assert factory.call_args.kwargs["verify"] is context
    assert factory.call_args.kwargs["follow_redirects"] is False
    instance.close()

    script = Path(__file__).parents[1] / "scripts" / "prepare_bdpmo_population.py"
    spec = importlib.util.spec_from_file_location("bdpmo_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare = Mock(return_value={"status": "complete_unlinked"})
    monkeypatch.setattr(module, "prepare", prepare)
    assert (
        module.main(
            [
                "--targets",
                "targets.json",
                "--output",
                "out",
                "--cache",
                "cache",
                "--ca-bundle",
                str(pem),
                "--limit",
                "3",
            ]
        )
        == 0
    )
    assert prepare.call_args.args[-1] == pem
    assert prepare.call_args.args[-2] == 3


@pytest.mark.parametrize(
    "location",
    [
        "https://example.org/",
        "http://rosstat.gov.ru/",
        "https://rosstat.gov.ru.evil.example/",
        "https://rosstat.gov.ru:8443/",
    ],
)
def test_redirect_is_blocked_before_unapproved_destination_request(tmp_path, location):
    calls = []

    def transport(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": location})

    client = BdpmoClient(tmp_path, interval=0)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(transport))
    try:
        with pytest.raises(ValueError, match="official"):
            client.request("GET", "https://rosstat.gov.ru/start")
        assert calls == ["https://rosstat.gov.ru/start"]
    finally:
        client.close()


def test_allowed_redirect_is_recorded_and_cached(tmp_path):
    def transport(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/finish"})
        return httpx.Response(200, content=b"official")

    client = BdpmoClient(tmp_path, interval=0)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(transport))
    try:
        raw, metadata = client.request("GET", "https://rosstat.gov.ru/start")
        assert raw == b"official"
        assert metadata["final_url"] == "https://rosstat.gov.ru/finish"
        assert len(metadata["redirects"]) == 1
        assert client.request("GET", "https://rosstat.gov.ru/start")[0] == raw
    finally:
        client.close()


def test_official_code_bridge_smoke_preserves_leading_zero_and_unmatched(tmp_path):
    import importlib.util

    import duckdb
    import pandas as pd

    script = Path(__file__).parents[1] / "scripts" / "prepare_population_code_bridge.py"
    spec = importlib.util.spec_from_file_location("code_bridge_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [
        ("1", "79", "701", "79701000"),
        ("622", "03", "701", "3701000"),
        ("1439", "46", "704", "46704000"),
    ]
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "\n".join(
            f"{region};{area};000;000;0;1;Municipality;;;;0;01.01.2023;01.01.2023"
            for _, region, area, _ in records
        ),
        encoding="utf-8",
    )
    cross = pd.DataFrame(
        [
            {
                "territory_id": tid,
                "oktmo_digits": region + area + "000000",
                "oktmo": f"{region}-{area}-000-000",
                "region_name": region,
                "municipal_district_name": tid,
            }
            for tid, region, area, _ in records + [("99", "88", "888", "88888000")]
        ]
    )
    cross_path = tmp_path / "cross.csv"
    cross.to_csv(cross_path, index=False)
    assignments = tmp_path / "assignments.parquet"
    with duckdb.connect() as connection:
        connection.register("cross_rows", cross)
        connection.execute("COPY cross_rows TO ? (FORMAT PARQUET)", [str(assignments)])
    population = tmp_path / "population.csv"
    pd.DataFrame(
        [
            {
                "source_region_url": f"https://rosstat.gov.ru/dbscripts/munst/munst{region}/",
                "source_oktmo": source,
                "reference_date": f"{year}-01-01",
                "population_type": kind,
                "value": 100,
            }
            for _, region, _, source in records
            for year in (2023, 2024)
            for kind in (11, 12)
        ]
    ).to_csv(population, index=False)
    selector = tmp_path / "selector.html"
    selector.write_text(
        "\n".join(
            f"https://rosstat.gov.ru/dbscripts/munst/munst{region}/" for _, region, _, _ in records
        )
    )
    output = tmp_path / "derived"
    result = module.prepare(
        registry, cross_path, cross_path, assignments, population, selector, output
    )
    assert result["primary_count"] == 4
    assert result["exact_official_section1_matches"] == 3
    assert result["unmatched_count"] == 1
    assert result["linked_pilot_cells"] == 12
    linked = pd.read_csv(output / "population_linked_pilot.csv", dtype=str)
    assert set(linked.loc[linked.territory_id == "622", "official_oktmo_8"]) == {"03701000"}
    assert set(linked.loc[linked.territory_id == "622", "source_oktmo"]) == {"3701000"}
    with pytest.raises(FileExistsError):
        module.prepare(registry, cross_path, cross_path, assignments, population, selector, output)
