"""Offline guards for the bounded documentary comparison; no live HTTP."""

import importlib.util
import io
from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter

SPEC = importlib.util.spec_from_file_location(
    "collect_round4_cases", Path(__file__).parents[1] / "scripts/collect_round4_cases.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def pdf_body():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_rounding_not_exact_equality():
    result = module.compare_rounded("24.8", "24767", "million_rubles", "thousand_rubles")
    assert result["within_rounding_interval"]
    assert result["difference_million_rubles"] == "0.033"
    assert not module.compare_rounded("24.8", "24700", "million_rubles", "thousand_rubles")[
        "within_rounding_interval"
    ]


@pytest.mark.parametrize("unit", ["rubles", "million_rubles", "unknown"])
def test_no_silent_unit_assumption(unit):
    with pytest.raises(ValueError, match="unit"):
        module.compare_rounded("24.8", "24767", "million_rubles", unit)


def test_redirect_host_and_size_guard(tmp_path):
    calls = []

    def response(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://unapproved.test/file"})

    spec = {"source_id": "a", "url": "https://official.test/doc", "suffix": ".pdf"}
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        result = module.retrieve(spec, tmp_path, client, 100)
    assert result["status"] == "unavailable"
    assert len(calls) == 1
    assert not (tmp_path / "a.pdf").exists()
    spec["source_id"] = "b"
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=pdf_body()))
    ) as client:
        result = module.retrieve(spec, tmp_path, client, 10)
    assert result["status"] == "unavailable"
    assert not (tmp_path / "b.pdf").exists()


def test_cached_raw_hash_must_match(tmp_path):
    spec = {"source_id": "a", "url": "https://official.test/doc", "suffix": ".pdf"}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=pdf_body()))
    ) as client:
        module.retrieve(spec, tmp_path, client, 10000)
        (tmp_path / "a.pdf").write_bytes(b"changed")
        with pytest.raises(ValueError, match="Saved raw"):
            module.retrieve(spec, tmp_path, client, 10000)


def test_offline_missing_record_cannot_fetch(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    module.save(
        plan, [{"source_id": "missing", "url": "https://official.test/doc", "suffix": ".pdf"}]
    )
    monkeypatch.setattr(
        module, "retrieve", lambda *args: pytest.fail("Offline replay attempted fetch")
    )
    with pytest.raises(ValueError, match="Offline replay"):
        module.run(plan, tmp_path / "raw", tmp_path / "reports", True, 1000)


@pytest.fixture
def reviewed_bundle(tmp_path):
    raw, report, inputs = (tmp_path / name for name in ("raw", "report", "inputs"))
    for path in (raw, report, inputs):
        path.mkdir()
    (raw / "yakutia_publications.html").write_bytes(b"catalogue")
    rows = [
        {
            "territory_id": "318",
            "year": "2024",
            "indicator_id": "8109001",
            "value_status": "observed",
            "source_oktmo": "98615000",
            "official_oktmo_8": "98615000",
            "value": "909171",
            "unit": "thousand_rubles",
            "source_sha256": "prior",
            "request_key": "request",
            "source_csv_row": "3",
        }
    ]
    module.write_csv(inputs / "round3_investment_reference.csv", rows)
    module.save(
        inputs / "lineage.json",
        {"reference_sha256": module.sha((inputs / "round3_investment_reference.csv").read_bytes())},
    )
    source_id = "yakutia_yearbook2025_current"
    body = pdf_body()
    (raw / (source_id + ".pdf")).write_bytes(body)
    module.save(
        raw / (source_id + ".meta.json"),
        {
            "raw_file": source_id + ".pdf",
            "sha256": module.sha(body),
            "url": "https://official.test/document",
            "last_modified": "",
            "retrieved_at_utc": "2026-09-16",
        },
    )
    recipe = {
        "source_catalogue_sha256": module.sha(b"catalogue"),
        "observations": [
            {
                "source_id": source_id,
                "source_sha256": module.sha(body),
                "visually_reviewed": True,
                "physical_pdf_page": 1,
                "scope_physical_page": 1,
                "images": [],
                "model_join_accepted": False,
                "unit": "million_rubles",
                "years": [2024],
                "rows": [
                    {
                        "territory_id_context_candidate": 318,
                        "regional_row_label": "Верхнеколымский",
                        "values": ["909.2"],
                    }
                ],
                "printed_page": 1,
                "table": "13.8",
                "price_basis": "actual_current_prices",
                "scope": "excludes_small_business",
                "edition_year": 2025,
                "catalogue_display_date": "2025-12-12",
                "join_status": "context_only",
            }
        ],
    }
    recipe_path = tmp_path / "recipe.json"
    module.save(recipe_path, recipe)
    return raw, report, recipe_path, recipe


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "Duplicate evidence"),
        ("wrong_page", "Missing page"),
        ("model_join", "Unsupported inference"),
    ],
)
def test_review_cannot_promote_or_shift_attribution(reviewed_bundle, mutation, match):
    raw, report, path, recipe = reviewed_bundle
    spec = recipe["observations"][0]
    if mutation == "duplicate":
        spec["rows"].append(spec["rows"][0].copy())
    elif mutation == "wrong_page":
        spec["physical_pdf_page"] = 2
    else:
        spec["model_join_accepted"] = True
    module.save(path, recipe)
    with pytest.raises(ValueError, match=match):
        module.replay_evidence(raw, path, report)
