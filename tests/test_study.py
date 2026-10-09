import pytest

from municipal_profiles.study import validate_lock


def test_evaluation_refuses_code_data_or_config_changes():
    lock = {
        "config_sha256": "config",
        "data_sha256": "data",
        "code_sha256": "code",
        "development_status": "complete",
    }
    validate_lock(lock, "config", "data", "code")
    for changed in [
        ("other", "data", "code"),
        ("config", "other", "code"),
        ("config", "data", "other"),
    ]:
        with pytest.raises(ValueError, match="mismatch"):
            validate_lock(lock, *changed)
    lock["development_status"] = "failed"
    with pytest.raises(ValueError, match="completed"):
        validate_lock(lock, "config", "data", "code")
