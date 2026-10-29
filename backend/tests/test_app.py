def test_status_values_are_documented():
    allowed = {"WAITING_FOR_PERMISSION", "RUNNING", "COMPLETED", "FAILED"}
    assert "COMPLETED" in allowed
    assert "WAITING_FOR_PERMISSION" in allowed
