import pytest

from aml_reg_rag.llm import describe_error


def test_describe_error_includes_int_status_code_only():
    class ApiError(Exception):
        status_code = 429

    assert describe_error(ApiError("key=secret")) == "ApiError（狀態碼 429）"


@pytest.mark.parametrize("code", ["e3q8", None])
def test_describe_error_ignores_non_int_codes(code):
    class OddError(Exception):
        pass

    OddError.code = code  # type: ignore[attr-defined]
    assert describe_error(OddError("secret")) == "OddError"
