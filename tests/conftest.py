import pytest

from aml_reg_rag.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        eval_dir=tmp_path / "eval",
        var_dir=tmp_path / "var",
    )
