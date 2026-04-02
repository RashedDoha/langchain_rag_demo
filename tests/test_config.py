from rag.config import get_settings, project_root


def test_project_root_contains_pyproject() -> None:
    assert (project_root() / "pyproject.toml").is_file()


def test_get_settings_paths_exist_as_configured() -> None:
    s = get_settings()
    assert s.data_dir.is_absolute()
    assert s.chunk_size > 0
    assert s.chunk_overlap >= 0
