import pytest

from msq.core import config


def test_local_file_wins_over_the_example(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)  # and no repository root: a lab's own cohorts.local.toml there must not reach the test
    monkeypatch.setattr("msq.core.runinfo.repo_root", lambda: None)
    (tmp_path / "cohorts.example.toml").write_text('[groups]\ncontrol = "control"\n')
    c = config.local_config("cohorts", root=tmp_path)
    assert c["groups"] == {"control": "control"} and c["_local"] is False
    (tmp_path / "cohorts.local.toml").write_text('[groups]\ncontrol = "untreated"\n')
    c = config.local_config("cohorts", root=tmp_path)
    assert c["groups"] == {"control": "untreated"} and c["_local"] is True and c["_source"].endswith("cohorts.local.toml")


def test_config_dir_from_the_environment_comes_first(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "groups.local.toml").write_text('reference = "wt"\n')
    (tmp_path / "groups.example.toml").write_text('reference = "reference"\n')
    monkeypatch.setenv(config.ENV_VAR, str(env_dir))
    assert config.local_config("groups", root=tmp_path)["reference"] == "wt"
    assert config.config_dirs(tmp_path)[0] == env_dir


def test_missing_configuration_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="nothing.local.toml"):
        config.local_config("nothing", root=tmp_path)
