from pathlib import Path

from laya_server import runtime as runtime_module


def _make_model(root: Path) -> None:
    (root / "tokenizer").mkdir(parents=True)
    (root / "encoder").mkdir(parents=True)
    (root / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"cached")


def test_cached_model_overrides_use_hf_snapshot(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    repo = hf_home / "hub" / "models--convaiinnovations--laya"
    snapshot = repo / "snapshots" / "abc123"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")

    _make_model(snapshot)
    _make_model(snapshot / "multilingual")
    _make_model(snapshot / "typed-decisions")

    monkeypatch.setenv("HF_HOME", str(hf_home))

    overrides = runtime_module._cached_model_overrides()

    assert overrides["english"] == str(snapshot)
    assert overrides["multilingual"] == (str(snapshot), "multilingual")
    assert overrides["typed-decisions"] == (str(snapshot), "typed-decisions")


def test_incomplete_cache_is_not_used(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    snapshot = (
        hf_home
        / "hub"
        / "models--convaiinnovations--laya"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "rl_agent_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(hf_home))

    assert runtime_module._cached_model_overrides() == {}
