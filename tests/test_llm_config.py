from pause_detector.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, resolve_config


def test_resolve_config_priority(monkeypatch):
    # nothing set
    for k in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY",
              "OPENAI_BASE_URL", "PAUSE_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    api, url, model = resolve_config()
    assert api is None
    assert url == DEFAULT_BASE_URL
    assert model == DEFAULT_MODEL

    # env wins over default
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("PAUSE_LLM_MODEL", "env-model")
    api, url, model = resolve_config()
    assert api == "env-key"
    assert url == "https://env.example/v1"
    assert model == "env-model"

    # CLI overrides env
    api, url, model = resolve_config(api_key="cli-key",
                                     base_url="https://cli.example/v1",
                                     model="cli-model")
    assert api == "cli-key"
    assert url == "https://cli.example/v1"
    assert model == "cli-model"


def test_resolve_config_dashscope_key_fallback(monkeypatch):
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "PAUSE_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    api, _, _ = resolve_config()
    assert api == "ds-key"
