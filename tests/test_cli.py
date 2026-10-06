"""Tests for docuchat.cli: each flag maps onto Settings the way its help says."""

import pytest

from docuchat.cli import SMALL_RERANKER, flag_env, parse_args
from docuchat.config import Settings


def settings_for(monkeypatch, *argv):
    for name, value in flag_env(parse_args(list(argv))).items():
        monkeypatch.setenv(name, value)
    return Settings.from_env()


def test_no_flags_set_nothing():
    assert flag_env(parse_args([])) == {}


def test_flags_reach_settings(monkeypatch):
    cfg = settings_for(monkeypatch, "--no-ocr", "--cpu", "--local")
    assert cfg.do_ocr is False
    assert cfg.cross_encoder_name == SMALL_RERANKER and cfg.cross_encoder_max_length == 512
    assert cfg.llm_provider == "llamacpp" and cfg.context_window == 8192


def test_fallback_keeps_the_api_provider(monkeypatch):
    monkeypatch.delenv("DOCUCHAT_LLM_PROVIDER", raising=False)
    assert settings_for(monkeypatch, "--fallback").llm_provider == Settings().llm_provider


def test_local_and_fallback_are_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--local", "--fallback"])
