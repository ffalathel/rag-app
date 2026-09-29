"""Call-budget tests: ask() must not re-run rewrite/decompose/rerank per
sub-query, and must skip models an ablation config doesn't need."""

from docuchat.config import Settings
from docuchat.pipeline import ask


def test_full_config_makes_exactly_three_llm_calls(fake_llm, fake_store, fake_models):
    ask("what is the interest rate?", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert fake_llm.calls == 3  # rewrite + decompose + answer


def test_baseline_config_makes_exactly_one_llm_call(fake_llm, fake_store, fake_models):
    cfg = Settings(retrieval_mode="vector", use_rewrite=False,
                   use_decomposition=False, use_rerank=False)
    ask("what is the interest rate?", fake_store, cfg, llm=fake_llm, **fake_models)
    assert fake_llm.calls == 1  # answer only


def test_rerank_disabled_skips_the_cross_encoder(fake_llm, fake_store, fake_models):
    cfg = Settings(use_rerank=False)
    ask("q", fake_store, cfg, llm=fake_llm, **fake_models)
    assert fake_models["cross_encoder"].calls == 0


def test_full_config_reranks_once_and_encodes_at_most_once(fake_llm, fake_store, fake_models):
    ask("what is the interest rate?", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert fake_models["cross_encoder"].calls == 1
    assert fake_models["encoder"].calls <= 1


def test_multiple_sub_queries_still_rerank_exactly_once(fake_store, fake_models, fake_llm_returning):
    llm = fake_llm_returning("line one is long enough?\nline two is long enough?")
    ask("what is the interest rate?", fake_store, Settings(), llm=llm, **fake_models)
    assert fake_models["cross_encoder"].calls == 1
