import pytest

from docuchat.profiles import PROFILES


def test_both_profiles_are_registered():
    assert set(PROFILES) == {"mortgage", "generic"}


def test_profiles_are_hashable_so_settings_stays_lru_cacheable():
    assert hash(PROFILES["mortgage"])


def test_mortgage_profile_keeps_the_notebook_taxonomy():
    labels = [label for _, label in PROFILES["mortgage"].doc_type_keywords]
    assert "Closing Disclosure" in labels and "Promissory Note" in labels
    keys = [k for k, _ in PROFILES["mortgage"].doc_type_keywords]
    assert keys.index("closing disclosure") < keys.index("disclosure")


def test_generic_profile_has_no_keyword_taxonomy():
    assert PROFILES["generic"].doc_type_keywords == ()


def test_generic_prompt_is_domain_neutral():
    assert "mortgage" not in PROFILES["generic"].answer_system_prompt.lower()


@pytest.mark.parametrize("name", ["mortgage", "generic"])
def test_every_profile_keeps_the_grounding_contract(name):
    prompt = PROFILES[name].answer_system_prompt
    assert "ONLY from the provided context" in prompt
    assert "cannot find this information" in prompt


def test_search_domain_is_domain_specific():
    assert PROFILES["mortgage"].search_domain == "mortgage document"
    assert PROFILES["generic"].search_domain == "document"
