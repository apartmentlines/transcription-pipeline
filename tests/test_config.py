import os
import argparse
from unittest.mock import patch
import pytest
from transcription_pipeline.config import load_configuration, set_environment_variables


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "env_api_key", "TRANSCRIPTION_DOMAIN": "env_domain"},
)
def test_load_configuration_cli_over_env():
    args = argparse.Namespace(api_key="cli_api_key", domain="cli_domain")
    api_key, domain = load_configuration(args)
    assert api_key == "cli_api_key"
    assert domain == "cli_domain"


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "env_api_key", "TRANSCRIPTION_DOMAIN": "env_domain"},
)
def test_load_configuration_env():
    args = argparse.Namespace(api_key=None, domain=None)
    api_key, domain = load_configuration(args)
    assert api_key == "env_api_key"
    assert domain == "env_domain"


def test_load_configuration_missing():
    args = argparse.Namespace(api_key=None, domain=None)
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            load_configuration(args)


def test_set_environment_variables():
    set_environment_variables("test_api_key", "test_domain")
    assert os.environ["TRANSCRIPTION_API_KEY"] == "test_api_key"
    assert os.environ["TRANSCRIPTION_DOMAIN"] == "test_domain"


def test_set_environment_variables_api_key_only():
    # Clear existing env vars
    with patch.dict(os.environ, {}, clear=True):
        set_environment_variables("test_api_key", None)
        assert os.environ["TRANSCRIPTION_API_KEY"] == "test_api_key"
        assert "TRANSCRIPTION_DOMAIN" not in os.environ


def test_set_environment_variables_domain_only():
    # Clear existing env vars
    with patch.dict(os.environ, {}, clear=True):
        set_environment_variables(None, "test_domain")
        assert os.environ["TRANSCRIPTION_DOMAIN"] == "test_domain"
        assert "TRANSCRIPTION_API_KEY" not in os.environ


def test_set_environment_variables_both_none():
    # Clear existing env vars
    with patch.dict(os.environ, {}, clear=True):
        set_environment_variables(None, None)
        assert "TRANSCRIPTION_API_KEY" not in os.environ
        assert "TRANSCRIPTION_DOMAIN" not in os.environ
