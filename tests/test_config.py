"""Tests for config properties (filter_thinking and ssl_verify)."""

import os
import pytest


class TestFilterThinkingConfig:
    """Test GROK_FILTER_THINKING environment variable handling."""

    def setup_method(self):
        # Reset the singleton to force re-read of env vars
        from grok_search.config import Config
        Config._instance = None

    def teardown_method(self):
        os.environ.pop("GROK_FILTER_THINKING", None)
        os.environ.pop("GROK_SSL_VERIFY", None)
        from grok_search.config import Config
        Config._instance = None

    def test_filter_thinking_default_true(self):
        os.environ.pop("GROK_FILTER_THINKING", None)
        from grok_search.config import Config
        c = Config()
        assert c.filter_thinking is True

    def test_filter_thinking_explicit_true(self):
        os.environ["GROK_FILTER_THINKING"] = "true"
        from grok_search.config import Config
        c = Config()
        assert c.filter_thinking is True

    def test_filter_thinking_false(self):
        os.environ["GROK_FILTER_THINKING"] = "false"
        from grok_search.config import Config
        c = Config()
        assert c.filter_thinking is False

    def test_filter_thinking_yes(self):
        os.environ["GROK_FILTER_THINKING"] = "yes"
        from grok_search.config import Config
        c = Config()
        assert c.filter_thinking is True

    def test_filter_thinking_one(self):
        os.environ["GROK_FILTER_THINKING"] = "1"
        from grok_search.config import Config
        c = Config()
        assert c.filter_thinking is True

    def test_filter_thinking_no(self):
        os.environ["GROK_FILTER_THINKING"] = "no"
        from grok_search.config import Config
        c = Config()
        assert c.filter_thinking is False


class TestSSLVerifyConfig:
    """Test GROK_SSL_VERIFY environment variable handling."""

    def setup_method(self):
        from grok_search.config import Config
        Config._instance = None

    def teardown_method(self):
        os.environ.pop("GROK_SSL_VERIFY", None)
        from grok_search.config import Config
        Config._instance = None

    def test_ssl_verify_default_true(self):
        os.environ.pop("GROK_SSL_VERIFY", None)
        from grok_search.config import Config
        c = Config()
        assert c.ssl_verify is True

    def test_ssl_verify_false(self):
        os.environ["GROK_SSL_VERIFY"] = "false"
        from grok_search.config import Config
        c = Config()
        assert c.ssl_verify is False

    def test_ssl_verify_explicit_true(self):
        os.environ["GROK_SSL_VERIFY"] = "true"
        from grok_search.config import Config
        c = Config()
        assert c.ssl_verify is True

    def test_ssl_verify_zero(self):
        os.environ["GROK_SSL_VERIFY"] = "0"
        from grok_search.config import Config
        c = Config()
        assert c.ssl_verify is False
