"""Tests for the ask_grok tool and GrokSearchProvider.consult() method."""
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from grok_search.providers.grok import GrokSearchProvider
from grok_search.config import Config


# ---------- Config: grok_advisor_model ----------

class TestGrokAdvisorModelConfig:
    """测试 grok_advisor_model 配置项"""

    def setup_method(self):
        # 重置单例以隔离测试
        Config._instance = None

    def teardown_method(self):
        Config._instance = None

    @patch.dict(os.environ, {"GROK_API_URL": "https://test.api", "GROK_API_KEY": "test-key"})
    def test_advisor_model_defaults_to_grok_model(self):
        """未设置 GROK_ADVISOR_MODEL 时，默认与 grok_model 相同"""
        cfg = Config()
        assert cfg.grok_advisor_model == cfg.grok_model

    @patch.dict(os.environ, {
        "GROK_API_URL": "https://test.api",
        "GROK_API_KEY": "test-key",
        "GROK_ADVISOR_MODEL": "grok-4-2-beta",
    })
    def test_advisor_model_from_env(self):
        """GROK_ADVISOR_MODEL 环境变量可覆盖"""
        cfg = Config()
        assert cfg.grok_advisor_model == "grok-4-2-beta"

    @patch.dict(os.environ, {
        "GROK_API_URL": "https://openrouter.example.com/v1",
        "GROK_API_KEY": "test-key",
        "GROK_ADVISOR_MODEL": "grok-4-2-beta",
    })
    def test_advisor_model_openrouter_suffix(self):
        """OpenRouter URL 时自动附加 :online 后缀"""
        cfg = Config()
        assert cfg.grok_advisor_model == "grok-4-2-beta:online"

    @patch.dict(os.environ, {"GROK_API_URL": "https://test.api", "GROK_API_KEY": "test-key"})
    def test_config_info_includes_advisor_model(self):
        """get_config_info 返回值包含 GROK_ADVISOR_MODEL"""
        cfg = Config()
        info = cfg.get_config_info()
        assert "GROK_ADVISOR_MODEL" in info


# ---------- GrokSearchProvider.consult ----------

class TestConsultMethod:
    """测试 GrokSearchProvider.consult() 方法"""

    @pytest.fixture
    def provider(self):
        return GrokSearchProvider("https://test.api/v1", "test-key", "grok-test")

    @pytest.mark.asyncio
    async def test_consult_parses_json_response(self, provider):
        """consult 应正确解析 JSON 格式的响应"""
        mock_response = json.dumps({
            "answer": "This is an analysis.",
            "sources": ["https://example.com"],
            "follow_up_searches": ["search query 1"]
        })

        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.consult("test question")
            parsed = json.loads(result)

            assert parsed["answer"] == "This is an analysis."
            assert parsed["sources"] == ["https://example.com"]
            assert parsed["follow_up_searches"] == ["search query 1"]
            assert parsed["model_used"] == "grok-test"

    @pytest.mark.asyncio
    async def test_consult_parses_json_code_block(self, provider):
        """consult 应能解析 ```json ... ``` 包裹的 JSON"""
        mock_response = '```json\n{"answer": "block analysis", "sources": [], "follow_up_searches": []}\n```'

        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.consult("test question")
            parsed = json.loads(result)

            assert parsed["answer"] == "block analysis"
            assert parsed["model_used"] == "grok-test"

    @pytest.mark.asyncio
    async def test_consult_fallback_on_non_json(self, provider):
        """consult 在非 JSON 响应时应降级到 fallback 结构"""
        mock_response = "This is just plain text analysis without JSON."

        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.consult("test question")
            parsed = json.loads(result)

            assert parsed["answer"] == mock_response
            assert parsed["sources"] == []
            assert parsed["follow_up_searches"] == []
            assert parsed["model_used"] == "grok-test"
            assert "note" in parsed

    @pytest.mark.asyncio
    async def test_consult_with_context(self, provider):
        """consult 应将 context 附加到用户消息中"""
        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value='{"answer":"ok","sources":[],"follow_up_searches":[]}') as mock_exec:
            await provider.consult("question", context="extra info")

            # 验证 payload 中 user message 包含 context
            call_args = mock_exec.call_args
            payload = call_args[0][1]  # headers, payload, ctx
            user_content = payload["messages"][1]["content"]
            assert "[Additional Context]" in user_content
            assert "extra info" in user_content

    @pytest.mark.asyncio
    async def test_consult_without_sources(self, provider):
        """require_sources=False 时 system prompt 不包含 source 要求"""
        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value='{"answer":"ok","sources":[],"follow_up_searches":[]}') as mock_exec:
            await provider.consult("question", require_sources=False)

            call_args = mock_exec.call_args
            payload = call_args[0][1]
            system_content = payload["messages"][0]["content"]
            assert "## Sources" not in system_content

    @pytest.mark.asyncio
    async def test_consult_with_sources(self, provider):
        """require_sources=True（默认）时 system prompt 包含 source 要求"""
        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value='{"answer":"ok","sources":[],"follow_up_searches":[]}') as mock_exec:
            await provider.consult("question", require_sources=True)

            call_args = mock_exec.call_args
            payload = call_args[0][1]
            system_content = payload["messages"][0]["content"]
            assert "## Sources" in system_content

    @pytest.mark.asyncio
    async def test_consult_uses_streaming(self, provider):
        """consult 应使用 stream: True"""
        with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value='{"answer":"ok","sources":[],"follow_up_searches":[]}') as mock_exec:
            await provider.consult("question")

            call_args = mock_exec.call_args
            payload = call_args[0][1]
            assert payload["stream"] is True
