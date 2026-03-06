"""Tests for the ask_grok tool and GrokSearchProvider.consult() method."""
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from grok_search.providers.grok import GrokSearchProvider, _estimate_messages_chars
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


# ---------- Payload size & truncation ----------

class TestEstimateMessagesChars:
    """测试 _estimate_messages_chars 辅助函数"""

    def test_empty_messages(self):
        assert _estimate_messages_chars([]) == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _estimate_messages_chars(messages) == 5

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "abc"},
            {"role": "user", "content": "defgh"},
        ]
        assert _estimate_messages_chars(messages) == 8

    def test_message_without_content(self):
        messages = [{"role": "user"}]
        assert _estimate_messages_chars(messages) == 0


class TestConsultPayloadLimit:
    """测试 consult() 对 payload 过大的处理"""

    @pytest.fixture
    def provider(self):
        return GrokSearchProvider("https://test.api/v1", "test-key", "grok-test")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROK_API_URL": "https://test.api", "GROK_API_KEY": "test-key", "GROK_MAX_PAYLOAD_CHARS": "2000"})
    async def test_consult_truncates_context_when_too_large(self, provider):
        """当 context 过大时，consult 应自动截断 context"""
        Config._instance = None
        try:
            short_question = "What is AI?"
            large_context = "x" * 5000  # 远超 2000 字符限制

            with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value='{"answer":"ok","sources":[],"follow_up_searches":[]}') as mock_exec:
                result = await provider.consult(short_question, context=large_context)
                # 应成功调用 API（截断后 payload 适当大小）
                assert mock_exec.called
                # 验证 payload 中的 context 已被截断
                call_args = mock_exec.call_args
                payload = call_args[0][1]
                user_content = payload["messages"][1]["content"]
                assert "自动截断" in user_content
                assert len(user_content) < len(large_context)
        finally:
            Config._instance = None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROK_API_URL": "https://test.api", "GROK_API_KEY": "test-key", "GROK_MAX_PAYLOAD_CHARS": "100"})
    async def test_consult_returns_error_when_question_too_large(self, provider):
        """当 question 本身过大且无法截断时，consult 返回错误 JSON"""
        Config._instance = None
        try:
            huge_question = "q" * 200  # 远超 100 字符限制

            with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock) as mock_exec:
                result = await provider.consult(huge_question)
                # 不应调用 API
                assert not mock_exec.called
                parsed = json.loads(result)
                assert parsed["note"] == "payload_too_large"
                assert "请求内容过大" in parsed["answer"]
                assert "建议" in parsed["answer"]
        finally:
            Config._instance = None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROK_API_URL": "https://test.api", "GROK_API_KEY": "test-key"})
    async def test_consult_normal_payload_passes_through(self, provider):
        """正常大小的 payload 应不被截断或拒绝"""
        Config._instance = None
        try:
            with patch.object(provider, "_execute_stream_with_retry", new_callable=AsyncMock, return_value='{"answer":"ok","sources":[],"follow_up_searches":[]}') as mock_exec:
                result = await provider.consult("short question", context="short context")
                assert mock_exec.called
                parsed = json.loads(result)
                assert parsed["answer"] == "ok"
                # 验证 context 未被截断
                payload = mock_exec.call_args[0][1]
                user_content = payload["messages"][1]["content"]
                assert "自动截断" not in user_content
        finally:
            Config._instance = None


class TestExecuteStreamBadRequest:
    """测试 _execute_stream_with_retry 对 400 错误的处理"""

    @pytest.fixture
    def provider(self):
        return GrokSearchProvider("https://test.api/v1", "test-key", "grok-test")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROK_API_URL": "https://test.api", "GROK_API_KEY": "test-key"})
    async def test_400_raises_value_error(self, provider):
        """400 Bad Request 应抛出 ValueError 并包含有用信息"""
        Config._instance = None
        try:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.headers = {}
            mock_response.text = "Bad Request"

            error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)

            mock_stream_cm = AsyncMock()
            mock_stream_cm.__aenter__ = AsyncMock(side_effect=error)
            mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient") as MockClient:
                mock_client_instance = AsyncMock()
                mock_client_instance.stream = MagicMock(return_value=mock_stream_cm)
                MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
                MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

                payload = {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hello"},
                    ]
                }

                with pytest.raises(ValueError, match="400 Bad Request"):
                    await provider._execute_stream_with_retry({}, payload)
        finally:
            Config._instance = None
