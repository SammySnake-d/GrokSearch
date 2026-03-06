"""Tests for multi-turn conversation support."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from grok_search.providers.grok import GrokSearchProvider
from grok_search.utils import trim_history, advisor_prompt


# ---------- trim_history ----------

class TestTrimHistory:
    """测试 trim_history 对话历史截断函数"""

    def test_empty_history(self):
        assert trim_history([]) == []

    def test_within_limit(self):
        """历史在限制范围内，不截断"""
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert trim_history(history, max_turns=10) == history

    def test_exact_limit(self):
        """历史恰好等于限制，不截断"""
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        assert trim_history(history, max_turns=2) == history

    def test_exceeds_limit(self):
        """历史超出限制，截断保留最近 N 轮"""
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        result = trim_history(history, max_turns=2)
        assert len(result) == 4
        assert result[0]["content"] == "q2"
        assert result[3]["content"] == "a3"

    def test_single_turn_limit(self):
        """max_turns=1 只保留最后一轮"""
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = trim_history(history, max_turns=1)
        assert len(result) == 2
        assert result[0]["content"] == "q2"

    def test_default_max_turns(self):
        """默认 max_turns=10，20条消息以内不截断"""
        history = [
            {"role": "user", "content": f"q{i}"}
            if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
            for i in range(20)
        ]
        result = trim_history(history)
        assert len(result) == 20


# ---------- advisor_prompt ----------

class TestAdvisorPrompt:
    """测试 advisor_prompt 常量"""

    def test_advisor_prompt_exists(self):
        assert advisor_prompt is not None
        assert len(advisor_prompt) > 0

    def test_advisor_prompt_contains_key_instructions(self):
        assert "expert advisor" in advisor_prompt
        assert "Sources" in advisor_prompt
        assert "sources" in advisor_prompt
        assert "follow_up_searches" in advisor_prompt


# ---------- consult_with_messages ----------

class TestConsultWithMessages:
    """测试 GrokSearchProvider.consult_with_messages() 方法"""

    @pytest.fixture
    def provider(self):
        return GrokSearchProvider("https://test.api/v1", "test-key", "grok-test")

    @pytest.mark.asyncio
    async def test_consult_with_messages_passes_messages_directly(self, provider):
        """consult_with_messages 应直接透传 messages 列表"""
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(
            provider, "_execute_stream_with_retry",
            new_callable=AsyncMock, return_value="Hello!"
        ) as mock_exec:
            result = await provider.consult_with_messages(messages)

            call_args = mock_exec.call_args
            payload = call_args[0][1]  # headers, payload, ctx
            assert payload["messages"] == messages
            assert payload["model"] == "grok-test"
            assert payload["stream"] is True
            assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_consult_with_messages_multi_turn(self, provider):
        """consult_with_messages 支持多轮对话消息"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up question"},
        ]

        with patch.object(
            provider, "_execute_stream_with_retry",
            new_callable=AsyncMock, return_value="Follow-up answer"
        ) as mock_exec:
            result = await provider.consult_with_messages(messages)

            call_args = mock_exec.call_args
            payload = call_args[0][1]
            assert len(payload["messages"]) == 4
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][1]["role"] == "user"
            assert payload["messages"][2]["role"] == "assistant"
            assert payload["messages"][3]["role"] == "user"
            assert result == "Follow-up answer"

    @pytest.mark.asyncio
    async def test_consult_with_messages_uses_correct_headers(self, provider):
        """consult_with_messages 应使用正确的 Authorization 头"""
        messages = [{"role": "user", "content": "test"}]

        with patch.object(
            provider, "_execute_stream_with_retry",
            new_callable=AsyncMock, return_value="ok"
        ) as mock_exec:
            await provider.consult_with_messages(messages)

            call_args = mock_exec.call_args
            headers = call_args[0][0]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"
