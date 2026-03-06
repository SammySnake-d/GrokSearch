"""Tests for the strip_thinking_tags function."""

import pytest
from grok_search.providers.grok import strip_thinking_tags


class TestStripThinkingTags:
    """Test strip_thinking_tags removes thinking content correctly."""

    def test_removes_thinking_tags(self):
        text = "<thinking>some internal thought</thinking>actual content"
        assert strip_thinking_tags(text) == "actual content"

    def test_removes_think_tags(self):
        text = "<think>some internal thought</think>actual content"
        assert strip_thinking_tags(text) == "actual content"

    def test_removes_multiline_thinking(self):
        text = "<thinking>\nline1\nline2\nline3\n</thinking>\nactual content"
        assert strip_thinking_tags(text) == "actual content"

    def test_removes_multiline_think(self):
        text = "<think>\nline1\nline2\n</think>\nactual content"
        assert strip_thinking_tags(text) == "actual content"

    def test_handles_unclosed_thinking_tag(self):
        text = "<thinking>some internal thought that never closes"
        assert strip_thinking_tags(text) == ""

    def test_handles_unclosed_think_tag(self):
        text = "<think>some internal thought that never closes"
        assert strip_thinking_tags(text) == ""

    def test_preserves_content_without_tags(self):
        text = "just regular content without any thinking tags"
        assert strip_thinking_tags(text) == "just regular content without any thinking tags"

    def test_empty_string(self):
        assert strip_thinking_tags("") == ""

    def test_multiple_thinking_blocks(self):
        text = "<thinking>thought1</thinking>content1<thinking>thought2</thinking>content2"
        assert strip_thinking_tags(text) == "content1content2"

    def test_thinking_with_json_content(self):
        text = '<thinking>I need to search for this</thinking>[{"title": "Result", "url": "http://example.com", "description": "A result"}]'
        result = strip_thinking_tags(text)
        assert '"title": "Result"' in result
        assert "<thinking>" not in result

    def test_nested_angle_brackets_in_content(self):
        """Content with angle brackets that are not thinking tags should be preserved."""
        text = "Use <code>hello</code> for greeting"
        assert strip_thinking_tags(text) == "Use <code>hello</code> for greeting"

    def test_thinking_at_end_with_content_before(self):
        text = "actual content<thinking>some thought</thinking>"
        assert strip_thinking_tags(text) == "actual content"

    def test_thinking_in_middle(self):
        text = "before<thinking>thought</thinking>after"
        assert strip_thinking_tags(text) == "beforeafter"

    def test_unclosed_thinking_after_content(self):
        text = "actual content\n<thinking>some thought without closing"
        assert strip_thinking_tags(text) == "actual content"

    def test_empty_thinking_tags(self):
        text = "<thinking></thinking>content"
        assert strip_thinking_tags(text) == "content"

    def test_empty_think_tags(self):
        text = "<think></think>content"
        assert strip_thinking_tags(text) == "content"
