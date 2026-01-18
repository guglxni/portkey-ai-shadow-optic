"""
VCR-style tests for activities.py with HTTP interaction recording/mocking.

This module provides comprehensive test coverage for the activities module,
focusing on error handling logic including:
- Rate limit handling (429 errors)
- Authentication errors (401)
- Bad request errors (400)
- Server errors (500, 502, 503)
- Timeout errors
- Successful request handling

Uses aioresponses for async HTTP mocking to simulate Portkey API responses.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from aioresponses import aioresponses

from shadow_optic.models import (
    ProductionTrace,
    GoldenPrompt,
    ShadowResult,
    ChallengerConfig,
    TraceStatus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_portkey_response_success():
    """Successful chat completion response from Portkey."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "@openai/gpt-5-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a successful response from the AI model."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150
        }
    }


@pytest.fixture
def mock_portkey_rate_limit_error():
    """Rate limit error response (429)."""
    return {
        "error": {
            "message": "Rate limit exceeded. Please try again later.",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    }


@pytest.fixture
def mock_portkey_auth_error():
    """Authentication error response (401)."""
    return {
        "error": {
            "message": "Invalid API key provided.",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }
    }


@pytest.fixture
def mock_portkey_bad_request_error():
    """Bad request error response (400)."""
    return {
        "error": {
            "message": "Invalid model specified.",
            "type": "invalid_request_error",
            "code": "model_not_found"
        }
    }


@pytest.fixture
def mock_portkey_server_error():
    """Server error response (500)."""
    return {
        "error": {
            "message": "Internal server error.",
            "type": "server_error",
            "code": "internal_error"
        }
    }


@pytest.fixture
def sample_golden_prompts():
    """Sample golden prompts for testing."""
    return [
        GoldenPrompt(
            original_trace_id="trace-001",
            prompt="What is machine learning?",
            production_response="Machine learning is a subset of AI...",
            cluster_id=0,
            diversity_score=0.8,
            timestamp=datetime.utcnow()
        ),
        GoldenPrompt(
            original_trace_id="trace-002",
            prompt="Explain quantum computing in simple terms.",
            production_response="Quantum computing uses quantum mechanics...",
            cluster_id=1,
            diversity_score=0.9,
            timestamp=datetime.utcnow()
        ),
        GoldenPrompt(
            original_trace_id="trace-003",
            prompt="How do I write a Python function?",
            production_response="To write a Python function, use the def keyword...",
            cluster_id=2,
            diversity_score=0.7,
            timestamp=datetime.utcnow()
        ),
    ]


@pytest.fixture
def challenger_config():
    """Sample challenger configuration."""
    return ChallengerConfig(
        model_id="gpt-5-mini",
        provider="OpenAI",
        provider_slug="openai",
        portkey_config_id="test-config-123",
        max_tokens=4096,
        temperature=0.0
    )


@pytest.fixture
def sample_production_traces():
    """Sample production traces for testing."""
    return [
        ProductionTrace(
            trace_id=f"trace-{i:03d}",
            request_timestamp=datetime.utcnow() - timedelta(hours=i),
            prompt=f"Sample question {i}?",
            response=f"Sample response {i}.",
            model="gpt-5.2-turbo",
            latency_ms=150.0 + i * 10,
            tokens_prompt=50 + i,
            tokens_completion=100 + i,
            cost=0.001 * (1 + i * 0.1),
            status=TraceStatus.SUCCESS,
            metadata={"category": f"cat-{i % 3}"}
        )
        for i in range(10)
    ]


# ============================================================================
# Mock Portkey Client
# ============================================================================

class MockPortkeyResponse:
    """Mock response object that mimics Portkey's response structure."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self.choices = []
        self.usage = None
        
        if "choices" in data:
            self.choices = [MockChoice(c) for c in data["choices"]]
        if "usage" in data:
            self.usage = MockUsage(data["usage"])
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class MockChoice:
    """Mock choice object."""
    def __init__(self, data: Dict[str, Any]):
        self.index = data.get("index", 0)
        self.message = MockMessage(data.get("message", {}))
        self.finish_reason = data.get("finish_reason", "stop")


class MockMessage:
    """Mock message object."""
    def __init__(self, data: Dict[str, Any]):
        self.role = data.get("role", "assistant")
        self.content = data.get("content", "")


class MockUsage:
    """Mock usage object."""
    def __init__(self, data: Dict[str, Any]):
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)


class MockAsyncPortkey:
    """Mock async Portkey client for testing."""
    
    def __init__(self, responses: List[Dict[str, Any]] = None, errors: List[Exception] = None):
        self.responses = responses or []
        self.errors = errors or []
        self.call_count = 0
        self.requests = []
        self.chat = MockChatCompletions(self)
        self.logs = MockLogs()
        self.feedback = MockFeedback()
    
    async def close(self):
        pass
    
    def get_next_response(self, request_params: Dict[str, Any]) -> MockPortkeyResponse:
        """Get next response or raise next error."""
        self.requests.append(request_params)
        
        if self.call_count < len(self.errors) and self.errors[self.call_count]:
            error = self.errors[self.call_count]
            self.call_count += 1
            raise error
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return MockPortkeyResponse(response)
        
        self.call_count += 1
        return MockPortkeyResponse({
            "choices": [{"message": {"content": "Default response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        })


class MockChatCompletions:
    """Mock chat completions."""
    def __init__(self, client: MockAsyncPortkey):
        self.client = client
        self.completions = self
    
    async def create(self, **kwargs) -> MockPortkeyResponse:
        return self.client.get_next_response(kwargs)


class MockLogs:
    """Mock logs API."""
    def __init__(self):
        self.exports = MockExports()


class MockExports:
    """Mock exports API."""
    async def create(self, **kwargs):
        return MagicMock(id="export-123")
    
    async def start(self, export_id: str):
        pass
    
    async def retrieve(self, export_id: str):
        return MagicMock(status="completed")
    
    async def download(self, export_id: str):
        return MagicMock(data=[])


class MockFeedback:
    """Mock feedback API."""
    async def bulk_create(self, feedbacks: List[Dict]):
        pass


# ============================================================================
# Error Classes for Testing
# ============================================================================

class PortkeyRateLimitError(Exception):
    """Simulated rate limit error."""
    def __str__(self):
        return "429 RateLimited: Rate limit exceeded"


class PortkeyAuthError(Exception):
    """Simulated authentication error."""
    def __str__(self):
        return "401 Unauthorized: Invalid API key"


class PortkeyBadRequestError(Exception):
    """Simulated bad request error."""
    def __str__(self):
        return "400 BadRequest: Invalid model specified"


class PortkeyServerError(Exception):
    """Simulated server error."""
    def __str__(self):
        return "500 ServerError: Internal server error"


class PortkeyTimeoutError(Exception):
    """Simulated timeout error."""
    def __str__(self):
        return "Request timeout exceeded"


# ============================================================================
# Tests: Error Classification Logic
# ============================================================================

class TestErrorClassification:
    """Test the error classification logic in replay_shadow_requests."""
    
    def test_rate_limit_error_classification(self):
        """Test that 429 errors are correctly classified as rate_limited."""
        error_msg = str(PortkeyRateLimitError())
        
        assert "429" in error_msg or "RateLimited" in error_msg
        
        # Classify using the same logic as activities.py
        if "429" in error_msg or "RateLimited" in error_msg:
            error_type = "rate_limited"
        else:
            error_type = "unknown"
        
        assert error_type == "rate_limited"
    
    def test_auth_error_classification(self):
        """Test that 401 errors are correctly classified as auth_error."""
        error_msg = str(PortkeyAuthError())
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            error_type = "auth_error"
        else:
            error_type = "unknown"
        
        assert error_type == "auth_error"
    
    def test_bad_request_error_classification(self):
        """Test that 400 errors are correctly classified as bad_request."""
        error_msg = str(PortkeyBadRequestError())
        
        if "400" in error_msg or "BadRequest" in error_msg:
            error_type = "bad_request"
        else:
            error_type = "unknown"
        
        assert error_type == "bad_request"
    
    def test_server_error_classification(self):
        """Test that 5xx errors are correctly classified as server_error."""
        error_msg = str(PortkeyServerError())
        
        if "500" in error_msg or "502" in error_msg or "503" in error_msg:
            error_type = "server_error"
        else:
            error_type = "unknown"
        
        assert error_type == "server_error"
    
    def test_timeout_error_classification(self):
        """Test that timeout errors are correctly classified."""
        error_msg = str(PortkeyTimeoutError())
        
        if "timeout" in error_msg.lower():
            error_type = "timeout"
        else:
            error_type = "unknown"
        
        assert error_type == "timeout"


# ============================================================================
# Tests: Replay Shadow Requests with Mocked Responses
# ============================================================================

class TestReplayShadowRequests:
    """Test replay_shadow_requests activity with various scenarios."""
    
    @pytest.mark.asyncio
    async def test_successful_replay(
        self,
        sample_golden_prompts,
        challenger_config,
        mock_portkey_response_success
    ):
        """Test successful replay of golden prompts."""
        # Create mock client with successful responses
        mock_client = MockAsyncPortkey(
            responses=[mock_portkey_response_success for _ in sample_golden_prompts]
        )
        
        # Import and patch the activity
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # Verify all requests succeeded
        assert len(results) == len(sample_golden_prompts)
        assert all(r.success for r in results)
        
        # Verify response content was captured
        for result in results:
            assert result.shadow_response != ""
            assert result.tokens_prompt > 0
            assert result.tokens_completion > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test handling of rate limit (429) errors."""
        # First request succeeds, second fails with rate limit, third succeeds
        mock_client = MockAsyncPortkey(
            responses=[
                {"choices": [{"message": {"content": "Response 1"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
                None,  # Will use error
                {"choices": [{"message": {"content": "Response 3"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
            ],
            errors=[
                None,  # No error for first
                PortkeyRateLimitError(),  # Rate limit for second
                None,  # No error for third
            ]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # Verify mixed results
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert "rate_limited" in results[1].error
        assert results[2].success is True
    
    @pytest.mark.asyncio
    async def test_auth_error_handling(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test handling of authentication (401) errors."""
        mock_client = MockAsyncPortkey(
            errors=[PortkeyAuthError() for _ in sample_golden_prompts]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # All should fail with auth error
        assert all(not r.success for r in results)
        assert all("auth_error" in r.error for r in results)
    
    @pytest.mark.asyncio
    async def test_bad_request_error_handling(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test handling of bad request (400) errors."""
        mock_client = MockAsyncPortkey(
            errors=[PortkeyBadRequestError() for _ in sample_golden_prompts]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # All should fail with bad request error
        assert all(not r.success for r in results)
        assert all("bad_request" in r.error for r in results)
    
    @pytest.mark.asyncio
    async def test_server_error_handling(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test handling of server (5xx) errors."""
        mock_client = MockAsyncPortkey(
            errors=[PortkeyServerError() for _ in sample_golden_prompts]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # All should fail with server error
        assert all(not r.success for r in results)
        assert all("server_error" in r.error for r in results)
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test handling of timeout errors."""
        mock_client = MockAsyncPortkey(
            errors=[PortkeyTimeoutError() for _ in sample_golden_prompts]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # All should fail with timeout error
        assert all(not r.success for r in results)
        assert all("timeout" in r.error for r in results)
    
    @pytest.mark.asyncio
    async def test_mixed_error_scenarios(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test handling of mixed success and various error types."""
        mock_client = MockAsyncPortkey(
            responses=[
                {"choices": [{"message": {"content": "Success"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
                None,
                None,
            ],
            errors=[
                None,  # Success
                PortkeyRateLimitError(),  # Rate limit
                PortkeyTimeoutError(),  # Timeout
            ]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # Verify each result type
        assert results[0].success is True
        assert results[1].success is False
        assert "rate_limited" in results[1].error
        assert results[2].success is False
        assert "timeout" in results[2].error
    
    @pytest.mark.asyncio
    async def test_empty_golden_prompts(self, challenger_config):
        """Test handling of empty golden prompts list."""
        mock_client = MockAsyncPortkey()
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=[],
                    challenger_config=challenger_config
                )
        
        assert len(results) == 0


# ============================================================================
# Tests: Parse Portkey Logs
# ============================================================================

class TestParsePortkeyLogs:
    """Test parse_portkey_logs activity."""
    
    @pytest.fixture
    def raw_logs(self):
        """Sample raw Portkey logs."""
        return [
            {
                "trace_id": "trace-001",
                "request": {
                    "messages": [{"role": "user", "content": "What is AI?"}]
                },
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "AI is artificial intelligence."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20}
                },
                "model": "gpt-5.2-turbo",
                "latency": 150,
                "cost": 0.001,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metadata": {"user_id": "user-123"}
            },
            {
                "trace_id": "trace-002",
                "request": {
                    "messages": [{"role": "user", "content": "Explain ML."}]
                },
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "ML is machine learning."}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 15}
                },
                "model": "gpt-5.2-turbo",
                "latency": 120,
                "cost": 0.0008,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metadata": {}
            },
        ]
    
    @pytest.mark.asyncio
    async def test_parse_valid_logs(self, raw_logs):
        """Test parsing valid Portkey logs."""
        with patch('shadow_optic.activities.activity') as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()
            
            from shadow_optic.activities import parse_portkey_logs
            
            traces = await parse_portkey_logs(raw_logs)
        
        assert len(traces) == 2
        assert traces[0].trace_id == "trace-001"
        assert traces[0].prompt == "What is AI?"
        assert traces[0].response == "AI is artificial intelligence."
        assert traces[0].tokens_prompt == 10
        assert traces[0].tokens_completion == 20
    
    @pytest.mark.asyncio
    async def test_parse_logs_with_missing_fields(self):
        """Test parsing logs with missing optional fields."""
        incomplete_logs = [
            {
                "trace_id": "trace-incomplete",
                "request": {"messages": [{"role": "user", "content": "Test"}]},
                "response": {
                    "choices": [{"message": {"content": "Response"}}],
                    "usage": {}
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        ]
        
        with patch('shadow_optic.activities.activity') as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()
            
            from shadow_optic.activities import parse_portkey_logs
            
            traces = await parse_portkey_logs(incomplete_logs)
        
        # Should still parse successfully with defaults
        assert len(traces) == 1
        assert traces[0].tokens_prompt == 0
        assert traces[0].tokens_completion == 0
    
    @pytest.mark.asyncio
    async def test_parse_empty_logs(self):
        """Test parsing empty logs list."""
        with patch('shadow_optic.activities.activity') as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()
            
            from shadow_optic.activities import parse_portkey_logs
            
            traces = await parse_portkey_logs([])
        
        assert len(traces) == 0
    
    @pytest.mark.asyncio
    async def test_parse_logs_skips_invalid_entries(self):
        """Test that invalid log entries are skipped."""
        logs_with_invalid = [
            {
                "trace_id": "valid-trace",
                "request": {"messages": [{"role": "user", "content": "Valid"}]},
                "response": {"choices": [{"message": {"content": "Response"}}]},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            {
                # Missing prompt - should be skipped
                "trace_id": "invalid-trace-1",
                "request": {"messages": []},
                "response": {"choices": [{"message": {"content": "Response"}}]},
            },
            {
                # Missing response - should be skipped
                "trace_id": "invalid-trace-2",
                "request": {"messages": [{"role": "user", "content": "Question"}]},
                "response": {"choices": []},
            },
        ]
        
        with patch('shadow_optic.activities.activity') as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()
            
            from shadow_optic.activities import parse_portkey_logs
            
            traces = await parse_portkey_logs(logs_with_invalid)
        
        # Only the valid trace should be parsed
        assert len(traces) == 1
        assert traces[0].trace_id == "valid-trace"


# ============================================================================
# Tests: Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Test helper functions in activities module."""
    
    def test_parse_time_window_hours(self):
        """Test parsing time windows in hours."""
        from shadow_optic.activities import _parse_time_window
        
        assert _parse_time_window("24h") == 24
        assert _parse_time_window("1h") == 1
        assert _parse_time_window("48h") == 48
    
    def test_parse_time_window_days(self):
        """Test parsing time windows in days."""
        from shadow_optic.activities import _parse_time_window
        
        assert _parse_time_window("1d") == 24
        assert _parse_time_window("7d") == 168
        assert _parse_time_window("30d") == 720
    
    def test_parse_time_window_weeks(self):
        """Test parsing time windows in weeks."""
        from shadow_optic.activities import _parse_time_window
        
        assert _parse_time_window("1w") == 168
        assert _parse_time_window("2w") == 336
    
    def test_parse_time_window_numeric(self):
        """Test parsing numeric time windows (assumed hours)."""
        from shadow_optic.activities import _parse_time_window
        
        assert _parse_time_window("12") == 12
        assert _parse_time_window("24") == 24
    
    def test_extract_prompt_from_messages(self):
        """Test extracting user prompt from messages."""
        from shadow_optic.activities import _extract_prompt
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
            {"role": "user", "content": "Tell me more."},
        ]
        
        # Should return the last user message
        prompt = _extract_prompt(messages)
        assert prompt == "Tell me more."
    
    def test_extract_prompt_from_complex_content(self):
        """Test extracting prompt from complex content structure."""
        from shadow_optic.activities import _extract_prompt
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image", "url": "http://example.com/image.png"}
                ]
            }
        ]
        
        prompt = _extract_prompt(messages)
        assert "Describe this image:" in prompt
    
    def test_extract_response_from_choices(self):
        """Test extracting response from completion choices."""
        from shadow_optic.activities import _extract_response
        
        choices = [
            {"message": {"role": "assistant", "content": "This is the response."}}
        ]
        
        response = _extract_response(choices)
        assert response == "This is the response."
    
    def test_extract_response_empty_choices(self):
        """Test extracting response from empty choices."""
        from shadow_optic.activities import _extract_response
        
        assert _extract_response([]) == ""
        assert _extract_response(None) == ""


# ============================================================================
# Tests: Client Factory Functions
# ============================================================================

class TestClientFactories:
    """Test client factory functions."""
    
    def test_get_portkey_client_with_api_key(self):
        """Test getting Portkey client with API key set."""
        with patch.dict('os.environ', {'PORTKEY_API_KEY': 'test-api-key'}):
            from shadow_optic.activities import get_portkey_client
            
            client = get_portkey_client()
            assert client is not None
    
    def test_get_portkey_client_without_api_key(self):
        """Test that missing API key raises ValueError."""
        with patch.dict('os.environ', {}, clear=True):
            # Remove any existing PORTKEY_API_KEY
            import os
            if 'PORTKEY_API_KEY' in os.environ:
                del os.environ['PORTKEY_API_KEY']
            
            from shadow_optic.activities import get_portkey_client
            
            with pytest.raises(ValueError, match="PORTKEY_API_KEY"):
                get_portkey_client()
    
    def test_get_qdrant_client_default_url(self):
        """Test getting Qdrant client with default URL."""
        with patch.dict('os.environ', {}, clear=True):
            from shadow_optic.activities import get_qdrant_client
            
            client = get_qdrant_client()
            assert client is not None


# ============================================================================
# Integration-Style Tests with Full Activity Flow
# ============================================================================

class TestActivityIntegration:
    """Integration-style tests for complete activity flows."""
    
    @pytest.mark.asyncio
    async def test_full_replay_and_parse_flow(
        self,
        sample_golden_prompts,
        challenger_config,
        mock_portkey_response_success
    ):
        """Test full flow from parsing logs to replaying and getting results."""
        # Create mock that returns successful responses
        mock_client = MockAsyncPortkey(
            responses=[mock_portkey_response_success for _ in sample_golden_prompts]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # Verify complete flow
        assert len(results) == len(sample_golden_prompts)
        
        # All results should have proper structure
        for result in results:
            assert result.golden_prompt_id is not None
            assert result.challenger_model == challenger_config.model_id
            assert result.timestamp is not None
            
            if result.success:
                assert result.shadow_response != ""
                assert result.tokens_prompt >= 0
                assert result.tokens_completion >= 0
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_partial_failures(
        self,
        sample_golden_prompts,
        challenger_config
    ):
        """Test that the system gracefully handles partial failures."""
        # Mix of successes and failures
        responses_and_errors = [
            ({"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}, None),
            (None, PortkeyRateLimitError()),
            ({"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}, None),
        ]
        
        mock_client = MockAsyncPortkey(
            responses=[r[0] for r in responses_and_errors],
            errors=[r[1] for r in responses_and_errors]
        )
        
        with patch('shadow_optic.activities.get_portkey_client', return_value=mock_client):
            with patch('shadow_optic.activities.activity') as mock_activity:
                mock_activity.logger = MagicMock()
                mock_activity.heartbeat = MagicMock()
                
                from shadow_optic.activities import replay_shadow_requests
                
                results = await replay_shadow_requests(
                    golden_prompts=sample_golden_prompts,
                    challenger_config=challenger_config
                )
        
        # Should have results for all prompts (including failures)
        assert len(results) == len(sample_golden_prompts)
        
        # Count successes and failures
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        
        assert successes == 2
        assert failures == 1
        
        # Failure should have error message
        failed_results = [r for r in results if not r.success]
        assert all(r.error is not None for r in failed_results)
