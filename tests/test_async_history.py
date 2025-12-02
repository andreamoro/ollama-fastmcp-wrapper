"""
Tests for async conversation history functionality.

These tests verify that the async save/load operations work correctly
and don't block the event loop.
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from ollama_wrapper import MessageHistory, OllamaWrapper
from fastapi.testclient import TestClient


@pytest.fixture
def temp_history_file(tmp_path):
    """Create temporary file path for history"""
    return tmp_path / "test_history.json"


@pytest.fixture
def sample_messages():
    """Sample message history for testing"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "I don't have access to weather data."}
    ]


class TestMessageHistoryAsync:
    """Test async save/load operations for MessageHistory"""

    @pytest.mark.asyncio
    async def test_save_creates_file(self, temp_history_file, sample_messages):
        """Test that save() creates a JSON file with correct content"""
        history = MessageHistory(system_prompt="You are a helpful assistant.")

        # Add messages
        for msg in sample_messages[1:]:  # Skip system prompt (already in __init__)
            history.add(msg["role"], msg["content"])

        # Save to file
        await history.save(str(temp_history_file))

        # Verify file exists
        assert temp_history_file.exists()

        # Verify content
        with open(temp_history_file, 'r') as f:
            data = json.load(f)

        assert "messages" in data
        assert "summary" in data
        assert len(data["messages"]) == len(sample_messages)

    @pytest.mark.asyncio
    async def test_load_restores_history(self, temp_history_file, sample_messages):
        """Test that load() correctly restores saved conversation"""
        # Create and save history
        history1 = MessageHistory(system_prompt="You are a helpful assistant.")
        for msg in sample_messages[1:]:
            history1.add(msg["role"], msg["content"])
        await history1.save(str(temp_history_file))

        # Load into new instance
        history2 = MessageHistory()
        await history2.load(str(temp_history_file))

        # Verify messages match
        assert len(history2.messages) == len(sample_messages)
        for i, msg in enumerate(sample_messages):
            assert history2.messages[i]["role"] == msg["role"]
            assert history2.messages[i]["content"] == msg["content"]

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, temp_history_file):
        """Test full save/load cycle preserves all data"""
        # Create history with messages and summary
        history1 = MessageHistory(system_prompt="Test system prompt")
        history1.add("user", "Test message 1")
        history1.add("assistant", "Test response 1")
        history1.summary = "This is a test summary"

        # Save
        await history1.save(str(temp_history_file))

        # Load into new instance
        history2 = MessageHistory()
        await history2.load(str(temp_history_file))

        # Assert all fields match
        assert len(history2.messages) == len(history1.messages)
        assert history2.summary == history1.summary
        for i in range(len(history1.messages)):
            assert history2.messages[i] == history1.messages[i]

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, tmp_path):
        """Test that multiple saves don't block each other"""
        # Create multiple MessageHistory instances
        histories = []
        files = []
        for i in range(5):
            history = MessageHistory(system_prompt=f"Assistant {i}")
            history.add("user", f"Message from user {i}")
            history.add("assistant", f"Response from assistant {i}")
            histories.append(history)
            files.append(tmp_path / f"history_{i}.json")

        # Time concurrent saves
        start_time = time.time()
        await asyncio.gather(*[
            history.save(str(file))
            for history, file in zip(histories, files)
        ])
        concurrent_time = time.time() - start_time

        # Verify all files created
        for file in files:
            assert file.exists()

        # Concurrent saves should be reasonably fast (less than 1 second for 5 files)
        assert concurrent_time < 1.0

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, tmp_path):
        """Test that load() raises appropriate error for missing file"""
        history = MessageHistory()
        non_existent_file = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError):
            await history.load(str(non_existent_file))

    @pytest.mark.asyncio
    async def test_save_overwrites_existing_file(self, temp_history_file):
        """Test that save() overwrites existing files"""
        # Create and save initial history
        history1 = MessageHistory(system_prompt="First version")
        history1.add("user", "First message")
        await history1.save(str(temp_history_file))

        # Create and save different history to same file
        history2 = MessageHistory(system_prompt="Second version")
        history2.add("user", "Second message")
        await history2.save(str(temp_history_file))

        # Load and verify it has the second version
        history3 = MessageHistory()
        await history3.load(str(temp_history_file))
        assert history3.messages[0]["content"] == "Second version"
        assert history3.messages[1]["content"] == "Second message"


class TestOllamaWrapperHistoryEndpoints:
    """Test FastAPI history endpoints with async operations"""

    @pytest.fixture
    def wrapper(self):
        """Create OllamaWrapper instance for testing"""
        return OllamaWrapper(model="llama3.2:3b")

    @pytest.fixture
    def client(self, wrapper):
        """Create TestClient for FastAPI app"""
        return TestClient(wrapper.app)

    def test_save_history_endpoint(self, wrapper, client, tmp_path):
        """Test /history/save/{file_name} endpoint"""
        # Add messages to history
        wrapper.message_history.add("user", "Test message")
        wrapper.message_history.add("assistant", "Test response")

        # Call save endpoint
        response = client.get("/history/save/test_save")

        assert response.status_code == 200
        assert "test_save" in response.json()["detail"]

        # Verify file created
        file_path = Path("messages_history/test_save.json")
        assert file_path.exists()

        # Cleanup
        if file_path.exists():
            file_path.unlink()

    def test_load_history_endpoint(self, wrapper, client):
        """Test /history/load/{file_name} endpoint"""
        # Save a history file first
        wrapper.message_history.add("user", "Load test message")
        wrapper.message_history.add("assistant", "Load test response")
        client.get("/history/save/test_load")

        # Clear history
        wrapper.message_history.reset()
        assert len(wrapper.message_history.messages) == 1  # Only system prompt

        # Load history
        response = client.get("/history/load/test_load")

        assert response.status_code == 200
        assert len(wrapper.message_history.messages) > 1

        # Cleanup
        file_path = Path("messages_history/test_load.json")
        if file_path.exists():
            file_path.unlink()

    def test_overwrite_endpoint(self, wrapper, client):
        """Test /history/overwrite/{file_name} endpoint"""
        # Save initial history
        wrapper.message_history.add("user", "First version")
        client.get("/history/save/test_overwrite")

        # Modify history
        wrapper.message_history.add("user", "Second version")

        # Try to save without overwrite (should fail)
        response = client.get("/history/save/test_overwrite")
        assert response.status_code == 500

        # Save with overwrite endpoint (should succeed)
        response = client.get("/history/overwrite/test_overwrite")
        assert response.status_code == 200

        # Cleanup
        file_path = Path("messages_history/test_overwrite.json")
        if file_path.exists():
            file_path.unlink()

    def test_load_nonexistent_file(self, wrapper, client):
        """Test loading a file that doesn't exist"""
        response = client.get("/history/load/nonexistent_file_12345")
        assert response.status_code == 500
        assert "Load history failed" in response.json()["detail"]

    def test_get_history_endpoint(self, wrapper, client):
        """Test /history endpoint returns current conversation"""
        wrapper.message_history.add("user", "Test message")
        wrapper.message_history.add("assistant", "Test response")

        response = client.get("/history")

        assert response.status_code == 200
        messages = response.json()["history"]
        assert len(messages) >= 2
        assert any(msg["role"] == "user" for msg in messages)
        assert any(msg["role"] == "assistant" for msg in messages)

    def test_clear_history_endpoint(self, wrapper, client):
        """Test /history/clear endpoint"""
        wrapper.message_history.add("user", "Message to clear")
        wrapper.message_history.add("assistant", "Response to clear")

        response = client.get("/history/clear")

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        # Should only have system prompt left
        assert len(wrapper.message_history.messages) == 1


class TestLifespanHistoryLoading:
    """Test that history loads during FastAPI lifespan startup"""

    @pytest.mark.asyncio
    async def test_lifespan_loads_history(self, temp_history_file):
        """Test that wrapper loads history file on startup"""
        # Create and save a history file
        history = MessageHistory(system_prompt="Test assistant")
        history.add("user", "Saved message")
        history.add("assistant", "Saved response")
        await history.save(str(temp_history_file))

        # Create wrapper with history_file parameter
        wrapper = OllamaWrapper(
            model="llama3.2:3b",
            history_file=str(temp_history_file)
        )

        # Trigger lifespan startup by creating TestClient
        with TestClient(wrapper.app):
            # Verify history loaded
            assert len(wrapper.message_history.messages) > 1
            assert any(msg["content"] == "Saved message" for msg in wrapper.message_history.messages)

    @pytest.mark.asyncio
    async def test_lifespan_handles_missing_file(self, tmp_path):
        """Test that missing history file doesn't crash startup"""
        non_existent_file = tmp_path / "does_not_exist.json"

        # Create wrapper with non-existent history file
        wrapper = OllamaWrapper(
            model="llama3.2:3b",
            history_file=str(non_existent_file)
        )

        # Should not raise error, starts with empty history
        with TestClient(wrapper.app):
            # Should have only system prompt
            assert len(wrapper.message_history.messages) == 1
            assert wrapper.message_history.messages[0]["role"] == "system"


class TestMessageHistoryTrimming:
    """Test message history trimming and summarization"""

    @pytest.mark.asyncio
    async def test_history_preserves_data_after_trim(self, temp_history_file):
        """Test that save/load works correctly after history trimming"""
        # Create history with small max_messages to trigger trimming
        history = MessageHistory(
            system_prompt="Test assistant",
            max_messages=5
        )

        # Add enough messages to trigger trimming
        for i in range(10):
            history.add("user", f"Message {i}")
            history.add("assistant", f"Response {i}")

        # Save history (including potential summary)
        await history.save(str(temp_history_file))

        # Load into new instance
        history2 = MessageHistory()
        await history2.load(str(temp_history_file))

        # Verify messages loaded correctly
        assert len(history2.messages) > 0
        # Summary might exist if trimming occurred
        assert history2.summary == history.summary


# Cleanup fixture to remove test files
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test"""
    yield
    # Cleanup messages_history directory if it exists
    messages_dir = Path("messages_history")
    if messages_dir.exists():
        for file in messages_dir.glob("test_*.json"):
            try:
                file.unlink()
            except Exception:
                pass
