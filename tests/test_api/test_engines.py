"""Tests for get_search_engines API function."""

from unittest.mock import MagicMock, patch

import pytest

from py_search_helper import get_search_engines
from py_search_helper.exceptions import EngineError


def test_get_search_engines_returns_list_of_tuples() -> None:
    """Test that get_search_engines returns list of tuples."""
    engines = get_search_engines()

    assert isinstance(engines, list)
    assert len(engines) > 0
    assert all(isinstance(e, tuple) for e in engines)
    assert all(len(e) == 2 for e in engines)
    assert all(isinstance(e[0], str) and isinstance(e[1], str) for e in engines)


def test_get_search_engines_includes_ddgs() -> None:
    """Test that DDGS engine is registered by default."""
    engines = get_search_engines()
    engine_ids = [e[0] for e in engines]

    assert "ddgs" in engine_ids


def test_get_search_engines_returns_descriptions() -> None:
    """Test that engines include descriptions."""
    engines = get_search_engines()

    for engine_id, description in engines:
        assert engine_id
        assert description
        assert len(description) > 0


@patch("py_search_helper.api.engines.get_registry")
def test_get_search_engines_uses_registry(mock_get_registry: MagicMock) -> None:
    """Test that get_search_engines uses the registry."""
    mock_registry = MagicMock()
    mock_registry.get_all_engines.return_value = [
        ("test1", "Test Engine 1"),
        ("test2", "Test Engine 2"),
    ]
    mock_get_registry.return_value = mock_registry

    engines = get_search_engines()

    assert engines == [("test1", "Test Engine 1"), ("test2", "Test Engine 2")]
    mock_registry.get_all_engines.assert_called_once()


@patch("py_search_helper.api.engines.get_registry")
def test_get_search_engines_raises_engine_error_on_failure(mock_get_registry: MagicMock) -> None:
    """Test that registry failures raise EngineError."""
    mock_registry = MagicMock()
    mock_registry.get_all_engines.side_effect = Exception("Registry failure")
    mock_get_registry.return_value = mock_registry

    with pytest.raises(EngineError, match="Failed to retrieve engines"):
        get_search_engines()


def test_get_search_engines_consistency() -> None:
    """Test that get_search_engines returns consistent results."""
    engines1 = get_search_engines()
    engines2 = get_search_engines()

    # Should return same engines (order may vary in general, but our registry is deterministic)
    assert len(engines1) == len(engines2)
    assert set(e[0] for e in engines1) == set(e[0] for e in engines2)
