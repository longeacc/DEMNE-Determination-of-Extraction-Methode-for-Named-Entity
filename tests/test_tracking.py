"""Tests for demne._tracking -- eco2ai energy tracking helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from demne._tracking import start_tracker, stop_tracker


# ---------------------------------------------------------------------------
# start_tracker
# ---------------------------------------------------------------------------
class TestStartTracker:
    """Test start_tracker under various conditions."""

    def test_returns_none_when_eco2ai_missing(self, monkeypatch):
        """Without eco2ai installed, start_tracker returns None."""
        monkeypatch.setattr("demne._tracking.HAS_ECO2AI", False)
        result = start_tracker("proj", "desc")
        assert result is None

    def test_returns_none_when_disabled_via_env(self, monkeypatch):
        """DISABLE_ECO2AI env var should prevent tracking."""
        monkeypatch.setattr("demne._tracking.HAS_ECO2AI", True)
        monkeypatch.setenv("DISABLE_ECO2AI", "1")
        result = start_tracker("proj", "desc")
        assert result is None

    def test_starts_tracker_when_available(self, monkeypatch):
        """When eco2ai is available and not disabled, a Tracker is returned."""
        mock_tracker_instance = MagicMock()
        mock_tracker_cls = MagicMock(return_value=mock_tracker_instance)
        mock_set_params = MagicMock()

        monkeypatch.setattr("demne._tracking.HAS_ECO2AI", True)
        monkeypatch.setattr("demne._tracking.Tracker", mock_tracker_cls)
        monkeypatch.setattr("demne._tracking.set_params", mock_set_params)
        monkeypatch.delenv("DISABLE_ECO2AI", raising=False)

        result = start_tracker("my-project", "my description", "custom.csv")

        mock_set_params.assert_called_once_with(
            project_name="my-project",
            experiment_description="my description",
            file_name="custom.csv",
        )
        mock_tracker_cls.assert_called_once()
        mock_tracker_instance.start.assert_called_once()
        assert result is mock_tracker_instance

    def test_default_file_name(self, monkeypatch):
        """Default file_name is data/energy_consumption_DEMNE.csv."""
        mock_set_params = MagicMock()
        monkeypatch.setattr("demne._tracking.HAS_ECO2AI", True)
        monkeypatch.setattr("demne._tracking.Tracker", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr("demne._tracking.set_params", mock_set_params)
        monkeypatch.delenv("DISABLE_ECO2AI", raising=False)

        start_tracker("p", "d")

        call_kwargs = mock_set_params.call_args[1]
        assert call_kwargs["file_name"] == "data/energy_consumption_DEMNE.csv"


# ---------------------------------------------------------------------------
# stop_tracker
# ---------------------------------------------------------------------------
class TestStopTracker:
    """Test stop_tracker safety."""

    def test_none_is_noop(self):
        """Passing None should not raise."""
        stop_tracker(None)  # should not raise

    def test_calls_stop(self):
        """A real tracker's .stop() is called."""
        mock = MagicMock()
        stop_tracker(mock)
        mock.stop.assert_called_once()

    def test_exception_is_caught(self, capsys):
        """If tracker.stop() raises, the error is caught and printed."""
        mock = MagicMock()
        mock.stop.side_effect = RuntimeError("GPU unavailable")
        stop_tracker(mock)  # should not raise
        captured = capsys.readouterr()
        assert "GPU unavailable" in captured.out
        assert "Warning" in captured.out
