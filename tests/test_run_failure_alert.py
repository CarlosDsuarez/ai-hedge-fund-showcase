# tests/test_run_failure_alert.py — Telegram alert on pipeline failure (auditoría Fase A, prompt 5)
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_mod


def test_run_failure_sends_telegram_and_reraises():
    with (
        patch.object(main_mod, "_acquire_run_lock", return_value=True),
        patch.object(main_mod, "_release_run_lock") as mock_release,
        patch.object(main_mod, "_run", side_effect=RuntimeError("boom")),
        patch("alerts.telegram_alert.send_message") as mock_send,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            main_mod.main()

    mock_send.assert_called_once()
    assert "RUN FAILED" in mock_send.call_args.args[0]
    assert "boom" in mock_send.call_args.args[0]
    mock_release.assert_called_once()  # lock released even on failure


def test_run_success_sends_no_failure_alert():
    with (
        patch.object(main_mod, "_acquire_run_lock", return_value=True),
        patch.object(main_mod, "_release_run_lock"),
        patch.object(main_mod, "_run", return_value=None),
        patch("alerts.telegram_alert.send_message") as mock_send,
    ):
        main_mod.main()

    mock_send.assert_not_called()


def test_alert_failure_does_not_mask_original_error():
    with (
        patch.object(main_mod, "_acquire_run_lock", return_value=True),
        patch.object(main_mod, "_release_run_lock"),
        patch.object(main_mod, "_run", side_effect=ValueError("original")),
        patch("alerts.telegram_alert.send_message", side_effect=ConnectionError("telegram down")),
    ):
        with pytest.raises(ValueError, match="original"):
            main_mod.main()
