# tests/test_dashboard_agents.py
import os
import sys

import pytest

pytest.importorskip("streamlit", reason="dashboard deps live in dashboard/requirements.txt")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
import app  # dashboard/app.py


# ── classify_run_output ───────────────────────────────────────────────────────
def test_classify_lock():
    s = app.classify_run_output(0, "[LOCK] main.py is already running. Exiting.", "")
    assert s["locked"] is True
    assert s["ok"] is True
    assert s["auth_error"] is False


def test_classify_auth_error():
    s = app.classify_run_output(
        1, "starting agents", "anthropic.AuthenticationError: Error code 401 - invalid x-api-key"
    )
    assert s["auth_error"] is True
    assert s["ok"] is False


def test_classify_success():
    s = app.classify_run_output(0, "PIPELINE COMPLETE", "")
    assert s["ok"] is True
    assert s["locked"] is False
    assert s["auth_error"] is False


def test_classify_failure_nonzero():
    s = app.classify_run_output(2, "", "Traceback (most recent call last): ...")
    assert s["ok"] is False
    assert "code 2" in s["message"]


# ── formatting helpers ────────────────────────────────────────────────────────
def test_fmt_pct_fraction():
    assert app._fmt_pct(-0.0153) == "-1.53%"
    assert app._fmt_pct(1.1549) == "115.49%"
    assert app._fmt_pct(None) == "—"
    assert app._fmt_pct(True) == "—"  # bool guarded


def test_fmt_money_sign_and_commas():
    assert app._fmt_money(-786.79) == "-$786.79"
    assert app._fmt_money(2260.33) == "$2,260.33"
    assert app._fmt_money(None) == "—"


def test_fmt_num():
    assert app._fmt_num(2.0444, 2) == "2.04"
    assert app._fmt_num(None) == "—"


# ── run_pipeline ──────────────────────────────────────────────────────────────
def test_run_pipeline_dry_run_argv():
    captured = {}

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    res = app.run_pipeline(True, runner=fake_runner)
    assert "--dry-run" in captured["cmd"]
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1].endswith("main.py")
    assert captured["kwargs"]["cwd"] == str(app.ROOT)
    assert res["returncode"] == 0
    assert res["stdout"] == "ok"


def test_run_pipeline_full_run_omits_dry_flag():
    captured = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd

        class P:
            returncode, stdout, stderr = 0, "", ""

        return P()

    app.run_pipeline(False, runner=fake_runner)
    assert "--dry-run" not in captured["cmd"]


# ── available_runs / load_run_file ────────────────────────────────────────────
import json as _json


def test_available_runs_and_load_run_file(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "last_run.json").write_text(_json.dumps({"timestamp": "2026-06-06T10:00:00"}))
    (runs / "2026-06-05_09-00.json").write_text(
        _json.dumps({"timestamp": "2026-06-05T09:00:00",
                     "pm": {"recommendations": [{"ticker": "AAPL"}]}})
    )
    (runs / "2026-06-04_09-00.json").write_text(
        _json.dumps({"timestamp": "2026-06-04T09:00:00"})
    )
    monkeypatch.setattr(app, "LOGS", tmp_path)

    opts = app.available_runs()
    assert opts[0] == ("Latest (last_run.json)", None)
    labels = [o[0] for o in opts]
    assert labels[1].startswith("2026-06-05")  # newest historical first
    assert labels[2].startswith("2026-06-04")

    chosen_path = opts[1][1]
    data = app.load_run_file(chosen_path)
    assert data["pm"]["recommendations"][0]["ticker"] == "AAPL"
