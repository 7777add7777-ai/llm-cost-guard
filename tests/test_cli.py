"""CLI tests: the demo must run end to end, offline and deterministically."""

import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def demo_output():
    proc = subprocess.run(
        [sys.executable, "-m", "cost_guard.cli", "demo"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc


def test_demo_exit_code(demo_output):
    assert demo_output.returncode == 0, demo_output.stderr


def test_demo_reports_budget_and_block(demo_output):
    out = demo_output.stdout
    assert "Spent $" in out
    assert "Blocked at" in out
    assert "BLOCK" in out


def test_demo_shows_model_breakdown(demo_output):
    out = demo_output.stdout
    assert "Spend by model:" in out
    assert "gpt-4o-mini" in out


def test_demo_is_deterministic():
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-m", "cost_guard.cli", "demo"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
