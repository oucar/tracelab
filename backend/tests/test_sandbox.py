"""Sandbox executor: limits, isolation, artifact collection."""

import textwrap
from pathlib import Path

import pytest

from app.sandbox.executor import run_code


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n3,4\n5,6\n")
    return csv


def test_runs_code_and_reads_dataset(dataset):
    result = run_code(
        "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint('sum_a', df['a'].sum())",
        dataset,
    )
    assert result.exit_code == 0
    assert "sum_a 9" in result.stdout


def test_timeout_kills_infinite_loop(dataset):
    result = run_code("while True:\n    pass", dataset)
    assert result.timed_out or result.exit_code != 0


def test_stderr_captured_on_crash(dataset):
    result = run_code("raise ValueError('boom')", dataset)
    assert result.exit_code != 0
    assert "boom" in result.stderr


def test_collects_json_artifacts(dataset):
    code = textwrap.dedent("""
        import json
        with open('artifacts/chart.json', 'w') as f:
            json.dump({"kind": "bar", "x": [1, 2]}, f)
        print('done')
    """)
    result = run_code(code, dataset)
    assert result.exit_code == 0
    assert result.artifacts and result.artifacts[0]["data"]["kind"] == "bar"


def test_literal_escaped_newlines_are_normalized(dataset):
    # Structured output sometimes delivers the whole script on one line with
    # literal \n escapes; the executor must unescape that corruption.
    result = run_code("print('a')\\nprint('b')", dataset)
    assert result.exit_code == 0
    assert "a\nb" in result.stdout


def test_no_api_keys_in_environment(dataset):
    result = run_code(
        "import os\nprint('KEY' if os.environ.get('OPENAI_API_KEY') else 'CLEAN')", dataset
    )
    assert "CLEAN" in result.stdout
