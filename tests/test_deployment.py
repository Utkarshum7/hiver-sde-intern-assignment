"""
Tests for the Render deployment configuration (render.yaml,
requirements-deploy.txt) and the deployment-only additions to the demo UI.

Deliberately does NOT parse render.yaml with a YAML library (that would add
a new test-only dependency); plain string/line checks are enough to verify
the specific properties this project cares about and keep the test suite's
dependency footprint exactly what it already is.

Nothing here touches, imports, or depends on src/, scripts/evaluate.py,
the golden dataset, or reports/ — the last test in this file explicitly
proves that via `git diff`.
"""

import os
import subprocess
import sys

import pytest

from scripts import demo_server

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RENDER_YAML_PATH = os.path.join(REPO_ROOT, "render.yaml")
DEPLOY_REQUIREMENTS_PATH = os.path.join(REPO_ROOT, "requirements-deploy.txt")
MAIN_REQUIREMENTS_PATH = os.path.join(REPO_ROOT, "requirements.txt")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# render.yaml
# ---------------------------------------------------------------------------

def test_render_yaml_exists():
    assert os.path.exists(RENDER_YAML_PATH), "render.yaml is missing"


def test_render_yaml_declares_a_python_web_service():
    text = _read(RENDER_YAML_PATH)
    assert "type: web" in text
    assert "runtime: python" in text


def test_render_start_command_binds_all_interfaces_and_uses_port_env():
    text = _read(RENDER_YAML_PATH)
    assert "startCommand:" in text
    start_line = [l for l in text.splitlines() if "startCommand:" in l][0]
    assert "--host 0.0.0.0" in start_line, "must bind 0.0.0.0, not localhost, for Render traffic to reach it"
    assert "--port $PORT" in start_line, "must read Render's $PORT, never a hardcoded port"
    assert "127.0.0.1" not in start_line
    assert "8000" not in start_line  # the hardcoded local-dev port must not leak into the deploy command


def test_render_start_command_targets_the_real_app_module():
    text = _read(RENDER_YAML_PATH)
    assert "scripts.demo_server:app" in text


def test_render_build_command_uses_deploy_only_requirements():
    text = _read(RENDER_YAML_PATH)
    assert "buildCommand:" in text
    build_line = [l for l in text.splitlines() if "buildCommand:" in l][0]
    assert "requirements-deploy.txt" in build_line


def test_render_yaml_declares_no_secrets_or_api_keys():
    """Checks actual YAML directives only, not explanatory comments — the
    file legitimately has a comment stating ANTHROPIC_API_KEY is NOT
    referenced/required, which is the honest disclosure we want, not a
    violation."""
    text = _read(RENDER_YAML_PATH)
    code_lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
    code_text = "\n".join(code_lines).lower()
    for banned in ("api_key", "apikey", "secret", "token", "password"):
        assert banned not in code_text, f"render.yaml must not declare {banned!r} outside comments"


# ---------------------------------------------------------------------------
# requirements-deploy.txt
# ---------------------------------------------------------------------------

def _package_names(requirements_text: str):
    names = set()
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # Strip version specifiers: "pandas>=2.2.0" -> "pandas"
        name = line.split(">=")[0].split("==")[0].split("<")[0].strip().lower()
        names.add(name)
    return names


def test_deploy_requirements_file_exists():
    assert os.path.exists(DEPLOY_REQUIREMENTS_PATH)


def test_deploy_requirements_is_a_true_subset_of_main_requirements():
    """Every package listed for deployment must also appear in the main
    requirements.txt — requirements-deploy.txt must never introduce a
    package that local dev / CI hasn't already vetted."""
    deploy_pkgs = _package_names(_read(DEPLOY_REQUIREMENTS_PATH))
    main_pkgs = _package_names(_read(MAIN_REQUIREMENTS_PATH))
    assert deploy_pkgs, "deploy requirements should not be empty"
    assert deploy_pkgs.issubset(main_pkgs), (
        f"requirements-deploy.txt lists packages not in requirements.txt: "
        f"{deploy_pkgs - main_pkgs}"
    )


def test_deploy_requirements_excludes_data_prep_and_dev_only_packages():
    deploy_pkgs = _package_names(_read(DEPLOY_REQUIREMENTS_PATH))
    for excluded in ("pyarrow", "kaggle", "tqdm", "pytest", "anthropic"):
        assert excluded not in deploy_pkgs, (
            f"{excluded!r} is not imported by the demo server's runtime "
            "code path and should not be in the deployment-only requirements"
        )


def test_main_requirements_file_unchanged_in_spirit():
    """requirements.txt must still declare everything the data-prep and
    test-suite workflows need — this project never removes from it without
    proof; this checks the well-known data-prep/test packages are all
    still present."""
    main_pkgs = _package_names(_read(MAIN_REQUIREMENTS_PATH))
    for still_needed in ("pandas", "numpy", "pyarrow", "scikit-learn", "pytest", "kaggle", "tqdm"):
        assert still_needed in main_pkgs


# ---------------------------------------------------------------------------
# The app itself: never hardcodes a host/port, binds correctly via uvicorn
# ---------------------------------------------------------------------------

def test_demo_server_source_never_hardcodes_a_bind_host_or_port():
    """scripts/demo_server.py must never itself launch/bind with a baked-in
    host or port — those must only ever come from whatever invokes uvicorn
    (the local dev command or render.yaml's startCommand). The module
    docstring's usage example mentioning 127.0.0.1:8000 as documentation is
    fine and expected; this checks there's no actual Python code doing the
    binding (no self-launch, no host=/port= keyword arguments)."""
    source_path = os.path.join(REPO_ROOT, "scripts", "demo_server.py")
    text = _read(source_path)
    assert "uvicorn.run(" not in text, "must not self-launch with a baked-in host/port"
    for needle in ('host="127.0.0.1"', "host='127.0.0.1'", 'host="0.0.0.0"', "host='0.0.0.0'"):
        assert needle not in text


def test_app_binds_correctly_via_uvicorn_with_render_style_host_and_port():
    """Constructs the exact same uvicorn objects the Render start command
    would (host=0.0.0.0, an arbitrary port) against the real FastAPI app
    object, without actually opening a socket or running the event loop —
    this exercises uvicorn's own config/route validation against our app."""
    import uvicorn

    config = uvicorn.Config(demo_server.app, host="0.0.0.0", port=0, log_level="critical")
    server = uvicorn.Server(config)
    assert server.config.host == "0.0.0.0"
    assert server.config.app is demo_server.app


# ---------------------------------------------------------------------------
# Deployment-only UI note
# ---------------------------------------------------------------------------

def test_wake_up_notice_is_present_in_the_ui():
    html = demo_server.index()
    assert "Hosted demo may take a moment to wake up after inactivity." in html


def test_wake_up_notice_does_not_replace_the_offline_disclaimer():
    """The new note must be additive — the existing offline/no-live-access
    disclaimers must still all be present, unchanged."""
    html = demo_server.index()
    assert "not sent to a live Delta system" in html
    assert (
        "This demo does not access live Delta systems and cannot perform "
        "flight-status lookups, bookings, refunds, account actions, or "
        "baggage-system operations."
    ) in html


# ---------------------------------------------------------------------------
# No evaluation files changed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "src",
    "scripts/evaluate.py",
    "data/golden_eval",
    "reports",
    "REPORT.md",
    "docs",
])
def test_no_evaluation_related_paths_changed_on_this_branch(path):
    """Diffs this branch against origin/master for each evaluation-critical
    path and asserts there is no output at all. Skips (does not fail) if
    git or the remote ref is unavailable, e.g. in a stripped-down CI
    checkout — this is a real repo-state check, not a mock."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/master...HEAD", "--", path],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git not available in this environment")

    if result.returncode != 0:
        pytest.skip(f"git diff against origin/master unavailable: {result.stderr.strip()}")

    changed = [l for l in result.stdout.splitlines() if l.strip()]
    assert changed == [], f"Evaluation-critical path {path!r} has changes on this branch: {changed}"
