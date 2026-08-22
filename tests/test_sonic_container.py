from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_sonic_image_contains_project_and_uses_unified_app_command() -> None:
    dockerfile = (ROOT / "docker" / "sonic" / "Dockerfile").read_text()

    assert "APP_ROOT=/workspace/g1-stack" in dockerfile
    assert 'python -m pip install --no-cache-dir -e ".[sim]"' in dockerfile
    assert 'CMD ["app"]' in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini"' in dockerfile


def test_sonic_stack_runs_simulator_and_controller_inside_container() -> None:
    launcher = (ROOT / "docker" / "sonic" / "run_stack.sh").read_text()
    entrypoint = (ROOT / "docker" / "sonic" / "entrypoint.sh").read_text()

    assert "run_sim_loop.py" in launcher
    assert "./deploy.sh" in launcher
    assert "docker " not in launcher
    assert "app|stack)" in entrypoint
    assert "exec /usr/local/bin/sonic-stack" in entrypoint


def test_project_python_does_not_manage_docker() -> None:
    controller = (ROOT / "src" / "g1_stack" / "controllers" / "sonic.py").read_text()

    assert "docker" not in controller.lower()
