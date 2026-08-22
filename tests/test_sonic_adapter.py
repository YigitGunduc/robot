from pathlib import Path

from g1_stack.controllers.sonic import SonicProcessAdapter, SonicRuntimeConfig


def test_sonic_commands_keep_official_runtime_external() -> None:
    root = Path("/opt/GR00T-WholeBodyControl")
    adapter = SonicProcessAdapter(
        SonicRuntimeConfig(root=root, expected_ref="abc", input_type="keyboard", target="sim")
    )

    assert adapter.simulator_command() == (
        "/opt/GR00T-WholeBodyControl/.venv_sim/bin/python",
        "/opt/GR00T-WholeBodyControl/gear_sonic/scripts/run_sim_loop.py",
    )
    assert adapter.deployment_command() == (
        "/opt/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh",
        "--input-type",
        "keyboard",
        "sim",
    )


def test_sonic_container_config_comes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SONIC_ROOT", "/opt/pinned-sonic")
    monkeypatch.setenv("SONIC_REF", "deadbeef")
    monkeypatch.setenv("SONIC_INPUT_TYPE", "zmq_manager")

    config = SonicRuntimeConfig.from_environment()

    assert config.root == Path("/opt/pinned-sonic")
    assert config.expected_ref == "deadbeef"
    assert config.input_type == "zmq_manager"
    assert config.target == "sim"
