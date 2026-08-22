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
