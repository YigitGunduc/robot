import pytest

from mini_groot_sonic.checkpoint import (
    BODY_CONTROL_STACK_VERSION,
    require_current_body_control_stack,
)


def test_rejects_legacy_body_checkpoint_action_semantics():
    with pytest.raises(RuntimeError, match="Start a new body run"):
        require_current_body_control_stack({"policy": {}})


def test_accepts_current_body_checkpoint():
    require_current_body_control_stack(
        {"control_stack_version": BODY_CONTROL_STACK_VERSION, "policy": {}}
    )
