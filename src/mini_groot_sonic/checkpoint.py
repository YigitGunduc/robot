from __future__ import annotations

BODY_CONTROL_STACK_VERSION = 2


def require_current_body_control_stack(checkpoint: dict) -> None:
    version = int(checkpoint.get("control_stack_version", 1))
    if version != BODY_CONTROL_STACK_VERSION:
        raise RuntimeError(
            f"Checkpoint control stack v{version} is incompatible with "
            f"v{BODY_CONTROL_STACK_VERSION}. Start a new body run: action scaling, "
            "reference features, and critic observations changed."
        )
