import numpy as np
import pytest

from g1_stack.core.types import ActuatorCommand


def test_actuator_command_is_copied_and_frozen() -> None:
    source = np.array([0.25])
    command = ActuatorCommand(names=("joint",), values=source)
    source[0] = 0.5

    assert command.values.tolist() == [0.25]
    with pytest.raises(ValueError):
        command.values[0] = 1.0


def test_actuator_command_rejects_name_value_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ActuatorCommand(names=("one", "two"), values=np.array([0.0]))

