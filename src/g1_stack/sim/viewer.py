from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

from g1_stack.sim.mujoco_backend import MujocoBackend


class PassiveMujocoViewer:
    """Thin context-managed wrapper around MuJoCo's passive desktop viewer."""

    def __init__(
        self,
        backend: MujocoBackend,
        *,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        try:
            import mujoco.viewer
        except ImportError as error:  # pragma: no cover - depends on optional GUI support
            raise RuntimeError("MuJoCo viewer support is unavailable") from error
        self._handle = mujoco.viewer.launch_passive(
            backend.model,
            backend.data,
            key_callback=key_callback,
        )

    def is_running(self) -> bool:
        return bool(self._handle.is_running())

    def sync(self) -> None:
        self._handle.sync()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> PassiveMujocoViewer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
