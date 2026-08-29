"""MjLab task registration for SONIC-Lite G1.

The import guard keeps the data-selection utilities usable on machines where
mjlab is not installed yet. When mjlab loads this package through its task
entry-point, registration happens normally.
"""

TASK_ID = "Mjlab-SonicLite-Tracking-Flat-Unitree-G1"

try:
    from mjlab.tasks.registry import register_mjlab_task
except ModuleNotFoundError as exc:
    if exc.name != "mjlab":
        raise
else:
    from .env_cfg import sonic_lite_g1_env_cfg
    from .rl_cfg import sonic_lite_ppo_runner_cfg

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=sonic_lite_g1_env_cfg(),
        play_env_cfg=sonic_lite_g1_env_cfg(play=True),
        rl_cfg=sonic_lite_ppo_runner_cfg(),
    )

__all__ = ["TASK_ID"]
