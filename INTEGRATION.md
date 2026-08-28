# Integration into the existing G1 codebase

The user's existing G1 files were not mounted when this artifact was generated, so the package uses strict adapters instead of guessing their internal APIs.

## Low-level ownership

Keep one owner of all 29 body joints: SONIC-Lite. Existing locomotion and manipulation modules should provide commands/references or demonstrations, not concurrently write joint targets.

## Recommended insertion points

### `height_conditioned_g1.py`

Reuse the existing MJCF, joint order, actuator selection and reset helpers. Replace/augment the task-level command generator with BONES future-reference sampling and feed the resulting 640-D reference observation plus the 930-D history into `UniversalTokenModule`.

### `train_g1.py`

Add a task switch, e.g. `--task sonic_tracking`, while leaving existing locomotion training intact for regression. Route the new task to `scripts/train_sonic_mjwarp.py` or import its components.

### `render_g1.py`

Load a BONES `MotionClip`, render the reference and simulated robot together, and show root error/local MPJPE. Do not use rendering in the training loop.

### `g1_arm_manipulation.py`

Keep/reuse task-space reach/push/pull/grasp/lift/release and IK. When V2 is enabled, GR00T-Lite predicts task-space hand targets; the IK layer validates them, but SONIC remains the final whole-body joint controller.

### `g1_arm_manipulation_config.py`

Add per-hand workspace boxes, collision margins and primitive tolerances. Feed rejected targets back to the planner rather than clipping joint actions after SONIC.

### `manipulation_checkpoint.py`

Load/freeze the trained SONIC universal-token model while training GR00T-Lite. Save the high-level policy separately so low-level motor-control regression is easy to isolate.

## Existing checkpoint

Do not directly reinterpret old velocity-command locomotion observations as SONIC observations. `load_matching_tensors` can reuse only exact-shaped tensors. Keep the old locomotion checkpoint as a baseline until SONIC passes held-out tracking/fall tests.
