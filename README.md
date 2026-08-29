# SONIC-Lite G1

A deliberately small SONIC-inspired motor-token controller for Unitree G1 using **PyTorch + mjlab (MuJoCo-Warp)**.

This is a V1 proof-of-concept, not a line-for-line SONIC reproduction. It keeps the pieces that matter for getting a G1 motion tracker working and removes the large universal-model machinery.

## What is kept

- G1-retargeted BONES-SEED motion imitation.
- Small hand-selected curriculum: stand → walk → turns/starts/stops → crouch/squat → jog.
- NVIDIA BONES impossible-motion blacklist plus stricter V1 semantic filtering.
- Within-skill kinematic difficulty ranking before training.
- 50 Hz policy on 200 Hz MuJoCo physics (mjlab G1 tracking default).
- 29 normalized residual joint-position actions.
- G1-specific action scaling and the robot's actuator effort/velocity/joint limits from mjlab.
- PD/implicit position actuation in the simulator rather than learned raw torques.
- Five future reference frames (0–0.4 s) encoded into a **64-D scalar-quantized motor token**.
- 10-step proprioceptive/tracking-error history.
- PPO from scratch; no BC/teacher/student/warm-up stage.
- Asymmetric critic, tracking terminations, joint-limit/action-rate/self-collision regularization inherited from mjlab's G1 tracker.
- Adaptive failed-frame sampling inherited from mjlab's MotionCommand.
- Small randomized reference-state initialization.

## Intentionally omitted from V1

- Transformer, diffusion, MoE, RNN.
- SMPL/teleop/SOMA encoders and cross-modal token alignment.
- Vision/language/GR00T.
- Manipulation/object interaction/dexterous hands.
- Large sim-to-real pushes/COM/friction randomization at the beginning of training.
- Separate supervised reconstruction loss. SONIC uses auxiliary reconstruction/alignment objectives, but the smallest proof-of-concept can learn the bottleneck jointly from PPO via the straight-through quantizer. Add reconstruction only if the token space proves unstable or uninformative.
- Raw-torque actions. We keep the safer/easier joint-position-residual + actuator-PD design.

## 1. Install

Use a Linux CUDA machine for useful training (Colab works; an NVIDIA GPU is strongly recommended).

```bash
# In this repository
uv sync
# or, in an existing mjlab environment:
pip install -e .
```

The plugin registers:

```text
Mjlab-SonicLite-Tracking-Flat-Unitree-G1
```

## 2. Select a tiny BONES-SEED curriculum

Point the selector at the native BONES-SEED `g1/csv` directory. The selector understands BONES' native centimetre/degree/Euler CSV format:

```bash
uv run sonic-lite-select-bones \
  --root /path/to/bones-seed/robot \
  --out data/selected_bones.json \
  --input-fps 120
```

Default maximums are 150 stand + 700 walk + 200 turn/start/stop + 300 crouch/squat + 500 jog = at most 1,850 clips. The script ranks each semantic bucket by a cheap kinematic difficulty score and keeps the easiest clips in each bucket.

For the very first smoke test, reduce this dramatically, e.g. `--max-stand 30 --max-walk 100 --max-turn 0 --max-crouch 0 --max-jog 0`.

## 3. Convert selected BONES CSVs with mjlab

**Do not use an Isaac/PhysX NPZ converter.** mjlab's tracking data stores body arrays in MuJoCo body order, and the mjlab docs warn that a mismatched body ordering prevents convergence.

BONES-SEED's native G1 files are **not** directly in the no-header format expected by mjlab: BONES uses centimetres + extrinsic XYZ Euler degrees + joint degrees. First convert each selected source file:

```bash
uv run sonic-lite-bones-to-mjlab-csv \
  /path/to/bones-seed/g1/csv/.../walking__A123.csv \
  data/mjlab_csv/walking__A123.csv
```

Then use mjlab's own FK converter. The converted generalized-coordinate CSV is still 120 FPS; the policy reference is 50 FPS:

```bash
WANDB_MODE=offline MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file data/mjlab_csv/walking__A123.csv \
  --output-name walking__A123 \
  --input-fps 120 \
  --output-fps 50 \
  --render False
cp /tmp/motion.npz data/converted_npz/walking__A123.npz
```

Current mjlab saves `/tmp/motion.npz` and also logs it through W&B. `WANDB_MODE=offline` avoids requiring an online run. Do **not** replace mjlab's FK conversion with an Isaac/PhysX converter: body ordering differs and breaks tracking. For a first smoke test, convert only 50–150 stand/walk clips rather than the full 1,850-clip target.

## 4. Pack the small motion set

```bash
uv run sonic-lite-pack-motions \
  --dir data/converted_npz \
  --out data/c0_c1_stand_walk.npz
```

The pack is just ordinary mjlab motion arrays concatenated along time plus `clip_starts`/`clip_lengths`. The custom command prevents references and future windows from crossing clip boundaries.

## 5. Train

Start small enough to debug:

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-SonicLite-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file data/c0_c1_stand_walk.npz \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 5000 \
  --video True
```

Then move to 4096 environments if memory allows:

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-SonicLite-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file data/c0_c1_stand_walk.npz \
  --env.scene.num-envs 4096
```

### Curriculum

Train **one continuing policy**, not separate policies:

1. C0: stand/idle.
2. C1: C0 + neutral walk.
3. C2: + starts/stops/gentle turns.
4. C3: + crouch/squat.
5. C4: + jog.

At each stage, build a new packed NPZ and resume the previous checkpoint. Do not introduce jump, furniture, object interactions, acrobatics, or advanced styles until the easy controller is reliable.

## Network

Reference frame = `29 q + 29 qdot + 3 root/torso linear velocity + 3 angular velocity = 64`.

Five future frames → 320-D raw reference:

```text
320 -> 256 -> 128 -> 64 -> scalar quantizer (32 levels/component)
                                  |
                                  + 10-step proprio history
                                  |
                         512 -> 256 -> 128 -> 29 actions
```

This is roughly around the low-million-parameter scale depending on the exact proprioceptive observation dimension—far smaller than the large SONIC experiments.

## Low-level controller / safety choices

The package reuses mjlab's current Unitree G1 robot configuration rather than coding another actuator model. The G1 tracking config applies robot-specific action scales and the underlying actuator layer enforces effort/velocity/joint constraints. The learned action is a bounded position residual (`clip_actions=1.0`), not a raw torque command.

For eventual hardware deployment, add a separate safety layer that rejects/freezes commands on excessive joint velocity and checks platform-specific position/torque limits. NVIDIA's deployment code has such guards; they are intentionally not mixed into the learned V1 policy.

## What to measure before expanding the curriculum

Do not move to the next stage just because reward rises. Track:

- episode termination/fall rate per motion;
- anchor/root position and orientation error;
- joint-position and joint-velocity error;
- mean body position/orientation error;
- self-collision rate;
- joint-limit penalty;
- action-rate penalty;
- per-motion success rate.

Once C1 is stable, use actual tracking failure/error to re-rank candidate clips (ExBody2-style empirical difficulty) instead of relying only on the offline kinematic score.

## Important V1 limitation

This code was assembled against the **mjlab 1.5.3 / RSL-RL 5.x APIs published in 2026**, but the execution environment used to create this package did not have internet/package access to install mjlab, so only the dependency-free data utilities and quantizer could be runtime-tested here. If mjlab changes a config field/API after 1.5.3, pin to 1.5.3 first.
