# SONIC + GR00T-Lite for Unitree G1 on MuJoCo-Warp

Clean-room Python/PyTorch reimplementation of the **public NVIDIA SONIC training contracts** plus a smaller **GR00T-like frozen-backbone flow policy**, designed around:

- Unitree G1 29-DOF whole-body control
- BONES-SEED as the motion dataset
- NVIDIA-compatible motion filtering and adaptive failure sampling
- MuJoCo GPU simulation through **MuJoCo-Warp / MJWarp**
- PPO + auxiliary universal-token reconstruction
- 64-D SONIC motion tokens (2 × 32 FSQ dimensions)
- a much smaller trainable VLA/action model with a frozen Hugging Face backbone

This is not a copy of NVIDIA source code. It independently implements the interfaces, dimensions, hyperparameters and training behavior exposed by NVIDIA's paper/configs/repository, while changing Isaac Lab-specific infrastructure to MuJoCo-Warp and shrinking networks where requested.

## What is implemented

### BONES-SEED

- public flat G1 CSV reader (120 Hz source)
- NVIDIA's released filename/category exclusion filter
- NVIDIA-style 120 Hz -> 30 Hz preprocessing cache
- runtime resampling to the 50 Hz control/reference stream
- canonical 29-joint MuJoCo order plus IsaacLab<->MuJoCo index maps
- optional MuJoCo FK cache for body-position/orientation/velocity tracking rewards
- freeze-frame augmentation
- upper-body recombination augmentation
- adaptive failure-biased motion/time-bin sampling

### SONIC-like controller

- current G1 encoder contract: **10 × (29 q + 29 qdot + 6D relative root orientation) = 640 D**
- current actor proprioception contract: **10 × (3 base angular velocity + 29 q + 29 qdot + 29 previous action + 3 projected gravity) = 930 D**
- 2 × 32 FSQ universal motion token = **64 D**
- dynamic decoder input = **994 D**
- G1 encoder -> FSQ -> dynamic decoder + auxiliary kinematic decoder
- same public FSQ package as NVIDIA when `vector-quantize-pytorch` is installed; dependency-free STE fallback otherwise
- Gaussian stochastic PPO actor with learned/clamped standard deviation
- asymmetric critic interface closely matching the release composition: future q/qd, reference-anchor error, current privileged body pose, and clean 10-frame base/joint/action history (**1,645 D** for the 14-body G1 set)
- PPO clipping, GAE, entropy, value clipping, gradient clipping
- KL-adaptive actor learning rate; **no invented SONIC LR warm-up**
- auxiliary kinematic reconstruction loss
- released tracking reward weights
- strict released termination thresholds
- 200 Hz physics / 50 Hz neural policy
- normalized policy action -> 29 joint-position targets -> PD torques
- current deployment-style G1 default angles, stiffness/damping and action scaling
- 4–6 s root-velocity pushes
- startup friction + selected-body mass randomization through compiled MJWarp physics variants

### Two network presets

`small` (default): approximately **9.83M parameters** for the G1 encoder + FSQ + dynamic + kinematic paths.

`nvidia`: approximately **45.0M parameters** for the G1 encoder + FSQ + dynamic + kinematic actor path using the current `sonic_v1_1` released widths. The asymmetric critic is a separate large network (~40M with this port's 1,645-D privileged input), so actor+critic training parameters are not directly comparable with NVIDIA's paper actor/controller parameter-count table.

Current NVIDIA-like G1 actor widths used by `--network nvidia`:

```text
G1 encoder:        640 -> 2048 -> 1024 -> 512 -> 512 -> 64
Dynamic decoder:   994 -> 4096 -> 4096 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29
Kinematic decoder:  64 -> 2048 -> 1024 -> 512 -> 512 -> 640
Critic:            1645 -> 4096 -> 4096 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 1
```

### GR00T-Lite

- frozen `google/siglip2-base-patch16-224` by default
- separate frozen image and text feature extraction
- trainable robot-state condition projector
- compact 8-layer / 512-D flow-matching Transformer
- default 16-step token horizon
- four Euler flow-integration steps at inference
- action masks for variable embodiments/action fields
- text-only BONES Stage-A training
- image features can be enabled later using synchronized visual manipulation demonstrations
- receding-horizon SONIC-token execution buffer
- optional V2 action layout for task-space end-effector targets and grippers
- task-space safety gate ready to connect to your existing IK implementation

Default trainable GR00T-Lite condition/action stack is about **27.2M parameters**, plus the frozen pretrained SigLIP2 backbone.

## Source-of-truth policy

Where NVIDIA's paper and current code differ, this package makes the choice explicit:

| Item | Default here | Reason |
|---|---|---|
| PPO rollout | 24 | current SONIC release override / paper |
| entropy coefficient | 0.01 | current release code; paper reported 0.013 |
| training iterations | 100,000 max | current release configuration |
| actor LR | 2e-5 | current release |
| critic LR | 1e-3 | current release |
| desired KL | 0.01 | current release |
| future G1 refs | 10 | current release |
| future spacing | 0.1 s | current release |
| proprio history | 10 | current release |
| FSQ | 2×32, 32 levels/dim | current release |
| upper-body recombination | p=0.5 | current release |
| adaptive bin | 50 frames | current release |
| uniform sampler mixture | 0.1 | current release |
| pre-failure window | 200 frames | current release |
| max failure / mean | 200 | SONIC release override |
| SONIC optimizer warm-up | none | no conventional public SONIC LR warm-up |
| GR00T-Lite warm-up | 5% | mirrors the GR00T transformer training convention |

`gear_sonic_mjx/config/sonic_paper.yaml` is provided for paper-specific overrides such as entropy `0.013`.

## Installation

Recommended Python: 3.10–3.12 with an NVIDIA GPU.

```bash
cd sonic_groot_mjx_reference
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[sim,hf,quantizer,dev]'
```

The simulator path uses direct `mujoco_warp` so the policy can remain PyTorch and Warp arrays can be exposed as zero-copy PyTorch CUDA tensors.

## 1. Preprocess BONES-SEED

Assuming the public G1 flat CSV hierarchy is under `$BONES`:

```bash
python scripts/preprocess_bones.py \
  --input "$BONES" \
  --output data/bones_30hz
```

The filter is on by default. It removes whole context-dependent clips such as chair/bed/bike/stairs/handstand/box-jump/etc. It **does not remove G1 actuator dimensions**.

To inspect only filtering behavior:

```bash
python -m gear_sonic_mjx.data_process.filter_and_copy_bones_data \
  --source "$BONES" \
  --dest /tmp/bones_filtered \
  --dry-run
```

## 2. Cache reference forward kinematics

The SONIC reward tracks body-space quantities, so cache reference FK once using **your exact G1 MJCF**:

```bash
python scripts/augment_bones_fk.py \
  --motions data/bones_30hz \
  --mjcf path/to/g1.xml
```

By default every named non-world body is cached. This avoids running a second reference FK simulation inside every GPU RL step.

## 3. Train SONIC-Lite on MJWarp

```bash
python scripts/train_sonic_mjwarp.py \
  --mjcf path/to/g1.xml \
  --motions data/bones_30hz \
  --network small \
  --output runs/sonic_small
```

For the current open NVIDIA G1-only widths:

```bash
python scripts/train_sonic_mjwarp.py \
  --mjcf path/to/g1.xml \
  --motions data/bones_30hz \
  --network nvidia \
  --output runs/sonic_nvidia_widths
```

The default simulation configuration is:

```text
4096 worlds
0.005 s physics dt = 200 Hz
4 physics steps per policy step
0.020 s policy dt = 50 Hz
10 s episodes
24 PPO steps per rollout
5 PPO epochs
4 minibatches
```

If your GPU cannot fit 4096 worlds, change `num_envs` in `sonic_release_mjx.yaml`. Do not change the 50-Hz controller contract just to reduce memory.

### MJWarp contact buffers

For a new MJCF you may need explicit `nconmax` / `njmax`. Tune them in the MuJoCo-Warp viewer/test-speed tool and put them in the YAML. Fixed buffer overflow in GPU simulation must be treated as a training failure, not ignored.

### Body names

The released `sonic_release` experiment overrides the historical “5point” reward with three reward points: a point 0.5 m above `torso_link`, plus both wrist-yaw links. The full body-tracking set is 14 named G1 bodies and the foot termination uses both ankle-roll links. Those canonical names live in `gear_sonic_mjx/g1_parameters.py`.

If your MJCF uses different body names, pass the reward/foot overrides or adapt the semantic mapping explicitly:

```bash
--reward-point-bodies torso_link left_wrist_yaw_link right_wrist_yaw_link \
--foot-bodies left_ankle_roll_link right_ankle_roll_link
```

The 29 **joint names** are deliberately strict. If they differ in your MJCF, adapt only the mapping in `g1_parameters.py`; do not reorder the policy silently.

## 4. Encode BONES into learned SONIC tokens

After the low-level policy is good enough:

```bash
python scripts/encode_bones_tokens.py \
  --motions data/bones_30hz \
  --checkpoint runs/sonic_small/checkpoint_XXXXXXX.pt \
  --output data/bones_sonic_tokens
```

Each output file contains:

```text
tokens [T,64]
state  [T,32]   # 29 G1 body q + 3 projected gravity
text   scalar   # weak filename-derived description unless you provide richer annotations
```

This implements the useful SONIC/GR00T hierarchy: GR00T predicts a compact motion representation; SONIC owns physical stabilization and 29 joint commands.

## 5. Train GR00T-Lite

```bash
python -m groot_lite.train \
  --data data/bones_sonic_tokens \
  --output runs/groot_lite
```

The Hugging Face backbone is frozen and held in `.eval()` mode. Only the condition projector and flow-action Transformer train.

The GR00T-Lite optimizer uses AdamW with a 5% linear warm-up followed by cosine decay. This warm-up is **GR00T-side**, not SONIC PPO-side.

### Important: BONES alone cannot train visual manipulation

BONES-SEED contains motion, not synchronized egocentric RGB + object interaction + task labels. Therefore BONES can train:

```text
text/body state -> future SONIC motion tokens
```

but it cannot teach:

```text
camera pixels of a red cup -> locate cup -> grasp cup
```

For visual manipulation, collect MuJoCo/real demonstrations and call `FrozenSiglip2Backbone.encode_images(...)` during training. Keep SONIC frozen initially.

## 6. Runtime hierarchy

```text
camera + instruction + robot state
          |
          v
frozen SigLIP2 + trainable condition/action model
          |
          v
H x 64-D motion-token chunk
          |
          v
RecedingHorizonTokenBuffer
          |
          v
SONIC dynamic decoder @ 50 Hz
          |
          v
29 normalized actions
          |
          v
q_target = q_default + action_scale * action
          |
          v
PD @ 200 Hz (or your robot's faster motor loop)
```

See `groot_lite/runtime.py`.

## Manipulation / V2 interface

`groot_lite/action_layout.py` supports a packed action with explicit validity masks:

```text
motion_token        64 D    always valid
left_ee_target       9 D    optional xyz + rotation-6D
right_ee_target      9 D    optional
left_gripper         1 D    optional
right_gripper        1 D    optional
```

The flow loss only trains fields whose mask is true. The task-space targets should go through your IK + collision checker (`gear_sonic_mjx/manipulation/task_space.py`) before execution. They should **not** overwrite SONIC-owned shoulder/elbow joints independently.

## Domain randomization notes

The released SONIC setup randomizes physical properties and applies periodic pushes. This port implements:

- 4–6 s root-velocity pushes with the released-scale velocity ranges
- startup selected-body mass scale 0.8–2.5
- broad friction variation
- per-world physics through a finite set of safely compiled MuJoCo variants

Why variants? Changing `body_mass` alone on a compiled MuJoCo model leaves dependent inertia/weight constants inconsistent. The code recompiles variant models and transfers the dependent rigid-body fields into MJWarp's per-world model arrays.

MuJoCo does not expose separate static/dynamic Coulomb friction coefficients exactly like the Isaac configuration; the port maps the sampled range onto MuJoCo's sliding-friction coefficient. That is an explicit simulator-model difference, not hidden as “exact parity.”

## Checkpoint migration from your existing locomotion policy

Use `gear_sonic_mjx/checkpoint_utils.py::load_matching_tensors` only for exact name-and-shape matches. Do **not** stretch an old locomotion policy's input layer to SONIC's 994-D observation. Preserve the existing locomotion checkpoint as a regression baseline and compare it against BONES tracking.

Recommended mapping when your repository is available:

- `height_conditioned_g1.py`: keep G1 morphology, joint ordering, actuator semantics; add/route to `G1SonicTrackingTask`
- `train_g1.py`: add `sonic_tracking` mode and MJWarp backend selection
- `render_g1.py`: add BONES reference overlays / tracking-error display
- `g1_arm_manipulation.py`: reuse its IK/task-space primitives as the manipulation target solver / demonstration generator
- `g1_arm_manipulation_config.py`: define reach/push/pull/grasp/lift/release workspace and collision constraints
- `manipulation_checkpoint.py`: freeze/load SONIC while training the high-level GR00T-Lite policy

## Evaluation gates

Before training the high-level policy, evaluate held-out BONES categories separately. Useful initial gates are:

```text
tracking success > 95%
local MPJPE      < 40 mm
```

Then aim toward the public SONIC regime around >97% success and ~30 mm local MPJPE where comparable. `gear_sonic_mjx/evaluation.py` contains global/local MPJPE and success helpers.

Do not trust aggregate success alone; report per motion family so standing/walking clips cannot hide failures on running, crouching, kneeling or transitions.

## Tests

```bash
pytest -q
```

Current artifact validation: **9 tests passed**. They cover:

- 640-D G1 encoder contract
- 930-D history and 994-D dynamic input
- FSQ gradients/shapes
- MuJoCo/Isaac joint reorder round-trip
- BONES filtering/resampling
- adaptive failure sampling
- flow-matching action masks and sampling
- PPO + auxiliary update

The current execution environment did not include your G1 MJCF, `mujoco_warp`, or a downloaded Hugging Face checkpoint, so those optional external-runtime paths were syntax/contract checked but could not be physically simulated here. The package intentionally fails loudly on missing joints, actuators, bodies, FK cache or optional dependencies.

## Primary references

- SONIC paper: https://arxiv.org/abs/2511.07820
- NVIDIA whole-body-control repository: https://github.com/NVlabs/GR00T-WholeBodyControl
- NVIDIA SONIC training guide: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/training.md
- NVIDIA SONIC configuration guide: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/configuration.md
- NVIDIA BONES filtering implementation: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data_process/filter_and_copy_bones_data.py
- NVIDIA BONES converter: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py
- Isaac GR00T: https://github.com/NVIDIA/Isaac-GR00T
- GR00T N1 paper: https://arxiv.org/abs/2503.14734
- SigLIP2 checkpoint: https://huggingface.co/google/siglip2-base-patch16-224
- MuJoCo-Warp documentation: https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
- MJX documentation: https://mujoco.readthedocs.io/en/latest/mjx.html
- vector-quantize-pytorch FSQ: https://github.com/lucidrains/vector-quantize-pytorch

## BONES text supervision for GR00T-Lite

Do **not** train GR00T-Lite from filename-derived labels. Export SONIC tokens with the official
BONES-SEED metadata and temporal annotations. The dataset loader prefers a timestamped local action
label for each action chunk and otherwise samples one of the official full-motion caption variants.

```bash
python scripts/encode_bones_tokens.py \
  --motions data/bones_30hz \
  --checkpoint runs/sonic_small/checkpoint.pt \
  --metadata /path/to/bones-seed/metadata/seed_metadata_v004.parquet \
  --timelines /path/to/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --output data/bones_sonic_tokens

python -m groot_lite.train --data data/bones_sonic_tokens --output runs/groot_lite
```

Whole-motion captions are paraphrase augmentation. Timeline labels are preferred for individual
short action windows because they align the language target to the motion actually occurring in that
window.
