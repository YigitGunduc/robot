# mini-groot-sonic

A compact research scaffold for a **GR00T + SONIC-style Unitree G1 stack** using:

- PyTorch for all learned models.
- MuJoCo Warp (MJWarp) for GPU-parallel physics.
- BONES-SEED G1 trajectories and natural-language annotations.
- A SONIC-inspired 64D FSQ motion-token bottleneck and universal G1 controller.
- A GR00T-N1.7-inspired flow-matching model that predicts future motion-token chunks.
- Optional frozen Hugging Face SigLIP2 image/text features.
- A custom replay loop for turning BONES motions into physically executed training episodes.

This is **not copied NVIDIA code** and it is not a drop-in replacement for Isaac GR00T or GEAR-SONIC. It is an intentionally small original implementation of their useful architectural boundaries so the system is realistic for a small research project.

## Why this shape

The important separation is:

```text
planner command / text / optional vision / optional sparse 3D goals
                            |
                            v
                small GR00T-style flow model
                            |
                  40 x 64D motion tokens
                            |
                            v
                  tiny SONIC-style controller
              64D token + proprio history -> 29 actions
                            |
                            v
                           G1
```

The upper model learns **what motion to request**. The body controller learns **how to physically execute it**.

The 29 outputs are bounded normalized joint-position targets. By default each joint maps
asymmetrically onto its legal range around the default pose. The simulator either sends
those targets to MuJoCo position actuators or converts them to clipped PD torques for
MuJoCo motor actuators.

---

## What mirrors SONIC

The released SONIC stack uses multiple motion encoders, FSQ, one shared G1 dynamic decoder, and a G1 kinematic decoder for auxiliary reconstruction. The default controller runs at 50 Hz and uses a 64D token representation.

This project keeps the smallest useful subset:

```text
future G1 q/qdot + root trajectory/orientation/height/velocity (10 frames)
        |
        v
small G1 MLP encoder
        |
        v
64D FSQ token (32 scalar levels)
        |-----------------------> kinematic decoder -> reconstruct normalized reference
        |
        + proprio history (10 frames)
        |
        v
small dynamic decoder
        |
        v
29 bounded joint-position targets
```

The body model is trained jointly with PPO and a reconstruction auxiliary loss. That follows the released SONIC training pattern more closely than pretraining a standalone autoencoder first.

The default reward weights in `RewardConfig` mirror the released SONIC composition:

- anchor/root position: `+0.5`
- anchor/root orientation: `+0.5`
- relative body position: `+1.0`
- relative body orientation: `+1.0`
- body linear velocity: `+1.0`
- body angular velocity: `+1.0`
- five-point local tracking: `+2.0`
- action rate: `-0.1`
- joint limits: `-10.0`
- undesired contacts: `-0.1`
- head/wrist anti-shake: `-0.005`
- ankle-joint acceleration: `-2.5e-7`

The implementation is deliberately smaller than NVIDIA's released MLP sizes.

References:

- https://github.com/NVlabs/GR00T-WholeBodyControl
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/references/training_code.md
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/configuration.md
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/trl/modules/universal_token_modules.py

---

## What mirrors GR00T N1.7

The upper model does not reproduce NVIDIA's large VLM. It reproduces the **action-head idea**:

```text
frozen text embedding
frozen optional image embedding
robot state
optional sparse body goals
noisy future 64D token chunk
flow timestep
              |
              v
small Transformer
              |
              v
predicted flow velocity
```

Training uses the current GR00T-N1.7 flow convention:

```text
z_t = (1 - t) * noise + t * target
t = (1 - Beta(1.5, 1.0)) * 0.999
velocity_target = target - noise
loss = MSE(predicted_velocity, velocity_target)
```

Inference uses a few Euler integration steps starting from Gaussian noise.

References:

- https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/model/gr00t_n1d7.py
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_workflow.md

---

# Installation

Python 3.11 or 3.12 is recommended on the actual GPU machine.

For Google Colab with BONES on Drive, use
[`notebooks/mini_groot_sonic_colab.ipynb`](notebooks/mini_groot_sonic_colab.ipynb).
It selectively extracts a small stand/walk/run curriculum, caches derived data and
checkpoints on Drive, and runs a target-GPU MJWarp smoke test before PPO.

```bash
cd mini_groot_sonic
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[all]'
```

For model-only development without the simulator:

```bash
pip install -e '.[dev]'
pytest -q
```

For MJWarp GPU physics you need an NVIDIA GPU and a compatible CUDA environment.

MJWarp is designed for large batched GPU simulation and is maintained by Google DeepMind and NVIDIA:

- https://mujoco.readthedocs.io/en/latest/mjwarp/
- https://github.com/google-deepmind/mujoco_warp

The code uses MJWarp directly instead of switching the project to JAX or Isaac Lab.

---

# BONES-SEED assumptions

This code expects the official extracted layout:

```text
bones-seed/
  metadata/
    seed_metadata_v00*.parquet or .csv
    seed_metadata_*_temporal_labels.jsonl
  g1/
    csv/
      <date>/
        <motion>.csv
```

Official BONES G1 CSVs contain:

- `Frame`
- `root_translateX/Y/Z` in centimeters
- `root_rotateX/Y/Z` in extrinsic XYZ Euler degrees
- `<joint>_dof` in degrees, matching the G1 joint order

The loader automatically reads natural-language descriptions, actor/source metadata,
and temporal segmentation. Multi-phase motions are emitted as separately captioned
subclips by default.

BONES-SEED is gated/licensed; this project assumes you already have authorized access and a local copy.

References:

- https://huggingface.co/datasets/bones-studio/seed
- https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst

---

# Recommended training order

## 0. Start with the official 29-DOF G1 MJCF

The defaults assume body names used by the standard Unitree G1 XML, including:

```text
pelvis
head_link
left_wrist_yaw_link
right_wrist_yaw_link
left_ankle_roll_link
right_ankle_roll_link
```

If your XML uses different names, change `SimConfig.keypoint_body_names`.

The environment validates actuator semantics before training. `<position>` actuators
receive position targets directly. `<motor>` actuators receive explicit PD torque with
gear correction and finite torque clipping. Mixed or ambiguous actuator definitions fail
fast. Tune `joint_stiffness` and `joint_damping` for the exact MJCF before PPO.

---

## 1. Preprocess a small BONES subset

Start with only 100-500 clips.

```bash
mgsp-preprocess-bones \
  --bones-root /data/bones-seed \
  --mjcf /path/to/g1_29dof.xml \
  --out data/bones_preprocessed \
  --limit 256 \
  --seed 0 \
  --include-keywords "stand,idle,walk,run,jog,turn" \
  --exclude-keywords "jump,flip,cartwheel,crawl,stairs,climb"
```

This:

1. Loads the official 120 Hz BONES G1 CSV.
2. Converts root centimeters to meters.
3. Converts root extrinsic XYZ Euler degrees to quaternions.
4. Converts joint degrees to radians.
5. Resamples to 50 Hz.
6. Computes joint and root linear/angular velocities with MuJoCo quaternion differentiation.
7. Runs MuJoCo forward kinematics once on CPU.
8. Saves body positions, orientations and velocities for reward computation.
9. Uses BONES temporal descriptions to split multi-phase clips.
10. Preserves actor/source IDs for leakage-free validation splits.

`--limit` is a seeded sample of BONES rather than the first files in lexical order, so
small experiments are reproducible without silently biasing the subset toward one capture batch.
Keyword filters search all available captions plus actor/source/category metadata and are
applied before the limit, which makes small motion curricula straightforward.

The precomputed body tracks are important: the GPU PPO loop should not repeatedly recalculate reference kinematics on the CPU.

---

## 2. Train the SONIC-like body controller

```bash
mgsp-train-body \
  --motions data/bones_preprocessed \
  --mjcf /path/to/g1_29dof.xml \
  --device cuda:0 \
  --num-envs 256 \
  --iterations 5000 \
  --max-motions 256 \
  --out runs/body
```

The first critical milestone is **not language**. It is:

> one policy tracks held-out BONES motions robustly.

The training path is:

```text
BONES future q/qdot + invariant root motion
       |
       v
G1 encoder -> FSQ 64D token
       |              |
       |              +-> kinematic reconstruction loss
       v
64D token + 10-step proprio history
       |
       v
body decoder -> 29 bounded joint-position targets
       |
       v
MJWarp G1 physics
       |
       v
SONIC-style tracking reward + PPO
```

Training enables observation/reference noise, randomized friction, mass, COM, motor
strength, PD gains and one-step latency, plus occasional root pushes. The critic receives
clean privileged state. Motion sampling is duration-aware and gradually upweights failed
motions with a cap.

Training automatically reserves a validation group. Actor IDs are used when present;
otherwise source-motion IDs are used so temporal segments never cross the split.

The default simulator rate is:

```text
physics:    200 Hz (dt = 0.005)
controller:  50 Hz (decimation = 4)
```

### Scaling

The in-memory `MotionBank` is deliberately simple. Do not load all 142K BONES motions into it.

Recommended progression:

```text
256 motions -> prove code
1K motions  -> prove universal tracking
5K+ motions -> add sharded/streaming bank
```

Only build the sharded loader after the controller is behaving correctly.

---

# 3. Collect synthetic replay data

After body training, replay BONES motions through the learned token controller. Replay
rows store the pre-action state beside the token generated from that state, preserving
the causal `state_t -> token_t...` contract:

```bash
mgsp-collect-replay \
  --motions data/bones_preprocessed \
  --mjcf /path/to/g1_29dof.xml \
  --checkpoint runs/body/body_005000.pt \
  --mode policy \
  --out replays/train \
  --limit 1000
```

Each episode stores:

```text
caption(s)
actual q / qdot
root pose + velocities
64D token
29D body action
reference q / qdot
actual body positions/orientations
sparse root/head/hand/foot goal slots
```

This produces the target tuples for the GR00T-like upper model:

```text
language + current state + optional sparse goals -> future 64D token chunk
```

## Bootstrap collection before the body policy is good

You can also execute reference joint targets directly through the normalized action interface:

```bash
mgsp-collect-replay \
  --motions data/bones_preprocessed \
  --mjcf /path/to/g1_29dof.xml \
  --mode reference_pd \
  --out replays/bootstrap
```

Use this mainly for debugging/data-pipeline validation. The learned controller replay is more valuable because it represents physically realized controller behavior.

Collect a smaller randomized recovery set after nominal replay works:

```bash
mgsp-collect-replay ... --mode policy --randomized --out replays/recovery
```

---

# 4. Optional RGB collection

For small debugging datasets:

```bash
mgsp-collect-replay ... --rgb --camera ego_camera
```

The included RGB hook copies the single MJWarp world back to CPU MuJoCo at a low camera rate. That is intentionally simple and slow.

For large synthetic vision collection, replace the hook with MJWarp's GPU batch renderer instead of changing the replay loop.

The replay API is intentionally pluggable:

```python
class MySensors:
    def reset(self):
        ...

    def observe(self, step, env):
        return {
            "rgb": rgb,
            "depth": depth,
            "lidar": points,
        }
```

Then call:

```python
collect_preprocessed_episode(..., hook=MySensors())
```

This is the intended place to add your future RGB/depth/LiDAR stack.

### Empty-room recommendation

For the first language-to-body model, do **not** collect millions of empty-room images. They add little information.

Start with:

```text
language + robot state -> motion token
```

Add vision only when the correct motion actually depends on what the robot sees.

---

# 5. Train the GR00T-style flow model

Text-only first:

```bash
mgsp-train-flow \
  --replays replays/train \
  --out runs/flow \
  --device cuda:0 \
  --epochs 20
```

With replay RGB later:

```bash
mgsp-train-flow ... --vision
```

The pretrained SigLIP2 model is frozen. Only the compact robotics flow Transformer trains.

Training creates actor/source-disjoint train/validation partitions, learns per-feature
state/goal statistics from training data only, caches repeated text embeddings, uses BF16
autocast on CUDA, and saves `flow_best.pt` by validation loss. Logged validation metrics
also include sampled-token MSE and temporal token delta.

The replay dataset randomly chooses one BONES caption variant each time, which gives natural language augmentation for free.

---

# Sparse 3D goal conditioning

The flow model receives a fixed 48D goal vector:

```text
6 slots x (xyz + quaternion wxyz + active mask)

root
head
left hand/wrist
right hand/wrist
left foot
right foot
```

Replay stores the exact simulated 3D poses, but the training dataset deliberately hides most of them.

Default sampling is approximately:

```text
55% language only
20% root/path-style target only
20% one end-effector target
 5% multiple targets
```

All target poses are converted from simulator world coordinates into the **current root frame** before they reach the upper model.

Interpretation:

```text
mask = 0 -> target pose is zeroed; model decides this body part itself
mask = 1 -> this pose is an important continuous constraint
```

This lets the same model handle:

```text
"run forward"                    -> no body targets
"walk there"                     -> root/path target
"touch there with your right hand" -> right-hand pose target
"step here"                      -> foot target
```

No discrete `WALK`, `REACH`, or `KICK` action vocabulary is introduced.

## Important limitation

Random target masking teaches the model to use or ignore known targets, but it does **not fully teach target perturbation/generalization**.

After the base model works, generate a smaller goal-perturbation dataset with whole-body IK / trajectory optimization / goal-conditioned RL:

```text
original BONES motion
+ moved hand/foot target
        |
        v
physically corrected motion
        |
        v
new target <-> new token trajectory
```

That is the right phase for precise arbitrary reaching.

---

# 6. Run natural language through both models

After both checkpoints exist:

```bash
mgsp-run-command \
  --mjcf /path/to/g1_29dof.xml \
  --body runs/body/body_005000.pt \
  --flow runs/flow/flow_0019.pt \
  --command "walk backward slowly while leaning to the left" \
  --device cuda:0
```

The runtime uses a receding horizon:

```text
flow model predicts: 40 future tokens
body runs at:          50 Hz
upper replan:           2.5 Hz
consume before replan: 20 tokens
```

The unused tail is blended into the new chunk across the overlap, avoiding a hard token
and joint-command discontinuity while still reconditioning on the newest robot state.

The generated flow output is projected back to the nearest FSQ grid before the body decoder by default. This is a conservative choice because the small controller is trained on quantized tokens.

---

# Frozen vision/language backbone

Default:

```text
google/siglip2-base-patch16-224
```

It is used only as a frozen feature extractor.

For empty-room language-motion training, the text tower alone is enough.

Later, you can replace it with:

- a stronger frozen VLM,
- separate vision and language backbones,
- cached embeddings,
- LoRA on only the final backbone layers.

The flow model only assumes it receives fixed-size text/image embeddings.

---

# What is intentionally smaller than NVIDIA

## SONIC

NVIDIA uses much larger MLPs, multiple encoders (G1, teleop, SMPL, optional SOMA), and multi-GPU Isaac Lab training.

This project starts with:

```text
one G1 encoder
one FSQ
one small dynamic decoder
one small kinematic decoder
one privileged critic
PPO + reconstruction
subset-oriented domain randomization and adaptive sampling
```

There is a `SparseGoalEncoder` scaffold in `models/sonic_tiny.py`, but it is **not trained by the base body PPO loop yet**. Train the robust G1 token/controller first. Add goal/hybrid token alignment only after held-out tracking works.

## GR00T

NVIDIA N1.7 uses a much larger VLM plus a 16-layer action model and multi-embodiment handling.

This project uses:

```text
frozen SigLIP2
3-layer 192D Transformer
64D x 40 token flow output
```

With `configs/default.yaml`, the trainable footprint is approximately:

```text
body actor:   1.27M parameters
body critic:  0.80M parameters
flow model:   1.76M parameters
total:        3.83M parameters (excluding frozen SigLIP2)
```

That is intentional.

---

# Suggested first experiment

Do this before scaling anything:

```text
1. Preprocess 64 locomotion/gesture BONES clips.
2. Train with 64-128 MJWarp environments.
3. Evaluate the controller on actor/source-disjoint held-out clips.
4. Collect 100 replay episodes.
5. Train text-only flow model.
6. Test 20 paraphrased commands.
```

Only after those six steps work should you spend GPU budget on thousands of motions or visual data.

Evaluate a body checkpoint explicitly with:

```bash
mgsp-eval-body \
  --motions data/bones_validation \
  --mjcf /path/to/g1_29dof.xml \
  --body runs/body/body_best.pt
```

The report includes success rate, root position/orientation error, MPJPE, joint error,
action rate, undesired contacts, and mean absolute actuator force. Training also logs
individual reward terms, FSQ occupancy/saturation, policy/value/reconstruction losses,
KL, and held-out metrics.

Render a held-out rollout beside its BONES reference for visual inspection:

```bash
mgsp-render-body \
  --motions data/bones_validation \
  --mjcf /path/to/g1_29dof.xml \
  --body runs/body/body_best.pt \
  --out held_out_policy_vs_reference.mp4
```

The renderer overlays the caption plus root, MPJPE and joint errors and writes a matching
`*.metrics.json` sidecar. The Colab notebook exports three such videos to Drive by default.

---

# Tests

Model/data unit tests do not require MuJoCo Warp:

```bash
pytest -q
```

They cover:

- FSQ shape/range/straight-through gradients.
- SONIC-like tensor contracts.
- Flow matching forward/backward/sample shapes.
- BONES CSV parsing/resampling.
- SONIC-style reward sanity.
- Root-reference translation/yaw invariance.
- Bounded PPO action likelihoods.
- GR00T N1.7 timestep sampling.
- Replay causality, goal masking and metadata deduplication.
- Replanning overlap continuity.
- MuJoCo actuator detection and root-velocity preprocessing when MuJoCo is installed.

The GPU MJWarp integration cannot be validated without an NVIDIA/CUDA machine and a concrete G1 MJCF. Run the first real simulator smoke test on the target GPU before launching PPO.

---

# Integration with the existing G1 stack

The controller exposes 29 normalized joint-position targets. `MJWarpG1VecEnv` owns the
single conversion boundary to either position control or PD torque control; deployment
must reproduce that same target mapping, gains, gear convention and clipping.

Suggested integration boundary:

```text
validated position or PD-torque actuator boundary
             ^
             |
TinySonicPolicy.decode_token(...)
             ^
             |
64D token
             ^
             |
TinyFlowMotionPolicy
```

Keep the current command-conditioned locomotion checkpoint as a fallback until held-out BONES tracking is demonstrably better.
