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

The 29 outputs are SONIC-style Gaussian residual actions. The policy is not squashed;
the simulator clips actions to `[-20, 20]` once, applies the released per-joint residual
scale around the default pose, and finally respects physical joint limits. It either sends
the resulting targets to MuJoCo position actuators or converts them to clipped PD torques
for MuJoCo motor actuators.

---

## What mirrors SONIC

The released SONIC stack uses multiple motion encoders, FSQ, one shared G1 dynamic decoder, and a G1 kinematic decoder for auxiliary reconstruction. The default controller runs at 50 Hz and uses a 64D token representation.

This project keeps the smallest useful subset:

```text
future G1 q/qdot + robot-relative root orientation (10 frames)
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
29 Gaussian residual joint-position actions
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

For a fresh Google Colab run with BONES on Drive, use the SONIC-aligned v4
notebook: [`notebooks/mini_groot_sonic_sonic_stack_v2_colab.ipynb`](notebooks/mini_groot_sonic_sonic_stack_v2_colab.ipynb).
The filename is retained for existing links; its contents and run directories are v4.
The earlier [`notebooks/mini_groot_sonic_colab.ipynb`](notebooks/mini_groot_sonic_colab.ipynb)
is retained for comparison.
It selects a SONIC-filtered candidate pool, builds a structured-metadata and kinematic
five-stage curriculum, caches derived data and checkpoints on Drive, and dynamically
promotes stages using held-out metrics.

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

The loader reads natural-language descriptions and actor/source metadata after a file
has been admitted. Selection itself searches only the source path/filename, matching
SONIC's offline filtering design. Multi-phase motions are emitted as separately
captioned subclips by default unless `--no-temporal-segments` is used.

BONES-SEED is gated/licensed; this project assumes you already have authorized access and a local copy.

Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the
underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.

References:

- https://huggingface.co/datasets/bones-studio/seed
- https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst

---

# Recommended training order

## 0. Start with the official 29-DOF G1 MJCF

The defaults assume body names used by the standard Unitree G1 XML, including:

```text
pelvis
torso_link
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
  --no-temporal-segments
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
9. Optionally uses BONES temporal descriptions to split multi-phase clips.
10. Preserves actor/source IDs for leakage-free validation splits.

`--limit` is a seeded sample of BONES rather than the first files in lexical order, so
small experiments are reproducible without silently biasing the subset toward one capture batch.
Keyword filters search only motion paths/filenames and are applied before the limit.
The exclusion filter defaults to SONIC's published offline BONES denylist; supplying
`--exclude-keywords` replaces that denylist. Captions and actor/source/category metadata
cannot make a motion enter the subset. The Colab workflow also disables temporal-label
expansion so a different annotated phase of a selected recording cannot enter by accident.

The precomputed body tracks are important: the GPU PPO loop should not repeatedly recalculate reference kinematics on the CPU.

---

## 2. Build and train the dynamic curriculum

Create cumulative balance, neutral-walk, walk-variation, turn, and jog/run stages:

```bash
mgsp-build-curriculum \
  --motions data/bones_preprocessed \
  --out data/bones_curriculum \
  --stage-sizes 8,20,32,48,64 \
  --seed 0
```

This writes `curriculum.json`, one filename list per stage, and an `audit.csv` containing
the admission/rejection reason, physical-quality result, measured kinematic features,
difficulty score, and first stage for every candidate. The hard gates reject mirrored
duplicates, props, complex actions, non-locomotion packages, body positions below the
floor, implausible body/joint velocities, and extended airborne motion. Difficulty is a
percentile score over root speed, yaw rate, joint speed, pelvis-height range, and
upper-body speed.

Train with validation-gated promotion:

```bash
mgsp-train-curriculum \
  --manifest data/bones_curriculum/curriculum.json \
  --motions data/bones_preprocessed \
  --config configs/default.yaml \
  --mjcf /path/to/g1_29dof.xml \
  --device cuda:0 \
  --num-envs 64 \
  --out runs/body_curriculum \
  --evaluation-chunk-iterations 100 \
  --minimum-stage-iterations 1000 \
  --maximum-stage-iterations 20000 \
  --promotion-patience 2 \
  --randomization
```

Stages remain cumulative, and actor/source validation groups remain permanently held out
as the curriculum expands. The default gate requires success rate `>= 0.80`, MPJPE
`<= 0.08 m`, root-position error `<= 0.08 m`, and root-orientation error `<= 0.20 rad`
on two consecutive evaluations. Training stops before the next stage if the current one
uses its full budget without passing. Progress and every gate decision are saved in
`curriculum_state.json`; rerunning the command resumes safely. With `--randomization`,
randomization is disabled for balance and walking, then enabled from the turning stage.
Without either randomization CLI flag, the value in the YAML is preserved.

## 3. Train the SONIC-like body controller manually

```bash
mgsp-train-body \
  --motions data/bones_preprocessed \
  --mjcf /path/to/g1_29dof.xml \
  --device cuda:0 \
  --num-envs 256 \
  --iterations 100000 \
  --max-motions 256 \
  --out runs/body
```

The `100000`-iteration default follows the convergence scale documented for released
SONIC while retaining the smaller local network and environment count. Stop earlier only
when the held-out gates have passed consistently. The first critical milestone is **not
language**. It is:

> one policy tracks held-out BONES motions robustly.

The training path is:

```text
BONES future q/qdot + robot-relative root orientation
       |
       v
G1 encoder -> FSQ 64D token
       |              |
       |              +-> kinematic reconstruction loss
       v
64D token + 10-step proprio history
       |
       v
body decoder -> 29 SONIC-calibrated joint-position residuals
       |
       v
MJWarp G1 physics
       |
       v
SONIC-style tracking reward + PPO
```

The Menagerie G1 position servos are recalibrated at load time with SONIC's per-joint
stiffness, damping, armature and effort limits. Actions use SONIC's small per-joint
residual scales around its crouched default pose instead of spanning full joint limits.

Training adds 10% freeze-frame balance augmentation and samples one-second motion bins,
shifting starts up to four seconds before difficult bins. Failure-targeted sampling is
blended with 10% uniform coverage. PPO adapts the actor learning rate from KL instead of
discarding the rest of an update, and the critic receives clean root-relative full-body
state. Tracking termination uses root/end-effector height, root orientation, and local
foot error; horizontal trajectory drift alone does not terminate an episode.

These changes alter action semantics, reference dimensions, and critic dimensions.
Checkpoints made before control-stack version 2 cannot be resumed; start a new run.

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

# 4. Collect synthetic replay data

After body training, replay BONES motions through the learned token controller. Replay
rows store the pre-action state beside the token generated from that state, preserving
the causal `state_t -> token_t...` contract:

```bash
mgsp-collect-replay \
  --motions data/bones_preprocessed \
  --mjcf /path/to/g1_29dof.xml \
  --checkpoint runs/body/body_100000.pt \
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
body-stack version and exact body-policy/codebook fingerprint
```

This produces the target tuples for the GR00T-like upper model:

```text
language + current state + optional sparse goals -> future 64D token chunk
```

Replay from legacy, reference-PD, or mixed body checkpoints is rejected by flow training.
The resulting flow checkpoint is locked to the exact body checkpoint that produced its
tokens, preventing a different FSQ codebook from being decoded silently at runtime.

## Bootstrap collection before the body policy is good

You can also execute reference joint targets directly through the SONIC residual-action interface:

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

# 5. Optional RGB collection

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

# 6. Train the GR00T-style flow model

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

# 7. Run natural language through both models

After both checkpoints exist:

```bash
mgsp-run-command \
  --mjcf /path/to/g1_29dof.xml \
  --body runs/body/body_100000.pt \
  --flow runs/flow/flow_0019.pt \
  --initial-motion data/bones_preprocessed/standing_balance.npz \
  --command "walk backward slowly while leaning to the left" \
  --device cuda:0
```

The runtime first resets to the supplied preprocessed standing/balance reference and runs
the body policy for a one-second warm start. It refuses to continue if that controller
cannot remain upright. It then uses a receding horizon:

Use a standing/balance clip that was included when collecting the flow replay so the
initial state is inside the upper model's training distribution.

```text
flow model predicts: 40 future tokens
body runs at:          50 Hz
upper replan:           2.5 Hz
consume before replan: 20 tokens
```

The unused tail is blended into the new chunk across the overlap, avoiding a hard token
and joint-command discontinuity while still reconditioning on the newest robot state.

## Control-stack compatibility

These SONIC action, raw-tokenizer, and optimizer corrections define body control stack v4. Older body
checkpoints, replay datasets, and flow checkpoints are intentionally rejected. Retrain the
body controller, recollect policy replay, and then retrain flow in that order. Body
checkpoints own the trained simulator/control configuration; deployment only overrides the
MJCF path, device, and evaluation-time noise switches.

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
body actor:   1.13M parameters
body critic:  0.87M parameters (30-body G1 asset)
flow model:   1.76M parameters
total:        3.76M parameters (excluding frozen SigLIP2)
```

That is intentional.

---

# Suggested first experiment

Do this before scaling anything:

```text
1. Preprocess a 256-motion filename-filtered candidate pool.
2. Audit it into cumulative stages of 8, 20, 32, 48, and 64 motions.
3. Dynamically train with 64 MJWarp environments until each validation gate passes.
4. Inspect the numerical evaluation and MP4 at every blocked or completed stage.
5. Collect 100 replay episodes only after locomotion tracking is stable.
6. Train the text-only flow model and test paraphrased commands.
```

Only after those six steps work should you spend GPU budget on thousands of motions or visual data.

Evaluate a body checkpoint explicitly with:

```bash
mgsp-eval-body \
  --motions data/bones_validation \
  --mjcf /path/to/g1_29dof.xml \
  --body runs/body/body_best.pt
```

The report includes success rate, root-height error, diagnostic root-XY drift,
root-orientation error, heading-local MPJPE, joint error, action rate, undesired contacts,
and mean absolute actuator force. Training also logs individual reward terms, FSQ
occupancy/saturation, action saturation, failures per reset, actor learning rate,
policy/value/reconstruction losses, KL, and held-out metrics.

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
- Unsquashed Gaussian PPO action likelihoods with environment-boundary clipping.
- GR00T N1.7 timestep sampling.
- Replay causality, goal masking and metadata deduplication.
- Replanning overlap continuity.
- MuJoCo actuator detection and root-velocity preprocessing when MuJoCo is installed.

The GPU MJWarp integration cannot be validated without an NVIDIA/CUDA machine and a concrete G1 MJCF. Run the first real simulator smoke test on the target GPU before launching PPO.

---

# Integration with the existing G1 stack

The controller exposes 29 Gaussian residual joint-position actions. `MJWarpG1VecEnv` owns the
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
