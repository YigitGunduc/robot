# NVIDIA parity / deliberate deviations

## Preserved closely

- BONES public G1 input semantics and NVIDIA filename filter
- 29-DOF position-target action interface
- 50-Hz policy over 200-Hz physics
- 10 future G1 reference frames at 0.1-s spacing
- 10-frame actor proprio/action history
- 64-D 2x32 FSQ universal token
- public `sonic_v1_1` G1 encoder/dynamic decoder/kinematic decoder widths in `--network nvidia` (including the 4096-wide dynamic decoder)
- PPO gamma/lambda/clip/epochs/minibatches/LRs/KL/entropy/std/grad clip
- auxiliary reconstruction objective
- upper-body recombination and freeze-frame hook
- 1-s adaptive failure bins, uniform mixture and pre-failure window
- released reward weights and termination thresholds
- periodic perturbation pushes
- public deployment action scaling/PD semantics
- GR00T-like flow matching, action chunks/masks, frozen pretrained perception/language option

## Deliberate changes

- Isaac Lab -> direct MuJoCo-Warp
- one G1 encoder instead of G1 + SMPL + teleop + SOMA for the first version
- default controller MLP is smaller; exact open G1 widths remain selectable
- only public BONES-SEED (~subset), not NVIDIA's private/full SONIC motion corpus
- reference FK is cached offline for efficient MuJoCo training
- physical randomization uses safely compiled MJWarp variant buckets
- MuJoCo's friction model cannot exactly reproduce Isaac static/dynamic friction separation
- Mini-GR00T uses frozen SigLIP2 + ~27M trainable action stack rather than N1.7 ~3B
- Mini-GR00T starts with a 16-step horizon rather than current N1.7's 40-step padded 132-D general embodiment head
- visual manipulation requires additional synchronized RGB demonstrations; BONES alone is not misrepresented as a visual dataset


## Current release details encoded explicitly

- actor observation corruption: gravity ±0.05, base angular velocity ±0.2, q-relative ±0.01 rad, qdot ±0.5; previous action uncorrupted
- G1 tokenizer orientation corruption: ±0.05 on the 6-D orientation representation
- freeze-frame probability: 0.1
- released reward-point override: torso offset [0,0,0.5] + left/right wrist-yaw points
- tracked-body set: 14 canonical G1 bodies from pelvis through legs, torso, arms and wrists
- reset/reference perturbations: xyz/orientation, joint-position and root-velocity ranges from the released command configuration
- body reference alignment follows NVIDIA's command-manager convention: robot XY + reference Z and heading-only orientation alignment

## Still simulator/project specific

- `undesired_contacts` needs your MJCF collision/body groups. It is intentionally zero until those groups are supplied; the code does not guess them.
- MJWarp spatial body velocity representation should be validated against your FK-cache velocity convention. The port uses MJWarp `cvel`; if your model exposes a preferred world-frame body velocity field, substitute it in `MjWarpBatchSim.body_state`.
- The public release's adaptive termination machinery is composed through Isaac-Lab manager configs. The strict released thresholds are implemented; `AdaptiveTerminationCurriculum` is provided as an optional, explicitly non-exact warm-up rather than inventing a hidden NVIDIA schedule.
