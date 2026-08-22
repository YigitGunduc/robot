# Future-Proof Embodied Intelligence Roadmap for a General-Purpose Unitree G1

> Revised architecture integrating the strongest parts of the previous perception/world-model/navigation plan with NVIDIA GEAR-SONIC, Isaac GR00T N1.7, and a Vesta-style embodied reasoning layer.
>
> Design objective: **one coherent embodied agent using reliable robotics tools**, not a collection of independent navigation/manipulation/planning modules competing for control.

---

# 1. Central design decision

Build the robot as a **multi-timescale embodied intelligence system** with a unified learned actor and a separate fast whole-body controller:

```text
Persistent experience / learning
        │
        ▼
Embodied reasoning and long-horizon memory      ← Vesta-like
        │
        ▼
Fast local physical intelligence                ← GR00T N1.7
        │
        ▼
Whole-body embodiment / motor prior             ← GEAR-SONIC
        │
        ▼
Unitree G1 hardware
```

Running beside this learned hierarchy are explicit robotics services:

```text
state estimation
local geometry
persistent mapping
object memory
IK / reachability
route planning
collision checking
safety supervision
data recording and evaluation
```

These services provide **facts, constraints, memory, tools, and fallbacks**. They should not micromanage the robot's continuous behavior.

The goal is not to reproduce human motion exactly. Human data is a prior for useful physical behavior. The robot should learn strategies that are natural and effective for **its own morphology, sensing, dynamics, reach, balance, hands, and actuation**.

---

# 2. What changes from the previous roadmap

The old roadmap's strongest insight was the multi-timescale split:

```text
System 3 — Persistent memory and learning
System 2 — Semantic reasoning and deliberative planning
System 1 — Reactive visuomotor intelligence
System 0 — Physical execution
```

That remains.

The major architectural change is **inside System 1**.

## 2.1 Remove the hard navigation/manipulation split

Do not make the local actor two independent authorities:

```text
Navigation controller
        +
Manipulation controller
```

A humanoid often needs to:

- step while reaching;
- reposition the pelvis during manipulation;
- carry while walking;
- crouch while grasping;
- brace against the environment;
- move the torso to increase reach;
- rotate and step during tool use;
- recover balance while preserving a grasp.

These should be learned as **one continuous whole-body behavior** whenever possible.

Global route planning still has value for long-range travel. The hard split disappears only at the local physical-control level.

## 2.2 Promote the learned actor much earlier

The previous roadmap placed a learned System 1 near the end. In the revised architecture:

```text
GR00T + SONIC
```

is established close to the beginning.

Otherwise the project risks spending most of its development time perfecting a traditional execution architecture that will later be replaced.

## 2.3 Demote behavior trees

Behavior trees remain useful for:

- mission supervision;
- startup/shutdown;
- battery management;
- return-home behavior;
- model failure fallback;
- task cancellation;
- permission gates;
- emergency recovery.

They should **not** become the primary representation of general manipulation intelligence.

Avoid encoding every task as:

```text
Navigate → Reach → Grasp → Lift → Carry → Place
```

when a learned actor can solve the physical sequence continuously.

## 2.4 Demote MoveIt / classical manipulation planning

MoveIt 2, IK, collision checkers, and reachability solvers remain valuable as:

- feasibility oracles;
- constraint generators;
- offline teachers;
- recovery tools;
- deterministic fallbacks.

They are not the primary intelligence.

## 2.5 Move predictive physical reasoning earlier

A useful robot must learn:

```text
If I do this, what is likely to happen?
```

Action-outcome prediction should no longer be only a distant future feature. Start with simple predictors for:

- grasp success;
- object displacement;
- collision risk;
- balance risk;
- containment;
- articulation movement;
- support/contact success.

Later evolve toward a learned latent dynamics model.

---

# 3. Target architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│                           USER / MISSION                              │
│ Speech, text, schedule, operator request, autonomous mission          │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                 EMBODIED REASONING / PERSISTENT MIND                 │
│                         Vesta-like System 2                           │
│                                                                       │
│ Goal understanding              Long-horizon progress                 │
│ Spatial reasoning               Episodic retrieval                    │
│ Object/tool reasoning           Search/exploration strategy           │
│ Task decomposition              Failure interpretation                │
│ Counterfactual strategy         Uncertainty / ask-for-help logic      │
│                                                                       │
│ Runs slowly / event-driven. Does not generate joint trajectories.     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                      grounded physical intent
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     EMBODIED ACTOR / SYSTEM 1                         │
│                          GR00T N1.7                                   │
│                                                                       │
│ Inputs                                                                │
│   ego cameras / optional depth                                        │
│   robot state                                                         │
│   local geometry features                                             │
│   current goal                                                        │
│   relevant object/body memory                                         │
│   recent action/observation history                                   │
│                                                                       │
│ Learns local physical behavior:                                       │
│   approach • step • reach • contact • grasp • carry • reposition      │
│   push • pull • place • regrasp • recover • use tools                 │
│                                                                       │
│ Primary output for G1: SONIC motion tokens + hand commands            │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                    latent whole-body intention
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    EMBODIMENT / SYSTEM 0                              │
│                         GEAR-SONIC                                    │
│                                                                       │
│ Learned body prior                                                     │
│ Whole-body coordination                                                │
│ Locomotion and stepping                                                │
│ Posture / balance-compatible motion                                    │
│ Contact-aware motion tracking                                          │
│ Smooth coordinated arm/torso/leg behavior                              │
│                                                                       │
│ Fast control loop; no semantic long-horizon reasoning.                │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
                            UNITREE G1
                                │
                 sensors / contacts / proprioception
                                │
               ┌────────────────┴───────────────────┐
               │                                    │
               ▼                                    ▼
        EXPLICIT WORLD MODEL                   BODY MODEL
               │                                    │
               └────────────── feedback ────────────┘
```

The architecture should feel like **one mind inhabiting one body**, even though it is implemented through multiple rates and models.

---

# 4. Responsibilities of SONIC, GR00T, and Vesta

# 4.1 GEAR-SONIC — embodiment and motor intelligence

SONIC answers:

> **How can this G1 body physically realize the requested behavior?**

Use SONIC for:

- whole-body motion coordination;
- locomotion and stepping;
- arm/torso/leg coupling;
- natural posture transitions;
- balance-compatible motion;
- motion-prior generalization;
- control-rate execution;
- teleoperation and reference-motion decoding.

Official GR00T Whole-Body Control documentation currently supports a SONIC universal-token controller for the Unitree G1. The released architecture uses a 64-dimensional motion token and a 50 Hz controller. In the VLA workflow, GR00T can predict 40 future SONIC tokens; SONIC decodes the latent commands into full-body joint commands. The full G1 SONIC action space used by the official workflow is currently 78 dimensions: 64 motion-token values plus 7 left-hand and 7 right-hand joint commands.

Official references:

- https://github.com/NVlabs/GR00T-WholeBodyControl
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_workflow.md
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_inference.md

### SONIC should not own

SONIC should not decide:

- which object matters;
- why a task is failing semantically;
- whether a spoon can substitute for a missing scraper;
- which room to search;
- what the user's ultimate goal means;
- what happened twenty minutes earlier.

Those belong higher in the stack.

---

# 4.2 GR00T N1.7 — local physical intelligence

GR00T answers:

> **Given what I see, my body state, and the current goal, what physical behavior should I perform now?**

Use GR00T as the primary local actor for:

- visual manipulation;
- whole-body mobile manipulation;
- continuous local approach behavior;
- grasp selection and adjustment;
- contact strategy;
- regrasping;
- carrying;
- moving around the immediate workspace;
- local recovery from failed interaction;
- physically grounded task execution.

Do not restrict it to a small hand-authored action vocabulary if the learned policy can solve the local problem directly.

For G1 + SONIC, use the official `UNITREE_G1_SONIC` embodiment path rather than creating a custom incompatible action representation unless evaluation proves a different interface is better.

Official references:

- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/GR00TWholeBodyControl/README.md
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/embodiment_tags.py

### GR00T should not own

GR00T should not be responsible for:

- permanent global metric mapping;
- storing years of object history;
- hard real-time emergency collision stopping;
- actuator watchdogs;
- multi-room shortest-path search when a deterministic route planner solves it reliably;
- long-horizon task bookkeeping.

---

# 4.3 Vesta-like planner — persistent embodied reasoning

Vesta answers:

> **What is happening, what should happen next, what matters, and should I change strategy?**

Vesta's published architecture unifies:

- localization-style grounding;
- navigation reasoning;
- embodied reasoning;
- task progress estimation;
- long-horizon action planning;
- multimodal episodic memory.

In NVIDIA's published real-robot evaluation, Vesta was used above GR00T N1.6 as a planner and substantially improved performance on memory-heavy tasks. NVIDIA also demonstrates Vesta producing navigation waypoints for a G1 simulator whose low-level motion is handled by a SONIC-based controller.

Official reference:

- https://research.nvidia.com/labs/gear/vesta/
- https://arxiv.org/abs/2606.20905

### Important implementation rule

Do **not** hard-wire the robot to an implementation-specific Vesta API.

Use:

```python
class ReasoningProvider(Protocol):
    def deliberate(self, context: EmbodiedContext) -> ReasoningDecision:
        ...
```

Vesta is the preferred architectural model/backend when available. A compatible multimodal planner can temporarily implement the same interface.

### Vesta should not micromanage motion

Bad:

```text
move forward 0.21 m
rotate 28 degrees
grasp at x/y/z
pull 0.17 m
```

Better:

```text
Goal:
    open(drawer)

Attention:
    drawer_handle
    drawer_front

Constraints:
    preserve clearance from vase
    do not move cabinet

Current hypothesis:
    drawer likely requires pulling

Success condition:
    drawer_open_fraction > threshold
```

GR00T then solves the local physical behavior.

---

# 5. Long-term model-collapse strategy

There is merit in reducing the number of separate learned models, but not all boundaries should disappear.

## 5.1 Do not collapse SONIC first

Reasoning and motor control have very different:

- update rates;
- training data;
- loss functions;
- failure modes;
- safety requirements.

Even if the system eventually becomes end-to-end trainable, preserve a fast embodiment layer conceptually.

## 5.2 Vesta + GR00T is the most promising fusion

Long-term target:

```text
              SHARED EMBODIED FOUNDATION MODEL

 camera ────────────┐
 language ──────────┤
 body state ────────┤
 world memory ──────┤
 recent history ────┘
          │
     shared latent
       world/body state
          │
     ┌────┴─────────────┐
     │                  │
 slow reasoning      fast acting
  Vesta-like         GR00T-like
     │                  │
     └──── guidance ─────┘
                        │
                  SONIC tokens
                        │
                        ▼
                      SONIC
```

This reduces the information bottleneck where a planner has to compress detailed spatial reasoning into a short English instruction.

## 5.3 Event-driven reasoning

Deep reasoning should be triggered by events such as:

- no task progress;
- repeated grasp failure;
- unexpected contact;
- missing tool;
- low confidence;
- target disappearance;
- contradiction in memory;
- environment change;
- completion of a meaningful subgoal;
- safety-layer intervention.

During normal successful behavior, GR00T + SONIC should continue without repeatedly invoking expensive deliberation.

---

# 6. Core engineering principles

## 6.1 Modular engineering, unified intelligence

Keep modular:

- hardware drivers;
- sensor synchronization;
- state estimation;
- geometry;
- mapping;
- safety;
- persistent databases;
- logging;
- evaluation;
- model serving.

Unify as much as practical inside the embodied actor:

- local navigation;
- body positioning;
- reaching;
- manipulation;
- carrying;
- tool interaction;
- local recovery.

## 6.2 Separate capabilities from implementations

Downstream code should request:

```text
LocalRobotState
LocalGeometryView
GlobalSpatialMemory
TrackedObjectSet
EmbodiedMemory
RouteHint
PhysicalIntent
ActorAction
SafetyConstraintSet
```

It should not require:

```text
NvbloxMap
HydraNode
DaaamTrack
Nav2Path
VestaRawResponse
GrootInternalTensor
```

This permits backend replacement without changing the architecture.

## 6.3 Preserve a deterministic fallback path

The robot should still be able to:

- localize;
- stop safely;
- avoid immediate obstacles;
- return home;
- execute simple point-goal navigation;
- report uncertainty;
- enter teleoperation mode;

if semantic/planning models fail.

The fallback does not need equal capability. It must fail safely and predictably.

## 6.4 Uncertainty is first-class data

Every important estimate should include:

```text
value
timestamp
reference frame
confidence or covariance
source
freshness
supporting evidence
```

Distinguish:

- observed now;
- observed recently;
- historical;
- predicted;
- inferred;
- unknown;
- contradictory.

## 6.5 Build the data engine from day one

Every meaningful autonomy run should record:

- synchronized sensor data;
- proprioception;
- SONIC state/tokens where available;
- GR00T observations and predicted actions;
- Vesta/planner decisions;
- world-model updates;
- explicit geometry state;
- alternative candidate strategies when available;
- action-outcome predictions;
- actual outcomes;
- unexpected contacts;
- safety interventions;
- human corrections;
- success/failure labels;
- recovery sequence;
- uncertainty and model confidence.

---

# 7. World representations

The world model is **not one map**.

Use several synchronized representations optimized for different jobs.

## 7.1 Smooth local robot state

Purpose:

- SONIC/controller observation support;
- local acting;
- balance/contact interpretation;
- sensor projection;
- safety filtering.

Contains:

```text
base pose in odom
base velocity
joint states
contact states
IMU state
hand state
held-object estimate
state covariance
sensor timestamps
```

Target update rate:

```text
100–500 Hz state propagation
```

Global loop closure must never cause a discontinuous jump inside the fast control frame.

## 7.2 Local dense geometry

Keep the previous TSDF/occupancy + ESDF concept.

Purpose:

- collision detection;
- support-surface estimation;
- body clearance;
- manipulation clearance;
- reachability support;
- drop-off detection;
- short-horizon trajectory checks.

Representation:

```text
TSDF / occupancy
ESDF / clearance
support surfaces
ground / traversability
negative obstacles
dynamic-object exclusion mask
uncertainty
observation age
```

Recommended initial backend:

```text
Isaac ROS nvblox behind GeometryMap
```

The actor may learn geometry implicitly too, but the explicit map is a valuable independent physical constraint source.

## 7.3 Persistent global metric memory

Purpose:

- localization;
- room-to-room routing;
- loop closure;
- multi-session operation;
- persistent search;
- map-change detection.

Store:

```text
pose graph
keyframes
submaps
optimized poses
room topology
traversability graph
map version
observation sessions
```

Keep full resolution local. Load distant submaps only when useful.

## 7.4 Object-centric semantic world

A scene graph remains useful for explicit, searchable structure.

Entities should include:

```text
Building
Floor
Room
Place
Surface
Object
ObjectPart
Human
Robot
Region
TaskZone
```

Relations may include:

```text
inside
contains
adjacent_to
on_top_of
under
near
connected_to
reachable_from
visible_from
part_of
held_by
supports
blocks
moving_toward
last_seen_at
```

Hydra can remain a strong initial backend behind a `SemanticWorldGraph` interface.

### Important rule

The scene graph is **explicit memory**, not the entirety of what the robot knows.

Do not force every subtle visual or physical cue into a symbolic graph before the actor can use it.

## 7.5 Temporal and episodic physical memory

Store object history:

```text
object UUID
pose history
velocity history
semantic distribution
appearance embeddings
state attributes
first seen
last seen
supporting observations
confidence
static / semi-static / dynamic
```

Store episodes:

```text
goal
initial world/body state
strategy attempted
actions
observations
predicted outcomes
actual outcomes
unexpected contacts
failure cause
recovery behavior
final result
human feedback
```

Retrieval should support physical similarity, not only language similarity.

Example:

```text
Current problem:
    heavy drawer, horizontal handle, normal pull failed

Retrieved experience:
    similar drawer opened by bracing frame with left hand,
    pulling with right hand, stepping backward
```

## 7.6 Learned body schema

Add a representation of:

```text
what this body can do
```

This should include learned estimates of:

- reachability;
- comfortable reach;
- stepping requirement;
- balance margin;
- visibility;
- grasp success;
- force capability;
- posture feasibility;
- two-hand coordination;
- tool-extended reach;
- terrain capability;
- action latency;
- stopping behavior.

Prefer probabilistic queries such as:

```text
P(successful_interaction |
    target_pose,
    body_state,
    object_geometry,
    hand,
    local_scene)
```

over hard-coded constants whenever enough data exists.

## 7.7 Learned latent physical world model

Start small.

Predict:

```text
grasp_success
object_displacement
contact_success
collision_risk
balance_risk
articulation_motion
containment_success
```

Later evolve toward:

```text
latent_state_t + candidate_action
             ↓
     predicted_latent_state_t+1
```

Use it initially to **score or compare candidate physical strategies**, not to bypass safety or directly command actuators.

## 7.8 Semantic embedding index

Keep embeddings for:

- appearance;
- language description;
- function;
- state;
- associated tasks;
- physical interaction history.

The graph/entity database remains authoritative. The embedding index accelerates retrieval.

---

# 8. Perception architecture

Keep the previous multi-lane perception strategy.

## 8.1 Lane A — reflexive geometry

Continuous and predictable.

Inputs:

```text
depth
LiDAR
IMU-compensated points
robot pose
```

Outputs:

```text
occupancy
surfaces
free space
clearance
ground
drop-offs
immediate collision hazards
```

This lane cannot depend on a large VLM.

## 8.2 Lane B — real-time object perception

Responsibilities:

```text
person detection
safety-critical object detection
object segmentation
short-term tracking
motion estimation
3D association
```

Use efficient models with bounded latency.

## 8.3 Lane C — open-ended semantic understanding

Runs asynchronously or on selected frames.

Responsibilities:

```text
open-vocabulary segmentation
novel-object description
fine-grained attributes
state estimation
affordance hypotheses
ambiguity resolution
language grounding
```

DAAAM remains a useful reference/backend behind an adapter.

## 8.4 Lane D — actor-native visual features

New requirement.

Not all visual information should be converted to symbolic labels before GR00T sees it.

The embodied actor should receive native visual observations/features so it can exploit:

- shape cues;
- contact geometry;
- partial occlusion;
- texture/state cues;
- subtle object orientation;
- immediate motion;
- task-specific visual patterns.

Avoid a mandatory bottleneck of:

```text
camera → labels/JSON → actor
```

The explicit world model supplements direct perception rather than replacing it.

---

# 9. Localization and SLAM

Keep the previous architecture almost unchanged.

Separate:

```text
odom → base_link
```

from:

```text
map → odom
```

Fast control consumes smooth local state. Persistent memory and global planning consume map-frame corrections.

## 9.1 Estimator interface

```python
class StateEstimator(Protocol):
    def get_local_state(self) -> LocalRobotState:
        ...

    def get_global_transform(self) -> TransformEstimate:
        ...
```

## 9.2 Simulation progression

```text
MuJoCo oracle pose
→ noisy oracle pose
→ simulated IMU/LiDAR/vision
→ recorded-data estimator
→ live estimator
```

## 9.3 Humanoid-specific fusion

Account for:

- head/torso oscillation;
- impact vibration;
- foot contacts;
- self-occlusion;
- gait-periodic motion;
- sensor flex;
- calibration drift;
- changing body geometry.

Add:

```text
contact factors
kinematic constraints
motion-distortion compensation
self-filtering
calibration checks
gait-aware noise models
```

---

# 10. Navigation architecture — revised

Navigation now has two different scales.

## 10.1 Global routing remains explicit

For tasks such as:

```text
kitchen → bedroom
warehouse aisle A → loading dock
search room 3 then room 5
```

use conventional spatial planning.

Good backends remain:

```text
room/place topology
A* / Theta* / Smac
Nav2
MPPI
```

There is little value in forcing a large neural actor to rediscover shortest-path graph search every time.

## 10.2 Local movement becomes part of the embodied actor

Near task-relevant objects or in interaction-rich spaces, increasingly use:

```text
GR00T → SONIC
```

for:

- approach positioning;
- final foot placement;
- torso positioning;
- step-while-reaching;
- carrying around local obstacles;
- manipulation-coupled locomotion;
- recovering a better stance.

## 10.3 Route planner becomes advisory

Define:

```python
@dataclass
class RouteHint:
    corridor: list[Pose]
    goal_region: GoalRegion
    forbidden_regions: list[Region]
    preferred_approach: ApproachHint | None
```

GR00T can condition on this guidance without being forced to replay a precomputed velocity trajectory exactly.

## 10.4 Preserve deterministic fallback navigation

Keep the old:

```text
GlobalPlanner
LocalPlanner
VelocityCommand
```

interfaces as a fallback and benchmark.

The existing A*/DWA or Nav2 path remains useful for:

- actor failure;
- model-offline mode;
- regression comparison;
- low-risk transit;
- collecting planner demonstrations.

---

# 11. Manipulation architecture — revised

## 11.1 The actor owns local physical behavior

Do not require Vesta to generate every primitive.

Fundamental inputs should be closer to:

```text
goal relationship
relevant objects
constraints
attention targets
local geometry
body state
recent history
```

Example:

```text
Goal:
    inside(block, bowl)

Constraints:
    do not disturb glass
```

The actor may discover:

- direct grasp;
- palm push;
- finger push;
- bimanual move;
- dragging the bowl closer;
- tool-assisted scoop;
- repositioning the body first.

## 11.2 Task-space primitives remain optional tools

Still expose validated tools such as:

```text
reach(target_pose)
push(object, direction)
pull(object, direction)
grasp(object, grasp_region)
lift(object)
place(object, goal_region)
release()
```

Use them for:

- fallback execution;
- data generation;
- Vesta-triggered recovery;
- benchmarking;
- difficult precision actions;
- bootstrapping early training.

They are **not the only behaviors the actor is allowed to perform**.

## 11.3 IK / collision / reachability as oracles

Before or during execution, make available:

```text
IK feasibility
joint-limit margin
collision margin
support polygon / balance estimate
reachability probability
body-clearance estimate
```

The learned actor can use these signals as features, teachers, or constraints.

## 11.4 Closed-loop outcome verification

Never define success as "the motion finished."

For a grasp, verify:

```text
fingers closed appropriately
object moved with hand
visual tracking consistent
contact/force signal plausible
object not slipping
```

For a drawer:

```text
drawer state changed
expected articulation observed
progress exceeds threshold
```

Failed local attempts should first trigger GR00T-level adaptation. Repeated or semantically meaningful failures trigger Vesta-level deliberation.

---

# 12. Affordances — from labels to physical hypotheses

Keep explicit affordance hypotheses:

```text
graspable
pushable
pullable
openable
pourable
container
supporting
handle
fragile
hot
hazardous
```

but expand them with action-conditioned predictions.

Instead of only:

```text
dustpan.pushable = true
```

learn:

```text
P(block enters bowl |
  contact point,
  dustpan geometry,
  push direction,
  current scene,
  body state)
```

Affordances should be:

- probabilistic;
- context-dependent;
- learned from experience;
- revisable;
- grounded by geometry and actual outcomes.

This is essential for novel tool use.

---

# 13. Embodied reasoning and memory

## 13.1 Use Vesta-style reasoning for the slow loop

Preferred reasoning output:

```python
@dataclass
class ReasoningDecision:
    objective: GoalPredicate
    attention_entities: list[UUID]
    constraints: list[Constraint]
    relevant_memory_ids: list[UUID]
    strategy_hint: str | None
    exploration_request: ExplorationRequest | None
    success_predicates: list[Predicate]
    abort_predicates: list[Predicate]
    confidence: float
```

Avoid using unrestricted prose as the execution API.

## 13.2 Keep multimodal history selective

The planner should receive:

- current keyframes;
- task-relevant past frames;
- structured world state;
- a compact progress record;
- retrieved physical episodes.

Do not repeatedly send the entire raw map and complete video history.

## 13.3 Event-driven escalation

Define triggers:

```text
progress_score < threshold
same_failure repeated N times
world contradiction
actor uncertainty high
safety intervention
missing object/tool
goal ambiguity
unexpected scene change
subgoal complete
```

Only then invoke expensive deliberation.

---

# 14. GR00T actor design

## 14.1 Baseline interface

Use the official SONIC embodiment when possible:

```text
Inputs:
    ego image
    G1 state
    projected gravity
    task/language condition

Outputs:
    motion_token[64]
    left_hand_joints[7]
    right_hand_joints[7]
```

The official N1.7 configuration predicts an action chunk over future steps rather than a single instantaneous motor command.

## 14.2 Extend context gradually

V1:

```text
ego RGB
robot state
language / goal
```

V2:

```text
+ selected world entities
+ local geometry summary
+ route hint
+ body-schema features
+ recent failures
```

V3:

```text
+ retrieved physical episodes
+ learned predictive-model features
+ Vesta shared latent / richer planner conditioning
```

Do not overload the first implementation with every representation simultaneously.

## 14.3 Training data

Use:

- official/compatible GR00T pretraining prior;
- whole-body teleoperation;
- SONIC teleop data;
- successful autonomous episodes;
- planner-generated demonstrations;
- simulation rollouts;
- human corrections;
- failed attempts plus recoveries;
- deliberately perturbed tasks;
- cross-object and cross-layout task variants.

## 14.4 Train on goals and problems, not only trajectories

Curriculum should include:

```text
object moved from expected location
tool missing
preferred hand occupied
obstacle blocks reach
grasp slips
drawer sticks
target moves
object heavier than expected
route becomes blocked
familiar tool replaced by unfamiliar object
```

Reward completion of the **desired world state** rather than imitation of a single reference path wherever possible.

## 14.5 Preserve physical prompting as a later capability

Demonstration-conditioned behavior remains valuable, but it is not the complete intelligence goal.

Later add:

```text
physical demonstration + current scene + goal
                    ↓
              GR00T actor
```

Use demonstrations as behavioral context while still allowing the robot to choose a different trajectory or tool strategy.

---

# 15. SONIC integration and migration from existing G1 controllers

The existing G1 locomotion/manipulation code remains valuable as:

- regression baseline;
- simulation fallback;
- teacher;
- deterministic navigation bridge;
- source of existing reward/evaluation infrastructure.

Preserve the current command semantics where useful, particularly existing command-conditioned locomotion interfaces such as velocity/body-height/gait parameters.

Do not require all old policy action semantics to become the new primary interface.

## 15.1 Suggested migration

```text
Current custom locomotion policy
        │
        ├── keep as benchmark/fallback
        └── keep for isolated RL experimentation

New production embodied path
        │
        GR00T N1.7
        ↓
        SONIC tokens
        ↓
        GEAR-SONIC
        ↓
        G1
```

## 15.2 Existing project-file mapping

Where these files remain in the project:

```text
height_conditioned_g1.py
    keep the current command-conditioned locomotion environment
    as fallback/regression and sim experimentation.

train_g1.py
    keep training/evaluation harnesses;
    add SONIC-vs-custom-controller benchmark profiles.

render_g1.py
    extend to visualize planner goal, route hint,
    GR00T action chunk, SONIC token/debug state,
    contact events, and body-schema estimates.

g1_arm_manipulation.py
    shift from independent arm authority toward
    task-space goals/oracles/fallback skills.

g1_arm_manipulation_config.py
    retain manipulation limits, collision settings,
    task-space targets, and evaluation scenarios.

manipulation_checkpoint.py
    preserve checkpoint migration and isolated-policy evaluation;
    do not require a separate upper-body policy in the final architecture.
```

## 15.3 Do not split upper and lower body into independent primary policies

The primary embodied path should exploit whole-body coordination.

A temporary decoupled arm/leg setup is acceptable for debugging or fallback, but not the long-term control philosophy.

---

# 16. Safety architecture

Safety remains a separate plane.

```text
GR00T / route / fallback command
             │
             ▼
       command validator
             │
             ▼
   predictive safety filter
             │
             ▼
 independent collision monitor
             │
             ▼
            SONIC
```

## 16.1 Command validator

Checks:

```text
velocity/acceleration limits
joint/actuator limits
stale commands
stale perception
support/contact state
forbidden regions
battery/task permissions
known hazardous conditions
```

## 16.2 Predictive safety filter

Use empirical G1 models for:

```text
stopping distance
turning lag
lateral drift
body sweep volume
carried-object footprint
fall risk
command latency
```

## 16.3 Independent collision monitor

Emergency slowdown/stop should use minimally processed geometric sensing and should not require Vesta or GR00T to be functioning.

## 16.4 Semantic safety cannot reduce geometric safety

Semantic information can make the robot **more conservative** around:

- people;
- animals;
- fragile objects;
- hot surfaces;
- sharp objects;
- restricted areas.

It cannot override collision margins or actuator constraints.

---

# 17. ROS 2 architecture and compartmentalization

Keep the backend-independent structure, but revise the intelligence packages.

```text
robot_ws/
├── src/
│   ├── robot_interfaces/
│   ├── robot_hardware/
│   │   ├── mujoco_bridge/
│   │   ├── g1_driver/
│   │   ├── sonic_bridge/
│   │   ├── sensor_drivers/
│   │   └── time_sync/
│   │
│   ├── robot_state/
│   │   ├── state_estimator_api/
│   │   ├── oracle_state_estimator/
│   │   ├── lidar_inertial_adapter/
│   │   └── contact_fusion/
│   │
│   ├── robot_geometry/
│   │   ├── geometry_map_api/
│   │   ├── occupancy_backend/
│   │   ├── nvblox_adapter/
│   │   └── traversability/
│   │
│   ├── robot_semantics/
│   │   ├── semantic_perception_api/
│   │   ├── daaam_adapter/
│   │   ├── fast_detector/
│   │   ├── object_pose/
│   │   └── dynamic_tracking/
│   │
│   ├── robot_world_model/
│   │   ├── world_model_api/
│   │   ├── hydra_adapter/
│   │   ├── temporal_store/
│   │   ├── body_schema/
│   │   ├── physical_outcome_model/
│   │   ├── embedding_index/
│   │   └── map_persistence/
│   │
│   ├── robot_navigation/
│   │   ├── navigation_api/
│   │   ├── global_route_planner/
│   │   ├── existing_navigation_core/
│   │   ├── nav2_adapter/
│   │   └── exploration/
│   │
│   ├── robot_actor/
│   │   ├── actor_api/
│   │   ├── groot_n17_adapter/
│   │   ├── sonic_action_adapter/
│   │   ├── fallback_task_space_actor/
│   │   └── actor_monitor/
│   │
│   ├── robot_reasoning/
│   │   ├── reasoning_api/
│   │   ├── vesta_adapter/
│   │   ├── alternate_vlm_adapter/
│   │   ├── progress_monitor/
│   │   ├── memory_retrieval/
│   │   └── deliberation_manager/
│   │
│   ├── robot_safety/
│   │   ├── command_validator/
│   │   ├── predictive_filter/
│   │   └── collision_monitor/
│   │
│   ├── robot_data_engine/
│   │   ├── recorder/
│   │   ├── event_logger/
│   │   ├── replay/
│   │   ├── auto_labeling/
│   │   ├── failure_mining/
│   │   ├── evaluator/
│   │   └── dataset_registry/
│   │
│   └── robot_bringup/
```

## 17.1 Process boundaries

Keep separate processes for:

```text
SONIC / low-level control
safety monitor
GR00T inference
Vesta/reasoning inference
research-grade mapping/semantic stack
data recorder
```

A VLM crash must never terminate balance or emergency stopping.

---

# 18. Canonical interfaces

## 18.1 Physical intent

```python
@dataclass
class PhysicalIntent:
    objective: GoalPredicate
    attention_entities: list[UUID]
    constraints: list[Constraint]
    route_hint: RouteHint | None
    strategy_hint: str | None
    valid_until_ns: int
```

## 18.2 Actor observation

```python
@dataclass
class ActorObservation:
    images: dict[str, ImageRef]
    local_state: LocalRobotState
    local_geometry: LocalGeometrySummary | None
    relevant_entities: list[WorldEntity]
    body_schema_features: BodySchemaFeatures | None
    recent_events: list[EmbodiedEvent]
    intent: PhysicalIntent
```

## 18.3 Actor action

```python
@dataclass
class SonicActorAction:
    motion_token: np.ndarray      # 64-D for current SONIC workflow
    left_hand_joints: np.ndarray # 7-D where supported
    right_hand_joints: np.ndarray# 7-D where supported
    confidence: float
    predicted_progress: float
```

Do not make dimensionality a permanent system-wide assumption. Version the action schema because future SONIC checkpoints can change their latent space.

## 18.4 World entity

Retain the old `WorldEntity` concept with:

```text
UUID
semantic distribution
pose + covariance
extent
motion state
attributes
relations
observation provenance
first/last seen
persistence class
confidence
```

## 18.5 Embodied episode

```python
@dataclass
class EmbodiedEpisode:
    goal: GoalPredicate
    initial_state_ref: str
    selected_strategy: str | None
    events: list[EmbodiedEvent]
    predicted_outcomes: list[OutcomePrediction]
    safety_interventions: list[SafetyEvent]
    final_success: bool
    final_score: float
    operator_feedback: str | None
```

---

# 19. Updated implementation roadmap

The ordering is intentionally changed so the learned embodied path appears early.

## Phase 0 — Freeze the current baseline and interfaces

### Goal

Preserve all working navigation/locomotion capability before changing the architecture.

### Keep

```text
existing MuJoCo scenarios
existing known-map navigation
rolling local map
DWA/A* baseline
current G1 locomotion controller
current manipulation experiments
```

### Add

```text
machine-readable metrics
failure reason codes
deterministic seeds
record/replay
versioned interfaces
```

### Acceptance gate

Current benchmark results remain reproducible after the refactor.

---

## Phase 1 — Establish SONIC as the new whole-body reference path

### Goal

Run official SONIC evaluation/inference for G1 and understand the latent/action boundary.

### Implement

- install and run GR00T Whole-Body Control in simulation;
- validate G1 model/joint ordering;
- inspect SONIC token encoder/decoder behavior;
- record controller state at 50 Hz;
- create `SonicController` adapter;
- benchmark SONIC against the current custom locomotion policy on basic locomotion/body-motion tests.

### Do not

Delete the custom controller yet.

### Acceptance gate

- stable SONIC simulation execution;
- known safe initialization;
- no uncontrolled pose snaps;
- repeatable token-to-motion behavior;
- baseline locomotion/motion tests pass.

---

## Phase 2 — GR00T N1.7 → SONIC actor baseline

### Goal

Make the robot execute language-conditioned whole-body mobile manipulation through the official G1 SONIC action representation.

### Implement

```text
ego camera
G1 state
projected gravity
language/task input
        ↓
GR00T N1.7
        ↓
64-D SONIC motion token + hand joints
        ↓
SONIC
```

Start in simulation.

### Initial tasks

```text
approach table
reach object
pick object
carry a short distance
place object
```

### Acceptance gate

The GR00T + SONIC path completes simple closed-loop whole-body tasks in simulation without using a separate navigation/manipulation authority.

---

## Phase 3 — Actor evaluation harness and data engine

### Goal

Make every run useful for learning and debugging.

### Record

```text
all cameras
robot state
SONIC tokens
GR00T observations/actions
contact events
goal/progress
success/failure
human corrections
```

### Add deliberate perturbation tests

```text
object moved
target pose changed
minor obstruction
grasp slip
start stance changed
lighting variation
```

### Acceptance gate

Every failure can be replayed and classified.

---

## Phase 4 — Local geometry and explicit safety integration

### Goal

Give the actor reliable geometric constraints without making geometry the actor.

### Implement

```text
RGB-D preprocessing
local TSDF/occupancy
ESDF clearance
support surfaces
drop-off detection
self-filtering
independent collision monitor
```

Add a compact geometry summary to actor context only after the baseline is stable.

### Acceptance gate

- geometric safety works with GR00T disabled;
- false/stale geometry cannot cause unsafe command continuation;
- actor performance does not regress unacceptably.

---

## Phase 5 — Estimated state and localization

### Goal

Remove dependence on simulator oracle state.

### Implement

```text
IMU
LiDAR/vision localization
contact-aided body-state fusion
map→odom correction
calibration monitoring
```

### Acceptance gate

- smooth local state;
- loop closure never destabilizes control;
- graceful degradation vs oracle pose;
- estimator failure leads to stop/fallback.

---

## Phase 6 — Persistent world model

### Goal

Build explicit long-term spatial and object memory.

### Implement in order

```text
persistent metric submaps
room/place topology
Hydra adapter
tracked object UUIDs
temporal event store
semantic embeddings
```

Keep DAAAM or an equivalent semantic stack behind `SemanticPerceptionProvider`.

### Acceptance gate

The robot can answer without a VLM:

```text
which room am I in?
where was object X last seen?
what objects are currently in this room?
which routes connect room A and B?
```

---

## Phase 7 — Body schema

### Goal

Teach the system the practical capabilities of this G1 body.

### Generate simulation data for

```text
reach success
stance-dependent reach
step-needed reach
grip success
balance margin
body clearance
visibility
carrying stability
push/pull feasibility
```

### Train models such as

```text
ReachabilityPredictor
InteractionSuccessPredictor
BalanceRiskPredictor
```

### Acceptance gate

The system predicts held-out physical feasibility better than fixed-rule baselines.

---

## Phase 8 — Vesta-style embodied reasoning

### Goal

Add long-horizon cognition only after a working actor exists.

### Implement

```text
ReasoningProvider
progress monitor
memory retrieval
deliberation triggers
structured PhysicalIntent output
```

Use Vesta directly if an appropriate runnable checkpoint/runtime is available; otherwise use a compatible multimodal planner while preserving the same interface.

### Initial tasks

```text
find an object in multiple rooms
remember which locations were checked
complete a multi-stage pick/place mission
recover when a required object is missing
```

### Acceptance gate

Planner improves held-out long-horizon task success relative to GR00T actor alone and does not reduce low-level reliability.

---

## Phase 9 — Hybrid global routing + embodied local control

### Goal

Use classical route planning where it is strongest and learned whole-body behavior where embodiment matters.

### Implement

```text
semantic goal → global route / goal region
                      ↓
                  RouteHint
                      ↓
                 GR00T + SONIC
```

### Acceptance gate

The robot can move across multiple rooms and transition naturally into manipulation without a hard navigation/manipulation handoff failure.

---

## Phase 10 — Problem-solving curriculum

### Goal

Train strategy generalization rather than only clean demonstrations.

### Scenarios

```text
missing familiar tool
unreachable target
preferred hand unavailable
object blocked
tool too short
grasp failure
object slip
target moves
container closed
drawer stuck
unexpected obstacle
```

### Reward emphasis

```text
goal-state completion
safe progress
efficient recovery
low unnecessary intervention
```

not only trajectory matching.

### Acceptance gate

Held-out success on perturbed tasks increases without memorizing scenario-specific scripts.

---

## Phase 11 — Predictive physical model

### Goal

Give the robot limited physical foresight.

### V1 predictors

```text
grasp success
push displacement
collision risk
balance risk
articulation progress
```

### V2

Latent transition model for candidate action chunks.

### Use

Rank strategies before execution and detect model surprise after execution.

### Acceptance gate

Prediction-guided selection improves task success or reduces failed contacts on held-out scenarios.

---

## Phase 12 — Episodic physical learning

### Goal

Reuse prior physical experience.

### Retrieve by

```text
object geometry
interaction type
body configuration
failure signature
environment structure
semantic goal
```

### Acceptance gate

The robot solves repeated/similar physical problems with fewer failures or shorter completion time.

---

## Phase 13 — Vesta/GR00T representation fusion research

### Goal

Reduce the planner→actor information bottleneck.

Investigate:

- shared visual encoder;
- shared spatial tokens;
- planner latent conditioning of actor;
- common episodic memory embeddings;
- slow/fast heads on a shared backbone;
- test-time memory or recurrent adaptation.

Do this only after the hierarchical baseline is measurable.

### Acceptance gate

Fusion improves generalization or efficiency on held-out tasks compared with the clean hierarchical baseline.

---

## Phase 14 — Continual data engine and model improvement

Build:

```text
episode recorder
automatic failure triage
hard-example mining
human correction UI
data/model versioning
simulation replay
regression benchmarks
shadow deployment
rollback
```

Every new model must run against:

```text
fixed benchmark episodes
previous failures
safety-critical scenarios
new-environment holdouts
latency/power budgets
```

---

# 20. Immediate implementation order

The next concrete sequence should now be:

## Step 1 — Freeze current benchmarks

Keep the existing navigation, MuJoCo, locomotion, and manipulation tests as regression baselines.

## Step 2 — Add the new core interfaces

```text
SonicController
EmbodiedActor
ReasoningProvider
PhysicalIntent
BodySchemaProvider
WorldModel
SafetySupervisor
```

## Step 3 — Run SONIC in G1 simulation

Do not integrate semantics yet.

Understand initialization, observation requirements, token behavior, latency, and failure modes.

## Step 4 — Run GR00T N1.7 → SONIC in simulation

Reproduce the official whole-body VLA workflow before modifying it.

## Step 5 — Build recording/replay around the new actor

Record GR00T, SONIC, body state, visual input, and outcomes.

## Step 6 — Integrate explicit local geometry and safety

Geometry advises/constrains the actor; it does not replace it.

## Step 7 — Add estimated pose/state

Keep oracle state only for evaluation.

## Step 8 — Build persistent object/spatial memory

Hydra/DAAAM remain optional backends behind stable interfaces.

## Step 9 — Train body-schema predictors

Use simulation to learn reachability, balance risk, and physical feasibility.

## Step 10 — Add Vesta-style reasoning

Begin with event-driven long-horizon planning and memory.

## Step 11 — Hybridize global routing with GR00T local acting

Avoid a hard control handoff at the manipulation boundary.

## Step 12 — Start problem-solving curriculum

Introduce missing tools, failures, blocked actions, and novel configurations.

This order produces useful embodied capability much earlier than the previous roadmap.

---

# 21. Evaluation framework

## 21.1 SONIC / physical execution

```text
fall rate
motion tracking quality
contact stability
whole-body coordination
command latency
recovery from perturbation
carrying stability
```

## 21.2 GR00T actor

```text
closed-loop task success
local recovery success
grip/regrip success
whole-body approach efficiency
unseen object generalization
unseen layout generalization
perturbation robustness
actor uncertainty calibration
```

## 21.3 Vesta/reasoning

```text
long-horizon task completion
progress-estimation accuracy
memory retrieval correctness
unnecessary replans
recovery strategy success
unsupported-claim rate
human-intervention frequency
```

## 21.4 Body schema

```text
reachability calibration
grasp-success prediction
balance-risk AUC/calibration
physical-feasibility prediction
OOD confidence
```

## 21.5 World model

```text
entity persistence
duplicate/merge errors
spatial relation accuracy
change-detection latency
last-seen correctness
physical episode retrieval quality
```

## 21.6 Whole robot

```text
task success
task duration
falls
collisions
near misses
energy
human interventions
number of failed physical attempts
recovery rate
mean time between autonomy failures
```

## 21.7 Intelligence-specific benchmark categories

Create benchmark families for:

```text
A. clean familiar tasks
B. unseen object instances
C. unseen layouts
D. missing tool
E. substituted tool
F. blocked plan
G. reachability change
H. target movement
I. failed grasp/contact
J. long-horizon memory
K. cross-room task
L. strategy transfer
```

A system is not considered more intelligent merely because a selected demonstration looks smoother.

---

# 22. Performance-rate targets

Approximate architectural targets:

```text
Motor/actuator loop:                  hardware dependent, ~500–1000 Hz possible
SONIC whole-body controller:           50 Hz current official workflow
State propagation:                     200–500 Hz
Local safety filter:                   100–200 Hz
Local geometry:                        10–30 Hz
LiDAR/visual odometry:                 sensor rate, often 10–30 Hz
Fast object perception:                10–30 Hz
Semantic tracking:                     5–20 Hz
GR00T VLA inference:                   lower-rate chunk prediction; benchmark hardware
Hydra/graph update:                    asynchronous
Open-vocabulary grounding:             asynchronous/keyframe
Vesta deliberation:                    event-driven
Global graph optimization:             background
Training/data upload:                  off control path
```

The exact values must be profiled on the deployment computer.

The invariant is:

> **slow cognition never blocks fast control or safety.**

---

# 23. Simulation and curriculum strategy

## 23.1 Level 1 — body intelligence

Train/evaluate:

```text
reach at different heights
change stance for reach
step closer when needed
switch hands
carry while walking
crouch/reach
move around obstacles
maintain balance during contact
```

## 23.2 Level 2 — object interactions

Randomize:

```text
size
shape
mass
friction
pose
support surface
```

Tasks:

```text
push
pull
grasp
lift
place
contain
open/close simple articulation
```

## 23.3 Level 3 — tool use

Provide:

```text
stick
brush
spatula
tray
dustpan
box
cardboard
```

Vary which familiar tool is present.

## 23.4 Level 4 — blocked plans

Examples:

```text
cabinet closed
chair blocks cabinet
target behind obstacle
hand already occupied
preferred approach blocked
```

## 23.5 Level 5 — failure recovery

Inject:

```text
object slip
grasp miss
target movement
tool removal
drawer resistance
camera occlusion
unexpected contact
```

Reward recovery and goal completion.

---

# 24. What to keep from the old roadmap

Definitely preserve:

- multi-timescale architecture;
- explicit local geometry;
- map/odom separation;
- persistent metric mapping;
- temporal object memory;
- scene-graph interfaces;
- uncertainty tracking;
- semantic embeddings;
- three-lane perception;
- 2D→3D grounding;
- object pose/part estimation;
- localization evaluation;
- global navigation fallback;
- dynamic-agent handling;
- exploration framework;
- independent safety;
- ROS 2 process isolation;
- backend-independent APIs;
- lifecycle/bringup discipline;
- data logging and replay;
- failure mining;
- regression evaluation.

These are not obsolete because the actor is learned. They make learned embodied intelligence usable on a real robot.

---

# 25. What to remove or demote

Remove as primary control philosophy:

- independent local navigation and manipulation authorities;
- long rigid VLM-generated action scripts;
- behavior tree as the general manipulation brain;
- MoveIt as the default executor for every interaction;
- fixed primitives as the only allowed action vocabulary;
- pure semantic JSON bottleneck between vision and actor;
- a VLA that is only an optional late-stage add-on.

Demote to support/fallback roles:

- DWA/MPPI local velocity control near manipulation;
- IK trajectory generation;
- pre-scripted pick/place sequences;
- hand-built recovery trees;
- explicit affordance labels without learned outcome prediction.

---

# 26. What not to do

Do not:

- make Vesta the balance controller;
- make GR00T the emergency-stop system;
- make SONIC responsible for long-term memory;
- hard-code Vesta's implementation into world-model APIs;
- delete the custom locomotion baseline before SONIC is benchmarked;
- split arms and legs into independent primary learned policies without a strong measured reason;
- convert every visual observation to text before GR00T sees it;
- let VLM confidence reduce geometric safety margins;
- assume a scene graph contains all physically relevant knowledge;
- force every task into `navigate → reach → grasp → lift → place`;
- train only on successful clean demonstrations;
- reward only trajectory imitation when goal completion is what matters;
- add semantics, Vesta, predictive models, SLAM, tactile sensing, and new control simultaneously;
- change the SONIC latent action definition without checkpoint-aware versioning;
- assume the same SONIC token has identical meaning across different SONIC checkpoints;
- trust a more complex architecture without held-out regression evidence.

---

# 27. Final recommended stack

```text
Operating system:
    Ubuntu 24.04 where compatible with chosen NVIDIA/ROS dependencies

Middleware:
    ROS 2 Jazzy or the ROS 2 version required by the validated deployment stack

Embodied reasoning / System 2:
    Vesta-style multimodal embodied planner
    ReasoningProvider abstraction
    progress monitor
    memory retrieval
    event-driven deliberation

Embodied actor / System 1:
    Isaac GR00T N1.7
    UNITREE_G1_SONIC embodiment
    direct visual observation
    task/goal conditioning
    later world/body/predictive context

Embodiment / System 0:
    GEAR-SONIC universal-token whole-body controller
    official G1 model/checkpoint path initially
    current custom locomotion policy retained as benchmark/fallback

State estimation:
    replaceable LiDAR/IMU/vision estimator
    contact and kinematic fusion
    map/odom separation

Local geometry:
    nvblox adapter or equivalent
    explicit occupancy/ESDF fallback

Persistent spatial memory:
    pose graph/submaps
    Hydra adapter where useful

Semantic/temporal memory:
    tracked object entity store
    DAAAM or equivalent adapter
    temporal event store
    semantic/physical embedding index

Body intelligence support:
    learned body schema
    reachability
    interaction success
    balance/feasibility predictors

Predictive physical model:
    simple outcome predictors first
    latent transition model later

Navigation:
    global topology + A*/Nav2/Smac route planning
    RouteHint to actor
    deterministic fallback local planner
    GR00T+SONIC primary local embodied movement near tasks

Manipulation support:
    object pose/parts
    task-space targets
    IK/collision/reachability oracles
    optional deterministic primitives
    actor remains primary local authority

Safety:
    independent command validator
    predictive safety filter
    geometric collision monitor
    hardware emergency mechanisms

Data engine:
    synchronized recording
    GR00T/SONIC/Vesta traces
    outcome prediction vs reality
    failure triage
    hard-example mining
    replay
    model/dataset registry
    regression suite
```

---

# 28. Architectural equation

The revised project can be summarized as:

```text
Robust embodied intelligence =

    learned physical actor
  + learned whole-body embodiment
  + persistent reasoning and memory
  + accurate geometry
  + learned body schema
  + physical prediction
  + explicit spatial memory
  + independent safety
  + continual data improvement
```

Or more simply:

```text
          MIND
 Vesta-like + GR00T-like
          │
          ▼
          BODY
        SONIC
          │
          ▼
          G1

surrounded by reliable perception,
geometry, memory, safety, and data systems.
```

The long-term research direction is **not** to keep adding more isolated modules. It is to gradually improve and partially unify the embodied mind while keeping the fast body-control and safety timescales reliable.

The robot should become better by learning:

- what the world contains;
- what objects can do;
- what this body can do;
- what usually happens after physical actions;
- which strategies worked previously;
- when its current strategy is failing;
- when it should continue acting versus stop and reason.

That creates a path toward a robot that is not simply human-like, but **competent in a way that is natural for its own body and mind**.

---

# 29. Primary references

## NVIDIA GEAR-SONIC / GR00T Whole-Body Control

- https://github.com/NVlabs/GR00T-WholeBodyControl
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/index.rst
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_workflow.md
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_inference.md
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/training.md

## NVIDIA Isaac GR00T N1.7

- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/GR00TWholeBodyControl/README.md
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/embodiment_tags.py

## NVIDIA Vesta

- https://research.nvidia.com/labs/gear/vesta/
- https://arxiv.org/abs/2606.20905

