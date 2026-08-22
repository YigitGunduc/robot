Yes. I would rewrite the roadmap around one principle:

> **The architecture stays ambitious; the implementation stays small.**

Your existing roadmap already has the right long-term ideas—multi-timescale control, GR00T + SONIC, explicit geometry, persistent memory, independent safety, global routing, Vesta-style reasoning, body schema, prediction, episodic learning. 

What changes is that we stop treating all of them as dependencies.

Below is the roadmap I would use.

---

# Simplified Modular G1 Embodied Intelligence Roadmap

## 0. Architectural rule: everything is replaceable

The codebase should never know that it is using:

* GR00T
* SONIC
* Vesta
* Hydra
* nvblox
* Nav2
* DAAAM
* your custom PPO controller

outside of their adapter.

The rest of the system only knows capabilities.

Conceptually:

```text
                         Mission
                            │
                            ▼
                     ReasoningProvider
                            │
                     PhysicalIntent
                            │
                            ▼
                     EmbodiedActor
                            │
                       ActorAction
                            │
                            ▼
                   LowLevelController
                            │
                            ▼
                           G1


        ┌──────────────── support plane ────────────────┐
        │                                              │
 StateProvider   GeometryProvider   WorldModel   RoutePlanner
        │                                              │
        └────────────── SafetySupervisor ──────────────┘

                         DataRecorder
```

Today:

```text
ReasoningProvider  = SimpleReasoner / NoOp
EmbodiedActor      = GR00T
LowLevelController = SONIC
StateProvider      = MuJoCoOracle
GeometryProvider   = SimpleOccupancy
WorldModel         = SimpleObjectStore
RoutePlanner       = A*
```

Later:

```text
ReasoningProvider  = Vesta
GeometryProvider   = nvblox
WorldModel         = Hydra
RoutePlanner       = Nav2 / Smac
SemanticProvider   = DAAAM / future model
```

Nothing above or below those components changes.

Your existing roadmap already contains this backend-independent idea, but it expands it into a very large ROS package tree too early. 

We keep the interfaces.

We **do not build all those packages yet**.

---

# 1. The most important architectural rule: one authority per level

Avoid components fighting over the robot.

The authority chain should always be:

```text
Reasoner
   │
   │ WHAT should happen
   ▼
Actor
   │
   │ HOW should I physically behave
   ▼
Controller
   │
   │ HOW do I realize this with the body
   ▼
Robot
```

And:

```text
Safety
  │
  └──────── may VETO / LIMIT anything
```

But safety doesn't decide the task.

The route planner doesn't control the legs directly.

The world model doesn't produce joint positions.

Vesta doesn't control balance.

GR00T doesn't control emergency stopping.

SONIC doesn't decide which cup you wanted.

This separation is one of the strongest parts of your original architecture. 

---

# 2. Do not use independent upper-body and lower-body brains

There are three broad architectures you could choose.

### A. One giant policy

```text
camera + language + state
          ↓
     giant network
          ↓
       joints
```

Very elegant eventually.

Bad starting point because training/debugging/safety are difficult.

### B. Independent locomotion + manipulation policies

```text
walking policy ─────┐
                    ├── robot
arm policy ─────────┘
```

Easy initially.

Bad long term because:

```text
step while reaching
crouch while grasping
walk while carrying
pelvis repositioning
balance during pulling
```

all cross that boundary.

### C. Hierarchical whole-body system

```text
Reasoning
   ↓
GR00T
   ↓
SONIC
   ↓
G1
```

This is what I recommend.

Classical navigation, IK and primitives still exist, but as **tools/fallbacks**, not competing brains.

That preserves the original roadmap's strongest architectural change: local navigation and manipulation should increasingly become one whole-body physical behavior. 

---

# 3. Simplify the codebase before adding intelligence

Instead of your current proposed ~40 subpackages, start with this:

```text
robot/
│
├── core/
│   ├── interfaces.py
│   ├── types.py
│   ├── registry.py
│   └── config.py
│
├── backends/
│   ├── controllers/
│   │   ├── custom_rl.py
│   │   └── sonic.py
│   │
│   ├── actors/
│   │   ├── simple_task_space.py
│   │   └── groot.py
│   │
│   ├── reasoning/
│   │   ├── noop.py
│   │   └── simple_vlm.py
│   │
│   ├── state/
│   │   └── mujoco_oracle.py
│   │
│   ├── geometry/
│   │   └── simple_geometry.py
│   │
│   ├── navigation/
│   │   └── astar.py
│   │
│   └── world_model/
│       └── simple_store.py
│
├── runtime/
│   ├── robot_runtime.py
│   ├── safety.py
│   └── progress.py
│
├── tasks/
│   ├── definitions.py
│   ├── pick_place.py
│   └── evaluators.py
│
├── data/
│   ├── recorder.py
│   ├── replay.py
│   └── dataset.py
│
├── eval/
│   ├── benchmark.py
│   └── metrics.py
│
├── sim/
│   └── mujoco.py
│
└── bringup/
    └── run_robot.py
```

Only split these into separate ROS packages when there is an actual runtime reason.

---

# 4. Core plug-and-play interfaces

These matter more than the directory structure.

## Controller

```python
class LowLevelController(Protocol):
    def reset(self, state): ...
    def step(self, command, state) -> JointCommand: ...
```

Implementations:

```text
CustomRLController
SonicController
future_controller
```

---

## Actor

```python
class EmbodiedActor(Protocol):
    def reset(self): ...
    def act(
        self,
        observation: ActorObservation,
        intent: PhysicalIntent,
    ) -> ActorAction:
        ...
```

Implementations:

```text
TaskSpaceActor
GrootActor
future_actor
```

---

## Reasoner

```python
class ReasoningProvider(Protocol):
    def deliberate(
        self,
        context: EmbodiedContext,
    ) -> PhysicalIntent:
        ...
```

Implementations:

```text
NoOpReasoner
SimpleVLMReasoner
VestaAdapter
future_reasoner
```

This is already the right idea in the current roadmap: Vesta should implement a generic reasoning interface rather than infecting the rest of the architecture. 

---

## World model

```python
class WorldModel(Protocol):
    def update(self, observations): ...

    def get_entity(self, entity_id): ...

    def query(self, query): ...

    def recent_events(self): ...
```

Today:

```text
DictionaryWorldModel
```

Future:

```text
HydraWorldModel
```

---

## Geometry

```python
class GeometryProvider(Protocol):
    def clearance(self, region): ...
    def collision_risk(self, action): ...
    def support_surfaces(self): ...
```

Today:

```text
SimpleDepthGeometry
```

Future:

```text
NvbloxGeometry
```

---

## Navigation

```python
class RoutePlanner(Protocol):
    def plan(
        self,
        current_pose,
        destination,
    ) -> RouteHint:
        ...
```

Today:

```text
AStarPlanner
```

Future:

```text
Nav2Planner
SmacPlanner
```

---

# 5. Backend selection happens through configuration

For example:

```yaml
controller:
  backend: sonic

actor:
  backend: groot

reasoning:
  backend: noop

state:
  backend: mujoco_oracle

geometry:
  backend: simple_depth

world_model:
  backend: simple

navigation:
  backend: astar
```

Later:

```yaml
reasoning:
  backend: vesta

geometry:
  backend: nvblox

world_model:
  backend: hydra

navigation:
  backend: nav2
```

No application code changes.

That is what I mean by genuinely plug-and-play.

---

# STAGE 0 — Clean up without rewriting

This should be the first implementation task.

## Goal

Turn your current working robot into the **legacy reference implementation**.

Do not redesign it yet.

Keep:

```text
height_conditioned_g1.py
train_g1.py
render_g1.py
g1_arm_manipulation.py
g1_arm_manipulation_config.py
manipulation_checkpoint.py
```

Your existing roadmap correctly treats the old controller as benchmark, fallback and source of evaluation infrastructure rather than disposable code. 

### Tasks

1. Move nothing unless necessary.
2. Create wrappers around the current code.
3. Create common observations/actions.
4. Add deterministic benchmark scenes.
5. Add machine-readable metrics.
6. Add checkpoint/version metadata.

For example:

```python
CustomRLController(
    checkpoint="walk_v12.pt"
)
```

wraps your existing system.

Do **not** rewrite the PPO environment.

---

# STAGE 1 — Establish the basic interfaces

Implement only:

```text
RobotState
PhysicalIntent
ActorObservation
ActorAction
RouteHint

LowLevelController
EmbodiedActor
ReasoningProvider
GeometryProvider
WorldModel
RoutePlanner
SafetySupervisor
```

Don't fill every field yet.

For example V1:

```python
@dataclass
class ActorObservation:
    rgb: Image
    robot_state: RobotState
```

Later it can become:

```python
@dataclass
class ActorObservation:
    rgb: Image
    robot_state: RobotState

    geometry: GeometrySummary | None = None
    entities: list[WorldEntity] | None = None
    recent_events: list[Event] | None = None
    body_features: BodyFeatures | None = None
```

This matters because your current roadmap's `ActorObservation` already anticipates these future inputs. 

The mistake would be making all of them required today.

---

# STAGE 2 — SONIC only

No GR00T.

No Vesta.

No Hydra.

No semantics.

Architecture:

```text
test command
    ↓
SONIC
    ↓
MuJoCo G1
```

### Tasks

1. Run official G1 SONIC.
2. Validate joint ordering.
3. Validate initialization.
4. Understand token/control representation.
5. Build `SonicController`.
6. Record actions and resulting motion.
7. Benchmark against your existing policy.

### Benchmark

```text
standing
walking
turning
crouching
transitioning posture
balance after small disturbance
```

### Gate

Don't continue until:

```text
no initialization explosions
no uncontrolled pose snapping
repeatable behavior
stable simulation
```

These are essentially the original Phase 1 acceptance criteria. 

---

# STAGE 3 — GR00T → SONIC

This is the **main MVP**.

Architecture:

```text
camera
robot state
task
   │
   ▼
GR00T
   │
   ▼
SONIC
   │
   ▼
G1
```

Nothing else.

## V1 observation

```text
RGB
proprioception
projected gravity
task instruction
```

## V1 output

Whatever the validated SONIC embodiment expects, hidden behind:

```python
ActorAction
```

Do not make the entire codebase assume:

```text
64-D forever
```

Your original roadmap correctly notes that the action representation must be versioned because SONIC checkpoints could change. 

---

# STAGE 3A — One task only

Start with:

```text
pick object
      ↓
place in container
```

Don't start with:

```text
navigate room
find cup
open drawer
grab cup
close drawer
walk somewhere
place cup
```

## Randomize

```text
object position
object orientation
container position
robot starting pose
```

No motion-capture dependency.

No dexterous hand curriculum yet.

Use simple hand/gripper behavior sufficient for the task.

---

# STAGE 3B — Training pipeline

The pipeline should become:

```text
demonstrations
      ↓
dataset
      ↓
train / fine-tune
      ↓
checkpoint
      ↓
benchmark
```

Every checkpoint gets:

```text
model version
dataset version
controller version
task version
git commit
metrics
```

So:

```text
GR00T checkpoint 17
```

can never silently be evaluated with incompatible:

```text
SONIC checkpoint 9
```

---

# STAGE 4 — Data and evaluation

This isn't glamorous, but it may be one of the highest-value parts of the system.

Your original roadmap correctly requires every failure to be replayable and classified. 

Record:

```text
camera
robot state
actor input
actor output
SONIC command
joint state
contacts
task progress
success
failure reason
safety events
```

Each run becomes:

```python
Episode
```

with:

```text
task_id
seed
environment_version
actor_version
controller_version
start_state
events
outcome
```

---

# STAGE 4A — Perturbation benchmarks

Don't just test clean success.

Add:

```text
object moved
object rotated
robot starts somewhere else
container moved
small obstacle added
grasp slips
target moves
lighting changes
```

The roadmap already contains this idea. 

This is how you discover whether you have:

```text
imitation
```

or:

```text
physical intelligence
```

---

# STAGE 5 — Simple manipulation tools

Even with GR00T as the main actor, retain deterministic tools.

Expose:

```python
reach(target)
grasp(target)
release()
push(target, direction)
pull(target, direction)
lift(target)
place(target)
```

But they are optional tools.

Not mandatory decomposition.

Architecture:

```text
                EmbodiedActor
                     │
          ┌──────────┴─────────┐
          │                    │
     GR00T behavior       optional tools
                              │
                         IK / collision
```

For the tools:

```text
task-space target
      ↓
IK
      ↓
joint-limit check
      ↓
collision check
      ↓
execution
      ↓
visual/contact feedback
```

Never declare:

```text
grasp successful
```

just because the arm trajectory finished.

Verify:

```text
object moved with hand
contact plausible
object remains held
visual tracker agrees
```

---

# STAGE 6 — Basic safety + local geometry

Now add the minimum required for reliable physical operation.

Not Hydra.

Not full semantic maps.

Just:

```text
nearby obstacle geometry
robot self-collision
joint limits
velocity limits
stale command detection
drop-off detection
emergency stop
```

Architecture:

```text
ActorAction
     │
     ▼
SafetySupervisor
     │
 ┌───┴────┐
 │        │
allow    reject/limit
 │
 ▼
SONIC
```

Safety can:

```text
allow
limit
reject
stop
```

It does not rewrite the task.

---

# STAGE 7 — Mobile manipulation

Only after local manipulation works.

Add simple global navigation.

Architecture:

```text
Task:
"bring object from room B"

       ↓

RoutePlanner
       │
    RouteHint
       ↓
GR00T
       ↓
SONIC
```

Global planner handles:

```text
room A → corridor → room B
```

GR00T handles:

```text
approach table
position feet
reach
step while reaching
grasp
carry locally
```

That preserves the original roadmap's good hybrid idea: classical routing where graph search is useful, learned embodied behavior where body interaction matters. 

---

# STAGE 8 — Very simple persistent memory

Don't build Hydra yet.

Start with:

```python
WorldEntity(
    id,
    label,
    pose,
    confidence,
    last_seen,
)
```

And a simple database.

Enough to answer:

```text
Where did I last see the cup?

What objects are in the kitchen?

When did I last observe the drawer?
```

Then add:

```text
pose history
appearance embedding
state
room
observations
```

only when useful.

---

# STAGE 9 — Simple reasoner

Before Vesta, create the **architectural slot**.

Initially:

```text
ReasoningProvider
        │
        ▼
simple LLM/VLM
```

Input:

```text
goal
important observations
relevant objects
task progress
recent failures
```

Output should be structured:

```python
PhysicalIntent(
    objective=...,
    attention=...,
    constraints=...,
    route_hint=...,
    strategy_hint=...,
)
```

Not:

```text
walk 0.34m
rotate left 17°
move hand to x=0.32
```

The reasoner decides strategy.

GR00T decides physical execution.

---

# STAGE 9A — Event-driven reasoning

Don't ask the big model on every frame.

Invoke it when:

```text
task begins
subgoal succeeds
no progress
repeated failure
object missing
world changed
confidence low
safety intervened
goal ambiguous
```

That preserves another strong part of your existing design. 

During normal operation:

```text
GR00T → SONIC
GR00T → SONIC
GR00T → SONIC
GR00T → SONIC
```

not:

```text
VLM → GR00T
VLM → GR00T
VLM → GR00T
50 times per second
```

---

# STAGE 10 — Vesta becomes a plug-in

Now Vesta is simply:

```python
class VestaReasoner(ReasoningProvider):
    ...
```

You replace:

```yaml
reasoning:
  backend: simple_vlm
```

with:

```yaml
reasoning:
  backend: vesta
```

Nothing else changes.

Initial benchmark:

```text
find object across multiple rooms
remember searched rooms
multi-stage task
missing-object recovery
long-horizon progress tracking
```

And only keep Vesta if:

```text
Vesta task success
>
simple reasoner task success
```

on held-out tasks.

That's essentially the proper acceptance criterion already in your roadmap. 

---

# FUTURE OPTION A — nvblox

Trigger:

> Simple local geometry is becoming a limitation.

Swap:

```text
SimpleGeometry
       ↓
NvbloxGeometry
```

Gain:

```text
TSDF
ESDF
clearance
support surfaces
persistent/local dense geometry
```

No actor rewrite.

---

# FUTURE OPTION B — Hydra

Hydra should enter when your simple world store becomes inadequate.

Trigger:

> The robot needs persistent structured spatial reasoning.

Examples:

```text
rooms
places
objects
surfaces
object relationships
persistent map topology
multi-session memory
```

Swap:

```text
SimpleWorldModel
       ↓
HydraWorldModel
```

Your original roadmap already treats Hydra as an adapter rather than the system's identity. 

Correct approach.

---

# FUTURE OPTION C — Better semantic perception / DAAAM

Trigger:

> GR00T vision isn't enough for persistent/open-vocabulary world understanding.

Add:

```text
SemanticPerceptionProvider
```

Implementations:

```text
FastDetector
DAAAM
future VLM
```

Use it for:

```text
object identity
parts
attributes
open-vocabulary queries
3D association
semantic memory
```

Do **not** convert every camera frame into JSON before GR00T sees it.

Direct vision still goes to the actor.

---

# FUTURE OPTION D — Learned body schema

This is a very good future research idea.

But only add it if the robot exhibits failures such as:

```text
keeps reaching for impossible target
chooses bad hand
doesn't know stepping is necessary
attempts unstable push
poorly predicts carrying feasibility
```

Start with simple models:

```python
ReachabilityPredictor
InteractionSuccessPredictor
BalanceRiskPredictor
```

Possible inputs:

```text
body pose
target pose
local geometry
object geometry
hand
```

Output:

```text
P(success)
```

Your roadmap currently puts this in its own phase and proposes exactly these kinds of predictors. 

Keep it.

Just don't require it early.

---

# FUTURE OPTION E — Physical outcome prediction

Trigger:

> The actor performs valid actions but repeatedly chooses poor strategies.

Start extremely small.

Predict:

```text
grasp_success
collision_risk
push_displacement
balance_risk
drawer_progress
```

Not:

```text
giant universal physics world model
```

Then:

```text
candidate action A ─► 0.78 success
candidate action B ─► 0.26 success
candidate action C ─► 0.61 success
```

Use it to **rank** behavior.

Don't let the predictor directly control motors.

The current roadmap's gradual `simple predictor → latent transition model` progression is good. 

---

# FUTURE OPTION F — Episodic physical memory

Eventually record not only:

```text
what happened
```

but:

```text
what problem existed
what strategy was tried
why it failed
what succeeded
```

Example:

```text
Problem:
drawer stuck

Attempt:
one-hand pull

Result:
failure

Successful recovery:
left hand braces cabinet
right hand pulls
body steps backward
```

Then future tasks retrieve similar episodes.

Your existing roadmap already proposes retrieval based on geometry, interaction type, body configuration and failure signature. 

Keep this as a high-value later research direction.

---

# FUTURE OPTION G — Tool-use intelligence

Once basic manipulation generalizes, build curriculum levels.

### Level 1

```text
reach
step/reach
crouch/reach
carry
switch hand
```

### Level 2

```text
push
pull
grasp
lift
place
open
close
```

Randomize:

```text
shape
size
mass
friction
pose
```

### Level 3

Give:

```text
stick
brush
tray
spatula
box
dustpan
```

Then remove the obvious tool.

Ask the robot to still accomplish the world-state goal.

### Level 4

Blocked plans:

```text
chair in way
target unreachable
hand occupied
drawer closed
preferred path blocked
```

### Level 5

Failures:

```text
slip
missed grasp
moving target
camera occlusion
tool removed
unexpected resistance
```

This preserves the excellent curriculum structure from the existing roadmap. 

---

# FUTURE OPTION H — Vesta + GR00T fusion

This is **far future research**, but I would absolutely keep it on the roadmap.

Current:

```text
Vesta
   │
PhysicalIntent
   │
   ▼
GR00T
```

Potential future:

```text
       Shared embodied representation
             │               │
        slow reasoning    fast acting
             │               │
           Vesta           GR00T
              \             /
               \           /
                  SONIC
```

Possible experiments:

```text
shared visual encoder
shared spatial tokens
planner latent → actor
shared episodic embeddings
slow/fast heads
recurrent world state
```

Do this only if measurements show that the clean interface itself is the bottleneck.

Your current roadmap makes exactly that caveat. 

---

# FUTURE OPTION I — Tactile sensing

Also optional.

Don't design the whole stack around tactile sensors you don't yet use.

Eventually expose:

```python
TactileProvider
```

Actor observation gains:

```text
finger contacts
pressure
slip estimate
force distribution
```

Good for:

```text
fragile grasp
slipping
tool contact
drawer forces
in-hand adjustment
```

---

# FUTURE OPTION J — Continual learning

The final system eventually becomes:

```text
robot operates
     ↓
episodes recorded
     ↓
failures mined
     ↓
hard cases selected
     ↓
human corrections
     ↓
training
     ↓
candidate model
     ↓
regression suite
     ↓
shadow evaluation
     ↓
deployment
```

But training never happens blindly on the production robot.

Every new model must beat the existing one.

---

# The roadmap therefore becomes only 10 real stages

```text
STAGE 0
Freeze existing robot
      │
      ▼
STAGE 1
Interfaces + adapters
      │
      ▼
STAGE 2
SONIC
      │
      ▼
STAGE 3
GR00T → SONIC
      │
      ▼
STAGE 4
Recording + evaluation
      │
      ▼
STAGE 5
Safety + simple geometry
      │
      ▼
STAGE 6
Mobile manipulation + routing
      │
      ▼
STAGE 7
Simple persistent memory
      │
      ▼
STAGE 8
Simple reasoner
      │
      ▼
STAGE 9
Long-horizon intelligent tasks
      │
      ▼
STAGE 10
Real G1 deployment
```

And then:

```text
OPTIONAL RESEARCH UPGRADES

├── Vesta
├── Hydra
├── nvblox
├── richer semantic perception
├── learned body schema
├── physical outcome model
├── episodic physical memory
├── tool-use curriculum
├── tactile sensing
├── continual adaptation
└── Vesta/GR00T representation fusion
```

---

# Most important: advanced features are triggered by failures

This is the rule I would actually put at the top of the repository.

```text
DO NOT add a subsystem because
"a general robot should have it."

Add a subsystem because a measured failure
shows that the current system needs it.
```

For example:

| Observed problem                                               | Add                      |
| -------------------------------------------------------------- | ------------------------ |
| Can't reliably execute whole-body motion                       | improve SONIC/controller |
| Doesn't understand immediate physical task                     | improve GR00T            |
| Chooses unreachable interactions                               | body schema              |
| Hits unseen nearby obstacles                                   | geometry                 |
| Can't get to another room                                      | route planner            |
| Forgets where objects are                                      | world memory             |
| Can't solve long multi-stage tasks                             | Vesta/reasoner           |
| Repeats previously solved mistakes                             | episodic memory          |
| Chooses bad physical strategies                                | outcome predictor        |
| Needs structured multi-room world relations                    | Hydra                    |
| Semantic grounding is weak                                     | DAAAM/semantic provider  |
| Structured planner→actor interface loses important information | Vesta/GR00T fusion       |

That turns your giant roadmap into an **evidence-driven roadmap**.

---

# What the code should eventually look like

A task runner shouldn't contain anything like:

```python
vesta.plan(...)
hydra.query(...)
groot.predict(...)
sonic.decode(...)
```

It should look like:

```python
context = runtime.observe()

intent = runtime.reasoner.deliberate(context)

route = runtime.router.plan_if_needed(
    context,
    intent,
)

observation = runtime.build_actor_observation(
    context,
    intent,
    route,
)

action = runtime.actor.act(
    observation,
    intent,
)

safe_action = runtime.safety.filter(
    action,
    context,
)

runtime.controller.execute(
    safe_action
)

runtime.recorder.record(...)
```

That is what gives you the ability to replace essentially **any piece of the robot without rewriting the robot**.

---

# Final long-term architecture

So we preserve essentially all of the ambition of the original roadmap:

```text
                    USER / MISSION
                          │
                          ▼
              ┌────────────────────┐
              │ EMBODIED REASONER  │
              │                    │
              │ Simple → Vesta     │
              └──────────┬─────────┘
                         │
                  PhysicalIntent
                         │
                         ▼
              ┌────────────────────┐
camera ──────►│   EMBODIED ACTOR   │
state ───────►│                    │
memory ──────►│       GR00T        │
world ───────►│                    │
              └──────────┬─────────┘
                         │
                   ActorAction
                         │
                         ▼
              ┌────────────────────┐
              │    EMBODIMENT      │
              │                    │
              │       SONIC        │
              └──────────┬─────────┘
                         │
                         ▼
                    UNITREE G1


       OPTIONAL / REPLACEABLE SUPPORT SYSTEMS

 State             Geometry             Semantics
   │                   │                    │
oracle → SLAM     simple → nvblox      detector → DAAAM
   │                   │                    │
   └───────────────────┼────────────────────┘
                       ▼
                  WorldModel
                       │
                simple → Hydra
                       │
                       ▼
            persistent / episodic memory


 Route planning         Body schema        Prediction
 A* → Nav2              optional            optional

                       │
                       ▼

              INDEPENDENT SAFETY
                       │
                       ▼
                 DATA + EVALUATION
                       │
                       ▼
                CONTINUAL LEARNING
```

That keeps the central idea of your original roadmap—**a learned mind/actor over a fast learned body, surrounded by reliable robotics systems**—without forcing you to implement the entire future architecture before you even have the first intelligent manipulation task working. 

The next thing I would do from here is convert **Stages 0–4 into concrete implementation tickets**, probably ~25–35 tasks total, each small enough that you can give one to a coding agent and reasonably expect it to finish without needing to understand the entire robotics project.

