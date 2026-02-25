# Drone Swarm AI — Intelligent 3D Formation System

> **Transform any photograph into a fully choreographed 3D drone formation using Computer Vision, Depth AI, Physics Simulation, and Reinforcement Learning.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What We Built — Feature Inventory](#2-what-we-built)
3. [Algorithms Used](#3-algorithms-used)
4. [Core Concepts](#4-core-concepts)
5. [System Architecture](#5-system-architecture)
6. [Project Structure](#6-project-structure)
7. [How to Run](#7-how-to-run)
8. [Role of Reinforcement Learning](#8-role-of-reinforcement-learning)
9. [Metrics & What They Mean](#9-metrics--what-they-mean)
10. [How to Present This Project](#10-how-to-present-this-project)
11. [Technology Stack](#11-technology-stack)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

**Drone Swarm AI** is an end-to-end AI pipeline that takes a single photograph (e.g., a person standing), automatically extracts the subject's silhouette, distributes hundreds of drones across the shape in 3D space, and simulates their convergence to the formation using either classical physics or a trained Reinforcement Learning policy.

### What problem does it solve?

Real-world drone light shows (think Super Bowl halftime, concerts) require expert choreographers to manually place each drone waypoint. This project **automates that entire process** — feed it any image, and it produces production-ready 3D drone coordinates automatically.

### What makes it advanced?

| Capability | Approach |
|---|---|
| Object detection | DeepLabV3 semantic segmentation (COCO-trained) |
| Depth estimation | MiDaS DPT-Large (monocular 2D to 3D) |
| Optimal assignment | Hungarian Algorithm (global minimum cost) |
| Formation motion | Physics (attraction + repulsion) OR PPO Reinforcement Learning |
| RL training | Stable Baselines 3, curriculum learning, parameter sharing |
| Pattern learning | Variational Autoencoder (VAE) with collision+connectivity loss |
| Quality analysis | Connectivity graph, spacing metrics, convergence tracking |

---

## 2. What We Built

### Phase 1 — Core Pipeline
- `pipeline.py` — single entry point that orchestrates all stages end-to-end
- `sim/swarm_sim.py` — 2D simulation with attraction forces and Hungarian assignment
- `sim/swarm_sim_3d.py` — 3D volumetric simulation with MiDaS depth + layered filling

### Phase 2 — Formation Precision Upgrade
- Rewrote `utils/semantic_image_to_formation.py` with **Poisson-disk sampling**
  - 75% interior fill (even spacing, no clustering)
  - 25% arc-length interpolated contour edge (exact boundary tracing)
  - 1024 px high-res mask with morphological cleanup
  - Result: drones precisely match the human silhouette boundary

### Phase 3 — 3D Depth Fix
- Fixed `utils/formation_3d.py` — removed 4x layer stacking that was creating rectangular blobs
- Now 1:1 mapping: each 2D point maps to one 3D point, Z = MiDaS depth x height_scale
- Correct Y-flip: image pixel space (Y-down) to world space (Y-up)

### Phase 4 — Visualization Overhaul
- 6-panel precision report:
  1. Original input image
  2. Drone overlay on image (target positions shown directly on photo)
  3. 2D front view (XY plane)
  4. 3D front-on view (azimuth -88 degrees)
  5. 3D angled view (azimuth -50 degrees)
  6. **Convergence curve** (mean error per frame — RL vs Physics comparison)

### Phase 5 — Reinforcement Learning Integration
- `RL/swarm_env_sb3.py` — Gymnasium environment, per-drone (21,) obs space
- `RL/train_sb3.py` — PPO training with 8-stage curriculum
- `RL/rl_controller.py` — inference engine, N-agnostic (works for 20 or 1000 drones)
- `RL/__init__.py` — package alias for clean imports
- Pipeline extended: RL checkpoint replaces physics loop with PPO policy

### Phase 6 — ML Extensions
- `models/vae.py` — Variational Autoencoder for learning formation distributions
- `train/train_vae.py` — constrained training (collision avoidance loss + connectivity loss)
- `analysis/` — dataset generation, network topology analysis, scalability benchmarks
- `analysis/train_connectivity_model.py` — ML model for predicting connectivity

---

## 3. Algorithms Used

### 3.1 Hungarian Algorithm (Optimal Assignment)
- **What**: Solves the assignment problem — match N drones to N targets with minimum total cost
- **Why**: Greedy assignment (nearest drone to nearest target) wastes energy. Hungarian gives the globally optimal solution.
- **Complexity**: O(n^3) — fast enough for 1000 drones
- **Where**: `scipy.optimize.linear_sum_assignment` in `swarm_env_sb3.py`, `rl_controller.py`

### 3.2 Poisson-Disk Sampling
- **What**: Places points such that no two points are closer than a minimum distance r
- **Why**: Uniform, natural distribution inside shapes — looks like a real lit-up silhouette, not random clusters
- **Where**: `utils/semantic_image_to_formation.py`

### 3.3 Inverse-Square Repulsion Physics
- **Formula**: F_repulsion = sum(diff / distance^3) x strength for all neighbor pairs within radius
- **Why**: Mimics magnetic or electrostatic repulsion — keeps drones from colliding while allowing close formation
- **Where**: `sim/swarm_sim_3d.py`, `RL/swarm_env_sb3.py`

### 3.4 Proximal Policy Optimization (PPO)
- **What**: Policy gradient RL algorithm from OpenAI — on-policy, clip-ratio prevents destructive updates
- **Why**: Stable training for continuous action spaces; works well for multi-agent parameter sharing
- **Where**: `RL/train_sb3.py` via Stable Baselines 3

### 3.5 Curriculum Learning
- **What**: Training progression — start easy (grid 1.8m spacing), gradually harder (v-shape 1.0m tight)
- **Why**: Direct training on hard shapes fails; curriculum lets the policy bootstrap from simple patterns
- **8 stages**: grid(1.8) -> grid(1.5) -> circle -> line -> v-shape -> grid(1.2) -> circle(1.2) -> v(1.0)
- **Where**: `RL/train_sb3.py`

### 3.6 Parameter Sharing (Decentralized Execution)
- **What**: One shared policy network is trained on a single drone's local observation; applied independently to every drone during execution
- **Why**: N-agnostic — the same model trained with 20 drones works for 1000 at zero extra cost
- **Where**: `RL/rl_controller.py` (per-drone inference loop)

### 3.7 k-d Tree Spatial Indexing
- **What**: Binary space-partitioning data structure for O(log N) nearest-neighbor queries
- **Why**: Finding each drone's neighbors naively is O(N^2). k-d tree cuts this to O(N log N).
- **Where**: `scipy.spatial.cKDTree` used in env, controller, physics loop

### 3.8 Variational Autoencoder (VAE)
- **What**: Encodes formation patterns to a compressed latent space, decodes back to drone positions
- **Why**: Learns the statistical distribution of valid formations — can generate new formation variants
- **Loss**: Reconstruction (MSE) + KL-divergence + collision penalty + connectivity penalty
- **Where**: `models/vae.py`, `train/train_vae.py`

### 3.9 Monocular Depth Estimation (MiDaS)
- **What**: DPT-Large transformer that predicts per-pixel relative depth from a single RGB image
- **Why**: Enables genuine 3D formation from a simple photo without stereo cameras or LiDAR
- **Where**: `utils/depth_to_3d.py`

### 3.10 DeepLabV3 Semantic Segmentation
- **What**: FCN with atrous convolutions and ASPP, ResNet-101 backbone, COCO-trained
- **Why**: Automatically identifies the "person" class (class ID 15) in any photo — no manual masking
- **Where**: `utils/semantic_image_to_formation.py`

---

## 4. Core Concepts

### Swarm Intelligence
A large group of simple agents (drones) following local rules that produce globally intelligent behavior. No single central controller — each drone reacts to its neighbors and its target. Inspired by birds flocking (Reynolds Boids), ant colonies, and fish schooling.

### Formation Control
The specific swarm problem of driving agents from arbitrary starting positions to a defined geometric pattern. Two sub-problems:
- **Assignment**: which agent goes to which target position
- **Navigation**: how each agent moves there while avoiding collisions

### Decentralized Multi-Agent RL (MARL)
Each drone is an independent RL agent observing only local information (relative target position + 6 nearest neighbors). Agents are trained with a shared policy (parameter sharing) — identical neural network weights for every drone. At execution: each drone runs its own inference independently.

### Centralized Training, Decentralized Execution (CTDE)
Training is done in a simulated environment where all drone states are accessible (centralized). But the trained policy only uses local observations (decentralized) — making it deployable on real drones with onboard sensors only.

### Observation Space Design
Each drone's observation is a 21-dimensional vector:

```
obs = [
  rel_target_x,  rel_target_y,  dist_to_target,    # Where am I going?
  nb1_dx, nb1_dy, nb1_dist,                          # Neighbor 1
  nb2_dx, nb2_dy, nb2_dist,                          # Neighbor 2
  ...                                                # (6 nearest neighbors)
]
```

All values are **relative** to the focal drone — so the policy is translation-invariant.

### Reward Shaping
The RL reward balances three competing objectives:

```
reward = -0.1 x mean_error          # formation accuracy
        + 0.5 x connectivity_ratio  # network connectivity
        - 0.2 x unsafe_ratio        # collision avoidance
        + 10.0 terminal bonus if (mean_error < 0.5 AND connectivity > 0.8)
```

### Depth-Lifting (2D to 3D)
MiDaS produces a relative depth map D(x,y) in [0,1]. Each 2D drone position p = (px, py) is lifted to 3D as:

```
P_3D = (px,  py,  D(px, py) x height_scale)
```

No stereo camera, no LiDAR — a single photo is enough.

### Connectivity (Network Graph)
Drones within `comm_range` distance are considered connected (could communicate or maintain swarm cohesion). Total possible edges = N(N-1)/2. Connectivity ratio = actual connected pairs / max possible pairs. Target: greater than 0.8 (80% connected).

---

## 5. System Architecture

```
INPUT IMAGE
     |
     v
+---------------------------------+
|  DeepLabV3 Semantic Segmentation|  <- Extracts person/object mask
|  (torchvision pretrained)       |
+------------+--------------------+
             |  binary mask
      +------+--------+
      |               |
      v               v
Poisson-Disk       Arc-Length
Interior Fill      Contour Sample   <- Combined: N target 2D positions
(75% of drones)    (25% of drones)
      |               |
      +------+--------+
             |  2D targets (N x 2)
             v
+---------------------------------+
|  MiDaS DPT-Large Depth Est.    |  <- Estimates depth per pixel
+------------+--------------------+
             |  depth map
             v
+---------------------------------+
|  3D Lifting: Z = depth x scale |  <- Produces 3D targets (N x 3)
+------------+--------------------+
             |
             v
+---------------------------------+
|  Hungarian Algorithm           |  <- Optimally assigns drones to targets
|  (scipy.optimize.lsa)          |
+------------+--------------------+
             |
             v
   +---------+-----------+
   |                     |
   v                     v
RL MODE               PHYSICS MODE
PPO Policy            Attraction +
(per-drone            Repulsion Forces
inference loop)       (O(N log N) k-d tree)
   |                     |
   +---------+-----------+
             |  positions (150 frames)
             v
+---------------------------------+
|  6-Panel Visualization         |  <- Saves formation_precision_report.png
|  + Convergence Curve           |
+------------+--------------------+
             |
             v
      CSV Export + Metrics JSON
      (x, y, z per drone)
```

---

## 6. Project Structure

```
drone_swarm/
|
+-- pipeline.py                     <- MAIN ENTRY POINT for images
+-- pipeline_video.py               <- VIDEO ENTRY POINT (NEW) — swarm tracks video
|
+-- RL/                             <- Reinforcement Learning module
|   +-- swarm_env_sb3.py           <- Gymnasium env (21-dim obs, per-drone)
|   +-- train_sb3.py               <- PPO training script (curriculum)
|   +-- rl_controller.py           <- Inference engine (N-agnostic)
|   +-- __init__.py                <- Package + alias registration
|   +-- env_2d.py / env_3d.py     <- Legacy RLlib environments
|   +-- train.py                   <- Legacy RLlib training script
|   +-- curriculum.py              <- Curriculum schedules
|   +-- reward.py                  <- Reward function definitions
|   +-- evaluate.py                <- Policy evaluation tools
|   +-- run_with_policy.py         <- Run simulation using saved policy
|   +-- sim_core.py                <- Simulation core for RL env
|   +-- checkpoints/sb3/
|       +-- best_model.zip         <- Best trained PPO model (USE THIS)
|       +-- final_model.zip        <- End-of-training model
|
+-- utils/                          <- Core AI & physics modules
|   +-- semantic_image_to_formation.py  <- DeepLabV3 + Poisson-disk
|   +-- depth_to_3d.py                  <- MiDaS depth estimation
|   +-- formation_3d.py                 <- 2D to 3D lifting (1:1 mapping)
|   +-- shape_generator.py              <- Geometric shapes (grid/circle/v)
|   +-- evaluate_swarm.py               <- Metrics computation
|   +-- metrics.py                      <- ML loss functions
|   +-- image_to_formation.py           <- Image processing helpers
|   +-- network_metrics.py              <- Connectivity graph analysis
|
+-- models/                         <- ML model definitions
|   +-- vae.py                     <- Variational Autoencoder
|
+-- train/                          <- Training pipelines
|   +-- train_vae.py               <- VAE training
|
+-- sim/                            <- Standalone simulation scripts
|   +-- swarm_sim.py               <- 2D formation simulation
|   +-- swarm_sim_3d.py            <- 3D volumetric simulation
|   +-- generate_new_formations.py
|   +-- visualize_formations.py
|
+-- analysis/                       <- Research tools
|   +-- swarm_core.py
|   +-- network_analysis.py
|   +-- scalability_plot.py
|   +-- optimize_spacing.py
|   +-- train_connectivity_model.py
|   +-- generate_ml_dataset.py
|   +-- swarm_ml_dataset.csv
|
+-- data/                           <- Formation datasets
|   +-- formations.npy
|   +-- generate_formations.py
|
+-- input_images/                   <- Place your images here
|   +-- Cristiano_Ronaldo.webp
|   +-- 516AaQ6o17L.webp
|   +-- smile.webp
|
+-- output/                         <- All outputs land here
|   +-- formation_precision_report.png   <- 6-panel visual
|   +-- formation_3d_<timestamp>.csv     <- Drone positions
|
+-- vae_model.pth                   <- Pre-trained VAE weights
```

---

## 7. How to Run

### Environment Setup

This project uses a `.venv` virtual environment. Use the full Python path to run commands:

```bash
# Windows — use full Python path from project root
D:\drone_swram\.venv\Scripts\python.exe  <script>

# OR activate and then use python normally
D:\drone_swram\.venv\Scripts\activate
python <script>
```

Install all dependencies (first time only):

```bash
D:\drone_swram\.venv\Scripts\pip.exe install torch torchvision timm
D:\drone_swram\.venv\Scripts\pip.exe install stable-baselines3[extra] gymnasium
D:\drone_swram\.venv\Scripts\pip.exe install scipy numpy opencv-python matplotlib networkx pandas pillow
```

---

### Option A — Full Pipeline (Recommended)

**Physics mode (no RL needed):**
```bash
python pipeline.py --mode 3d --image input_images/Cristiano_Ronaldo.webp --output output/ --base-drones 200
```

**RL mode (uses trained PPO policy):**
```bash
python pipeline.py --mode 3d --image input_images/Cristiano_Ronaldo.webp --output output/ --base-drones 200 --rl-checkpoint RL/checkpoints/sb3/best_model
```

What happens step by step:
1. Loads image, DeepLabV3 extracts silhouette mask
2. Poisson-disk samples N drone target positions in 2D
3. MiDaS estimates depth, lifts to 3D
4. Hungarian algorithm assigns drones to targets (globally optimal)
5. RL mode: PPO policy drives each drone independently for 150 frames
6. Physics mode: attraction + repulsion physics drives convergence
7. Saves `output/formation_precision_report.png` (6 panels)
8. Saves CSV + metrics JSON

---

### Option B — Train RL Model from Scratch

```bash
python RL/train_sb3.py \
  --total-timesteps 300000 \
  --curriculum \
  --drones 20 \
  --checkpoint-dir RL/checkpoints/sb3
```

What happens:
- 8 curriculum stages (grid 1.8m spacing to tighter v-shape 1.0m)
- Saves `best_model.zip` (highest eval reward) + periodic checkpoints every 30k steps
- Takes approximately 5 minutes on CPU

---

### Option B2 — Video Drone Swarm (NEW)

Process any video and generate an animated swarm that traces the person silhouette in every frame:

```bash
# Physics mode (fast, no RL)
python pipeline_video.py \
  --video input_images/4761762-uhd_2160_4096_25fps.mp4 \
  --output output/ \
  --drones 300 \
  --depth-every 0

# RL-PPO mode (best quality, uses trained policy per frame)
python pipeline_video.py \
  --video input_images/4761762-uhd_2160_4096_25fps.mp4 \
  --output output/ \
  --drones 300 \
  --depth-every 8 \
  --rl-checkpoint RL/checkpoints/sb3/best_model

# With 3D panel (slower — adds 3D matplotlib view)
python pipeline_video.py \
  --video input_images/4761762-uhd_2160_4096_25fps.mp4 \
  --output output/ \
  --drones 300 \
  --render-3d
```

**What the video pipeline produces:**
- `output/<video_name>_swarm.mp4` — side-by-side: original video | drone swarm overlay
- `output/<video_name>_positions.csv` — every drone position per frame (frame, drone_id, x, y, z)

**How temporal consistency works:**
- Frame 0: Hungarian assigns 300 drones to person silhouette targets
- Frame 1+: Hungarian reassigns drones to new targets (ensures minimal crossing/shuffling)
- EMA smoothing (alpha=0.65) on targets prevents jitter from frame-to-frame segmentation noise
- Physics/RL moves drones incrementally — they "chase" the moving person across scenes

**Key CLI flags for `pipeline_video.py`:**

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--drones` | 300 | Number of drones |
| `--depth-every` | 5 | Run MiDaS every N frames (0=disable, faster) |
| `--motion-steps` | 6 | Physics/RL iterations per video frame |
| `--step-size` | 0.15 | Attraction step size (higher=faster response) |
| `--rl-checkpoint` | None | Path to PPO model (no .zip) |
| `--render-3d` | False | Add 3D scatter panel to output |
| `--max-frames` | 0 | Process first N frames only (0=all) |
| `--skip-frames` | 0 | Skip every N frames for speed |
| `--dot-radius` | 5 | Drone dot size in pixels |

---

### Option C — Standalone 2D Simulation

```bash
python sim/swarm_sim.py
```

### Option D — Standalone 3D Simulation

```bash
python sim/swarm_sim_3d.py
```

### Option E — Train VAE Model

```bash
python train/train_vae.py
```

### Option F — Network Analysis

```bash
python analysis/network_analysis.py
```

---

### Key CLI Parameters for `pipeline.py`

| Flag | Default | Description |
|---|---|---|
| `--mode` | `3d` | `2d` or `3d` pipeline |
| `--image` | required | Path to input image |
| `--output` | `output/` | Output directory |
| `--base-drones` | `300` | Number of drones |
| `--rl-checkpoint` | None | Path to PPO `.zip` model (omit `.zip` extension) |
| `--auto-train-rl` | False | Train RL model automatically before running |

---

## 8. Role of Reinforcement Learning

### Why RL instead of pure physics?

Physics-based control (attraction + repulsion) is simple but limited:
- Every drone independently chases its target with no coordination
- Drones frequently block each other — no awareness of the swarm as a whole
- Repulsion is hard-coded, not adaptive to the situation
- Convergence can be slow and sometimes oscillates

RL **learns a navigation strategy** — not just a force equation:
- The policy learns when to approach and when to wait for clearance
- Repulsion behavior emerges naturally from the reward signal
- Collision avoidance becomes a trained skill, not a hard constraint
- Convergence is faster and smoother across diverse shapes

### What the RL policy does at runtime

For each simulation frame, for each drone:
1. Build a 21-dimensional local observation (relative target + 6 nearest neighbors)
2. Feed through the PPO neural network (2 hidden layers, 64 units each, tanh)
3. Output a 2D velocity delta [dvx, dvy] — clamped to [-0.2, +0.2]
4. Apply this delta on top of base physics attraction to move the drone
5. Z-axis (depth) is kept physics-based (MiDaS depth attraction), not RL-controlled

### Training architecture details

```
Environment: SwarmSB3Env (gymnasium.Env)
  obs_space:  Box(-inf, +inf, shape=(21,))   <- single drone local view
  act_space:  Box(-0.2, +0.2, shape=(2,))    <- velocity delta (XY)

Algorithm: PPO (Stable Baselines 3 v2.7.1)
  Policy: MlpPolicy (2 hidden layers x 64 units, tanh activations)
  n_steps: 4096 per policy update
  n_epochs: 10 per update
  clip_range: 0.2

Curriculum: 8 stages of increasing difficulty
  Stage 1: shape=grid,   dist=1.8m  (easiest, lots of space)
  Stage 2: shape=grid,   dist=1.5m
  Stage 3: shape=circle, dist=1.5m
  Stage 4: shape=line,   dist=1.5m
  Stage 5: shape=v,      dist=1.5m
  Stage 6: shape=grid,   dist=1.2m
  Stage 7: shape=circle, dist=1.2m
  Stage 8: shape=v,      dist=1.0m  (hardest, tight v-shape)

Reward per step:
  -0.1 x mean distance to targets       <- approach targets
  +0.5 x connectivity ratio             <- stay networked
  -0.2 x collision unsafe ratio         <- avoid collisions
  +10.0 terminal bonus (if good result) <- converge fully

Training results (300k steps, ~5 min CPU):
  Stage 1 (grid 1.8m):    reward ~ 60   <- easy, policy learns basics
  Stage 6 (grid 1.2m):    reward ~ 55   <- tighter, still strong
  Stage 8 (v 1.0m):       reward ~ 12   <- hardest shape, policy adapts
```

### N-agnostic inference — the key design decision

The policy is trained with N=20 drones but works for any number at deployment:

```python
# At inference time (rl_controller.py):
for i in range(N_drones):         # N can be 20, 200, or 1000
    obs_i = build_obs(i)          # always (21,) — same shape every time
    action, _ = model.predict(obs_i)   # same network, same weights
    apply_action(i, action)
```

This is **parameter sharing** — one neural network, applied independently N times. No retraining is needed for different swarm sizes.

### RL vs Physics comparison

The 6th panel of the output report shows a convergence curve. Expect:
- RL (cyan line): faster convergence, fewer oscillations, better final connectivity
- Physics (orange line): slower, may plateau at higher error
- Best of both: RL is tried first; falls back to physics if no checkpoint is provided

---

## 9. Metrics & What They Mean

| Metric | Good Value | Meaning |
|---|---|---|
| Convergence Error | < 1.0 | Mean distance (units) from each drone to its assigned target. Lower = better formation accuracy. |
| Connectivity Ratio | > 0.80 | Fraction of drone pairs within communication range. Above 80% means well-networked swarm. |
| Min Inter-Drone Distance | > 0.50 | Closest any two drones get. Below 0.5 means collision risk. |
| Avg Inter-Drone Distance | 1.0 to 3.0 | Typical spacing. Too high = spread out swarm. Too low = crowded. |
| RL Used | true/false | Whether PPO policy was active for this run. |

---

## 10. How to Present This Project

### One-line pitch

> "We built an AI system that takes any photo, extracts the person silhouette, and autonomously flies hundreds of drones into that exact 3D shape — using the same computer vision AI used in self-driving cars and the same learning algorithm used in robotics labs."

---

### 3-Minute Presentation Structure

**Slide 1 — The Problem (30 sec)**
- Drone light shows cost millions — each drone waypoint is placed manually by expert designers
- This does not scale and is not real-time adaptive
- Stated goal: fully automate this from a single photograph

**Slide 2 — Our Solution (30 sec)**
- Feed any photo → system produces flying coordinates automatically
- Demo: input image (Cristiano Ronaldo) → show output formation_precision_report.png
- Highlight: 200 drones, fully autonomous, no human waypoint design

**Slide 3 — How It Works — The Pipeline (60 sec)**
- Three AI systems chained together:
  - DeepLabV3: "What is the shape?" (semantic segmentation)
  - MiDaS: "How deep is each part?" (monocular depth estimation)
  - PPO RL policy: "How do drones navigate there?" (learned coordination)
- Show the architecture diagram from Section 5
- Emphasize: it is fully automatic — no manual intervention

**Slide 4 — The Smart Part: Reinforcement Learning (30 sec)**
- Drones are not just flying to GPS coordinates — they are learning to coordinate
- Key result: train on 20 drones, deploy on 1000 — same model, zero retraining
- Show the convergence curve (Panel 6): RL line converges faster than physics
- Explain the reward: formation accuracy + connectivity + collision avoidance

**Slide 5 — Results and Output (20 sec)**
- Show the 6-panel precision report image
- Point to: drone overlay matches Ronaldo's exact silhouette
- Metrics: connectivity ratio, convergence error
- Output CSV is directly importable to Unity, Blender, or real drone firmware

**Slide 6 — Impact and Future Work (10 sec)**
- Applicable to: drone light shows, search and rescue, AR overlays, swarm robotics
- Future: video input for animated sequences, multi-person silhouettes, real quadcopter SDK export

---

### Questions to Expect and Answers

**Q: How does each drone know which target is its assigned position?**
A: The Hungarian Algorithm solves this globally before simulation starts. It finds the minimum-cost one-to-one matching between all N drones and all N targets — not just greedy nearest-neighbor. This guarantees no drone crosses another unnecessarily.

**Q: What if a drone fails mid-flight?**
A: The RL policy is trained with random initialization and the observation only uses relative neighbor positions — if a drone disappears, the remaining drones naturally adapt. The policy never assumes a fixed number of neighbors (gaps are padded with zeros).

**Q: Why not just use GPS coordinates directly?**
A: The system outputs waypoints — GPS integration is the deploy layer, compatible with ROS, Crazyflies, and DJI SDK. The output CSV is industry-standard format. Our contribution is the intelligence that computes these waypoints from an arbitrary image.

**Q: Is this real-time?**
A: Neural network inference per drone is under 0.1ms on CPU. For 1000 drones: about 100ms per simulation frame — near real-time. On GPU this drops to under 10ms.

**Q: Why PPO specifically?**
A: PPO has a clip ratio that prevents the policy from making too-large updates — which is critical when training with curriculum (you do not want the policy to forget stage 1 when moving to stage 8). SAC or DDPG would also work but are more sensitive to hyperparameters.

**Q: What is the difference between RL mode and physics mode?**
A: Physics applies the same attraction and repulsion force every frame by formula. RL learned from 300,000 simulation steps what combination of movements leads to fast, safe, connected convergence — it adapts to the situation rather than applying a fixed equation.

---

## 11. Technology Stack

| Component | Library / Model | Version |
|---|---|---|
| Deep Learning framework | PyTorch | 2.x |
| Semantic Segmentation | DeepLabV3-ResNet101 | torchvision pretrained |
| Depth Estimation | MiDaS DPT-Large | timm / torch.hub |
| Reinforcement Learning | Stable Baselines 3 | 2.7.1 |
| RL Environment | Gymnasium | 1.2.3 |
| Spatial Indexing | SciPy cKDTree | 1.x |
| Optimal Assignment | SciPy linear_sum_assignment | 1.x |
| Computer Vision | OpenCV | 4.x |
| Numerical Computing | NumPy | 2.x |
| Visualization | Matplotlib | 3.x |
| Data Analysis | Pandas | 2.x |
| Network Analysis | NetworkX | 3.x |
| VAE / Custom Models | PyTorch (custom) | — |

---

## 12. Troubleshooting

### ModuleNotFoundError for torch / torchvision / timm
```bash
D:\drone_swram\.venv\Scripts\pip.exe install torch torchvision timm
```

### ModuleNotFoundError for stable-baselines3
```bash
D:\drone_swram\.venv\Scripts\pip.exe install stable-baselines3[extra] gymnasium
```

### ModuleNotFoundError for scipy / networkx
```bash
D:\drone_swram\.venv\Scripts\pip.exe install scipy networkx
```

### MiDaS model download on first run
- MiDaS downloads approximately 500 MB on first run — requires internet connection
- Cached at `~/.cache/torch/hub/` after first run, subsequent runs are instant
- Be patient on first execution — this is normal

### Slow convergence or simulation is too slow
- Reduce `--base-drones` (try 100 instead of 200)
- Reduce simulation frames: find `num_frames=150` in pipeline.py and change to 80

### RL model not improving during training
- Check `RL/checkpoints/sb3/evaluations.npz` — load with numpy to inspect mean rewards over time
- Try increasing `--total-timesteps` to 500000
- Try fewer curriculum stages — start with only grid shapes before adding circle and v

### Out of memory (OOM) during inference
- Add `map_location='cpu'` when loading model in `rl_controller.py`
- Reduce `--base-drones` to lower the number of concurrent inferences

### RL checkpoint path error
- Do NOT include the `.zip` extension in `--rl-checkpoint`
- Correct:   `--rl-checkpoint RL/checkpoints/sb3/best_model`
- Incorrect: `--rl-checkpoint RL/checkpoints/sb3/best_model.zip`

---

## Authors and Acknowledgments

**Hardhik Poosa** — B.Tech Computer Science and Engineering

Libraries and models used:
- DeepLabV3: Facebook Research / PyTorch team
- MiDaS: Intel ISL
- Stable Baselines 3: DLR-RM
- Hungarian Algorithm implementation: SciPy
- OpenCV, NumPy, Matplotlib, NetworkX

---

**Last Updated**: February 25, 2026
**Python**: 3.13.x (.venv) | Stable Baselines 3: 2.7.1 | Gymnasium: 1.2.3
**Status**: Active — RL training complete, full pipeline operational
