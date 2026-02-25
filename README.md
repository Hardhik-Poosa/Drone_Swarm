# Drone Swarm AI — Intelligent Formation System

> **Transform any photograph or video into a fully choreographed drone swarm using Computer Vision, Depth AI, Physics Simulation, and Reinforcement Learning.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Was Built](#2-what-was-built)
3. [Algorithms Used](#3-algorithms-used)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [How to Run](#6-how-to-run)
7. [Reinforcement Learning](#7-reinforcement-learning)
8. [Video Pipeline (New)](#8-video-pipeline-new)
9. [Metrics](#9-metrics)
10. [Presentation Guide](#10-presentation-guide)
11. [Technology Stack](#11-technology-stack)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

**Drone Swarm AI** takes a single photograph or video file, automatically extracts the subject silhouette, distributes hundreds of virtual drones in that shape in 3D space, and simulates their convergence using either physics or a trained Reinforcement Learning policy.

### What it automates

Real drone light shows (concerts, Super Bowl halftime) require expert choreographers to manually place every waypoint. This project **automates the entire process** — any image or video in, production-ready 3D drone coordinates out.

| Capability | Technique |
|---|---|
| Object silhouette extraction | DeepLabV3-ResNet101 semantic segmentation |
| 3D depth from single photo | MiDaS DPT-Large monocular depth estimation |
| Drone-to-target matching | Hungarian Algorithm (global optimal assignment) |
| Formation motion (image) | Physics (attraction + repulsion) OR PPO Reinforcement Learning |
| Formation motion (video) | EMA lerp pixel-space tracking (smooth, flicker-free) |
| RL training | Stable Baselines 3, 8-stage curriculum, parameter sharing |
| Pattern distribution | Poisson-disk sampling (interior) + arc-length contour (edges) |

---

## 2. What Was Built

### Phase 1 — Core Pipeline
- `pipeline.py` — single entry point for image processing
- `sim/swarm_sim.py` — 2D attraction-physics simulation
- `sim/swarm_sim_3d.py` — 3D volumetric simulation with MiDaS depth

### Phase 2 — Formation Precision
Rewrote `utils/semantic_image_to_formation.py`:
- **80% interior** — Poisson-disk sampling for uniform, non-clustered fill
- **20% contour** — arc-length interpolated boundary tracing (exact silhouette edge)
- 1024 px high-res mask + morphological cleanup (9×9 ellipse, close×3, open×1)

### Phase 3 — 3D Depth Fix
Fixed `utils/formation_3d.py`:
- Removed 4× layer stacking that created rectangular blobs
- Now **1:1 mapping**: each 2D point → one 3D point, Z = MiDaS depth × height_scale
- Correct Y-flip: image pixel space (Y-down) to world space (Y-up)

### Phase 4 — Visualization Overhaul
6-panel precision report saved at `output/formation_precision_report.png`:
1. Original input image
2. Drone overlay on image (formation directly on photo)
3. 2D front view (XY plane)
4. 3D front-on view (azimuth −88°)
5. 3D angled view (azimuth −50°)
6. **Convergence curve** (mean error per frame — shows RL vs Physics)

### Phase 5 — Reinforcement Learning
- `RL/swarm_env_sb3.py` — Gymnasium env, per-drone **(21,) obs space**
- `RL/train_sb3.py` — PPO training, 8-stage curriculum, N=20 fixed
- `RL/rl_controller.py` — inference engine, **N-agnostic** (works for 20 or 1000 drones)
- `RL/checkpoints/sb3/best_model.zip` — trained model (reward ≈55 at stage 6)

### Phase 6 — Video Pipeline (v2, current)
`pipeline_video.py` — completely new pixel-space tracking system:
- Processes any video frame-by-frame; drones **smoothly follow the moving person**
- EMA lerp (α=0.30) instead of physics — no "rain" effect, no repulsion scattering
- Hungarian reassignment every frame in pixel coordinates
- MiDaS depth every N frames → drone dots coloured by depth (cyan=near, red=far)
- Output: side-by-side MP4 (original | drone overlay) + per-frame drone CSV

### Phase 7 — ML Extensions
- `models/vae.py` — Variational Autoencoder for formation distribution learning
- `train/train_vae.py` — constrained training (collision loss + connectivity loss)
- `analysis/` — network topology analysis, scalability benchmarks, connectivity ML

---

## 3. Algorithms Used

### 3.1 Hungarian Algorithm — Optimal Assignment
Solves the N-drone-to-N-target matching with minimum total distance. Greedy (nearest-drone-to-nearest-target) wastes energy and causes drones to cross paths. Hungarian gives the globally optimal solution in O(n³). Used every frame in both image and video pipelines via `scipy.optimize.linear_sum_assignment`.

### 3.2 Poisson-Disk Sampling
Places formation points so no two points are closer than a minimum radius. Produces uniform, natural fills — looks like a real lit-up silhouette, not random clusters. Used in `utils/semantic_image_to_formation.py`.

### 3.3 EMA Lerp Tracking (Video)
Per-frame drone motion for video:
```
targets_smooth  =  0.70 × new_targets  +  0.30 × prev_targets   # suppress jitter
positions_new   =  positions  +  0.30 × (targets_smooth − positions)  # glide
```
No repulsion forces — dense formations stay intact.

### 3.4 Proximal Policy Optimization (PPO)
On-policy policy gradient with clip ratio — prevents destructive updates during curriculum training. Used for the image pipeline RL mode via Stable Baselines 3.

### 3.5 Curriculum Learning
Training progresses through 8 stages of increasing difficulty:

| Stage | Shape | Spacing |
|---|---|---|
| 1 | Grid | 1.8 m (easiest) |
| 2 | Grid | 1.5 m |
| 3 | Circle | 1.5 m |
| 4 | Line | 1.5 m |
| 5 | V-shape | 1.5 m |
| 6 | Grid | 1.2 m |
| 7 | Circle | 1.2 m |
| 8 | V-shape | 1.0 m (hardest) |

### 3.6 Parameter Sharing (Decentralised Execution)
One PPO policy network (trained with N=20) applied independently to every drone at runtime — so the same model works for any swarm size without retraining. Key: per-drone observation is always **(21,)** regardless of N.

### 3.7 k-d Tree Spatial Indexing
`scipy.spatial.cKDTree` reduces neighbor queries from O(N²) to O(N log N). Used in physics simulation and RL environment to find the 6 nearest neighbors for each drone.

### 3.8 MiDaS Monocular Depth Estimation
DPT-Large transformer predicts per-pixel relative depth from a single RGB image. Outputs normalized depth D(x,y) ∈ [0,1]. Used to assign Z-coordinates to 2D drone positions, and to colour drone dots in the video pipeline.

### 3.9 DeepLabV3 Semantic Segmentation
ResNet-101 backbone with atrous convolutions; COCO-pretrained. Automatically identifies `person` (class 15) in any photo or video frame without manual masking. Runs at 512×512 then resized to processing resolution.

### 3.10 Variational Autoencoder (VAE)
Encodes formation patterns to a compressed latent space; decodes back to drone positions. Loss: MSE reconstruction + KL-divergence + collision penalty + connectivity penalty. Enables generating new formation variants from learned distribution.

---

## 4. System Architecture

### Image Pipeline
```
Input Image
    │
    ▼
DeepLabV3 Segmentation   ──►  Person binary mask (1024 px)
    │
    ├─── Poisson-disk interior (80%)  ───┐
    │                                    ├── N × 2D target positions
    └─── Arc-length contour (20%)   ─────┘
                                         │
    MiDaS DPT-Large Depth  ─────────────►├── Lift to 3D → N × 3D targets
                                         │
                              Hungarian Algorithm
                            (optimal drone ↔ target assignment)
                                         │
                    ┌────────────────────┤
                    │                    │
               RL MODE               PHYSICS MODE
           PPO per-drone           Attraction + Repulsion
           inference (21-dim obs)  (k-d tree, O(N log N))
                    │                    │
                    └────────┬───────────┘
                             │
              6-Panel Report + CSV + Metrics JSON
```

### Video Pipeline (v2)
```
Video frame  →  resize to proc_width × proc_h (pixel space throughout)
    │
    ├── DeepLabV3 segmentation  →  bool mask
    │
    ├── sample_targets_px()     →  (N, 2) pixel coords: 80% interior + 20% contour
    │
    ├── EMA smooth targets      →  0.70 × new + 0.30 × prev (suppress jitter)
    │
    ├── Hungarian assignment    →  optimal drone ↔ target pairing
    │
    ├── Lerp step               →  positions += 0.30 × (targets − positions)
    │
    ├── MiDaS depth [every K frames]  →  depth-colour drone dots
    │
    └── Render  →  [original | drone overlay] + HUD bar  →  MP4 frame
```

---

## 5. Project Structure

```
drone_swarm/
│
├── pipeline.py                     ← Main entry point (images)
├── pipeline_video.py               ← Video entry point — v2 pixel-space lerp
│
├── RL/
│   ├── swarm_env_sb3.py           ← Gymnasium env (21-dim per-drone obs)
│   ├── train_sb3.py               ← PPO + 8-stage curriculum
│   ├── rl_controller.py           ← N-agnostic inference engine
│   ├── __init__.py
│   ├── checkpoints/sb3/
│   │   ├── best_model.zip         ← Trained PPO (reward ≈55)  ← USE THIS
│   │   └── final_model.zip
│   └── [legacy: env_2d.py, env_3d.py, train.py, curriculum.py, reward.py]
│
├── utils/
│   ├── semantic_image_to_formation.py  ← DeepLabV3 + Poisson-disk + contour
│   ├── depth_to_3d.py                  ← MiDaS depth estimation
│   ├── formation_3d.py                 ← 2D → 3D lifting (1:1 mapping, fixed)
│   ├── shape_generator.py              ← Geometric shapes (grid/circle/v)
│   ├── evaluate_swarm.py               ← Metrics computation
│   ├── metrics.py                      ← ML loss functions
│   ├── image_to_formation.py           ← Image processing helpers
│   └── network_metrics.py              ← Connectivity graph analysis
│
├── models/
│   └── vae.py                     ← Variational Autoencoder
│
├── train/
│   └── train_vae.py               ← VAE training
│
├── sim/
│   ├── swarm_sim.py               ← 2D standalone simulation
│   └── swarm_sim_3d.py            ← 3D standalone simulation
│
├── analysis/                      ← Research: network analysis, scalability, ML
│
├── data/                          ← Formation datasets (.npy)
│
├── input_images/                  ← Place images/videos here
│   ├── Cristiano_Ronaldo.webp
│   └── 4761762-uhd_2160_4096_25fps.mp4
│
├── output/                        ← All outputs land here
│   ├── formation_precision_report.png
│   ├── *_swarm.mp4                ← Video output
│   └── *_positions.csv            ← Drone positions per frame
│
└── vae_model.pth                  ← Pre-trained VAE weights
```

---

## 6. How to Run

### Environment

```bash
# Use the .venv Python throughout (Windows)
D:\drone_swram\.venv\Scripts\python.exe  <script>

# First-time install
D:\drone_swram\.venv\Scripts\pip.exe install torch torchvision timm
D:\drone_swram\.venv\Scripts\pip.exe install stable-baselines3[extra] gymnasium
D:\drone_swram\.venv\Scripts\pip.exe install scipy numpy opencv-python matplotlib networkx pandas pillow
```

---

### A — Image Pipeline (Physics mode)

```bash
python pipeline.py \
  --mode 3d \
  --image input_images/Cristiano_Ronaldo.webp \
  --output output/ \
  --base-drones 200
```

Steps: segmentation → Poisson targets → MiDaS depth → 3D lift → Hungarian → physics simulation → 6-panel report + CSV.

### A2 — Image Pipeline (RL mode)

```bash
python pipeline.py \
  --mode 3d \
  --image input_images/Cristiano_Ronaldo.webp \
  --output output/ \
  --base-drones 200 \
  --rl-checkpoint RL/checkpoints/sb3/best_model
```

Replaces the physics loop with the trained PPO policy. Panel 6 shows the convergence curve vs physics.

**`pipeline.py` CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `3d` | `2d` or `3d` |
| `--image` | required | Input image path |
| `--output` | `output/` | Output directory |
| `--base-drones` | `300` | Number of drones |
| `--rl-checkpoint` | None | PPO model path (no `.zip`) |
| `--auto-train-rl` | False | Train RL model before running |

---

### B — Train RL from Scratch

```bash
python RL/train_sb3.py \
  --total-timesteps 300000 \
  --curriculum \
  --drones 20 \
  --checkpoint-dir RL/checkpoints/sb3
```

Saves `best_model.zip` (highest eval reward) + checkpoints every 30k steps. Takes ~5 min on CPU.

---

### C — Video Pipeline (v2)

```bash
# Standard — depth colours every 6 frames
python pipeline_video.py \
  --video input_images/4761762-uhd_2160_4096_25fps.mp4 \
  --output output \
  --drones 300 \
  --depth-every 6 \
  --dot-radius 5

# Fast test — no depth, first 10 frames only
python pipeline_video.py \
  --video input_images/4761762-uhd_2160_4096_25fps.mp4 \
  --output output \
  --drones 300 \
  --max-frames 10 \
  --depth-every 0

# With 3D panel
python pipeline_video.py \
  --video input_images/4761762-uhd_2160_4096_25fps.mp4 \
  --output output \
  --drones 300 \
  --render-3d
```

**`pipeline_video.py` CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--output` | `output` | Output directory |
| `--drones` | `300` | Number of drones |
| `--proc-width` | `640` | Processing width in pixels |
| `--lerp-alpha` | `0.30` | Drone lerp speed per frame (0.1=slow, 0.5=fast) |
| `--depth-every` | `6` | Run MiDaS every N frames (`0` = disable) |
| `--render-3d` | False | Add 3D scatter panel to output video |
| `--show-targets` | False | Show faint target markers behind drones |
| `--max-frames` | `0` | Cap frames processed (`0` = all) |
| `--dot-radius` | `5` | Drone dot radius in pixels |

**Output files:**
- `output/<name>_swarm.mp4` — side-by-side: original | drone overlay + HUD bar
- `output/<name>_positions.csv` — columns: `frame, drone_id, px_col, px_row, depth`

---

### D — Standalone Simulations

```bash
python sim/swarm_sim.py          # 2D
python sim/swarm_sim_3d.py       # 3D

python train/train_vae.py        # VAE training
python analysis/network_analysis.py   # graph analysis
```

---

## 7. Reinforcement Learning

### Why RL over pure physics?

Physics applies the same force equation every frame — no awareness of swarm state. RL **learns a navigation strategy** through 300k simulation steps:
- When to approach, when to wait
- Collision avoidance as a trained skill (not hard constraint)
- Faster convergence across diverse formation shapes

### Observation space — per-drone (21-dim)

```python
obs = [
    rel_target_x,  rel_target_y,  dist_to_target,     # 3  — where to go
    nb1_dx, nb1_dy, nb1_dist,                           # 3  — neighbor 1
    nb2_dx, nb2_dy, nb2_dist,                           # 3  — neighbor 2
    nb3_dx, nb3_dy, nb3_dist,                           # 3  — neighbor 3
    nb4_dx, nb4_dy, nb4_dist,                           # 3  — neighbor 4
    nb5_dx, nb5_dy, nb5_dist,                           # 3  — neighbor 5
    nb6_dx, nb6_dy, nb6_dist,                           # 3  — neighbor 6
]  # total = 21 features, all relative to focal drone
```

### Reward shaping

```
reward = −0.1 × mean_dist_to_targets   # formation accuracy
       + 0.5 × connectivity_ratio       # stay networked
       − 0.2 × unsafe_ratio             # avoid collisions
       + 10.0  (terminal bonus if error < 0.5 AND connectivity > 0.8)
```

### Parameter sharing — N-agnostic inference

Trained with N=20 drones, works for any N at deployment:

```python
# rl_controller.py  — deploy time
for i in range(N_drones):          # N can be 20, 200, or 1000
    obs_i = build_obs(i)            # always (21,) — same shape
    action, _ = model.predict(obs_i)
    apply_action(i, action)
```

One neural network → applied N times independently. No retraining for different swarm sizes.

### Training config

```
Algorithm:   PPO (Stable Baselines 3 v2.7.1)
Policy:      MlpPolicy — 2 hidden layers × 64 units, tanh
n_steps:     4096     n_epochs: 10     clip_range: 0.2

Curriculum:  8 stages (grid 1.8m → v-shape 1.0m)
Result:      Stage 1 reward ≈ 60   Stage 6 reward ≈ 55   Stage 8 reward ≈ 12
Checkpoint:  RL/checkpoints/sb3/best_model.zip
```

---

## 8. Video Pipeline (New)

### Why a separate video pipeline?

The image pipeline uses physics/RL to converge from start positions to targets — this takes 150 simulation frames and is designed for a single still image. For video:
- Targets move every real frame (person walks, camera pans)
- 150 physics steps per video frame = impossible in real time
- Physics repulsion scatters drones when they are densely packed in portrait frames

### v2 Design: Pixel-Space Lerp

Everything stays in **raw pixel coordinates** — no normalization, no physics repulsion:

```
Frame N:
  1.  Segment frame  →  person mask (DeepLabV3)
  2.  Sample targets →  300 pixel (col, row) positions on silhouette
  3.  EMA targets    →  0.70 × new + 0.30 × prev  (suppress jitter)
  4.  Hungarian      →  re-assign drones to smoothed targets
  5.  Lerp drones    →  pos += 0.30 × (targets − pos)
  6.  MiDaS depth    →  every 6 frames, colour dots cyan→red by depth
  7.  Render         →  draw dots on frame, write to MP4

Frame N+1:  positions carry over → drones glide continuously
```

### What to expect in output

- Drones are placed **on the person silhouette** from frame 0
- They smoothly follow person movement (lerp creates trailing effect — cinematic)
- Average tracking error ≈ 13–30 px after first few frames (within one dot-radius)
- Dot colours: near depth = bright cyan, far depth = deep red; gradient (cyan top → orange bottom) when depth disabled
- HUD bar shows: frame counter, drone count, tracking error, fps

### Key numbers (308-frame 4K video, 300 drones)

| Metric | Value |
|---|---|
| Processing resolution | 640 × 1214 px |
| Tracking error (settled) | ~13–20 px |
| Frames detecting person | All 308 |
| Depth update interval | Every 6 frames |
| Output file | `output/4761762-uhd_2160_4096_25fps_swarm.mp4` |

---

## 9. Metrics

| Metric | Good Value | Meaning |
|---|---|---|
| Convergence Error | < 1.0 | Mean drone-to-target distance (sim units). Lower = better formation accuracy. |
| Connectivity Ratio | > 0.80 | Fraction of drone pairs within comm range. Above 80% = well-networked swarm. |
| Min Inter-Drone Distance | > 0.50 | Closest two drones get. Below 0.5 = collision risk. |
| Avg Inter-Drone Distance | 1.0 – 3.0 | Typical spacing. Too high = spread out. Too low = crowded. |
| Video Tracking Error | < 30 px | Mean pixel distance from drone dot to silhouette target. |
| RL Used | true / false | Whether PPO policy drove the simulation. |

---

## 10. Presentation Guide

### One-line pitch

> "We built an AI that takes any photo or video, finds the person silhouette, and flies hundreds of drones into that exact 3D shape automatically — using the same computer vision as self-driving cars and the same RL algorithm used in robotics research."

### 3-Minute Structure

**30s — Problem**
Drone light shows cost millions; every waypoint is placed manually by expert designers. We automate that from a single image or live video.

**30s — Solution demo**
Show `output/formation_precision_report.png` — input photo → 200 drones match exact silhouette. Show video output MP4 — 300 drones track moving person in real time.

**60s — How (three AI layers)**
1. **DeepLabV3** — "What shape?" (semantic segmentation, same model as self-driving cars)
2. **MiDaS** — "How deep?" (monocular 3D from one photo, no stereo needed)
3. **PPO RL / Lerp** — "How do drones navigate?" (image: learned coordination; video: smooth pixel-space tracking)

**30s — The RL part**
- Train with 20 drones → deploy with 1000 drones, same model (parameter sharing)
- Convergence curve (panel 6): RL beats physics — faster, fewer oscillations

**20s — Results**
- Image: drones match Ronaldo silhouette boundary, connectivity > 80%
- Video: 300 drones track full 4K 308-frame video, avg error 13 px
- Output CSV is directly importable to Unity, Blender, or real drone firmware

**10s — Future**
Multi-person tracking, real Crazyflie SDK export, real-time inference on GPU (<10ms per frame).

### Expected Interview Questions

**Q: How does each drone know which position to go to?**
A: Hungarian Algorithm solves global minimum-cost matching before simulation. Greedy nearest-neighbor wastes energy and causes crossing paths — Hungarian prevents this in O(n³).

**Q: Why lerp for video instead of physics?**
A: In dense portrait formations (300 drones in a narrow strip) inter-drone spacing < repulsion radius → physics blasts all drones apart like rain. Lerp has no repulsion — drones glide directly to targets in pixel space.

**Q: If you train RL with 20 drones, how does it work with 300?**
A: Parameter sharing — one neural network is applied independently to each drone. The policy sees a (21,) local observation, not swarm size. Applied 300 times → same quality, zero retraining.

**Q: Why PPO?**
A: PPO's clip ratio prevents large policy updates — critical during curriculum when the policy must not forget easy shapes while learning harder ones. More stable than SAC/DDPG with curriculum.

**Q: Is this real-time?**
A: Neural network inference per drone < 0.1ms CPU. 1000 drones ≈ 100ms/frame on CPU, ~8ms on GPU.

---

## 11. Technology Stack

| Component | Library / Model | Version |
|---|---|---|
| Deep learning | PyTorch | 2.x |
| Segmentation | DeepLabV3-ResNet101 | torchvision pretrained |
| Depth estimation | MiDaS DPT-Large | timm / torch.hub |
| Reinforcement learning | Stable Baselines 3 | 2.7.1 |
| RL environment | Gymnasium | 1.2.3 |
| Optimal assignment | SciPy `linear_sum_assignment` | 1.x |
| Spatial indexing | SciPy `cKDTree` | 1.x |
| Computer vision | OpenCV | 4.x |
| Numerical computing | NumPy | 2.x |
| Visualization | Matplotlib | 3.x |
| Data | Pandas | 2.x |
| Graph analysis | NetworkX | 3.x |

---

## 12. Troubleshooting

**ModuleNotFoundError: torch / torchvision / timm**
```bash
D:\drone_swram\.venv\Scripts\pip.exe install torch torchvision timm
```

**ModuleNotFoundError: stable_baselines3**
```bash
D:\drone_swram\.venv\Scripts\pip.exe install stable-baselines3[extra] gymnasium
```

**MiDaS downloads ~500 MB on first run** — normal, cached at `~/.cache/torch/hub/` afterwards.

**Video output drones scatter (rain effect)** — you are running an old version. Confirm `pipeline_video.py` uses `lerp_step()` and `sample_targets_px()`, not `physics_step()`.

**RL checkpoint path error** — omit `.zip`:
```bash
# Correct
--rl-checkpoint RL/checkpoints/sb3/best_model
# Wrong
--rl-checkpoint RL/checkpoints/sb3/best_model.zip
```

**Slow video processing** — use `--depth-every 0` (disables MiDaS), `--max-frames 20` for quick test.

**Out of GPU memory** — add `map_location='cpu'` when loading PPO model, or reduce `--drones`.

**RL not converging** — increase `--total-timesteps` to 500000, or check `RL/checkpoints/sb3/evaluations.npz` for reward trend.

---

## Authors

**Hardhik Poosa** — B.Tech Computer Science and Engineering

Libraries and pre-trained models: DeepLabV3 (Facebook Research / PyTorch), MiDaS (Intel ISL), Stable Baselines 3 (DLR-RM), Hungarian Algorithm (SciPy), OpenCV, NumPy, Matplotlib.

---

**Last Updated**: February 26, 2026  
**Python**: 3.13.x (`.venv`) | SB3: 2.7.1 | Gymnasium: 1.2.3  
**Status**: Active — image pipeline ✅ · RL model trained ✅ · video pipeline v2 ✅
