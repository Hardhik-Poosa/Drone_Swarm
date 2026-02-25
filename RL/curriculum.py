"""
rl/curriculum.py
================
Curriculum scheduler for MARL training.

Curriculum philosophy
---------------------
- Start easy (small N, grid shape, wide comm_range) so the agent quickly
  learns the core "fly toward target and don't collide" behaviour.
- Gradually increase difficulty by:
    1. Increasing swarm size (N)
    2. Adding harder target shapes (circle, v, line)
    3. Tightening communication range (harder connectivity requirement)
    4. Porting to 3-D (after 2-D is stable)

Usage (in train.py)
-------------------
    scheduler = CurriculumScheduler()
    cfg = scheduler.get_config(training_iteration=500)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Stage:
    """One curriculum stage."""
    name        : str
    min_iter    : int            # training iteration at which stage activates
    N_range     : tuple          # (N_min, N_max) drone count sampled each episode
    shapes      : List[str]      # candidate target shapes
    comm_range  : float          # communication radius
    distance    : float          # target spacing
    max_steps   : int            # episode length
    description : str = ""


# ---------------------------------------------------------------------------
# Predefined 2-D curriculum stages
# ---------------------------------------------------------------------------

CURRICULUM_2D: List[Stage] = [
    Stage(
        name       = "stage_1_tiny_grid",
        min_iter   = 0,
        N_range    = (10, 20),
        shapes     = ["grid"],
        comm_range = 6.0,
        distance   = 1.5,
        max_steps  = 120,
        description= "Small N, grid only, easy connectivity",
    ),
    Stage(
        name       = "stage_2_medium_shapes",
        min_iter   = 300,
        N_range    = (20, 40),
        shapes     = ["grid", "circle"],
        comm_range = 5.5,
        distance   = 1.5,
        max_steps  = 120,
        description= "Medium N, add circle shape",
    ),
    Stage(
        name       = "stage_3_larger_shapes",
        min_iter   = 700,
        N_range    = (30, 60),
        shapes     = ["grid", "circle", "v"],
        comm_range = 5.0,
        distance   = 1.5,
        max_steps  = 150,
        description= "Larger N, V-shape added",
    ),
    Stage(
        name       = "stage_4_all_shapes",
        min_iter   = 1200,
        N_range    = (50, 100),
        shapes     = ["grid", "circle", "v", "line"],
        comm_range = 4.5,
        distance   = 1.5,
        max_steps  = 150,
        description= "Full shape suite, tighter comm range",
    ),
    Stage(
        name       = "stage_5_stress",
        min_iter   = 2000,
        N_range    = (80, 150),
        shapes     = ["grid", "circle", "v", "line"],
        comm_range = 4.0,
        distance   = 1.3,
        max_steps  = 180,
        description= "High N, tighter spacing, stress test",
    ),
]


# ---------------------------------------------------------------------------
# Predefined 3-D curriculum stages
# ---------------------------------------------------------------------------

CURRICULUM_3D: List[Stage] = [
    Stage(
        name       = "3d_stage_1_tiny_grid",
        min_iter   = 0,
        N_range    = (10, 20),
        shapes     = ["grid"],
        comm_range = 7.0,
        distance   = 1.5,
        max_steps  = 150,
        description= "3-D bootstrap, small N grid",
    ),
    Stage(
        name       = "3d_stage_2_medium",
        min_iter   = 400,
        N_range    = (20, 50),
        shapes     = ["grid", "circle"],
        comm_range = 6.5,
        distance   = 1.5,
        max_steps  = 150,
        description= "3-D medium N",
    ),
    Stage(
        name       = "3d_stage_3_larger",
        min_iter   = 1000,
        N_range    = (40, 100),
        shapes     = ["grid", "circle", "v"],
        comm_range = 6.0,
        distance   = 1.5,
        max_steps  = 180,
        description= "3-D larger N, varied shapes",
    ),
    Stage(
        name       = "3d_stage_4_stress",
        min_iter   = 2000,
        N_range    = (80, 200),
        shapes     = ["grid", "circle", "v", "line"],
        comm_range = 5.5,
        distance   = 1.3,
        max_steps  = 200,
        description= "3-D stress test",
    ),
]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Returns the active curriculum stage and a sampled env config dict
    given the current training iteration.

    Parameters
    ----------
    mode : "2d" or "3d"
    seed : int  for reproducible N sampling
    """

    def __init__(self, mode: str = "2d", seed: int = 0):
        self.mode   = mode.lower()
        self._rng   = __import__("numpy").random.default_rng(seed)
        self._stages = CURRICULUM_2D if self.mode == "2d" else CURRICULUM_3D

    def get_stage(self, training_iteration: int) -> Stage:
        """Return the highest stage whose min_iter ≤ training_iteration."""
        active = self._stages[0]
        for stage in self._stages:
            if training_iteration >= stage.min_iter:
                active = stage
        return active

    def sample_config(self, training_iteration: int) -> dict:
        """
        Sample a concrete env config dict for one episode.
        Suitable to pass as env_config or as a reset() options dict.
        """
        stage  = self.get_stage(training_iteration)
        N      = int(self._rng.integers(stage.N_range[0], stage.N_range[1] + 1))
        shape  = self._rng.choice(stage.shapes)

        return {
            "N"          : N,
            "shape"      : str(shape),
            "comm_range" : stage.comm_range,
            "distance"   : stage.distance,
            "max_steps"  : stage.max_steps,
            "stage_name" : stage.name,
        }

    def summary(self) -> str:
        lines = [f"Curriculum ({self.mode.upper()}) - {len(self._stages)} stages:"]
        for s in self._stages:
            lines.append(
                f"  [{s.min_iter:>5d}+] {s.name:<32s}  "
                f"N={s.N_range}  shapes={s.shapes}  "
                f"comm_range={s.comm_range}"
            )
        return "\n".join(lines)
