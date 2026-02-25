"""
==================================================================================
  DRONE SWARM AI - END-TO-END PIPELINE
==================================================================================
  
  Transform any image into an intelligent 3D drone formation choreography!
  
  Usage:
  ------
  # 2D formation from predefined shape
  python pipeline.py --mode 2d --shape grid --num-drones 60 --output results/my_formation.csv
  
  # 3D formation from image (full pipeline!)
  python pipeline.py --mode 3d --image input_images/sample.jpg --base-drones 300 --output results/
  
  # 3D formation with custom physics
  python pipeline.py --mode 3d --image input.jpg --config config/config_3d.yaml --output results/
  
==================================================================================
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    """Create structured logger with file and console output."""
    # Force UTF-8 on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str], mode: str) -> Dict:
    """Load config from YAML or use defaults."""
    if config_path and os.path.exists(config_path):
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using defaults...")
    
    # Defaults
    if mode == "2d":
        return {
            "simulation": {"num_drones": 50, "num_frames": 120, "step_size": 0.08},
            "formation": {"shape": "grid", "distance": 1.5},
            "environment": {"communication_range": 5.0, "min_safe_distance": 0.5},
            "output": {"save_csv": True, "show_visualization": False}
        }
    else:  # 3d
        return {
            "simulation": {"base_drones": 300, "num_frames": 150, "step_size": 0.05},
            "image_to_3d": {"scale_factor": 6, "height_scale": 10, "layers": 4},
            "physics": {"repulsion_strength": 0.02, "repulsion_radius": 0.6},
            "environment": {"communication_range": 5.0, "min_safe_distance": 0.5},
            "output": {"save_csv": True, "show_visualization": False}
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2D PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_2d_pipeline(
    shape: str,
    num_drones: int,
    config: Dict,
    output_csv: str,
    logger: logging.Logger
) -> Tuple[np.ndarray, Dict]:
    """
    2D formation pipeline:
    1. Generate target formation (shape)
    2. Initialize drones randomly
    3. Run physics simulation
    4. Export results
    """
    logger.info(f"[2D PIPELINE] Starting with shape='{shape}', N={num_drones}")
    
    try:
        from utils.shape_generator import generate_shape
        from analysis.swarm_core import run_simulation
        from utils.evaluate_swarm import evaluate_formation
        
        cfg = config.get("simulation", {})
        env_cfg = config.get("environment", {})
        
        # Generate target formation
        logger.info(f"Generating {shape} formation...")
        distance = config.get("formation", {}).get("distance", 1.5)
        targets = generate_shape(shape, n_drones=num_drones, distance=distance)
        logger.info(f"[OK] Generated {len(targets)} waypoints")
        
        # Run simulation
        logger.info("Running physics simulation...")
        step_size = cfg.get("step_size", 0.08)
        num_frames = cfg.get("num_frames", 120)
        comm_range = env_cfg.get("communication_range", 5.0)
        
        # Initialize random positions
        np.random.seed(42)
        current_positions = np.random.uniform(-5, 5, size=(num_drones, 2)).astype(np.float32)
        
        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.linalg.norm(
            current_positions[:, None, :] - targets[None, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        targets = targets[col_ind]
        
        # Physics loop
        for frame in range(num_frames):
            if frame % 20 == 0:
                logger.debug(f"  Frame {frame}/{num_frames}")
            
            attraction = targets - current_positions
            current_positions += step_size * attraction
        
        logger.info(f"[OK] Simulation converged after {num_frames} frames")
        
        # Evaluate (evaluate_formation returns Title Case keys)
        positions_3d = np.hstack([current_positions, np.zeros((num_drones, 1))])
        targets_3d   = np.hstack([targets, np.zeros((num_drones, 1))])
        raw_metrics  = evaluate_formation(positions_3d, targets_3d, comm_range)
        
        metrics = {
            "connectivity_ratio": float(raw_metrics.get("Connectivity Ratio", 0)),
            "min_distance":       float(raw_metrics.get("Minimum Inter-Drone Distance", 0)),
            "avg_distance":       float(raw_metrics.get("Average Inter-Drone Distance", 0)),
            "convergence_error":  float(raw_metrics.get("Convergence Error", 0)),
            "num_drones":         int(num_drones),
        }
        
        logger.info(f"Connectivity Ratio:       {metrics['connectivity_ratio']:.4f}")
        logger.info(f"Min Inter-Drone Distance: {metrics['min_distance']:.4f}")
        logger.info(f"Convergence Error:        {metrics['convergence_error']:.4f}")
        
        return current_positions, metrics
        
    except Exception as e:
        logger.error(f"Error in 2D pipeline: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3D PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_3d_pipeline(
    image_path: str,
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
    base_drones_override: Optional[int] = None,
    rl_checkpoint: Optional[str] = None,
    auto_train_rl: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    3D formation pipeline:
    1. Extract 2D silhouette from image (DeepLabV3 + Poisson-disk)
    2. Lift to 3D using MiDaS depth (1-to-1, no layer stacking)
    3. Hungarian optimal assignment
    4. RL-driven simulation (PPO policy) OR physics fallback
    5. Precision visualization (6-panel) + export
    """
    logger.info(f"[3D PIPELINE] Starting with image='{image_path}'")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        from utils.semantic_image_to_formation import image_to_semantic_outline
        from utils.formation_3d import lift_to_true_3d
        from utils.evaluate_swarm import evaluate_formation
        from scipy.optimize import linear_sum_assignment
        
        cfg = config.get("simulation", {})
        img_cfg = config.get("image_to_3d", {})
        phys_cfg = config.get("physics", {})
        
        # Step 1: Extract 2D outline
        logger.info("Step 1/4: Extracting 2D outline from image...")
        base_drones = base_drones_override if base_drones_override is not None else cfg.get("base_drones", 300)
        scale_factor = img_cfg.get("scale_factor", 6)
        
        formation_2d = image_to_semantic_outline(
            image_path,
            n_drones=base_drones,
            scale_factor=scale_factor
        )
        logger.info(f"[OK] Extracted {len(formation_2d)} 2D waypoints")
        
        # Step 1b: Save 2D silhouette precision visualization
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from PIL import Image as PILImg
            import torchvision.transforms as T_viz
            import torchvision.models as models_viz
            from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

            fig2d, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig2d.suptitle("Formation Precision – 2D Silhouette", fontsize=14, fontweight='bold')

            # Left: original image
            orig_img = PILImg.open(image_path).convert('RGB')
            axes[0].imshow(np.array(orig_img))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Right: drone positions scatter matching silhouette
            x_vals = formation_2d[:, 0]
            y_vals = formation_2d[:, 1]
            axes[1].scatter(x_vals, y_vals, s=4, c='deepskyblue', alpha=0.7, linewidths=0)
            axes[1].set_aspect('equal')
            axes[1].set_facecolor('#111111')
            axes[1].set_title(f'Drone Formation ({len(formation_2d)} drones)')
            axes[1].set_xlabel('X (world units)')
            axes[1].set_ylabel('Y (world units)')
            fig2d.tight_layout()
            viz2d_path = os.path.join(output_dir, 'formation_2d_precision.png')
            fig2d.savefig(viz2d_path, dpi=120, bbox_inches='tight')
            plt.close(fig2d)
            logger.info(f"[OK] Saved 2D precision visualization to {viz2d_path}")
        except Exception as viz_err:
            logger.warning(f"Could not save 2D precision viz: {viz_err}")
        
        # Step 2: Lift to 3D with depth
        logger.info("Step 2/4: Lifting to 3D with depth estimation...")
        height_scale = img_cfg.get("height_scale", 10)
        layers = img_cfg.get("layers", 4)
        
        target_formation = lift_to_true_3d(
            formation_2d,
            image_path,
            height_scale=height_scale,
            layers=layers
        )
        logger.info(f"[OK] Created 3D volumetric formation with {len(target_formation)} drones")
        
        # Step 3: Initialize and assign
        logger.info("Step 3/4: Initializing drone positions and assigning targets...")
        N_DRONES = len(target_formation)
        np.random.seed(42)
        current_positions = np.random.uniform(-5, 5, size=(N_DRONES, 3)).astype(np.float32)
        
        cost_matrix = np.linalg.norm(
            current_positions[:, None, :] - target_formation[None, :, :], axis=2
        )
        _, col_ind = linear_sum_assignment(cost_matrix)
        target_formation = target_formation[col_ind]
        
        logger.info(f"[OK] Assigned {N_DRONES} drones to waypoints")

        # ── Step 4: RL Controller or Physics Fallback ──────────────────────
        num_frames = cfg.get("num_frames", 150)
        convergence_history = []   # track mean dist-to-target per frame
        rl_used = False

        # Try to load or auto-train RL checkpoint
        _rl_ckpt = rl_checkpoint
        if _rl_ckpt is None and auto_train_rl:
            _rl_train_dir = os.path.join(output_dir, "rl_model")
            _best = os.path.join(_rl_train_dir, "best_model.zip")
            _final = os.path.join(_rl_train_dir, "final_model.zip")
            if os.path.exists(_best):
                _rl_ckpt = _best
            elif os.path.exists(_final):
                _rl_ckpt = _final
            else:
                from RL.rl_controller import auto_train
                _rl_ckpt = auto_train(_rl_train_dir,
                                      total_timesteps=300_000,
                                      logger=logger)

        if _rl_ckpt and os.path.exists(
                _rl_ckpt if _rl_ckpt.endswith('.zip') else _rl_ckpt + '.zip'):
            # ── RL MODE ────────────────────────────────────────────────────
            logger.info(f"Step 4/4: Running RL-driven simulation (PPO policy)...")
            logger.info(f"  Checkpoint: {_rl_ckpt}")
            try:
                from RL.rl_controller import RLController
                controller = RLController(_rl_ckpt)
                rl_used = True
                step_size        = cfg.get("step_size", 0.05)
                repulsion_strength = phys_cfg.get("repulsion_strength", 0.015)
                repulsion_radius   = phys_cfg.get("repulsion_radius",   0.5)

                for frame in range(num_frames):
                    current_positions = controller.step(
                        current_positions,
                        target_formation,
                        step_size=step_size,
                        repulsion_strength=repulsion_strength,
                        repulsion_radius=repulsion_radius,
                    )
                    mean_err = float(np.mean(
                        np.linalg.norm(current_positions - target_formation, axis=1)
                    ))
                    convergence_history.append(mean_err)
                    if frame % 25 == 0:
                        logger.debug(f"  [RL] Frame {frame:>3}/{num_frames} | "
                                     f"mean dist={mean_err:.4f}")

                logger.info(f"[OK] RL simulation complete after {num_frames} frames")
            except Exception as rl_err:
                logger.warning(f"[RL] Controller failed ({rl_err}), falling back to physics")
                rl_used = False

        if not rl_used:
            # ── PHYSICS FALLBACK ────────────────────────────────────────────
            logger.info("Step 4/4: Running physics simulation...")
            from scipy.spatial import cKDTree
            step_size          = cfg.get("step_size", 0.05)
            repulsion_strength = phys_cfg.get("repulsion_strength", 0.02)
            repulsion_radius   = phys_cfg.get("repulsion_radius",   0.6)

            for frame in range(num_frames):
                attraction = target_formation - current_positions
                repulsion  = np.zeros_like(current_positions)
                tree = cKDTree(current_positions)
                pairs = tree.query_pairs(r=repulsion_radius, output_type="ndarray")
                if len(pairs):
                    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
                    diff = current_positions[i_idx] - current_positions[j_idx]
                    dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
                    force = diff / (dist * dist ** 2)
                    np.add.at(repulsion, i_idx,  force)
                    np.add.at(repulsion, j_idx, -force)
                current_positions += (step_size * attraction
                                      + repulsion_strength * repulsion)
                mean_err = float(np.mean(
                    np.linalg.norm(current_positions - target_formation, axis=1)
                ))
                convergence_history.append(mean_err)
                if frame % 25 == 0:
                    logger.debug(f"  Frame {frame:>3}/{num_frames} | "
                                 f"mean dist={mean_err:.4f}")

            logger.info(f"[OK] Physics simulation complete after {num_frames} frames")
        
        sim_mode = "RL (PPO)" if rl_used else "Physics"
        
        # Evaluate (evaluate_formation returns Title Case keys)
        comm_range = config.get("environment", {}).get("communication_range", 5.0)
        raw_metrics = evaluate_formation(current_positions, target_formation, comm_range)
        
        # Normalize keys to snake_case
        metrics = {
            "connectivity_ratio":  float(raw_metrics.get("Connectivity Ratio", 0)),
            "min_distance":        float(raw_metrics.get("Minimum Inter-Drone Distance", 0)),
            "avg_distance":        float(raw_metrics.get("Average Inter-Drone Distance", 0)),
            "convergence_error":   float(raw_metrics.get("Convergence Error", 0)),
            "num_drones":          int(N_DRONES),
        }
        
        logger.info(f"Connectivity Ratio:       {metrics['connectivity_ratio']:.4f}")
        logger.info(f"Min Inter-Drone Distance: {metrics['min_distance']:.4f}")
        logger.info(f"Avg Inter-Drone Distance: {metrics['avg_distance']:.4f}")
        logger.info(f"Convergence Error:        {metrics['convergence_error']:.4f}")
        metrics["rl_used"] = rl_used
        metrics["final_convergence"] = convergence_history[-1] if convergence_history else 0.0

        # ── Precision Visualization (6-panel) ───────────────────────────────
        logger.info("Generating precision visualization...")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
            from PIL import Image as _PILImg

            _tf = target_formation   # (N, 3) exact person shape

            fig = plt.figure(figsize=(30, 8))
            fig.patch.set_facecolor('#0a0a0a')

            # ── Panel 1: Original image ──────────────────────────────────────
            ax1 = fig.add_subplot(1, 6, 1)
            orig_img = np.array(_PILImg.open(image_path).convert('RGB'))
            ax1.imshow(orig_img)
            ax1.set_title('Original Image', color='white', fontsize=11, fontweight='bold')
            ax1.axis('off')

            # ── Panel 2: Drone overlay on image ─────────────────────────────
            ax2 = fig.add_subplot(1, 6, 2)
            ax2.imshow(orig_img)
            _f2 = formation_2d   # (N, 2) world XY
            _fx_n = (_f2[:, 0] - _f2[:, 0].min()) / max(float(np.ptp(_f2[:, 0])), 1e-9)
            _fy_n = 1.0 - (_f2[:, 1] - _f2[:, 1].min()) / max(float(np.ptp(_f2[:, 1])), 1e-9)
            _px = _fx_n * (orig_img.shape[1] - 1)
            _py = _fy_n * (orig_img.shape[0] - 1)
            ax2.scatter(_px, _py, s=2, c='cyan', alpha=0.7, linewidths=0)
            ax2.set_title(f'Drone Overlay\n({len(formation_2d)} drones)',
                          color='white', fontsize=11, fontweight='bold')
            ax2.axis('off')

            # ── Panel 3: Front view XY — exact person silhouette ─────────────
            # Uses TARGET formation (not post-physics positions) for perfect shape
            ax3 = fig.add_subplot(1, 6, 3)
            ax3.set_facecolor('#0a0a0a')
            ax3.scatter(_tf[:, 0], _tf[:, 1], s=4, c='deepskyblue',
                        alpha=0.85, linewidths=0)
            ax3.set_aspect('equal')
            ax3.set_title('Front View (XY)\nExact Person Silhouette',
                          color='white', fontsize=11, fontweight='bold')
            ax3.set_xlabel('X', color='#aaaaaa')
            ax3.set_ylabel('Y', color='#aaaaaa')
            ax3.tick_params(colors='#aaaaaa')
            for sp in ax3.spines.values():
                sp.set_edgecolor('#333333')

            # ── Panel 4: 3D — front-facing (elev=10, azim=-90 = straight-on) ─
            # This view looks exactly like the person photo but in 3D with depth color
            ax4 = fig.add_subplot(1, 6, 4, projection='3d')
            ax4.set_facecolor('#0a0a0a')
            sc4 = ax4.scatter(
                _tf[:, 0],          # X = left/right
                _tf[:, 2],          # Z (depth) = into screen
                _tf[:, 1],          # Y = up/down
                c=_tf[:, 2],        # colour by depth → closer = brighter
                cmap='plasma', s=4, alpha=0.85
            )
            ax4.set_xlabel('X', color='white', fontsize=7, labelpad=2)
            ax4.set_ylabel('Depth', color='white', fontsize=7, labelpad=2)
            ax4.set_zlabel('Y (up)', color='white', fontsize=7, labelpad=2)
            ax4.view_init(elev=10, azim=-88)   # almost perfectly front-on
            ax4.tick_params(colors='#888888', labelsize=6)
            ax4.set_title('3D Front View\n(coloured by depth)',
                          color='white', fontsize=11, fontweight='bold')
            plt.colorbar(sc4, ax=ax4, shrink=0.5, pad=0.12,
                         label='Depth').ax.yaxis.label.set_color('white')

            # ── Panel 5: 3D — angled perspective to appreciate the 3D shape ──
            ax5 = fig.add_subplot(1, 6, 5, projection='3d')
            ax5.set_facecolor('#0a0a0a')
            ax5.scatter(
                _tf[:, 0],
                _tf[:, 2],
                _tf[:, 1],
                c=_tf[:, 2],
                cmap='cool', s=4, alpha=0.8
            )
            ax5.set_xlabel('X', color='white', fontsize=7, labelpad=2)
            ax5.set_ylabel('Depth', color='white', fontsize=7, labelpad=2)
            ax5.set_zlabel('Y (up)', color='white', fontsize=7, labelpad=2)
            ax5.view_init(elev=25, azim=-50)   # angled — see the 3D volume
            ax5.tick_params(colors='#888888', labelsize=6)
            ax5.set_title('3D Angled View\n(depth perspective)',
                          color='white', fontsize=11, fontweight='bold')

            # ── Panel 6: RL Convergence curve ───────────────────────────────
            ax6 = fig.add_subplot(1, 6, 6)
            ax6.set_facecolor('#0a0a0a')
            if convergence_history:
                color_curve = '#00e5ff' if rl_used else '#ff9800'
                ax6.plot(convergence_history, color=color_curve, linewidth=1.5)
                ax6.fill_between(range(len(convergence_history)),
                                 convergence_history, alpha=0.2, color=color_curve)
                ax6.axhline(y=convergence_history[-1], color='#ff4444',
                            linestyle='--', linewidth=1, label=f'Final: {convergence_history[-1]:.3f}')
                ax6.set_xlabel('Frame', color='#aaaaaa', fontsize=8)
                ax6.set_ylabel('Mean Dist to Target', color='#aaaaaa', fontsize=8)
                ax6.legend(fontsize=7, labelcolor='white',
                           facecolor='#1a1a1a', edgecolor='#333333')
            mode_label = 'RL (PPO Policy)' if rl_used else 'Physics (fallback)'
            ax6.set_title(f'Convergence Curve\n[{mode_label}]',
                          color='white', fontsize=11, fontweight='bold')
            ax6.tick_params(colors='#888888', labelsize=7)
            for sp in ax6.spines.values():
                sp.set_edgecolor('#333333')

            mode_str = 'RL-PPO' if rl_used else 'Physics'
            fig.suptitle(
                f'Drone Swarm Precision Report  [{mode_str}]  |  {N_DRONES} drones  '
                f'|  Connectivity: {metrics["connectivity_ratio"]:.3f}  '
                f'|  Conv Error: {metrics["convergence_error"]:.4f}',
                color='white', fontsize=13, fontweight='bold', y=1.01
            )
            fig.tight_layout()

            vis_path = os.path.join(output_dir, "formation_precision_report.png")
            fig.savefig(vis_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            logger.info(f"[OK] Saved visualization to {vis_path}")
        except Exception as ve:
            logger.warning(f"Visualization skipped: {ve}", exc_info=True)
        # ────────────────────────────────────────────────────────────────────
        
        return current_positions, metrics
        
    except Exception as e:
        logger.error(f"Error in 3D pipeline: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def export_results(
    positions: np.ndarray,
    metrics: Dict,
    output_path: str,
    logger: logging.Logger
) -> None:
    """Export drone positions and metrics to CSV and JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSV export
    if positions.shape[1] == 2:
        df = pd.DataFrame(positions, columns=['X', 'Y'])
    else:
        df = pd.DataFrame(positions, columns=['X', 'Y', 'Z'])
    
    df.to_csv(output_path, index=False)
    logger.info(f"[OK] Exported {len(df)} drone positions to {output_path}")
    
    # Metadata (convert numpy types to native Python for JSON)
    def _to_serializable(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    
    metrics_path = output_path.replace('.csv', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: _to_serializable(v) for k, v in metrics.items()}, f, indent=2)
    logger.info(f"[OK] Exported metrics to {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drone Swarm AI - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2D formation
  python pipeline.py --mode 2d --shape circle --num-drones 80 --output results/circle.csv
  
  # 3D from image
  python pipeline.py --mode 3d --image input_images/sample.jpg --output results/
  
  # With custom config
  python pipeline.py --mode 3d --image input.jpg --config config/config_3d.yaml --output results/
        """
    )
    
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d",
                       help="Simulation mode (default: 2d)")
    parser.add_argument("--shape", choices=["grid", "circle", "v", "line"], default="grid",
                       help="Target formation shape for 2D mode (default: grid)")
    parser.add_argument("--num-drones", type=int, default=50,
                       help="Number of drones for 2D mode (default: 50)")
    parser.add_argument("--image", type=str,
                       help="Input image path for 3D mode (required for 3D)")
    parser.add_argument("--base-drones", type=int, default=300,
                       help="Base drone count for 3D mode (default: 300)")
    parser.add_argument("--config", type=str,
                       help="Custom config file (YAML)")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                       help="Path to trained PPO checkpoint (.zip) to use RL-driven simulation")
    parser.add_argument("--auto-train-rl", action="store_true",
                       help="Auto-train RL model (300k steps) if no checkpoint found")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV file or directory path")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = os.path.dirname(args.output) or "."
    logger = setup_logging(output_dir)
    config = load_config(args.config, args.mode)
    
    logger.info("=" * 80)
    logger.info("DRONE SWARM AI - END-TO-END PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Run appropriate pipeline
        if args.mode == "2d":
            logger.info(f"Shape: {args.shape}")
            logger.info(f"Num Drones: {args.num_drones}")
            positions, metrics = run_2d_pipeline(
                args.shape, args.num_drones, config, args.output, logger
            )
            export_results(positions, metrics, args.output, logger)
        
        else:  # 3d
            if not args.image:
                logger.error("--image is required for 3D mode")
                sys.exit(1)
            
            logger.info(f"Image: {args.image}")
            logger.info(f"Base Drones: {args.base_drones}")
            
            output_path = args.output if args.output.endswith('.csv') \
                         else os.path.join(args.output, f"formation_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            positions, metrics = run_3d_pipeline(
                args.image, config, output_dir, logger,
                base_drones_override=args.base_drones,
                rl_checkpoint=args.rl_checkpoint,
                auto_train_rl=args.auto_train_rl,
            )
            export_results(positions, metrics, output_path, logger)
        
        logger.info("=" * 80)
        logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("[FAILED] PIPELINE FAILED")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
