"""
==================================================================================
  BENCHMARK SUITE - Compare Baselines
==================================================================================
  
  Compares performance of:
  1. Scripted controller (original physics)
  2. Enhanced controller (with warm-start concept)
  3. ML-enhanced controller (using spacing model if available)
  
  Run: python benchmark.py --scenarios 20
  
==================================================================================
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 1: SCRIPTED CONTROLLER (Original)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_scripted(
    N: int,
    shape: str,
    num_frames: int = 120,
    step_size: float = 0.08,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Original scripted physics controller."""
    from utils.shape_generator import generate_shape
    from utils.evaluate_swarm import evaluate_formation
    from scipy.optimize import linear_sum_assignment
    
    # Generate formation
    targets = generate_shape(shape, n_drones=N, distance=1.5)
    
    # Initialize random
    np.random.seed(seed)
    positions = np.random.uniform(-5, 5, size=(N, 2)).astype(np.float32)
    
    # Hungarian assignment
    cost_matrix = np.linalg.norm(
        positions[:, None, :] - targets[None, :, :], axis=2
    )
    _, col_ind = linear_sum_assignment(cost_matrix)
    targets = targets[col_ind]
    
    # Physics loop
    start_time = time.time()
    for _ in range(num_frames):
        attraction = targets - positions
        positions += step_size * attraction
    elapsed = time.time() - start_time
    
    # Evaluate
    positions_3d = np.hstack([positions, np.zeros((N, 1))])
    targets_3d = np.hstack([targets, np.zeros((N, 1))])
    metrics = evaluate_formation(positions_3d, targets_3d, comm_range=5.0)
    metrics['compute_time'] = elapsed
    
    return positions, metrics


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 2: ENHANCED (With Velocity Damping)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_enhanced(
    N: int,
    shape: str,
    num_frames: int = 120,
    step_size: float = 0.08,
    seed: int = 42,
    damping: float = 0.95,
) -> Tuple[np.ndarray, Dict]:
    """Enhanced controller with velocity tracking and damping for smoother convergence."""
    from utils.shape_generator import generate_shape
    from utils.evaluate_swarm import evaluate_formation
    from scipy.optimize import linear_sum_assignment
    
    # Generate formation
    targets = generate_shape(shape, n_drones=N, distance=1.5)
    
    # Initialize random
    np.random.seed(seed)
    positions = np.random.uniform(-5, 5, size=(N, 2)).astype(np.float32)
    velocities = np.zeros_like(positions)
    
    # Hungarian assignment
    cost_matrix = np.linalg.norm(
        positions[:, None, :] - targets[None, :, :], axis=2
    )
    _, col_ind = linear_sum_assignment(cost_matrix)
    targets = targets[col_ind]
    
    # Physics loop with velocity
    start_time = time.time()
    for _ in range(num_frames):
        attraction = targets - positions
        # Velocity-based: dampened acceleration
        accelerations = attraction
        velocities = damping * velocities + (1 - damping) * accelerations
        positions += step_size * velocities
    elapsed = time.time() - start_time
    
    # Evaluate
    positions_3d = np.hstack([positions, np.zeros((N, 1))])
    targets_3d = np.hstack([targets, np.zeros((N, 1))])
    metrics = evaluate_formation(positions_3d, targets_3d, comm_range=5.0)
    metrics['compute_time'] = elapsed
    
    return positions, metrics


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 3: ML-ENHANCED (With Spacing Prediction)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_ml_enhanced(
    N: int,
    shape: str,
    num_frames: int = 120,
    step_size: float = 0.08,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    ML-enhanced controller: predicts optimal spacing/communication range
    using pre-trained model (if available), then runs enhanced physics.
    Falls back to enhanced baseline if model not found.
    """
    try:
        # Try loading pre-trained spacing model
        model_path = "analysis/connectivity_model.pkl"
        if os.path.exists(model_path):
            import pickle
            with open(model_path, 'rb') as f:
                spacing_model = pickle.load(f)
            
            # Predict optimal spacing
            X = np.array([[N, 5.0]])  # features: [N, comm_range]
            predicted_spacing = spacing_model.predict(X)[0]
            
            logger.info(f"  ML predicted spacing: {predicted_spacing:.3f}")
            
            # Use predicted spacing
            from utils.shape_generator import generate_shape
            from utils.evaluate_swarm import evaluate_formation
            from scipy.optimize import linear_sum_assignment
            
            targets = generate_shape(shape, n_drones=N, distance=predicted_spacing)
            
            np.random.seed(seed)
            positions = np.random.uniform(-5, 5, size=(N, 2)).astype(np.float32)
            velocities = np.zeros_like(positions)
            
            cost_matrix = np.linalg.norm(
                positions[:, None, :] - targets[None, :, :], axis=2
            )
            _, col_ind = linear_sum_assignment(cost_matrix)
            targets = targets[col_ind]
            
            start_time = time.time()
            for _ in range(num_frames):
                attraction = targets - positions
                velocities = 0.95 * velocities + (1 - 0.95) * attraction
                positions += step_size * velocities
            elapsed = time.time() - start_time
            
            positions_3d = np.hstack([positions, np.zeros((N, 1))])
            targets_3d = np.hstack([targets, np.zeros((N, 1))])
            metrics = evaluate_formation(positions_3d, targets_3d, comm_range=5.0)
            metrics['compute_time'] = elapsed
            metrics['method'] = 'ml_enhanced_with_model'
            
            return positions, metrics
        else:
            raise FileNotFoundError("Model not found")
    
    except Exception as e:
        logger.debug(f"ML model not available ({e}), using enhanced baseline")
        # Fallback to enhanced
        pos, metrics = baseline_enhanced(N, shape, num_frames, step_size, seed)
        metrics['method'] = 'ml_enhanced_fallback'
        return pos, metrics


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    num_scenarios: int = 10,
    shapes: List[str] = None,
    drone_counts: List[int] = None,
) -> Dict:
    """Run comprehensive benchmark across all baselines."""
    
    if shapes is None:
        shapes = ["grid", "circle", "v"]
    if drone_counts is None:
        drone_counts = [30, 60, 100]
    
    baselines = [
        ("scripted", baseline_scripted),
        ("enhanced", baseline_enhanced),
        ("ml_enhanced", baseline_ml_enhanced),
    ]
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": num_scenarios,
        "shapes": shapes,
        "drone_counts": drone_counts,
        "methods": [name for name, _ in baselines],
        "data": []
    }
    
    scenario_count = 0
    for shape in shapes:
        for N in drone_counts:
            for trial in range(num_scenarios):
                scenario_count += 1
                seed = scenario_count
                
                logger.info(f"Scenario {scenario_count:3d}: shape={shape:7s} N={N:3d} trial={trial+1}/{num_scenarios}")
                
                for method_name, method_func in baselines:
                    try:
                        _, metrics = method_func(N, shape, seed=seed)
                        
                        result = {
                            "scenario": scenario_count,
                            "shape": shape,
                            "N": N,
                            "trial": trial + 1,
                            "method": method_name,
                            "connectivity": metrics.get("connectivity_ratio", 0),
                            "min_distance": metrics.get("min_distance", 0),
                            "convergence_error": metrics.get("convergence_error", 0),
                            "compute_time": metrics.get("compute_time", 0),
                        }
                        results["data"].append(result)
                        
                        logger.debug(f"  {method_name:15s}: conn={result['connectivity']:.3f}, "
                                   f"min_dist={result['min_distance']:.3f}, time={result['compute_time']:.2f}s")
                    
                    except Exception as e:
                        logger.warning(f"  {method_name} failed: {e}")
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: Dict) -> None:
    """Print aggregated benchmark summary."""
    df = pd.DataFrame(results["data"])
    
    logger.info("\n" + "=" * 90)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 90)
    
    # Group by method
    for method in results["methods"]:
        method_data = df[df["method"] == method]
        
        logger.info(f"\n[{method.upper()}]")
        logger.info(f"  Avg Connectivity:     {method_data['connectivity'].mean():.4f} ± {method_data['connectivity'].std():.4f}")
        logger.info(f"  Avg Min Distance:     {method_data['min_distance'].mean():.4f} ± {method_data['min_distance'].std():.4f}")
        logger.info(f"  Avg Convergence Err:  {method_data['convergence_error'].mean():.6f} ± {method_data['convergence_error'].std():.6f}")
        logger.info(f"  Avg Compute Time:     {method_data['compute_time'].mean():.4f}s ± {method_data['compute_time'].std():.4f}s")
    
    # Improvements
    logger.info(f"\n[IMPROVEMENTS]")
    scripted_conn = df[df["method"] == "scripted"]["connectivity"].mean()
    enhanced_conn = df[df["method"] == "enhanced"]["connectivity"].mean()
    ml_conn = df[df["method"] == "ml_enhanced"]["connectivity"].mean()
    
    logger.info(f"  Enhanced vs Scripted:   {100*(enhanced_conn/scripted_conn - 1):+.2f}%")
    logger.info(f"  ML vs Scripted:         {100*(ml_conn/scripted_conn - 1):+.2f}%")
    
    logger.info("\n" + "=" * 90 + "\n")


def save_results(results: Dict, output_path: str = "benchmark_results.json") -> None:
    """Save results to JSON and CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {output_path}")
    
    # CSV
    csv_path = output_path.replace(".json", ".csv")
    df = pd.DataFrame(results["data"])
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for Drone Swarm AI baselines"
    )
    parser.add_argument("--scenarios", type=int, default=5,
                       help="Number of scenarios per (shape, N) combination (default: 5)")
    parser.add_argument("--shapes", nargs="+", default=["grid", "circle", "v"],
                       help="Shapes to test (default: grid circle v)")
    parser.add_argument("--drones", nargs="+", type=int, default=[30, 60, 100],
                       help="Drone counts to test (default: 30 60 100)")
    parser.add_argument("--output", type=str, default="output/benchmark_results.json",
                       help="Output JSON file (default: output/benchmark_results.json)")
    
    args = parser.parse_args()
    
    logger.info("=" * 90)
    logger.info("DRONE SWARM AI - BENCHMARK SUITE")
    logger.info("=" * 90)
    logger.info(f"Scenarios per config: {args.scenarios}")
    logger.info(f"Shapes: {args.shapes}")
    logger.info(f"Drone counts: {args.drones}")
    logger.info("")
    
    results = run_benchmark(
        num_scenarios=args.scenarios,
        shapes=args.shapes,
        drone_counts=args.drones
    )
    
    print_summary(results)
    save_results(results, args.output)
    
    logger.info(f"✓ Benchmark complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
