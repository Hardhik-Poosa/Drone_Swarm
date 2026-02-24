import numpy as np

def pairwise_distances(points):
    dists = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(points[i] - points[j]))
    return np.array(dists)


def evaluate_formation(final_positions, target_positions, comm_range=5.0):
    distances = pairwise_distances(final_positions)

    min_dist = distances.min()
    avg_dist = distances.mean()

    # Connectivity: % of drone pairs within communication range
    connected_pairs = np.sum(distances <= comm_range)
    total_pairs = len(distances)
    connectivity_ratio = connected_pairs / total_pairs

    # Convergence error
    convergence_error = np.mean(
        np.linalg.norm(final_positions - target_positions, axis=1)
    )

    return {
        "Minimum Inter-Drone Distance": min_dist,
        "Average Inter-Drone Distance": avg_dist,
        "Connectivity Ratio": connectivity_ratio,
        "Convergence Error": convergence_error
    }
