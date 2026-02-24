import numpy as np

def is_fully_connected(positions, comm_range):
    N = len(positions)

    # Build adjacency matrix
    adjacency = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= comm_range:
                    adjacency[i][j] = 1

    # BFS to check connectivity
    visited = set()
    stack = [0]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            neighbors = np.where(adjacency[node] == 1)[0]
            stack.extend(neighbors)

    return len(visited) == N