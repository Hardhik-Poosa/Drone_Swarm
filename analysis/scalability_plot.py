import matplotlib.pyplot as plt

N_values = [5, 15, 20, 25, 30, 50]
connectivity = [0.50, 0.1429, 0.1053, 0.0833, 0.0690, 0.0408]
convergence = [0.0001, 0.0004, 0.0006, 0.0007, 0.0009, 0.0016]

plt.figure()
plt.plot(N_values, connectivity, marker='o')
plt.title("Connectivity vs Drone Count")
plt.xlabel("Number of Drones")
plt.ylabel("Connectivity Ratio")
plt.grid()
plt.show()

plt.figure()
plt.plot(N_values, convergence, marker='o', color='orange')