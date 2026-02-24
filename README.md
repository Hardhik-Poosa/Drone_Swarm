🚀 **Drone Swarm AI — 2D to 3D Intelligent Formation System**

---

## Overview

This project simulates an intelligent drone swarm capable of transforming images into 2D and 3D drone formations using computer vision, depth estimation, and swarm physics. It is designed as a foundation for real-world drone choreography systems, enabling both visualization and export for real-time light shows.

---

## Features

✅ **2D Formations**
	- Predefined geometric shapes (circle, grid, line, etc.)
	- Image outline-based drone formations
	- Hungarian optimal drone assignment

✅ **AI Semantic Understanding**
	- Extracts person silhouette from real images
	- Ignores background automatically

✅ **3D Formation Generation**
	- Converts 2D formation to 3D using MiDaS depth estimation
	- Creates volumetric (solid) drone bodies
	- Adds 3D collision avoidance physics

✅ **Unity Integration**
	- Exports CSV coordinates for real-time drone spawning
	- Multi-color emission light show
	- Frame-by-frame animation playback

---

## Core Technologies

- Python
- NumPy
- OpenCV
- PyTorch
- DeepLabV3 (Semantic Segmentation)
- MiDaS (Depth Estimation)
- Hungarian Algorithm (Optimal Assignment)
- 3D Swarm Physics Simulation
- Unity (Real-time Visualization)

---

## Project Structure

```
DRONE_SWARM/
│
├── analysis/      # ML experiments & scaling studies
├── data/          # Formation datasets
├── models/        # VAE & ML models
├── output/        # Generated CSV files
├── sim/           # Simulation scripts (2D & 3D)
├── utils/         # Core logic (vision, depth, metrics)
│
├── README.md
```

---

## Installation

1. **Clone the Repository**
	 ```sh
	 git clone https://github.com/Hardhik-Poosa/Drone_Swarm.git
	 cd Drone_Swarm
	 ```

2. **Create Python Environment**
	 - Using Conda:
		 ```sh
		 conda create -n drone python=3.10
		 conda activate drone
		 ```
	 - Using venv:
		 ```sh
		 python -m venv venv
		 venv\Scripts\activate
		 ```

3. **Install Dependencies**
	 ```sh
	 pip install numpy matplotlib scipy opencv-python torch torchvision timm pandas
	 ```

---

## Usage

### 🟢 Run 2D Swarm Simulation

```sh
python sim/swarm_sim.py
```
*Generates 2D formation, applies Hungarian assignment, animates convergence, prints evaluation metrics.*

### 🔵 Run 3D AI Depth Formation

```sh
python sim/swarm_sim_3d.py
```
*Extracts silhouette from image, estimates depth using MiDaS, generates volumetric 3D swarm, applies collision avoidance, exports final positions to `output/drone_positions_3d.csv`.*

#### 🖼 Using Your Own Image

1. Place your image in `input_images/`
2. Open `sim/swarm_sim_3d.py`
3. Change:
	 ```python
	 IMAGE_PATH = "input_images/your_image.jpg"
	 ```
4. Run the script again.

---

## Swarm Evaluation Metrics

- Minimum Inter-Drone Distance
- Average Inter-Drone Distance
- Connectivity Ratio
- Convergence Error

---

## Advanced Features

- Hungarian assignment
- Semantic segmentation
- Depth estimation
- Volumetric 3D stacking
- 3D collision avoidance (1/d² repulsion)
- Cinematic rotating camera
- CSV export for external engines

---

## Research Extensions

- Reinforcement learning swarm control
- GPU accelerated 10,000+ drone simulation
- Music synchronized light choreography
- Real quadcopter trajectory generation
- Unity XR / AR visualization
- Blender mesh export

---

## Author

**Hardhik Poosa**  
B.Tech CSE  
AI & Swarm Intelligence Research Enthusiast

GitHub: [Hardhik-Poosa](https://github.com/Hardhik-Poosa)

---

Enjoy building intelligent drone swarms!
