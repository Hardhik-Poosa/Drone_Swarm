# 🚀 Drone Swarm AI — 2D to 3D Intelligent Formation System

## 🎯 What Is This Project?

**Drone Swarm AI** is an advanced simulation system that transforms any image into intelligent 3D drone formations. Using computer vision and depth estimation AI, it can:

- Extract human silhouettes or objects from real photos
- Convert 2D outlines into full 3D volumetric structures
- Automatically assign 300+ drones to formation waypoints using optimal assignment algorithms
- Simulate realistic physics with collision avoidance
- Export choreographies to CSV for real-world drone or visualization engines

This is **not just a visualization tool** — it's a foundation for real-world drone choreography systems and swarm intelligence research.

---

## 🔬 How It Works

### **Step 1: Semantic Image Understanding**
- Uses **DeepLabV3 (ResNet-101)** trained on COCO dataset
- Automatically extracts person silhouettes from photos
- Ignores background clutter — focuses only on the subject
- Generates a 2D outline of the detected object

### **Step 2: Optimal Drone Placement (2D)**
- Converts the 2D outline into exactly `N` drone positions
- Uses the **Hungarian Algorithm** to optimally assign random drones to target positions
- Minimizes total assignment cost (distance traveled)
- Ensures no two drones are assigned to the same waypoint

### **Step 3: 3D Depth Estimation**
- Uses **MiDaS DPT-Large** for monocular depth estimation
- Analyzes depth at each 2D position
- Creates a Z-coordinate for each drone based on image depth
- Results: Full 3D coordinates from a single 2D image

### **Step 4: Volumetric 3D Filling**
- Stacks multiple layers of drones to create a "solid" 3D shape
- Configurable layers allow control over thickness
- Cleans outliers using statistical distance analysis
- Result: A complete 3D drone sculpture

### **Step 5: Physics Simulation**
- **Attraction Force**: Each drone is pulled toward its target waypoint
- **Repulsion Force**: Drones repel each other with 1/d² physics
- **Collision Avoidance**: Prevents drones from overlapping
- Iterative convergence over 150+ frames
- Real-time 3D visualization with matplotlib

### **Step 6: Data Export**
- Exports all drone (X, Y, Z) coordinates to CSV
- Compatible with Unity for real-time visualization
- Can be imported into other robotics/game engines

---

## ✨ Key Features

### **AI-Powered Computer Vision**
- ✅ DeepLabV3 semantic segmentation for automatic object detection
- ✅ MiDaS depth estimation (single image → 3D structure)
- ✅ Background removal and silhouette extraction
- ✅ Scale-invariant formation generation

### **Intelligent Drone Assignment**
- ✅ Hungarian Algorithm for optimal O(n³) assignment
- ✅ Minimizes total travel distance
- ✅ Guarantees one-to-one drone-to-waypoint mapping
- ✅ Globally optimal solution (not greedy)

### **Advanced Physics Simulation**
- ✅ 3D attraction forces (toward target positions)
- ✅ Velocity-based collision avoidance
- ✅ Realistic inverse-square law repulsion
- ✅ Adaptive step sizes for smooth convergence
- ✅ Statistical outlier removal

### **Multi-Scale Formation Control**
- ✅ Adjustable drone count (100-1000+)
- ✅ Configurable scale factors for image size
- ✅ Tunable height scaling for depth emphasis
- ✅ Layered 3D structure with variable thickness

### **Swarm Quality Metrics**
- ✅ Minimum inter-drone distance (collision safety)
- ✅ Average inter-drone distance (cohesion)
- ✅ Connectivity ratio (network graph density)
- ✅ Convergence error (formation accuracy)
- ✅ Energy consumption estimates

### **Research & ML Extensions**
- ✅ VAE model for learning formation patterns
- ✅ Constraint-aware loss functions (collision + connectivity)
- ✅ Dataset generation pipeline
- ✅ Scalability analysis tools
- ✅ Network topology analysis

---

## 📋 Project Structure

```
Drone_Swarm/
│
├── sim/                          # Simulation engines
│   ├── swarm_sim.py             # 2D formation simulation
│   ├── swarm_sim_3d.py          # 3D volumetric simulation (MAIN)
│   ├── generate_new_formations.py
│   └── visualize_formations.py
│
├── utils/                        # Core AI & physics modules
│   ├── semantic_image_to_formation.py    # DeepLabV3 + outline extraction
│   ├── depth_to_3d.py                    # MiDaS depth estimation
│   ├── formation_3d.py                   # 3D lifting & volumetric filling
│   ├── shape_generator.py                # Predefined geometric shapes
│   ├── evaluate_swarm.py                 # Metrics computation
│   ├── metrics.py                        # ML loss functions
│   ├── image_to_formation.py             # Image processing
│   └── network_metrics.py                # Connectivity analysis
│
├── models/                      # Machine learning models
│   ├── vae.py                  # Variational Autoencoder
│   └── __init__.py
│
├── train/                       # Training pipelines
│   └── train_vae.py            # VAE training with constraints
│
├── analysis/                    # Research & experiments
│   ├── swarm_core.py           # Core simulation loop
│   ├── train_connectivity_model.py
│   ├── optimize_spacing.py     # Spacing optimization
│   ├── scalability_plot.py     # Performance scaling analysis
│   ├── network_analysis.py     # Topology analysis
│   ├── generate_ml_dataset.py  # Dataset generation
│   └── swarm_ml_dataset.csv    # Generated dataset
│
├── data/                        # Formation datasets
│   ├── formations.npy          # NumPy array of reference formations
│   └── generate_formations.py  # Dataset generation
│
├── input_images/               # Your input images here
│   └── (place image files here)
│
├── output/                     # Generated outputs
│   └── drone_positions_3d.csv  # Final drone coordinates
│
├── vae_model.pth              # Trained VAE weights
└── README.md                  # This file
```

---

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM recommended

### **1. Clone Repository**
```bash
git clone https://github.com/Hardhik-Poosa/Drone_Swarm.git
cd Drone_Swarm
```

### **2. Create Virtual Environment**

**Option A: Conda (Recommended)**
```bash
conda create -n drone python=3.10
conda activate drone
```

**Option B: venv**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install numpy matplotlib scipy opencv-python torch torchvision timm pandas
```

**For GPU acceleration (optional):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Quick Start

### **Example 1: 3D Formation from Image (Recommended)**

```bash
python sim/swarm_sim_3d.py
```

**What happens:**
1. Loads `input_images/516AaQ6o17L.webp` (default test image)
2. Extracts person silhouette using DeepLabV3
3. Generates 2D outline with 300 drones
4. Estimates depth map using MiDaS
5. Lifts to 3D with 4 layers (volumetric filling)
6. Simulates 150 frames of drone convergence with physics
7. Shows 3D animation in real-time
8. Exports coordinates to `output/drone_positions_3d.csv`

**Output:**
- 3D scatter plot animation
- CSV file with final positions

### **Example 2: 2D Formation Simulation**

```bash
python sim/swarm_sim.py
```

**What happens:**
1. Loads image and extracts 2D silhouette
2. Creates 300 drone waypoints
3. Assigns random starting drones using Hungarian Algorithm
4. Animates convergence with attraction forces
5. Prints evaluation metrics

### **Example 3: Using Your Own Image**

1. Place your image in `input_images/` folder
   ```
   input_images/
   ├── your_photo.jpg
   └── another_image.png
   ```

2. Edit `sim/swarm_sim_3d.py`:
   ```python
   IMAGE_PATH = "input_images/your_photo.jpg"
   BASE_DRONES = 300        # Number of drones
   SCALE_FACTOR = 6         # Image scale (higher = more detail)
   HEIGHT_SCALE = 10        # Depth emphasis (higher = more 3D)
   LAYERS = 4               # 3D thickness (more = fuller shape)
   ```

3. Run:
   ```bash
   python sim/swarm_sim_3d.py
   ```

4. Results saved to `output/drone_positions_3d.csv`

---

## 📊 Configuration Parameters

### **In `sim/swarm_sim_3d.py`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_PATH` | `input_images/516AaQ6o17L.webp` | Path to input image |
| `BASE_DRONES` | 300 | Number of drones in 2D outline |
| `SCALE_FACTOR` | 6 | Image resolution scaling (higher = more detail) |
| `HEIGHT_SCALE` | 10 | Depth map scaling for Z-axis (higher = deeper structure) |
| `LAYERS` | 4 | Number of volumetric layers (higher = thicker 3D shape) |
| `step_size` | 0.05 | Attraction force magnitude per frame |
| `repulsion_strength` | 0.02 | Repulsion force multiplier |
| `repulsion_radius` | 0.6 | Distance at which drones repel each other |
| `num_frames` | 150 | Number of simulation frames |

### **Tuning Tips:**

- **For finer detail**: Increase `SCALE_FACTOR` (slower convergence)
- **For deeper 3D structure**: Increase `HEIGHT_SCALE`
- **For smoother convergence**: Increase `num_frames` and decrease `step_size`
- **For thicker volumes**: Increase `LAYERS`
- **Performance**: Reduce `BASE_DRONES` if simulation is slow

---

## 📈 Understanding the Output

### **CSV Format: `output/drone_positions_3d.csv`**

```
x,y,z
-2.345,1.234,0.567
-2.234,1.456,0.678
...
```

Each row is one drone's final (X, Y, Z) coordinate. Can be imported into:
- **Unity** for real-time visualization and light shows
- **Blender** for 3D rendering
- **ROS** for real robot simulation
- **Custom game engines** for interactive drone shows

---

## 📊 Evaluation Metrics

The system automatically computes swarm quality metrics:

```
Minimum Inter-Drone Distance:  0.45 units
Average Inter-Drone Distance:  1.23 units
Connectivity Ratio:            0.92 (92% of drones within comm range)
Convergence Error:             0.08 units
```

**What these mean:**
- **Min distance**: Safety margin (>0.5 is good for collision avoidance)
- **Avg distance**: Cohesion measure (balanced spacing)
- **Connectivity**: Network topology (>0.8 is good)
- **Convergence**: Formation accuracy (lower is better)

---

## 🧠 Technology Deep Dive

### **DeepLabV3 Semantic Segmentation**
- Pre-trained on COCO dataset (80 classes)
- Extracts "person" class (class ID 15)
- Generates binary mask automatically
- Fast inference: ~100ms per image on CPU

### **MiDaS Depth Estimation**
- DPT-Large backbone for high-quality depth
- Works on any single image (monocular)
- No calibration needed
- Relative depth used for Z-coordinate

### **Hungarian Algorithm**
- O(n³) optimal assignment
- Minimizes total travel distance
- Used here before simulation starts
- Ensures efficient drone-to-waypoint mapping

### **3D Physics**
- **Attraction**: `F = target_pos - current_pos`
- **Repulsion**: `F = Σ(diff / dist³)` for neighbors within radius
- **Integration**: `pos_new = pos + step_size * F_net`
- **Stability**: Tuned for smooth convergence without oscillation

---

## 🔬 Advanced Usage

### **Train Custom VAE Model**

```bash
python train/train_vae.py
```

Learns formation patterns from dataset with constraints:
- Collision avoidance loss
- Connectivity loss

### **Analyze Network Topology**

```bash
python analysis/network_analysis.py
```

Computes graph metrics:
- Degree distribution
- Clustering coefficient
- Average path length

### **Scalability Testing**

```bash
python analysis/scalability_plot.py
```

Tests performance with 100-1000 drones, generates plots.

---

## ⚠️ Troubleshooting

### **"ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch torchvision
```

### **"No module named 'deeplabv3_resnet101'"**
- Ensure `torchvision` version matches `torch`
- Reinstall: `pip install --upgrade torch torchvision`

### **"Cannot find MiDaS model"**
- First run downloads MiDaS (requires internet)
- Large download (~500MB), be patient
- Cached locally after first run

### **OOM (Out of Memory) Errors**
- Reduce `BASE_DRONES` (try 100 instead of 300)
- Reduce `SCALE_FACTOR` (try 3 instead of 6)
- Use CPU only: Remove CUDA requirement

### **Slow Convergence**
- Increase `step_size` (default 0.05 → try 0.1)
- Decrease `num_frames` (default 150 → try 100)
- Use simpler images with fewer details

---

## 🎨 Visualization Tips

### **Export to Unity**

1. Run simulation (generates `output/drone_positions_3d.csv`)
2. In Unity:
   ```csharp
   var positions = CSVReader.Read("drone_positions_3d.csv");
   foreach(var pos in positions) {
       Instantiate(dronePrefab, new Vector3(pos.x, pos.y, pos.z), Quaternion.identity);
   }
   ```

### **Render in Blender**

1. Export CSV
2. Import into Blender via script:
   ```python
   import csv
   with open('drone_positions_3d.csv') as f:
       reader = csv.DictReader(f)
       for row in reader:
           bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(float(row['x']), float(row['y']), float(row['z'])))
   ```

---

## 📚 Research Extensions

Potential directions for further development:

- **Reinforcement Learning**: Train agents to maintain formation autonomously
- **Music Synchronization**: Choreograph drone movements to music beats
- **Real Quadcopter Integration**: Export trajectories to Crazyflies or DJI drones
- **Multi-Person**: Handle multiple silhouettes in one image
- **Motion Sequences**: Create frame-by-frame animations from video
- **Energy Optimization**: Minimize total drone energy consumption
- **XR/AR Integration**: Real-time visualization in AR glasses

---

## 📜 Code Quality & Testing

- Type hints used throughout
- Modular architecture for easy extension
- Unit tests in `/tests` (run with `pytest`)
- Documentation with docstrings
- Example scripts provided

---

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@software{poosa2024droneswarm,
  author = {Poosa, Hardhik},
  title = {Drone Swarm AI: 2D to 3D Intelligent Formation System},
  year = {2024},
  url = {https://github.com/Hardhik-Poosa/Drone_Swarm}
}
```

---

## 📄 License

This project is provided for educational and research purposes.

---

## 👨‍💻 Author & Contact

**Hardhik Poosa**  
B.Tech Computer Science & Engineering  
AI & Swarm Intelligence Research Enthusiast

📧 GitHub: [github.com/Hardhik-Poosa](https://github.com/Hardhik-Poosa)  
🔗 Project: [Drone Swarm AI Repository](https://github.com/Hardhik-Poosa/Drone_Swarm)

---

## 🙏 Acknowledgments

- **DeepLabV3**: Facebook Research / PyTorch
- **MiDaS**: Intel ISL
- **Hungarian Algorithm**: SciPy optimization
- **Image Processing**: OpenCV & PIL
- **Scientific Computing**: NumPy, Matplotlib, Pandas

---

**Last Updated**: February 25, 2026  
**Status**: Active Development  
**Python Version**: 3.10+

---

🚀 **Transform any image into an intelligent drone choreography. Start building today!**
