# Whole-Body Obstacle Avoidance via Rotational Dynamics

This repository provides the simulation and implementation for the Master's thesis:

> **Whole-Body Obstacle Avoidance via Rotational Dynamics Along Normal and Tangent Directions**,  
> Aniri, Department of Informatics, Technical University of Munich, 2025.

The method integrates **normal and tangent direction control**, and **convergence dynamics**, **rotation-tree-based summing** to achieve reactive and smooth obstacle avoidance for full-body robots in complex static and dynamic environments.

---

## 🛠 Installation
The system is tested on **Ubuntu 20.04** with Python ≥ 3.9.  
Simulation environment: **MuJoCo**.  
Recommended environment manager:  `conda`.

### 1. Clone the Repository
```bash
git clone https://github.com/aniri15/Whole-robot-arm-obstacle-avoidance.git
cd Whole-robot-arm-obstacle-avoidance/
```

### 2. Setup Python Environment
```bash
conda create -n wbnt python=3.10.0
conda activate wbnt
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Install Required Submodules
```bash
pip install "git+https://github.com/hubernikus/various_tools.git"
pip install "git+https://github.com/hubernikus/dynamic_obstacle_avoidance.git"
```

---

##  Run the Demo Simulation

This project provides a simulation in MuJoCo for the Franka Emika Panda robot with different multi-obstacle scenarios.

###  Prerequisites
Make sure the `envs_` module and MuJoCo are correctly installed and configured.

### ▶️ Run Simulation
```bash
python whole_robot_arm_multi_obstacles_avoidance/extended_roam_examples/multi_obstacles/whole_body_franka_multi_obstacles.py
```

###  Options
- `goal`: 3D target position of the end-effector
- `dynamic_human`: whether human-shaped obstacles are dynamic (`True/False`)
- `env_name`: choose from `{table_box, human_table, cuboid_sphere, complex_table}`


###  Output Metrics

During and after execution, the following metrics will be printed:
- Initial and final end-effector position
- Number of collisions and activated sensors
- Average computation time per step
- Whether the robot reached the goal
- Number of singular configurations
- Sensors and links involved in collision or avoidance
- A replay of the whole trajectory via `env.replay()`


### Videos and figures

Demo videos and screenshots are also stored in folder `videos` and `figures`.

---

##  Method Overview

The WBNT (Whole-Body Normal/Tangent) avoidance method integrates:
- Rotational blending of local dynamics
- Convergence dynamics propagation through rotation trees
- Surface point and repulsion direction generation
- Tangent dynamics for motion preservation
- Normal repulsion for collision avoidance

Implemented using a modular control pipeline and geometry-aware vector field construction.

---

##  Acknowledgement

We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.
- [nonlinear_obstacle_avoidance](https://github.com/hubernikus/nonlinear_obstacle_avoidance)
- [dynamic_obstacle_avoidance](https://github.com/hubernikus/dynamic_obstacle_avoidance)
- [various_tools](https://github.com/hubernikus/various_tools)

## Citation

For related work:
```bibtex
@phdthesis{huber2024exact,
  title={Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation},
  author={Huber, Lukas},
  year={2024},
  address={Lausanne, Switzerland},
  school={EPFL},
  type={PhD thesis}
}
```

---


## 📬 Contact

For any questions or collaborations, feel free to reach out to:

**Aniri**  
Master's Thesis, TUM Informatics  
Email: *anirigermany@gmail.com]*
