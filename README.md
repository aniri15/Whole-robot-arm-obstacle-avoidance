# Whole-Body Obstacle Avoidance - via Rotational Dynamics Along Normal and Tangent Directions

This repository implements a whole-body obstacle avoidance method using **rotational dynamics along normal and tangent directions**, developed as part of a master's thesis at the Technical University of Munich (TUM). The method extends the ROAM (Rotational Obstacle Avoidance Method) framework to multi-link robot arms and concave/dynamic obstacles.

# Getting Started

## Installation

The system is tested on **Ubuntu 20.04** with Python ≥ 3.9.  
Simulation environment: **MuJoCo**.  
Recommended environment manager: `venv` or `conda`.

Install the code and dependencies:

```bash
git clone https://github.com/your-username/wbnt-obstacle-avoidance.git
cd wbnt-obstacle-avoidance
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Additional dependencies:

```bash
pip install "git+https://github.com/hubernikus/various_tools.git"
pip install "git+https://github.com/hubernikus/dynamic_obstacle_avoidance.git"
```

## Implementation

You can run the full-body avoidance simulation using:

```bash
python3 whole_robot_arm_multi_obstacles_avoidance/extended_roam_examples/multi_obstacles/whole_body_franka_multi_obstacles.py
```

Make sure your `PYTHONPATH` includes the project root directory or install as editable (`pip install -e .`).

# Running Evaluations

- Locally straight convergence dynamics construction  
- Rotation tree summing algorithm  
- Surface point and velocity propagation  
- Repulsion via tangent and normal directions  
- Support for multiple concave and dynamic obstacles  
- Whole-body manipulation using detection points and joint-space control

# Acknowledgement

This method builds on the ideas from the ROAM framework:

- Huber, Lukas. _Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation_, EPFL, 2024.  
- Repository: https://github.com/hubernikus/nonlinear_obstacle_avoidance

If you use this work, please cite:

```bibtex
@phdthesis{huber2024exact,
  title={Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation},
  author={Huber, Lukas},
  year={2024},
  month={April},
  address={Lausanne, Switzerland},
  school={EPFL},
  type={PhD thesis}
}
```
