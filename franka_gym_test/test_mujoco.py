import time

import mujoco
import mujoco.viewer
from gymnasium_robotics.utils import mujoco_utils
import os

#different file name
fullpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs", "assets", "franka_test.xml")
m = mujoco.MjModel.from_xml_path(fullpath)
d = mujoco.MjData(m)

def get_geom_size(name):
        geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        return m.geom_size[geom_id]

def get_obstacle_info():
        _mujoco = mujoco
        _utils = mujoco_utils
        obstacles = []
        total_obstacles = 7 
        for n in range(total_obstacles):
            obstacles.append({
                'pos': _utils.get_joint_qpos(m, d, f'obstacle{n}' + ':joint')[:3],
                'size': get_geom_size(f'obstacle{n}')
            })
        return obstacles

obstacles = get_obstacle_info()
print(obstacles)


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 80:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

# test functions in mujoco



