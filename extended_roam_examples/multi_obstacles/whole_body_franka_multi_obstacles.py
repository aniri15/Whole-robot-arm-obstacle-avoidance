from __future__ import annotations  # To be removed in future python versions

import numpy as np
import os
from envs_.franka_multi_obs_env_ import FrankaMultiObsEnv_







Vector = np.ndarray

global ctrl
global rot_ctrl



# initializations
folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scene_path = folder_path + "/envs_/franka_emika_panda/scene2.xml"
print("scene_path: ", scene_path)
goal = np.array([0.32557378, 0.43178297, 0.44178945])
dynamic_human = True
obstacle = True

env = FrankaMultiObsEnv_(scene_path,dynamic_human=dynamic_human, goal=goal, obstacle = obstacle)
if env.check_collision():
    print("collision")
    breakpoint()
env_name = "human_table" # "table_box" or "complex_table" or "human_table" or "cuboid_sphere" or "table"
if obstacle == True:
    start_positions = env.get_joints_sensors_end_position()
    print("end_effector_position: ", env.get_ee_position())
    print("goal: ", goal)
    env.move_franka_robot_sensors_norm_dir(start_positions, goal, env_name)
elif obstacle == False:
    start_positions = env.get_joints_end_position()
    print("end_effector_position: ", env.get_ee_position())
    print("goal: ", goal)
    env.move_franka_robot_norm_dir(start_positions, goal)



print("singular config number", env.singularities_number)
print('------------------------------------')
print("start replay")
print("goal: ", goal)
print("collision number: ", env.collision_num)
print("sensors activation number", env.sensors_activation_num)
print("sensors name: ", env.adjust_sensors_name)
print("collision name: ", env.collision_name)
print("averaged time: ", np.mean(env.time_storage))
print("collision number: ", env.collision_num)
print("reach goal: ", env.reach_goal)
#print("velocity average: ", np.mean(env.vel_storage, axis=0))
env.replay()



        
    

    

