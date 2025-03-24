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
#scene_path = "/home/aniri/nonlinear_obstacle_avoidance/franka_gym_test/envs2/franka_emika_panda/scene2.xml"
# good example
# goal = np.array([0.4, 0.2, 0.5]) with the obstacle root position be [0.4, -0,2, 0.25] good comparision
# goal = np.array([0.4, 0.3, 0.5]) with the obstacle root position be [0.4, -0,2, 0.25] unreachable for norm dir
goal = np.array([0.3, 0.2, 0.3])
goal = np.array([0.4, 0.4, 0.5]) # np.array([0.4,0.4,0.5]) with the obstacle root position be [0.4, -0,2, 0.25] with the obstacle moving achievable, without the obstacle moving unreachable???
dynamic_human = True
obstacle = True
it_max = 30
it = 0
collision_storage = []
reach_goal = []
time_storage = []
while it < it_max:
    # randomize the goal in range of [0.4, 0.4, 0.4] to [0.9, 0.9, 0.9]
    print("scenes number: ", it)
    goal = np.random.rand(3)*0.4 + 0.3
    env = FrankaMultiObsEnv_(scene_path,dynamic_human=dynamic_human, goal=goal, obstacle = obstacle)

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

    # start_positions = env.get_joints_end_position()
    # print("end_effector_position: ", env.get_ee_position())
    # print("goal: ", goal)
    # env.move_franka_robot_norm_dir(start_positions, goal)

    print("singular config number", env.singularities_number)
    print('------------------------------------')
    print("start replay")
    print("collision number: ", env.collision_num)
    print("sensors activation number", env.sensors_activation_num)
    print("sensors name: ", env.adjust_sensors_name)
    print("collision name: ", env.collision_name)
    collision_storage.append(["goal",goal, "collision num", env.collision_num])
    reach_goal.append(["goal",goal, "reach_goal", env.reach_goal])
    time_storage.append(np.mean(env.time_storage))
    print("collision_storage: ", collision_storage)
    print("reach_goal: ", reach_goal)
    print("time_storage: ", time_storage)
    it += 1
    #print("velocity average: ", np.mean(env.vel_storage, axis=0))
    #env.replay()
if len(collision_storage) == len(env.reach_goal):
    for i in range(len(collision_storage)):
        print("collision_storage: ", collision_storage[i])
        print("reach_goal: ", reach_goal[i])
        print("time_storage: ", time_storage[i])
else:
    print("collision_storage: ", collision_storage)
    print("reach_goal: ", reach_goal)
    print("time_storage: ", time_storage)


        
    

    

