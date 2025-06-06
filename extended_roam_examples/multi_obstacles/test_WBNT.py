from __future__ import annotations  # To be removed in future python versions

import numpy as np
import os
from envs_.franka_multi_obs_env_ import FrankaMultiObsEnv_
from envs_.franka_human_env_ import FrankaHumanEnv_







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
it_max = 100
it = 0
collision_storage = []
reach_goal = []
time_storage = []
f = open("results.txt", "a")
reach_num = 0
scene_no_collision = 0
success_scene = 0

while it < it_max:
    # randomize the goal in range of [0.3, 0.3, 0.4] to [0.6, 0.6, 0.6]
    print("scenes number: ", it)
    start_x = 0.2
    range_x = 0.2
    start_y = -0.4
    range_y = -0.2
    start_z = 0.4
    range_z = 0.2
    goal = np.zeros(3)
    x = np.random.rand(1)*range_x + start_x
    y = np.random.rand(1)*range_y + start_y
    z = np.random.rand(1)*range_z + start_z
    goal[0] = x
    goal[1] = y
    goal[2] = z
    #env = FrankaHumanEnv_(scene_path,dynamic_human=dynamic_human, goal=goal, obstacle = obstacle)
    env = FrankaMultiObsEnv_(scene_path,dynamic_human=dynamic_human, goal=goal, obstacle = obstacle)

    env_name = "cuboid_sphere" # "table_box" or "complex_table" or "human_table" or "cuboid_sphere" or "table"
    if obstacle == True:
        start_positions = env.get_joints_sensors_end_position()
        #start_positions = env.get_joints_end_position()
        print("end_effector_position: ", env.get_ee_position())
        print("goal: ", goal)
        env.move_franka_robot_sensors_norm_dir(start_positions, goal, env_name)
        #env.move_franka_robot_norm_dir(start_positions, goal, env_name)
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
    if env.reach_goal[0] == True:
        reach_num += 1
    if env.collision_num == 0:
        scene_no_collision += 1
    if env.reach_goal[0] == True and env.collision_num == 0:
        success_scene += 1

    f.write("ith " + str(it) + " goal: " + str(goal) + "\n")
    f.write("scene: " + str(env_name) + "\n")
    f.write("dynamic: " + str(dynamic_human) + "\n")
    f.write("collision number: " + str(env.collision_num) + "\n")
    f.write("reach_goal: " + str(env.reach_goal) + "\n")
    f.write("time per step: " + str(np.mean(env.time_storage)) + "\n")
    f.write("-------------------------------------" + "\n")
    it += 1
    #print("velocity average: ", np.mean(env.vel_storage, axis=0))
    #env.replay()
f.write("goal range x : " + str(start_x) + " " + str(range_x) + "\n")
f.write("goal range y: " + str(start_y) + " " + str(range_y) + "\n")
f.write("goal range z: " + str(start_z) + " " + str(range_z) + "\n")
f.write("total number of reach goal: " + str(reach_num) + "\n")
f.write("total number of no collision: " + str(scene_no_collision) + "\n")
f.write("total number of success scene: " + str(success_scene) + "\n")
f.write("---------------------------------------------------------------------------------------------" + "\n")
f.close()
if len(collision_storage) == len(reach_goal):
    for i in range(len(collision_storage)):
        print("collision_storage: ", collision_storage[i])
        print("reach_goal: ", reach_goal[i])
        print("time_storage: ", time_storage[i])
else:
    print("collision_storage: ", collision_storage)
    print("reach_goal: ", reach_goal)
    print("time_storage: ", time_storage)


        
    

    

