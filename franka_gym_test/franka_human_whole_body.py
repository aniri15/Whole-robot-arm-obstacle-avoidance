from __future__ import annotations  # To be removed in future python versions

import numpy as np
import os
from envs.franka_human_env import FrankaHumanEnv







Vector = np.ndarray

global ctrl
global rot_ctrl



# initializations
folder_path = os.path.dirname(os.path.abspath(__file__))
scene_path = folder_path + "/envs2/franka_emika_panda/scene2.xml"
print("scene_path: ", scene_path)
#scene_path = "/home/aniri/nonlinear_obstacle_avoidance/franka_gym_test/envs2/franka_emika_panda/scene2.xml"
# good example
# goal = np.array([0.4, 0.2, 0.5]) with the obstacle root position be [0.4, -0,2, 0.25] good comparision
# goal = np.array([0.4, 0.3, 0.5]) with the obstacle root position be [0.4, -0,2, 0.25] unreachable for norm dir
goal = np.array([0.3, 0.2, 0.3])
goal = np.array([0.15, 0.6, 0.3]) # np.array([0.4,0.4,0.5]) with the obstacle root position be [0.4, -0,2, 0.25] with the obstacle moving achievable, without the obstacle moving unreachable???
env = FrankaHumanEnv(scene_path,dynamic_human=True, goal=goal, obstacle = True) 
norm_dir = 1
#norm_dir = 0
norm_dir = 2
#norm_sensor = np.abs(1-norm_dir)
#norm_sensor = True

if norm_dir == 1:
    start_positions = env.get_joints_end_position()
    print("end_effector_position: ", env.get_ee_position())
    print("goal: ", goal)
    print("start_positions shape: ", start_positions.shape)
    #env.render2()
    env.move_franka_robot_norm_dir(start_positions, goal)
    print("singular config number", env.singularities_number)
    print('------------------------------------')
    print("start replay")
    print("collision number: ", env.collision_num)
    env.replay()
    #env.render2()
    
elif norm_dir == 2:
    #start_positions = env.get_sensors_end_position()
    start_positions = env.get_joints_sensors_end_position()
    print("end_effector_position: ", env.get_ee_position())
    print("goal: ", goal)
    #print("start_positions shape: ", start_positions.shape)
    env.move_franka_robot_sensors_norm_dir(start_positions, goal)
    print("singular config number", env.singularities_number)
    print('------------------------------------')
    print("start replay")
    print("collision number: ", env.collision_num)
    print("sensors activation number", env.sensors_activation_num)
    print("sensors name: ", env.adjust_sensors_name)
    print("collision name: ", env.collision_name)
    #print("velocity average: ", np.mean(env.vel_storage, axis=0))
    env.replay()

elif norm_dir == 0:
    start_position = np.zeros((1,3))
    end_effector_position = env.get_ee_position()
    #goal = env.get_goal_position()
    #goal = np.array([0.3, 0.3, 0.3])
    #env.bulid_goal_object(goal)
    # switch the tuple index to numpy array
    start_position[0] = np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2]])
    #goal = np.array([goal[0], goal[1], goal[2]])
    print("end_effector_position: ", start_position)
    print("goal: ", goal)
    #env.render2()
    env.move_robot(start_position, goal)
    print("singular config number", env.singularities_number)
    print('------------------------------------')
    print("start replay")
    env.replay()
    #env.render2()



        
    

    

