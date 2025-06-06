from __future__ import annotations  # To be removed in future python versions

import numpy as np
import os
from envs_.franka_human_env_ import FrankaHumanEnv_
from envs_.franka_human_env_ee_joints import End_Effector_Joints_Env_
from envs_.franka_human_env_ee_joints_sensors import End_Effector_Joints_Sensors_Env_
from envs_.franka_human_env_ee_rrmc import End_Effector_RRMC_Env_
from envs_.franka_human_env_ee import End_Effector_Env_







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
goal = np.array([0.45803004, 0.48489892, 0.31194257]) # np.array([0.4,0.4,0.5]) with the obstacle root position be [0.4, -0,2, 0.25] with the obstacle moving achievable, without the obstacle moving unreachable???
#[0.3,0.2,0.6]
dynamic_human = True
obstacle = True
env = FrankaHumanEnv_(scene_path,dynamic_human = dynamic_human, goal = goal, obstacle = obstacle) 
norm_dir = 2
env_name = "human_table" # "table_box" or "complex_table" or "human_table" or "cuboid_sphere" or "table" or "human"
#norm_dir = 0
#norm_dir = 2

if norm_dir == 1:
    #env = End_Effector_Joints_Env_(scene_path,dynamic_human=True, goal=goal, obstacle = True)
    start_positions = env.get_joints_end_position()
    print("end_effector_position: ", env.get_ee_position())
    print("goal: ", goal)
    print("start_positions shape: ", start_positions.shape)
    #env.render2()
    env.move_franka_robot_norm_dir(start_positions, goal,env_name)
    print("singular config number", env.singularities_number)
    print('------------------------------------')
    print("start replay")
    print("collision number: ", env.collision_num)
    env.replay()
    #env.render2()
    
elif norm_dir == 2:
    #env = End_Effector_Joints_Sensors_Env_(scene_path,dynamic_human=True, goal=goal, obstacle = True)
    #start_positions = env.get_sensors_end_position()
    #print("start_positions shape: ", start_positions.shape)
    if obstacle == True:
        start_positions = env.get_joints_sensors_end_position()
        print("end_effector_position: ", env.get_ee_position())
        print("goal: ", goal)
        env.move_franka_robot_sensors_norm_dir(start_positions, goal,env_name)
    elif obstacle == False:
        start_positions = env.get_joints_end_position()
        print("end_effector_position: ", env.get_ee_position())
        print("goal: ", goal)
        env.move_franka_robot_norm_dir(start_positions, goal)
    print("singular config number", env.singularities_number)
    print('------------------------------------')
    print("start replay")
    print("collision number: ", env.collision_num)
    print("sensors activation number", env.sensors_activation_num)
    print("sensors name: ", env.adjust_sensors_name)
    print("collision name: ", env.collision_name)
    print("time per step: ", np.mean(env.time_storage))
    #print("velocity average: ", np.mean(env.vel_storage, axis=0))
    env.replay()

elif norm_dir == 0:
    #env = End_Effector_RRMC_Env_(scene_path,dynamic_human=True, goal=goal, obstacle = True)
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



        
    

    

