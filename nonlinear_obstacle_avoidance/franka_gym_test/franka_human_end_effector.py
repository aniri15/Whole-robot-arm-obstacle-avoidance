from __future__ import annotations  # To be removed in future python versions

import numpy as np
import matplotlib.pyplot as plt
from envs.franka_human_env import FrankaHumanEnv
from franka_human_avoider import MayaviAnimator
import cv2




Vector = np.ndarray

global ctrl
global rot_ctrl



# initializations
scene_path = "/home/aniri/nonlinear_obstacle_avoidance/franka_gym_test/envs2/franka_emika_panda/scene2.xml"
goal = np.array([0.4, 0.4, 0.5])
env = FrankaHumanEnv(scene_path,dynamic_human=True, goal = goal) 
test_data = True
#test_data = False

start_position = np.zeros((1,3))

if test_data:
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
    
# test whether the point model can reach the goal
else:
    end_effector_position = env.get_ee_position()
    #goal = env.get_goal_position()
    #goal = np.array([1, 0.5, 0.3])
    env.bulid_goal_object(goal)
    # switch the tuple index to numpy array
    start_position[0] = np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2]])
    #goal = np.array([goal[0], goal[1], goal[2]])
    print("end_effector_position: ", start_position)
    print("goal: ", goal)
    env.move_point(start_position, goal)
    print('------------------------------------')
    print("start replay")
    env.render_point_trajectory()



        
    

    

