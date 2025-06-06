#from mujoco_py import load_model_from_xml
import mujoco
import mujoco.viewer
from gymnasium_robotics.utils import mujoco_utils
from typing import Optional
import mediapy as media
import time
import numpy as np
import matplotlib.pyplot as plt
from extended_roam_examples.multi_obstacles.franka_multi_obs_avoider import ROAM
import os
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import HyperSphere
#import pybullet
from scipy.spatial.transform import Rotation as R
import time
from itertools import accumulate
import mink

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.4,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([1.5, 0, 0.75]),
}
DEFAULT_SIZE = 480

class FrankaMultiObsEnv_():
    def __init__(
            self,
            scene_path,
            goal=None,
            control_mode='position',
            dynamic_human=False,
            obstacle = True,
            **kwargs
    ):
        self._mujoco = mujoco
        self._utils = mujoco_utils
        self.control_mode = control_mode
        #self.human_shape = HumanShape(pos=[0.5, 0.5, 0])  # Position human obstacle
        #self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_test.xml")
        self.fullpath = scene_path
        self.FRAMERATE = 60 #(Hz)
        self.height = 300
        self.width = 300
        self.n_frames = 60
        self.frames = []
        self.dynamic_human = dynamic_human
        self.obstacle = obstacle
        self.goal = goal

        # Initialize MuJoCo environment with the combined model
        #self.spec = mujoco.MjSpec.from_file(self.fullpath)
        self.spec = mujoco.MjSpec.from_file(self.fullpath)
        self.model = self.spec.compile()
        
        #self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        #self.model = load_model_from_xml(full_model_xml)
        self.render_mode = 'human'
        self.init_qpos = self.data.qpos.copy()
        

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 4
        self.it_max = 800      #500      #horizon

        # update step size
        self.step_size_position = 0.01
        self.step_size_ik = 0.1 #0.5
        self.delta_time = 0.03 #0.05     # 100hz

        # initialize for ik controller
        self.tol = 0.05   # unit: m, tolerance is 5cm
        self.length_tol = 0.01
        self.damping = 0.00001
        self.jacp = np.zeros((3, self.model.nv)) #translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) #rotational jacobian
        self.trajectory_qpos = []
        self.data_storage = []
        self.model_storage = []
        self.point_storage = []
        self.link_length_storage = []
        self.time_storage = []
        self.reach_goal = []
        self.sensors_activation_num = 0
        #self.obstacle_number = 7
        self.initial_qpos = {
            #'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],#[1.25, 0.53, 0.4, 1., 0., 0., 0.]
            'joint1': 0.144,
            'joint2': -0.2188,
            'joint3': 0.127,
            'joint4': -2.6851,
            'joint5': 0.0092,
            'joint6': 2.4664,
            'joint7': 0.157,
            'finger_joint1': 0.04,
            'finger_joint2': 0.04,
        }
        self.self_collision_pairs = [
            ("link5_sensor_2", "attachment"),
            #("link3", "link7"),
            
        ]
        self.rotation_joints = np.array([4]) #np.array([2,4,6])# rotation joints at the 2st, 4rd and 6th joints(at the end of the arms)
        self.rotation_joints_weights = np.array([0.1])
        self.ee_weight = 1
        self.avoid_weight = 1
        self.singularities_number = 0
        self.sample_points = 7
        self.sample_range = 0.05
        self.vel_storage = []
        self.weights_storage = []
        self.model_size_color_storage = []
        self.adjust_sensors_name = []
        self.collision_name = []
        self.collision_num = 0
        #self.initial_simulation()
        self.build_goal_object(goal)
        #self.build_inter_goal_object(goal)
        self.build_sensor_points()
        #self.build_less_sensor_points()
        self.collision_visualization()
        self.initial_simulation()
        #self.initial_ik()
        
    def initial_ik(self):
        # initial ik configuration
        self.configuration = mink.Configuration(self.model)
        # Define tasks
        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)
        self.tasks = [self.end_effector_task, self.posture_task]

    def initial_simulation(self):
        for name, value in self.initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        mujoco.mj_forward(self.model, self.data)


    '''
    # -------------------------------------------move ee in tangent direction-----------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    '''
    def move_robot(self,init_position, goal):
        #self.mujoco_renderer.render(self.render_mode)
        it_max = self.it_max
        self.animator = ROAM(it_max=it_max, current_position=init_position, 
                                  attractor_position=goal, dynamic_human=self.dynamic_human, obstacle=self.obstacle)    
        self.animator.setup()
        if self.obstacle:
            self.human_obstacle = self.animator.human_obstacle_3d
            self.adjust_obstacle_init()
        #self.animator.dynamic_human = self.dynamic_human


        for ii in range(it_max):
            time_start = time.time()
            #update trajectory
            velocity = self.animator.run(ii=ii)
            if not ii % 10:
                print(f"it={ii}")
            print('iteration: ', ii)
            print("velocity: ", velocity)
            for jj in range(len(velocity)):
                next_desired_pos = init_position[jj] + velocity[jj]*self.delta_time  #self.step_size_position
            #self.use_ik_controller(next_desired_pos)
            #self.panda_ik_controller(next_desired_pos)
            self.resolved_rate_motion_control(next_desired_pos, velocity[0])

            # update end effector position
            init_position = self.get_ee_position()
            init_position = np.array([[init_position[0], init_position[1], init_position[2]]])
            
            self.animator.update_trajectories(init_position, ii)
            #self.human_obstacle = self.animator.human_obstacle_3d
            #time_end = time.time()
            #print("time_diff: ", time_end - time_start)
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.adjust_obstacle()       
            time_end = time.time()
            print("time_diff: ", time_end - time_start)
            if self.check_goal_reach(init_position, goal):
                print("goal reached")
                break
            
    def resolved_rate_motion_control(self, goal, velocity):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        #body_id = self.model.site('robot0:grip').id
        
        print("current goal: ", goal)
        print("current init_q: ", init_q)
        print("current ee position: ", self.data.body(body_id).xpos)

        #self.data = self.RRMC(goal, init_q, body_id) #calculate the qpos
        self.data = self.RRMC(goal, velocity, init_q, body_id) #calculate the qpos
        #print("time: ", self.data.time)

    def RRMC(self, goal, velocity, init_q, body_id):
        self.data.qpos = init_q
        #mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        weights = self.animator.weight_ee
        self.weights_storage.append(weights)
        #self.trajectory_qpos.append(self.data.qpos.copy())
        self.store_data()
        jac = np.zeros((6, self.model.nv))
        time_start = self.data.time
        time_end = 0
        time_diff = 0
        

        while (np.linalg.norm(time_diff)<= self.delta_time):
            #calculate jacobian
            #mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], current_pose, body_id)
            j = jac
            #print("rank of jacobian: ", np.linalg.matrix_rank(j))
            if np.linalg.matrix_rank(j) < 6:
                print("jacobian is singular")
                self.singularities_number += 1
                breakpoint()
            #calculate delta of joint q
            current_pose = self.data.body(body_id).xpos
            #print("current_pose: ", current_pose)
            dv = self.desired_velocity(velocity)
            n = j.shape[1]
            I = np.identity(n)
            product = j.T @ j + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ j.T
            else:
                j_inv = np.linalg.inv(product) @ j.T

            #j_inv = np.linalg.pinv(j)
            delta_q = j_inv @ dv
          
            if np.isclose(np.linalg.norm(delta_q), 0):
                shape = delta_q.shape
                delta_q = np.eye(shape)*0.01


            self.data.qvel = delta_q

            #check limits
            self.check_joint_limits(self.data.qpos)

            mujoco.mj_step(self.model, self.data)
            error = np.subtract(goal, self.data.body(body_id).xpos)

            self.store_data()
            time_end = self.data.time
            time_diff = time_end - time_start
            #print("time_diff: ", time_diff)
        return self.data

    '''
    # ---------------------------------------move point model-----------------------------------------
    # -------------------------------------------------------------------------------------------------
    '''
    def move_point(self,init_position, goal):
         #self.mujoco_renderer.render(self.render_mode)
        it_max = self.it_max
        self.animator = ROAM(it_max=it_max, current_position=init_position, 
                                  attractor_position=goal, dynamic_human=self.dynamic_human, obstacle=self.obstacle)       
        self.animator.setup()
        if self.obstacle:
            self.human_obstacle = self.animator.human_obstacle_3d
            self.adjust_obstacle_init()
        #self.animator.dynamic_human = self.dynamic_human

        for ii in range(it_max):
            #update trajectory
            velocity = self.animator.run(ii=ii)
            #if not ii % 10:
            #    print(f"it={ii}")
            print('iteration: ', ii)
            print("velocity: ", velocity)
            next_desired_pos = init_position + velocity*self.step_size_position
            # update end effector position
            init_position = next_desired_pos
            #init_position = np.array([[init_position[0], init_position[1], init_position[2]]])
            self.animator.update_trajectories(init_position, ii)
            print('init_position: ', init_position)
            self.point_storage.append(init_position)
            self.store_data()
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.adjust_obstacle()
            if self.check_goal_reach(init_position, goal):
                print("goal reached")
                break
            
    '''
    #-----------------------------------move ee and joints in tangent and norm direction--------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    '''
    def move_franka_robot_norm_dir(self,init_position, goal):
        #self.mujoco_renderer.render(self.render_mode)
        it_max = self.it_max
        sampled_points = init_position
        self.animator = ROAM(it_max=it_max, current_position=sampled_points, 
                                  attractor_position=goal, dynamic_human=self.dynamic_human, obstacle=self.obstacle)    
        # self.animator.obstacle_initiation()
        # if self.obstacle:
        #     self.human_obstacle = self.animator.human_obstacle_3d
        #     self.table_obstacle = self.animator.table_obstacle_3d
        #     self.adjust_obstacle_init(self.human_obstacle)
        #     self.adjust_obstacle_init(self.table_obstacle)
        self.obstacle_sum_init('human_table')
        # self.obstacle_sum_init('table_box')
        # self.obstacle_sum_init('complex_table')

        next_desired_pos = []
        self.animator.set_up_franka()

        for ii in range(it_max):
            time_start = time.time()
            velocity= self.animator.run_norm(ii=ii)

            print('iteration: ', ii)
            #print("velocity: ", velocity)
            #print("velocity shape: ", velocity.shape)
            for jj in range(len(velocity)):
                joint_next_desired_pos = sampled_points[jj] + velocity[jj]* self.delta_time  #self.step_size_position
                next_desired_pos.append(joint_next_desired_pos)
            #self.use_ik_controller(next_desired_pos)
            #self.panda_ik_controller(next_desired_pos)
            self.joints_resolved_rate_motion_control_norm_dir(next_desired_pos, velocity)

            init_position = self.get_joints_end_position()
            sampled_points = init_position

            self.animator.update_multiple_points_trajectories(sampled_points, ii)
            # if self.obstacle:
            #     self.human_obstacle = self.animator.human_obstacle_3d
            #     self.table_obstacle = self.animator.table_obstacle_3d
            #     self.adjust_obstacle(self.human_obstacle)
            #     self.adjust_obstacle(self.table_obstacle)
            #self.human_obstacle = self.animator.human_obstacle_3d
            self.obstacle_sum_update('human_table')
            #self.obstacle_sum_update('table_box')
            #self.obstacle_sum_update('complex_table')
            time_end = time.time()
            print("time_diff: ", time_end - time_start)
            #self.adjust_obstacle()
            
            ee_position = self.get_ee_position()
            if self.check_goal_reach(ee_position, goal):
                print("goal reached")
                break

    def joints_resolved_rate_motion_control_norm_dir(self, goal, velocity):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        #body_id = self.model.site('robot0:grip').id
        joints_id = []
        for i in range(len(self.rotation_joints)):
            k = self.rotation_joints[i]
            joints_id.append(self.model.body(f'link{k}').id)
        
        print("current goal: ", goal[-1])
        print("current init_q: ", init_q)
        print("ee position: ", self.data.body(body_id).xpos)

        self.data = self.joint_velocity_control(goal, velocity, init_q, body_id, joints_id) #calculate the qpos
        #print("time: ", self.data.time)
        for i in range(len(self.rotation_joints)):
            print("joint position", self.get_body_xpos(f'link{self.rotation_joints[i]}'))
        
    def joint_velocity_control_old(self, goal, velocity, init_q, body_id, joints_id):
        """velocity: desired velocity of joint 4 and end effector
        goal: desired position of joint 4 and end effector
        """
        #print("velocity: ", velocity)
        self.data.qpos = init_q
        length = len(joints_id)
        joints_vel = velocity[:length]
        ee_vel = velocity[length:]
        joints_goal = goal[:length]
        ee_goal = goal[length:]
        #mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        #self.trajectory_qpos.append(self.data.qpos.copy())
        self.store_data()
        jac = np.zeros((6, self.model.nv))
        jac_joints = np.zeros((len(joints_id), 6, self.model.nv))
        time_start = self.data.time
        time_end = 0
        time_diff = 0
        while (np.linalg.norm(time_diff)<= self.delta_time):
            #calculate jacobian
            #mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], current_pose, body_id)
            j = jac
            #print("rank of jacobian: ", np.linalg.matrix_rank(j))
            if np.linalg.matrix_rank(j) < 6:
                print("jacobian is singular")
                self.singularities_number += 1
                breakpoint()
            #calculate delta of joint q
            current_pose = self.data.body(body_id).xpos
            #print("current_pose: ", current_pose)
            #ee_vel = np.array([[0,0.3,0]])
            dv = self.desired_velocity(ee_vel)
            n = j.shape[1]
            I = np.identity(n)
            product = j.T @ j + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ j.T
            else:
                j_inv = np.linalg.inv(product) @ j.T

            #j_inv = np.linalg.pinv(j)
            delta_q = j_inv @ dv
          
            if np.isclose(np.linalg.norm(delta_q), 0):
                shape = delta_q.shape
                #print("shape: ", shape)
                delta_q = np.ones(shape)*0.01

            #compute next step
            #self.data.qpos += self.step_size_ik * delta_q
            #self.data.qpos = self.step_size_ik * delta_q
            self.data.qvel = self.ee_weight* delta_q
            
            for i in range(len(joints_id)):
                joint_pose = self.data.body(joints_id[i]).xpos
                mujoco.mj_jac(self.model, self.data, jac_joints[i][:3], jac_joints[i][3:], joint_pose, joints_id[i])
                j = jac_joints[i]
                #print("jacobian shape: ", j)
                # if np.linalg.matrix_rank(j) < 4:
                #     print("jacobian is singular")
                #     self.singularities_number += 1
                #     breakpoint()

                #calculate delta of joint q
                #joints_vel[i] = np.array([[0.3,-0.3,0.3]])
                dv = self.desired_velocity(joints_vel[i])
                n = j.shape[1]
                I = np.identity(n)
                product = j.T @ j + self.damping * I
                
                if np.isclose(np.linalg.det(product), 0):
                    j_inv = np.linalg.pinv(product) @ j.T
                else:
                    j_inv = np.linalg.inv(product) @ j.T

                delta_q = j_inv @ dv
            
                # if np.isclose(np.linalg.norm(delta_q), 0):
                #     shape = delta_q.shape
                #     delta_q = np.ones(shape)*0.01
    
                self.data.qvel += self.rotation_joints_weights[i] * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)

            mujoco.mj_step(self.model, self.data)
            self.collision_num += self.check_collision()
            #error = np.subtract(ee_goal, self.data.body(body_id).xpos)

            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            #delta_q_norm = delta_q_norm1
            #self.trajectory_qpos.append(self.data.qpos.copy())
            self.store_data()
            time_end = self.data.time
            time_diff = time_end - time_start
            #print("time_diff: ", time_diff) 
        return self.data
    
    def joint_velocity_control(self, goal, velocity, init_q, body_id, joints_id):
        self.data.qpos = init_q
        #mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.store_data()
        jac = np.zeros((6, self.model.nv))
        ref_vel = velocity[-1]
        joints_vel = velocity[:-1]
        time_start = self.data.time
        time_end = 0
        time_diff = 0
        q_vel_convergence = self.get_ee_qpos(body_id, ref_vel)
        q_vel_joints = self.get_joints_qpos(self.rotation_joints, goal, joints_vel)

        while (np.linalg.norm(time_diff)<= self.delta_time):
            joint_weights = np.array(self.animator.weights_sensors)[:len(joints_id),:]
            joints_weights = joint_weights[:,:-1]
            # if np.any(joints_weights) > 0.3:
            #     self.data.qvel = q_vel_joints[:,0] + q_vel_convergence * 0.05
            # else:
            #     self.data.qvel = q_vel_convergence #+ q_vel_joints[:,0] * self.rotation_joints_weights[0]
            for i in range(q_vel_joints.shape[1]):
                self.data.qvel = q_vel_joints[:,i] * self.rotation_joints_weights[i] + q_vel_convergence

            #check limits
            self.check_joint_limits(self.data.qpos)
            mujoco.mj_step(self.model, self.data)
            self.collision_num += self.check_collision()
            
            #self.trajectory_qpos.append(self.data.qpos.copy())
            self.store_data()
            time_end = self.data.time
            time_diff = time_end - time_start

        return self.data
    
    '''
    #-----------------------------------move ee, joints and sensors in tangent and norm direction--------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    '''

    def move_franka_robot_sensors_norm_dir(self,init_position, goal, env_name):
        it_max = self.it_max
        #sampled_points = init_position[1:]
        self.animator = ROAM(it_max=it_max, current_position=init_position, 
                                  attractor_position=goal, dynamic_human=self.dynamic_human, obstacle=self.obstacle)    
        # self.animator.obstacle_initiation()
        # if self.obstacle:
        #     self.human_obstacle = self.animator.human_obstacle_3d
        #     self.table_obstacle = self.animator.table_obstacle_3d
        #     self.adjust_obstacle_init(self.human_obstacle)
        #     self.adjust_obstacle_init(self.table_obstacle)

        # self.animator.obstacle_initiation_table_box()
        # if self.obstacle:
        #     self.table_obstacle = self.animator.table_obstacle_3d
        #     self.box_obstacle = self.animator.box_obstacle_3d
        #     self.adjust_obstacle_init(self.table_obstacle)
        #     self.adjust_obstacle_init(self.box_obstacle)
        # self.human_obstacle = self.animator.human_obstacle_3d
        # self.adjust_obstacle_init()
        self.obstacle_sum_init(env_name)
        #self.obstacle_sum_init('human_table')
        next_desired_pos = []
        self.animator.set_up_franka()
        goal_unfeasible =self.check_collision()
        
        for ii in range(it_max):
            if goal_unfeasible:
                print("goal position is inside the obstacle")
                print("Reset the goal position !!")
                break
            time_start = time.time()
            #self.time_storage.append(time_start)
            #velocity_reference = self.get_sensors_velocity()
            velocity= self.animator.run_norm(ii=ii)
            print('iteration: ', ii)
            #print("velocity: ", velocity)
            #print("velocity shape: ", velocity.shape)
            for jj in range(len(velocity)):
                joint_next_desired_pos = init_position[jj] + velocity[jj]* self.delta_time  #self.step_size_position
                next_desired_pos.append(joint_next_desired_pos)
            
            self.joints_sensors_resolved_rate_motion_control_norm_dir(next_desired_pos, velocity)
            #self.test_sensor_joint_control()

            init_position = self.get_joints_sensors_end_position()

            self.animator.update_multiple_points_trajectories(init_position, ii)
            #self.obstacle_sum_update('human_table')
            #self.obstacle_sum_update('table_box')
            if self.obstacle:
                self.obstacle_sum_update(env_name)
            # if self.obstacle:
            #     self.human_obstacle = self.animator.human_obstacle_3d
            #     self.table_obstacle = self.animator.table_obstacle_3d
            #     self.adjust_obstacle(self.human_obstacle)
            #     self.adjust_obstacle(self.table_obstacle)
            # if self.obstacle:
            #     self.table_obstacle = self.animator.table_obstacle_3d
            #     self.box_obstacle = self.animator.box_obstacle_3d
            #     self.adjust_obstacle(self.table_obstacle)
            #     self.adjust_obstacle(self.box_obstacle)
            time_end = time.time()
            self.time_storage.append(time_end - time_start)
            print("time_diff: ", time_end - time_start)
            #self.adjust_obstacle()
            
            ee_position = self.get_ee_position()
            self.ii = ii
            if self.check_goal_reach(ee_position, goal):
                print("--------------------------------------------")
                print("goal reached")
                break
            print("--------------------------------------------")

    def joints_sensors_resolved_rate_motion_control_norm_dir(self, goal, velocity):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        
        print("current goal: ", goal[-1])
        #print("current init_q: ", init_q)
        print("ee position: ", self.data.body(body_id).xpos)
        joints_num = len(self.rotation_joints)
        joints_vel = velocity[:joints_num] # velocity of joints 4, if there are more joints, add them here
        avoid_vel = velocity[joints_num:-1]
        ref_vel = velocity[-1]
        self.data = self.sensors_velocity_control(goal, avoid_vel,ref_vel,joints_vel,init_q, body_id) #calculate the qpos


    def sensors_velocity_control(self, goal, avoid_vel, ref_vel, joints_vel, init_q, body_id):
        self.data.qpos = init_q
        #mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.store_data()
        jac = np.zeros((6, self.model.nv))
        time_start = self.data.time
        time_end = 0
        time_diff = 0
        repulsive_vel = self.compute_self_avoidance_velocity()
        #avoid_vel = avoid_vel + repulsive_vel
        ref_vel = ref_vel + repulsive_vel
        q_vel_avoid = self.maximize_q_vel(avoid_vel,ref_vel,body_id)
        q_vel_convergence = self.get_ee_qpos(body_id, ref_vel)
        q_vel_joints = self.get_joints_qpos(self.rotation_joints, goal, joints_vel)
        ee_start = self.get_ee_position()
        while (np.linalg.norm(time_diff)<= self.delta_time):
            
            for i in range(q_vel_joints.shape[1]):
                if np.all(q_vel_avoid == 0):
                    self.data.qvel = q_vel_convergence*self.ee_weight + q_vel_joints[:,i] * self.rotation_joints_weights[i]

                    error = np.subtract(self.goal, self.get_ee_position())
                    if np.linalg.norm(error) < 0.2:
                        #print("velocity", ref_vel, repulsive_vel)
                        self.data.qvel = q_vel_convergence * 1.5
                    #self.delta_time = 0.01
                    
                
                    #q_ctrl = self.get_ee_qctrl(goal[-1],body_id, ref_vel)
                    #self.data.ctrl = q_ctrl
                else:
                    #self.data.qvel = q_vel_avoid * 0.5 + q_vel_convergence*0.25 + q_vel_joints[:,i] * 0.25
                    self.data.qvel = q_vel_avoid + q_vel_convergence*0.1
                    self.sensors_activation_num += 1
                    #self.delta_time = 0.03
        

            #check limits
            self.check_joint_limits(self.data.qpos)
            mujoco.mj_step(self.model, self.data)
            self.collision_num += self.check_collision()
            
            ee_end = self.get_ee_position()
            if np.linalg.norm(ee_end - ee_start) > 0.02:
                print("robot is unstable")
                breakpoint()
            #self.trajectory_qpos.append(self.data.qpos.copy())
            self.store_data()
            time_end = self.data.time
            time_diff = time_end - time_start

        # print("norm of qvel: ", np.linalg.norm(self.data.qvel))
        # for m in range(len(self.data.qvel)):
        #     print(f"joint{m} qvel: ", self.data.qvel[m])
        #     if m == 3:
        #         print("joint4 qpos: ", self.data.qpos[m])

        if not np.all(q_vel_avoid == 0):
            self.unchange_sensor_color_size()
            mujoco.mj_step(self.model, self.data)
            self.store_data()
            #print("time_diff: ", time_diff)
        return self.data


    '''
    # --------------------------------------------------- assistive functions ---------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------
    '''
    def obstacle_sum_init(self, obstacle_name):
        if obstacle_name == 'human':
            self.animator.obstacle_initiation_human()
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.adjust_obstacle_init(self.human_obstacle)
        if obstacle_name == 'human_table':
            self.animator.obstacle_initiation()
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.table_obstacle = self.animator.table_obstacle_3d
                self.adjust_obstacle_init(self.human_obstacle)
                self.adjust_obstacle_init(self.table_obstacle)
        elif obstacle_name == 'table_box':
            self.animator.obstacle_initiation_table_box()
            if self.obstacle:
                self.table_obstacle = self.animator.table_obstacle_3d
                self.box_obstacle = self.animator.box_obstacle_3d
                self.adjust_obstacle_init(self.table_obstacle)
                self.adjust_obstacle_init(self.box_obstacle)
        elif obstacle_name == 'complex_table':
            self.animator.obstacle_initiation_table_multiobs()
            if self.obstacle:
                self.table_obstacle = self.animator.table_obstacle_3d
                self.cross_obstacle = self.animator.cross_obstacle_3d
                self.concave_obstacle = self.animator.concave_word_obstacle_3d
                self.star_obstacle = self.animator.star_shape_obstacle_3d
                self.adjust_obstacle_init(self.table_obstacle)
                self.adjust_obstacle_init(self.cross_obstacle)
                self.adjust_obstacle_init(self.concave_obstacle)
                self.adjust_obstacle_init(self.star_obstacle)
        elif obstacle_name == 'cuboid_sphere':
            self.animator.obstacle_initiation_cuboid_sphere()
            if self.obstacle:
                self.cuboid_obstacle = self.animator.cuboid_obstacle_3d
                self.sphere1_obstacle = self.animator.sphere_1_obstacle_3d
                self.sphere2_obstacle = self.animator.sphere_2_obstacle_3d
                self.sphere3_obstacle = self.animator.sphere_3_obstacle_3d
                self.adjust_obstacle_init(self.cuboid_obstacle)
                self.adjust_obstacle_init(self.sphere1_obstacle)
                self.adjust_obstacle_init(self.sphere2_obstacle)
                self.adjust_obstacle_init(self.sphere3_obstacle)
        elif obstacle_name == 'table':
            self.animator.obstacle_initiation_table()
            if self.obstacle:
                self.table_obstacle = self.animator.table_obstacle_3d
                self.box_obstacle = self.animator.box_obstacle_3d
                self.adjust_obstacle_init(self.table_obstacle)
                self.adjust_obstacle_init(self.box_obstacle)

    def obstacle_sum_update(self, obstacle_name):
        if obstacle_name == 'human':
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.adjust_obstacle(self.human_obstacle)
        if obstacle_name == 'human_table':
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.table_obstacle = self.animator.table_obstacle_3d
                self.adjust_obstacle(self.human_obstacle)
                self.adjust_obstacle(self.table_obstacle)
        elif obstacle_name == 'table_box':
            if self.obstacle:
                self.table_obstacle = self.animator.table_obstacle_3d
                self.box_obstacle = self.animator.box_obstacle_3d
                self.adjust_obstacle(self.table_obstacle)
                self.adjust_obstacle(self.box_obstacle)
        elif obstacle_name == 'complex_table':
            if self.obstacle:
                self.table_obstacle = self.animator.table_obstacle_3d
                self.cross_obstacle = self.animator.cross_obstacle_3d
                self.concave_obstacle = self.animator.concave_word_obstacle_3d
                self.star_obstacle = self.animator.star_shape_obstacle_3d
                self.adjust_obstacle(self.table_obstacle)
                self.adjust_obstacle(self.cross_obstacle)
                self.adjust_obstacle(self.concave_obstacle)
                self.adjust_obstacle(self.star_obstacle)
        elif obstacle_name == 'cuboid_sphere':
            if self.obstacle:
                self.cuboid_obstacle = self.animator.cuboid_obstacle_3d
                self.sphere1_obstacle = self.animator.sphere_1_obstacle_3d
                self.sphere2_obstacle = self.animator.sphere_2_obstacle_3d
                self.sphere3_obstacle = self.animator.sphere_3_obstacle_3d
                self.adjust_obstacle(self.cuboid_obstacle)
                self.adjust_obstacle(self.sphere1_obstacle)
                self.adjust_obstacle(self.sphere2_obstacle)
                self.adjust_obstacle(self.sphere3_obstacle)
        elif obstacle_name == 'table':
            if self.obstacle:
                self.table_obstacle = self.animator.table_obstacle_3d
                self.box_obstacle = self.animator.box_obstacle_3d
                self.adjust_obstacle(self.table_obstacle)
                self.adjust_obstacle(self.box_obstacle)

    def desired_velocity(self, velocity):
        #velocity = np.linalg.norm(error)
        #velocity = self.step_size_position * velocity
        # set the desired velocity to end effector
        if len(velocity) == 3:
            dv = np.array([velocity[0],velocity[1],velocity[2], 0, 0, 0])
        else:
            velocity = velocity[0]
            dv = np.array([velocity[0],velocity[1],velocity[2], 0, 0, 0])
        return dv
    
    def collision_visualization(self):
        # visualize contact frames and forces, make body transparent
        options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(options)
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # tweak scales of contact visualization elements
        self.model.vis.scale.contactwidth = 0.1
        self.model.vis.scale.contactheight = 0.03
        self.model.vis.scale.forcewidth = 0.05
        self.model.vis.map.force = 0.3
        
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))


    def check_goal_reach(self, position, goal):
        if np.linalg.norm(position - goal) < self.tol:
            self.reach_goal.append(True)
            return True
        if self.ii == self.it_max-1:
            self.reach_goal.append(False)
        return False


    def transform_degree_to_quat(self, degree, direction):
        """Next try is to calculate the values according to the tutorials formula:
        Form:
        quat = [w, x, y, z] where w = cos(theta/2) and x = x_dir*sin(theta/2), y = y_dir*sin(theta/2), z = z_dir*sin(theta/2)
        degree = [x_degree, y_degree, z_degree]
        direction = [x_dir, y_dir, z_dir]
        Example: rotate 45 degrees around x, y, z axis(direction: x, y, z)
        quat = [0.8446232  0.19134172 0.46193977 0.19134172]
        qx = [cos( (1/4 * pi) / 2) = 0,923, sin( (1/4 * pi) / 2)*(1,0,0) = (0,382, 0,0, 0,0)] = [0,923, 0,382, 0,0, 0,0]
        qy = [cos( (1/4 * pi) / 2) = 0,923, sin( (1/4 * pi) / 2)*(0,1,0) = (0,0, 0,382, 0,0)] = [0,923, 0,0, 0,382, 0,0]
        qz = [cos( (1/4 * pi) / 2) = 0,923, sin( (1/4 * pi) / 2)*(0,0,1) = (0,0, 0,0, 0,382)] = [0,923, 0,0, 0,0, 0,382]
        q = qz * qy * qx = [0,8446, 0,191, 0,461, 0,191] (cross product)
        """
        # quat = np.zeros(4)
        # # transform degree to radian
        # radian = np.radians(degree)
        # # calculate the values
        # quat[0] = direction[0] * np.sin(radian[0] / 2)
        # quat[1] = direction[1] * np.sin(radian[1] / 2)
        # quat[2] = direction[2] * np.sin(radian[2] / 2)
        # quat[3] = np.cos(radian[0] / 2)
        r = R.from_euler('xyz', degree, degrees=True)
        quat = r.as_quat(scalar_first=True)
        return quat
    
    def transform_quat_to_degree(self, quat):
        """Transform the quaternion to degree
        Form:
        quat = [x, y, z, w]
        degree = [x_degree, y_degree, z_degree]
        direction = [x_dir, y_dir, z_dir]
        """
        degree = np.zeros(3)
        # calculate the values
        degree[0] = np.degrees(np.arctan2(2 * (quat[3] * quat[0] + quat[1] * quat[2]), 1 - 2 * (quat[0] ** 2 + quat[1] ** 2)))
        degree[1] = np.degrees(np.arcsin(2 * (quat[3] * quat[1] - quat[2] * quat[0])))
        degree[2] = np.degrees(np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))
        return degree
    
    def store_data(self):
        # store the data of the simulation
        self.model_storage.append((self.model.geom_pos.copy(), self.model.geom_quat.copy()))
        self.model_size_color_storage.append((self.model.site_size.copy(), self.model.site_rgba.copy()))
        self.trajectory_qpos.append(self.data.qpos.copy())
        #self.time_storage.append(time.time())
        #print(self.data.contact.dist)

    def change_sensor_color_size(self, link_sensor, sensor_index, color):
        num = sensor_index+1
        link_sensor_name = f"{link_sensor}_point{num}"
        link_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, link_sensor_name)
        print("link_sensor_name: ", link_sensor_name)
        #print("sensor position: ", self._utils.get_site_xpos(self.model, self.data, link_sensor_name))
        self.adjust_sensors_name.append(link_sensor_name)
        self.model.site_rgba[link_sensor_id] = color
        self.model.site_size[link_sensor_id] = 0.015

    def unchange_sensor_color_size(self):
        color = [1,0,0,1]
        num = self.sensors_index +1
        link_sensor_name = f"{self.link_sensor}_point{num}"
        link_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, link_sensor_name)
        #print("link_sensor_name: ", link_sensor_name)
        self.model.site_rgba[link_sensor_id] = color
        self.model.site_size[link_sensor_id] = 0.005
    
    def skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def check_collision(self):
        # check if the robot is in collision with the obstacles
        num_contacts = self.data.ncon
        if num_contacts > 0:
            for j, c in enumerate(self.data.contact):
                #print("contact", j)
                name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                print(f"Contact {j}: Geom1 ID {c.geom1} Name {name1}, Geom2 ID {c.geom2} Name {name2}")
                body1 = self.model.geom_bodyid[c.geom1]
                body2 = self.model.geom_bodyid[c.geom2]
                name_body1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
                name_body2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)
                print(f"Contact {j}: Body1ID {body1} Name {name_body1}, Body2ID {body2} Name {name_body2}")
                self.collision_name.append((name_body1,name1, name_body2,name2))
                #print("contact", c.geom1, c.geom2)
                #print("contact", c.pos)
                #print("contact", c.frame)
                #print("contact", c.dist)

        #print("collision number", num_contacts)
        return num_contacts

    def replay_old(self):
        # use the mujoco viewer to replay the simulation
        m = self.model
        d = self.data
    
        with mujoco.viewer.launch_passive(m, d) as viewer:
                # Close the viewer automatically after 30 wall-seconds.
                start = time.time()
                while viewer.is_running() and time.time() - start < 80:
                    step_start = time.time()
                    for i in range(len(self.trajectory_qpos)):
                        d.qpos = self.trajectory_qpos[i]
                        gem_pos = np.zeros((89,3))
                        #d.xpos, d.xquat = self.model_storage[i]
                        m.geom_pos, m.geom_quat = self.model_storage[i]
                        m.site_size, m.site_rgba = self.model_size_color_storage[i]
                        mujoco.mj_step(m, d)
                        viewer.sync()
                        #if len(self.time_storage) > 0:
                        #    time.sleep(self.time_storage[i+1] - self.time_storage[i])
                    
                    #print("weights", self.weights_storage)
                    print("---------------------one loop done---------------------")
                    time.sleep(0.5)
                    # Rudimentary time keeping, will drift relative to wall clock.
                    #time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    #if time_until_next_step > 0:
                    #    time.sleep(time_until_next_step)

    def replay(self):
        # Get the model and data objects
        m = self.model
        d = self.data

        # Initialize MuJoCo viewer
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 1000:
                step_start = time.time()
                for i in range(len(self.trajectory_qpos)):
                    # Update joint positions from the stored trajectory
                    d.qpos[:] = self.trajectory_qpos[i]
                    
                    # Update geometry and visualization properties
                    gem_pos = np.zeros((89, 3))  # Example: 89 geometries
                    m.geom_pos[:], m.geom_quat[:] = self.model_storage[i]
                    m.site_size[:], m.site_rgba[:] = self.model_size_color_storage[i]

                    # Sphere size (radius)
                    size = np.array([0.01, 0.01, 0.01], dtype=np.float64)  # 3-element float array
                    pos_start = self.get_ee_position()

                    # Position and orientation
                    pos = np.array([pos_start[0], pos_start[1], pos_start[2]], dtype=np.float64)  # 3-element float array
                    mat = np.eye(3, dtype=np.float64).flatten()  # 3x3 identity matrix, flattened to 9 elements

                    # RGBA color
                    rgba = np.array([0.8, 0.8, 0.7, 0.1], dtype=np.float32)  # 4-element float array

                    # Initialize geometry
                    viewer.user_scn.ngeom+=1
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        size,
                        pos,
                        mat,
                        rgba
                    )

                    # Draw a line trajectory between consecutive positions
                    line_geom = mujoco.MjvGeom()

                    # Step the simulation
                    mujoco.mj_step(m, d)
                    pos_after = self.get_ee_position()
                    if i > 0:
                        mujoco.mjv_connector(
                            line_geom,
                            mujoco.mjtGeom.mjGEOM_LINE,
                            2,
                            pos_start,
                            pos_after,
                        )

                    # Sync the viewer to render the updated state
                    viewer.sync()

                    # Sleep to maintain real-time playback speed
                    time.sleep(0.005)

                print("---------------------one loop done---------------------")
                time.sleep(0.5)

    '''
    # --------------------------------------------- qpos calculation -------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------
    '''
    def converge_ik(self, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
        """
        Runs up to 'max_iters' of IK steps. Returns True if position and orientation
        are below thresholds, otherwise False.
        """
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, tasks, dt, solver, 1e-3)
            self.configuration.integrate_inplace(vel, dt)

            # Only checking the first FrameTask here (end_effector_task). 
            # If you want to check multiple tasks, sum or combine their errors.
            err = tasks[0].compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

            if pos_achieved and ori_achieved:
                return True
        return False

    def mink_ik(self,goal):
        # IK parameters
        SOLVER = "quadprog"
        POS_THRESHOLD = 1e-4
        ORI_THRESHOLD = 1e-4
        MAX_ITERS = 20
        dt = 0.001
        #goal = np.array([goal[0], goal[1], goal[2]])
        # Create a Mink self.configuration
        
        #self.data.mocap_pos[0] = goal
        self.configuration = mink.Configuration(self.model)

        #self.data.mocap_pos[0] = goal
        
        # Reset simulation data to the 'home' keyframe
        #mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        #self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        #mujoco.mj_forward(self.model, self.data)

        # Initialize to the home keyframe.
        #self.configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        #mink.move_mocap_to_frame(self.model, self.data, "intergoal", "attachment_site", "site")
        #initial_target_position = self.data.mocap_pos[0].copy()

        #self.data.mocap_pos[0] = goal

        # Update the end effector task target from the mocap body
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, "intergoal")
        self.end_effector_task.set_target(T_wt)

        # Attempt to converge IK
        converged = self.converge_ik(self.tasks, dt, SOLVER,
                                POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

        # Set robot controls (first 8 dofs in your self.configuration)
        self.data.ctrl = self.configuration.q[:8]

        # goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "intergoal")
        # # Set the mocap position to the goal
        # print("intergoal position: ", self.data.xpos[goal_id])
        # print("goal position: ", self.data.mocap_pos[0])
        # print("model position: ", self.data.body("intergoal").xpos)
        # Step simulation
        #mujoco.mj_step(self.model, self.data)
        self.store_data()
        return self.data.ctrl[:8]
    
    def get_ee_qctrl(self, goal, body_id, ref_vel):
        #self.build_inter_goal_object(goal)
        q_ctrl = self.mink_ik(goal)
        return q_ctrl

    def get_ee_qpos_old(self,body_id,ref_vel):
        jac = np.zeros((6, self.model.nv))
        current_pose = self.data.body(body_id).xpos
        mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], current_pose, body_id)
        j = jac
    
        if np.linalg.matrix_rank(j) < 6:
            print("jacobian is singular")
            self.singularities_number += 1
            breakpoint()
        
        #ee_vel = np.array([[0,0.3,0]])
        dv = self.desired_velocity(ref_vel)
        n = j.shape[1]
        I = np.identity(n)
        #print("j :", j.T @ j)
        product = j.T @ j + self.damping * I
        
        if np.isclose(np.linalg.det(product), 0):
            j_inv = np.linalg.pinv(product) @ j.T
        else:
            j_inv = np.linalg.inv(product) @ j.T

        #j_inv = np.linalg.pinv(j)
        delta_q = j_inv @ dv
        
        if np.isclose(np.linalg.norm(delta_q), 0):
            shape = delta_q.shape
            #print("shape: ", shape)
            delta_q = np.ones(shape)*0.01
        
        return delta_q
    
    def get_ee_qpos(self, body_id, ref_vel):
        jac = np.zeros((6, self.model.nv))
        current_pose = self.data.body(body_id).xpos
        mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], current_pose, body_id)
        j = jac

        if np.linalg.matrix_rank(j) < 6:
            print("Jacobian is singular")
            self.singularities_number += 1
            return np.zeros(self.model.nv)

        dv = self.desired_velocity(ref_vel)

        # Damping
        lambda2 = (0.01 + 0.1 * np.linalg.norm(dv))**2
        jjt = j @ j.T + lambda2 * np.eye(6)
        j_inv = j.T @ np.linalg.inv(jjt)

        # Primary solution
        delta_q = j_inv @ dv

        # Null-space projection (toward home pose)
        q_home = np.zeros(self.model.nv)
        null_weight = 0.05
        q_null = -null_weight * (self.data.qpos[:self.model.nv] - q_home)
        null_proj = np.eye(self.model.nv) - j.T @ np.linalg.inv(j @ j.T + lambda2 * np.eye(6)) @ j
        delta_q += null_proj @ q_null

        # Smooth output
        self.prev_qvel = getattr(self, "prev_qvel", np.zeros_like(delta_q))
        alpha = 0.2
        delta_q = alpha * delta_q + (1 - alpha) * self.prev_qvel
        self.prev_qvel = delta_q

        return delta_q

    def compute_self_avoidance_velocity(self, threshold=0.13, scale=0.5):
        repulsive_velocity = np.zeros(3)

        for body1_name, body2_name in self.self_collision_pairs:
            id1 = self.model.body(body1_name).id
            id2 = self.model.body(body2_name).id

            pos1 = self.data.body(id1).xpos
            pos2 = self.data.body(id2).xpos

            vec = pos1 - pos2
            dist = np.linalg.norm(vec)

            if dist < threshold and dist > 1e-4:
                # change the joint angle of joint 4
                pass

        return repulsive_velocity



    def get_joints_qpos(self, joints_idex, joints_goal, joints_vel):
        joints_id = []
        for i in range(len(self.rotation_joints)):
            k = self.rotation_joints[i]
            joints_id.append(self.model.body(f'link{k}').id)
        q_vel_joints = np.zeros((len(joints_id),self.model.nv))

        jac_joints = np.zeros((len(joints_id), 6, self.model.nv))
        for i in range(len(joints_id)):
                joint_pose = self.data.body(joints_id[i]).xpos
                mujoco.mj_jac(self.model, self.data, jac_joints[i][:3], jac_joints[i][3:], joint_pose, joints_id[i])
                j = jac_joints[i]
                #print("jacobian shape: ", j)
                # if np.linalg.matrix_rank(j) < 4:
                #     print("jacobian is singular")
                #     self.singularities_number += 1
                #     breakpoint()

                #calculate delta of joint q
                #joints_vel[i] = np.array([[0.3,-0.3,0.3]])
                dv = self.desired_velocity(joints_vel[i])
                n = j.shape[1]
                I = np.identity(n)
                product = j.T @ j + self.damping * I
                
                if np.isclose(np.linalg.det(product), 0):
                    j_inv = np.linalg.pinv(product) @ j.T
                else:
                    j_inv = np.linalg.inv(product) @ j.T

                delta_q = j_inv @ dv
                q_vel_joints[i] = delta_q
        return q_vel_joints.T

    def maximize_q_vel(self,avoid_vel,ref_vel, body_id):
 
        weights_sensors = self.animator.weights_sensors
        weights_sensors = np.array(weights_sensors)[len(self.rotation_joints):,:]  # remove the weights of the joints
        weights_sensors_obstacles = weights_sensors[:,:-1]   # remove the last column which is the weight of convergence dynamics
        #weights_sensors_obstacles = weights_sensors[:][:-1]
        #print("weights_sensors_obstacles: ", weights_sensors_obstacles.shape)
        #print("weights ", weights_sensors_obstacles)
        
        #if any(np.any(arr > 0.8) for arr in weights_sensors_obstacles):
        if np.any(weights_sensors_obstacles > 0.5): #0.5
            index = np.argmax(weights_sensors_obstacles)
            index = index // weights_sensors_obstacles.shape[1]
            obstacle_id = np.argmax(weights_sensors_obstacles[index])
            print("indx: ", index)
            # if obstacle_id < self.human_obstacle.n_components:
            #     print("obstacle position: ", self.human_obstacle[obstacle_id].center_position)
            # else:
            #     obstacle_index = obstacle_id - self.human_obstacle.n_components
            #     print("obstacle position: ", self.table_obstacle[obstacle_index].center_position)
            #print("shape of weights_sensors_obstacles: ", np.array(weights_sensors_obstacles).shape)
            print("obstacle id",obstacle_id,
                  "maximum value: ",np.max(weights_sensors_obstacles[index]))
            print("weights_sensors_obstacles: ", weights_sensors_obstacles[index])
            #print("avoid_vel: ", avoid_vel[index])
            #print("weights ", weights_sensors_obstacles)
            #print("index: ", index)
            #breakpoint()
            self.link_sensor, link_index, self.sensors_index = self.get_sensor_index(index)
            # if index < self.total_sensors:
            #     num_sensors_link = self.total_sensors // len(self.link_names)
            #     link_idex = index // num_sensors_link
            #     self.sensors_index = index - (link_idex * num_sensors_link)
            #     self.link_sensor = self.link_names[link_idex]
            # else:
            #     self.sensors_index = index - self.total_sensors
            #     self.link_sensor = self.hand_names[0]
            #     link_idex = 0
            jac = self.get_required_sensor_jac(self.link_sensor, link_index, self.sensors_index)
            self.change_sensor_color_size(self.link_sensor, self.sensors_index, [0,1,0,1])
            mujoco.mj_step(self.model, self.data)
            self.store_data()

            #jac = jac_sensors[sensors_index]
            #print("jac: ", jac.shape)
            #print("jac: ", jac.shape)
            #dv_sensor = self.desired_velocity(avoid_vel)
            n_sensor = jac.shape[1]
            I_sensor = np.identity(n_sensor)
            product_sensor = jac.T @ jac + self.damping * I_sensor

            if np.isclose(np.linalg.det(product_sensor), 0):
                j_inv_sensor = np.linalg.pinv(product_sensor) @ jac.T
            else:
                j_inv_sensor = np.linalg.inv(product_sensor) @ jac.T
            #print("j_inv_sensor: ", j_inv_sensor.shape)
            #print("avoid_vel: ", avoid_vel[i].shape)
            q_vel = j_inv_sensor @ avoid_vel[index]
            return q_vel
        else:
            return np.zeros((self.model.nv))
        
    def get_sensor_index(self, index):
        if index < self.total_sensors:
            num_sensors_link = list(accumulate(self.sensors_on_links))
            num_sensors_link.insert(0,0)
            for i in range(len(num_sensors_link)-1):
                if num_sensors_link[i] <= index < num_sensors_link[i+1]:
                    link_sensor = self.link_names[i]
                    link_index = i
                    sensors_index = index - num_sensors_link[i]
        else:
            sensors_index = index - self.total_sensors
            link_sensor = self.hand_names[0]
            link_index = 0
        return link_sensor, link_index, sensors_index
    '''
    # ------------------------------------------------------- jacobian calculation ---------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------
    '''
    def get_sensors_jac(self):
        sensors_jac = np.zeros((self.total_sensors, 3, self.model.nv))
        i = 0
        for _, link_sensor in enumerate(self.link_names):
            # sensor_vels=[]
            # for i in range(1, 23):
            #     link_sensor_point = f"{link_sensor}_point{i}"
            #     link_sensor_point_pos = self.get_site_pos(link_sensor_point)
            #     link_sensor_point_id = self.model.site(link_sensor_point).id
            #     jac = np.array((6, self.model.nv))
            #     print("link_sensor_point_pos: ", link_sensor_point_id)
            #     mujoco.mj_jacSite(self.model, self.data, jac, link_sensor_point_id)
            #     #inv_jac_ee = np.linalg.pinv(jac_ee)
            #     #vel = jac @ inv_jac_ee @ vel_ee
            #     #sensors_vels.append(vel)
            #     sensors_jac.append(jac)
            link_sensor_id = self.model.body(link_sensor).id
            jac_q2l = np.zeros((6, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jac_q2l[:3], jac_q2l[3:], self.data.body(link_sensor_id).xpos, link_sensor_id)
            jac_q2p = self.jacobian_link2point(self.data.body(link_sensor_id).xpos, link_sensor,jac_q2l, _)
            #jac = jac_q2l @ jac_l2p
            num_sensor_link = self.total_sensors // len(self.link_names)
            sensors_jac[i:i+num_sensor_link,:,:] = jac_q2p
            i += num_sensor_link
            #sensors_vels.append(sensor_vels)
        return np.array(sensors_jac)
    
    def get_required_sensor_jac(self, link_sensor, link_index, sensor_index):
        link_sensor_id = self.model.body(link_sensor).id
        jac_q2l = np.zeros((6, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac_q2l[:3], jac_q2l[3:], self.data.body(link_sensor_id).xpos, link_sensor_id)
        jac_q2p = self.jacobian_joint_angles2point(self.data.body(link_sensor_id).xpos, link_sensor,jac_q2l, link_index, sensor_index)
        return jac_q2p

    def jacobian_link2point(self, point, link, j_q2l, num):
        # get the homogeneous transformation matrix of the center of the link to the point
        num_link_sensor = self.total_sensors // len(self.link_names)
        j = []
        for i in range(num_link_sensor):
            j_l2p = self.totoal_j_l2p[num][i]
            j_q2p = j_l2p @ j_q2l
            j.append(j_q2p)
        return np.array(j)
    
    def jacobian_joint_angles2point(self, point, link, j_q2l, num, sensor_index):
        if link != self.hand_names[0]:
            j_l2p = self.totoal_j_l2p[num][sensor_index]
            j_q2p = j_l2p @ j_q2l
        else:
            j_l2q = self.hand_j_l2p[sensor_index]
            j_q2p = j_l2q @ j_q2l
        return np.array(j_q2p)

    '''
    # -------------------------------------------------build object-----------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    '''
    def adjust_obstacle(self, obstacles):
        # set the initial position, size and orientation of the obstacles
        #print('------------------------------------------------------')
        #obstacles = self.human_obstacle
        for ii, obs in enumerate(obstacles):
            if isinstance(obs, Ellipse):
                orientation = [0,0,0]
                #if obs.pose.orientation is not None:
                    # orientation is degree??
                quat = obs.pose.orientation.as_quat()
                if quat[0] < 0:
                    quat = quat * (-1)
                    obs.pose.orientation.from_quat(quat)
                orientation = obs.pose.orientation.as_euler("xyz", degrees=True)

                if np.isclose(abs(orientation[0]), 180) and ii in [7, 8]:
                    orientation[0] = 0
                    orientation[1] = orientation[1] - 180
                    orientation[2] = orientation[2]

                direction = [1,1,1]
                #xml_quat = self.transform_degree_to_quat(orientation,direction)
                xml_quat = obs.pose.orientation.as_quat()
                name_body = obs.name 
                name_geom = name_body + '_geom'

                #print('orientation', orientation)
                #print('position', obs.center_position)

                self.set_geom_quat(name_geom,xml_quat)
                self.set_geom_xpos(name_geom, obs.center_position)
                #self.set_geom_size(name_geom, obs.axes_length * 0.5)
                #self.set_body_xpos(name_body, obs.center_position)
                #self.set_body_xquat(name_body, xml_quat)
                #self.set_body_size(name_body,obs.axes_length*0.5)
            if isinstance(obs, Cuboid):
                name_body = obs.name
                name_geom = name_body + '_geom'
              
                #print('position', obs.center_position)
                self.set_geom_xpos(name_geom, obs.center_position)
                #self.set_geom_size(name_geom, obs.axes_length * 0.5)
                #self.set_body_xpos(name_body, obs.center_position)
                #self.set_body_size(name_body,obs.axes_length*0.5)
            if isinstance(obs, HyperSphere):
                name_body = obs.name
                name_geom = name_body + '_geom'
                #print('position', obs.center_position)
                self.set_geom_xpos(name_geom, obs.center_position)
                #self.set_geom_size(name_geom, obs.radius)
                #self.set_body_xpos(name_body, obs.center_position)
                #self.set_body_size(name_body,obs.radius)

        #self.data = mujoco.MjData(self.model)
        #self.data.qpos = self.trajectory_qpos[-1]
        mujoco.mj_forward(self.model, self.data)
        
        #self.model = self.spec.compile()

    def adjust_obstacle_init(self,obstacles):
        # set the initial position, size and orientation of the obstacles
        i = 0
        #obstacles = self.human_obstacle
        for ii, obs in enumerate(obstacles):
            if isinstance(obs, Ellipse):
                orientation = [0,0,0]
                #if obs.pose.orientation is not None:
                    # orientation is degree??
                quat = obs.pose.orientation.as_quat()
                if quat[0] < 0:
                    quat = quat * (-1)
                    obs.pose.orientation.from_quat(quat)
                orientation = obs.pose.orientation.as_euler("xyz", degrees=True)

                if np.isclose(abs(orientation[0]), 180) and ii in [7, 8]:
                    orientation[0] = 0
                    orientation[1] = orientation[1] - 180
                    orientation[2] = orientation[2]
                #print('orientation', orientation)
                direction = [1,1,1]
                #xml_quat = self.transform_degree_to_quat(orientation,direction)
                xml_quat = obs.pose.orientation.as_quat()
                name = obs.name
                position = obs.center_position
                size = obs.axes_length * 0.5
                color = ii*0.1
                print(f'obstacle {i} position: {position}')
                print(f'obstacle {i} orientation: {orientation}')
                i += 1
                #self.spec.worldbody.add_geom(name= name,
                # body_ellipse = self.spec.worldbody.add_body(name= name,quat= xml_quat,  pos= position,)
                # body_ellipse.add_geom(name= name +'_geom',
                #     type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                #     # change the color every iteration
                #     rgba= [color, color+0.3, 1, 1],
                #     size= size,
                #     )
                self.spec.worldbody.add_geom(name= name +'_geom',
                        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        # change the color every iteration
                        rgba= [color, color+0.3, 1, 1],
                        #rgba= [0.1, 0.1, 0.1, 1],
                        size= size,
                        pos= position,
                        quat= xml_quat,
                        )
            if isinstance(obs, Cuboid):
                name = obs.name
                position = obs.center_position
                size = obs.axes_length * 0.5
                print(f'obstacle {i} position: {position}')
                i += 1
                #self.spec.worldbody.add_geom(name= name,
                # body_cube = self.spec.worldbody.add_body(name= name,pos= position,)
                # body_cube.add_geom(name= name +'_geom',
                #         type=mujoco.mjtGeom.mjGEOM_BOX,
                #         rgba=[1, 0, 0, 1],
                #         size= size)
                self.spec.worldbody.add_geom(name= name +'_geom',
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        rgba=[0.9, 0.9, 0.9, 1],
                        size= size,
                        pos= position)
            if isinstance(obs, HyperSphere):
                name = obs.name
                position = obs.center_position
                size = [obs.radius,obs.radius,obs.radius]
                print(f'obstacle {i} position: {position}')
                i += 1
                #self.spec.worldbody.add_geom(name= name,
                # body_sphere = self.spec.worldbody.add_body(name= name,pos= position,)
                # body_sphere.add_geom(name= name +'_geom',
                #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
                #         rgba=[1, 0, 0, 1],
                #         size= size)
                self.spec.worldbody.add_geom(name= name +'_geom',
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        rgba=[0.9, 0.9, 0.9, 1],
                        size= size,
                        pos= position)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.initial_simulation()
        self.collision_visualization()
        #mujoco.mj_forward(self.model,self.data)
        #print(self.spec.to_xml())

    def build_goal_object(self,goal):
        name = "goal"
        position = goal
        size = [0.02, 0.02, 0.02]
        self.spec.worldbody.add_site(name= name +'_site',
                type=mujoco.mjtGeom.mjGEOM_BOX,
                rgba=[0, 1, 0, 1],
                size= size,
                pos= position)
        # self.spec.worldbody.add_geom(name= name +'_geom',
        #         type=mujoco.mjtGeom.mjGEOM_BOX,
        #         rgba=[0, 1, 0, 1],
        #         size= [0.0001, 0.0001, 0.0001],
        #         pos= position)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        #mujoco.mj_forward(self.model, self.data)

    def build_inter_goal_object(self,goal):
        name = "intergoal"
        position = goal
        size = [0.03, 0.03, 0.03]
        body = self.spec.worldbody.add_body(name= name,pos = position, mocap = True)
        # = self.spec.find_body(name)
        body.add_site(name= name +'_site',
                type=mujoco.mjtGeom.mjGEOM_BOX,
                rgba=[1, 0, 0, 1],
                size= size,
                pos= position,
                )
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        #mujoco.mj_forward(self.model, self.data)

    def build_sensor_points(self):
        ''' attach multiple points to the robot arms, in order to use them for obstacle avoidance method later
        '''
        #self.link_names = ['link2_sensor_1','link2_sensor_2','link4_sensor_1','link5_sensor_1', 'link5_sensor_2','link5_sensor_3', 'link7_sensor_1']
        self.link_names = ['link4_sensor_1','link5_sensor_1', 'link5_sensor_2','link5_sensor_3', 'link7_sensor_1']
        self.hand_names = ['link7_sensor_2']
        self.sensors_on_links = []
        links_radius = []
        links_length = []
        links_center = []
        links_interval = []
        links_quat = []
        offsets = []
        vertex_ = []
        syme = 1
        total_body_sensors_num = 0
        total_hand_sensors_num = 0

        hand_offests = []
        hand_quats = []
        
        #print("total_sensors: ", self.total_sensors)
        for i in range(len(self.link_names)):
            size = self.get_site_size(self.link_names[i])
            link_radius = size[0]
            link_length = size[1]
            link_center = self.get_body_xpos(self.link_names[i])
            link_interval = (link_length) / 2
            link_quat = self.get_site_quat(self.link_names[i])
            x_ = link_radius * np.sin(np.pi/3)
            y_ = link_radius * np.cos(np.pi/3)
            #offset = [[link_radius, 0, 0], [0, link_radius, 0], [-link_radius, 0, 0], [0, -link_radius, 0]]
            offset = [[0, -link_radius, 0], [x_, y_, 0], [-x_, y_, 0]]
            #offset = [[0, -link_radius, 0], [link_radius,0 ,0]]
            vertex = [[0, 0, link_length+link_radius], [0, 0, -link_length-link_radius]]
            links_radius.append(link_radius)
            links_length.append(link_length)
            links_center.append(link_center)
            links_interval.append(link_interval)
            offsets.append(offset)
            links_quat.append(link_quat)
            vertex_.append(vertex)
            # print("links_radius: ", link_radius)
            # print("links_length: ", link_length)
            # print("links_center: ", link_center)
            # print("links_interval: ", link_interval)
            # print("links_quat: ", link_quat)
            # print("offsets: ", offset)
            # print("------------------")
        #breakpoint()
        for u in range(len(self.hand_names)):
            size = self.get_site_size(self.hand_names[u])
            hand_quat = self.get_site_quat(self.hand_names[u])
            axis_x = size[0]
            axis_y = size[1]
            axis_z = size[2]
            offset_ = [#[axis_x, 0, 0], [0, axis_y, 0], [0, 0, axis_z], [-axis_x, 0, 0], [0, -axis_y, 0], [0, 0, -axis_z],
                      [axis_x, axis_y, axis_z], [axis_x, -axis_y, axis_z], [-axis_x, axis_y, axis_z], [-axis_x, -axis_y, axis_z],
                      [axis_x, axis_y, -axis_z], [axis_x, -axis_y, -axis_z], [-axis_x, axis_y, -axis_z], [-axis_x, -axis_y, -axis_z]]
            hand_offests.append(offset_)
            hand_quats.append(hand_quat)


        # self.total_sensors = (1*len(offset)+ (syme-1)*2*len(offset)+2)*len(self.link_names)
        # self.total_sensors_hand = len(offset_)*len(self.hand_names)
        # #self.total_sensors += len(hand_offests[0])
        # print("total_sensors: ", self.total_sensors + self.total_sensors_hand)
        size = [0.005,0.005,0.005]
        self.totoal_j_l2p = []
        self.hand_j_l2p = []

        # for i in range(len(self.link_names)):
        #     body = self.spec.find_body(self.link_names[i])
        #     j_l2p_ = []
        #     #body = self.spec.body(link_names[i])
        #     num = 1
        #     for j in range(syme):
        #         if j == 0:
        #             offset = offsets[i]
        #             for point, offset_ in enumerate(offset):
        #                 #print(f"{link_names[i]}_point{num}")
        #                 #self.set_geom_xpos(f"{link_names[i]}_point{num}", offset_)
        #                 #self.set_geom_quat(f"{link_names[i]}_point{num}", quat=links_quat[i])
        #                 body.add_site(name=f"{self.link_names[i]}_point{num}",
        #                               size= size,
        #                               pos= offset_,
        #                               quat= links_quat[i],
        #                               rgba= [1,0,0,1])
        #                 num += 1

        #                 j_l2p = np.zeros((3,6))
        #                 pos_l2p = offset_
        #                 transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
        #                 hat_transl = self.skew_symmetric(transl)
        #                 j_l2p[:,0:3] = np.eye(3)
        #                 j_l2p[:,3:6] = -hat_transl
        #                 j_l2p_.append(j_l2p)
        #         else:
        #             offset = offsets[i]
        #             for point, offset_ in enumerate(offset):
        #                 for sign in [-1, 1]:  # Place at -i*interval and i*interval
                            
        #                     pos = [offset_[0], offset_[1], offset_[2]+sign * j * links_interval[i]]
        #                     #self.set_geom_xpos(f"{link_names[i]}_point{num}", pos)
        #                     #self.set_geom_quat(f"{link_names[i]}_point{num}", links_quat[i])
        #                     body.add_site(name=f"{self.link_names[i]}_point{num}",
        #                               size=size,
        #                               pos= pos,
        #                               quat= links_quat[i],
        #                               rgba= [1,0,0,1])
        #                     num += 1

        #                     j_l2p = np.zeros((3,6))
        #                     pos_l2p = pos
        #                     transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
        #                     hat_transl = self.skew_symmetric(transl)
        #                     j_l2p[:,0:3] = np.eye(3)
        #                     j_l2p[:,3:6] = -hat_transl
        #                     j_l2p_.append(j_l2p)
        #         if j == syme-1:
        #             for point, vertex in  enumerate(vertex_[i]):
        #                 body.add_site(name=f"{self.link_names[i]}_point{num}",
        #                                     size= size,
        #                                     pos= vertex,
        #                                     quat= links_quat[i],
        #                                     rgba= [1,0,0,1])
        #                 num += 1

        #                 j_l2p = np.zeros((3,6))
        #                 pos_l2p = vertex
        #                 transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
        #                 hat_transl = self.skew_symmetric(transl)
        #                 j_l2p[:,0:3] = np.eye(3)
        #                 j_l2p[:,3:6] = -hat_transl
        #                 j_l2p_.append(j_l2p)
        #     self.totoal_j_l2p.append(j_l2p_)

        for i in range(len(self.link_names)):
            body = self.spec.find_body(self.link_names[i])
            j_l2p_ = []
            #body = self.spec.body(link_names[i])
            num = 1
            if self.link_names[i] in ["link2_sensor_1", "link5_sensor_3"] :
                for point, vertex in  enumerate(vertex_[i]):
                        body.add_site(name=f"{self.link_names[i]}_point{num}",
                                            size= size,
                                            pos= vertex,
                                            quat= links_quat[i],
                                            rgba= [1,0,0,1])
                        num += 1
                        total_body_sensors_num += 1

                        j_l2p = np.zeros((3,6))
                        pos_l2p = vertex
                        transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                        hat_transl = self.skew_symmetric(transl)
                        j_l2p[:,0:3] = np.eye(3)
                        j_l2p[:,3:6] = -hat_transl
                        j_l2p_.append(j_l2p)

            elif self.link_names[i] in ["link2_sensor_2", "link5_sensor_1", "link5_sensor_2"]:
                for j in range(syme):
                    if j == 0:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            #print(f"{link_names[i]}_point{num}")
                            #self.set_geom_xpos(f"{link_names[i]}_point{num}", offset_)
                            #self.set_geom_quat(f"{link_names[i]}_point{num}", quat=links_quat[i])
                            body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size= size,
                                        pos= offset_,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                            num += 1
                            total_body_sensors_num += 1

                            j_l2p = np.zeros((3,6))
                            pos_l2p = offset_
                            transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                            hat_transl = self.skew_symmetric(transl)
                            j_l2p[:,0:3] = np.eye(3)
                            j_l2p[:,3:6] = -hat_transl
                            j_l2p_.append(j_l2p)
                    else:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            for sign in [-1, 1]:  # Place at -i*interval and i*interval
                                
                                pos = [offset_[0], offset_[1], offset_[2]+sign * j * links_interval[i]]
                                #self.set_geom_xpos(f"{link_names[i]}_point{num}", pos)
                                #self.set_geom_quat(f"{link_names[i]}_point{num}", links_quat[i])
                                body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size=size,
                                        pos= pos,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                                num += 1
                                total_body_sensors_num += 1

                                j_l2p = np.zeros((3,6))
                                pos_l2p = pos
                                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                                hat_transl = self.skew_symmetric(transl)
                                j_l2p[:,0:3] = np.eye(3)
                                j_l2p[:,3:6] = -hat_transl
                                j_l2p_.append(j_l2p)
            else:
                for j in range(syme):
                    if j == 0:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            #print(f"{link_names[i]}_point{num}")
                            #self.set_geom_xpos(f"{link_names[i]}_point{num}", offset_)
                            #self.set_geom_quat(f"{link_names[i]}_point{num}", quat=links_quat[i])
                            body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size= size,
                                        pos= offset_,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                            num += 1
                            total_body_sensors_num += 1

                            j_l2p = np.zeros((3,6))
                            pos_l2p = offset_
                            transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                            hat_transl = self.skew_symmetric(transl)
                            j_l2p[:,0:3] = np.eye(3)
                            j_l2p[:,3:6] = -hat_transl
                            j_l2p_.append(j_l2p)
                    else:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            for sign in [-1, 1]:  # Place at -i*interval and i*interval
                                
                                pos = [offset_[0], offset_[1], offset_[2]+sign * j * links_interval[i]]
                                #self.set_geom_xpos(f"{link_names[i]}_point{num}", pos)
                                #self.set_geom_quat(f"{link_names[i]}_point{num}", links_quat[i])
                                body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size=size,
                                        pos= pos,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                                num += 1
                                total_body_sensors_num += 1

                                j_l2p = np.zeros((3,6))
                                pos_l2p = pos
                                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                                hat_transl = self.skew_symmetric(transl)
                                j_l2p[:,0:3] = np.eye(3)
                                j_l2p[:,3:6] = -hat_transl
                                j_l2p_.append(j_l2p)
                    if j == syme-1:
                        for point, vertex in  enumerate(vertex_[i]):
                            body.add_site(name=f"{self.link_names[i]}_point{num}",
                                                size= size,
                                                pos= vertex,
                                                quat= links_quat[i],
                                                rgba= [1,0,0,1])
                            num += 1
                            total_body_sensors_num += 1

                            j_l2p = np.zeros((3,6))
                            pos_l2p = vertex
                            transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                            hat_transl = self.skew_symmetric(transl)
                            j_l2p[:,0:3] = np.eye(3)
                            j_l2p[:,3:6] = -hat_transl
                            j_l2p_.append(j_l2p)
            self.sensors_on_links.append(num-1)
            self.totoal_j_l2p.append(j_l2p_)

        for u in range(len(self.hand_names)):
            body = self.spec.find_body(self.hand_names[u])
            num = 1
            offset = hand_offests[u]
            for point, offset_ in enumerate(offset):
                body.add_site(name=f"{self.hand_names[u]}_point{num}",
                                      size= size,
                                      pos= offset_,
                                      quat= hand_quats[u],
                                      rgba= [1,0,0,1])
                num += 1
                total_hand_sensors_num += 1
                j_l2p = np.zeros((3,6))
                pos_l2p = offset_
                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                hat_transl = self.skew_symmetric(transl)
                j_l2p[:,0:3] = np.eye(3)
                j_l2p[:,3:6] = -hat_transl
                self.hand_j_l2p.append(j_l2p)

        self.total_sensors = total_body_sensors_num
        self.total_sensors_hand = total_hand_sensors_num
        #self.total_sensors += len(hand_offests[0])
        print("total_sensors: ", self.total_sensors + self.total_sensors_hand)
        #self.total_j_l2p = np.array(self.totoal_j_l2p)
        self.hand_j_l2p = np.array(self.hand_j_l2p)
        #print("total_j_l2p: ", self.total_j_l2p.shape)
        #print("hand_j_l2p: ", self.hand_j_l2p.shape)
        self.model = self.spec.compile()
        #print(self.spec.to_xml())
        self.data = mujoco.MjData(self.model)

    def build_less_sensor_points(self):
        ''' attach multiple points to the robot arms, in order to use them for obstacle avoidance method later
        '''
        self.link_names = ['link2_sensor_1','link2_sensor_2','link4_sensor_1','link5_sensor_1', 'link5_sensor_2','link5_sensor_3', 'link7_sensor_1']
        self.hand_names = ['link7_sensor_2']
        self.sensors_on_links = []
        links_radius = []
        links_length = []
        links_center = []
        links_interval = []
        links_quat = []
        offsets = []
        vertex_ = []
        syme = 1
        total_body_sensors_num = 0
        total_hand_sensors_num = 0

        hand_offests = []
        hand_quats = []
        
        #print("total_sensors: ", self.total_sensors)
        for i in range(len(self.link_names)):
            size = self.get_site_size(self.link_names[i])
            link_radius = size[0]
            link_length = size[1]
            link_center = self.get_body_xpos(self.link_names[i])
            link_interval = (link_length) / 2
            link_quat = self.get_site_quat(self.link_names[i])
            x_ = link_radius * np.sin(np.pi/3)
            y_ = link_radius * np.cos(np.pi/3)
            #offset = [[link_radius, 0, 0], [0, link_radius, 0], [-link_radius, 0, 0], [0, -link_radius, 0]]
            #offset = [[0, -link_radius, 0], [x_, y_, 0], [-x_, y_, 0]]
            offset = [[0, 0, 0]]
            #offset = [[0, -link_radius, 0], [link_radius,0 ,0]]
            #vertex = [[0, 0, link_length+link_radius], [0, 0, -link_length-link_radius]]
            vertex = [[0, 0, 0]]
            links_radius.append(link_radius)
            links_length.append(link_length)
            links_center.append(link_center)
            links_interval.append(link_interval)
            offsets.append(offset)
            links_quat.append(link_quat)
            vertex_.append(vertex)

        for u in range(len(self.hand_names)):
            size = self.get_site_size(self.hand_names[u])
            hand_quat = self.get_site_quat(self.hand_names[u])
            axis_x = size[0]
            axis_y = size[1]
            axis_z = size[2]
            # offset_ = [#[axis_x, 0, 0], [0, axis_y, 0], [0, 0, axis_z], [-axis_x, 0, 0], [0, -axis_y, 0], [0, 0, -axis_z],
            #           [axis_x, axis_y, axis_z], [axis_x, -axis_y, axis_z], [-axis_x, axis_y, axis_z], [-axis_x, -axis_y, axis_z],
            #           [axis_x, axis_y, -axis_z], [axis_x, -axis_y, -axis_z], [-axis_x, axis_y, -axis_z], [-axis_x, -axis_y, -axis_z]]
            offset_ = [[0, 0, 0]]
            hand_offests.append(offset_)
            hand_quats.append(hand_quat)

        size = [0.005,0.005,0.005]
        self.totoal_j_l2p = []
        self.hand_j_l2p = []


        for i in range(len(self.link_names)):
            body = self.spec.find_body(self.link_names[i])
            j_l2p_ = []
            #body = self.spec.body(link_names[i])
            num = 1
            if self.link_names[i] in ['link2_sensor_1','link2_sensor_2','link4_sensor_1','link5_sensor_1', 
                                      'link5_sensor_2','link5_sensor_3', 'link7_sensor_1'] :
                for point, vertex in  enumerate(vertex_[i]):
                        body.add_site(name=f"{self.link_names[i]}_point{num}",
                                            size= size,
                                            pos= vertex,
                                            quat= links_quat[i],
                                            rgba= [1,0,0,1])
                        num += 1
                        total_body_sensors_num += 1

                        j_l2p = np.zeros((3,6))
                        pos_l2p = vertex
                        transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                        hat_transl = self.skew_symmetric(transl)
                        j_l2p[:,0:3] = np.eye(3)
                        j_l2p[:,3:6] = -hat_transl
                        j_l2p_.append(j_l2p)

            elif self.link_names[i] in ["link2_sensor_2", "link5_sensor_1", "link5_sensor_2"]:
                for j in range(syme):
                    if j == 0:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            #print(f"{link_names[i]}_point{num}")
                            #self.set_geom_xpos(f"{link_names[i]}_point{num}", offset_)
                            #self.set_geom_quat(f"{link_names[i]}_point{num}", quat=links_quat[i])
                            body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size= size,
                                        pos= offset_,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                            num += 1
                            total_body_sensors_num += 1

                            j_l2p = np.zeros((3,6))
                            pos_l2p = offset_
                            transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                            hat_transl = self.skew_symmetric(transl)
                            j_l2p[:,0:3] = np.eye(3)
                            j_l2p[:,3:6] = -hat_transl
                            j_l2p_.append(j_l2p)
                    else:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            for sign in [-1, 1]:  # Place at -i*interval and i*interval
                                
                                pos = [offset_[0], offset_[1], offset_[2]+sign * j * links_interval[i]]
                                #self.set_geom_xpos(f"{link_names[i]}_point{num}", pos)
                                #self.set_geom_quat(f"{link_names[i]}_point{num}", links_quat[i])
                                body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size=size,
                                        pos= pos,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                                num += 1
                                total_body_sensors_num += 1

                                j_l2p = np.zeros((3,6))
                                pos_l2p = pos
                                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                                hat_transl = self.skew_symmetric(transl)
                                j_l2p[:,0:3] = np.eye(3)
                                j_l2p[:,3:6] = -hat_transl
                                j_l2p_.append(j_l2p)
            else:
                for j in range(syme):
                    if j == 0:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            #print(f"{link_names[i]}_point{num}")
                            #self.set_geom_xpos(f"{link_names[i]}_point{num}", offset_)
                            #self.set_geom_quat(f"{link_names[i]}_point{num}", quat=links_quat[i])
                            body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size= size,
                                        pos= offset_,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                            num += 1
                            total_body_sensors_num += 1

                            j_l2p = np.zeros((3,6))
                            pos_l2p = offset_
                            transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                            hat_transl = self.skew_symmetric(transl)
                            j_l2p[:,0:3] = np.eye(3)
                            j_l2p[:,3:6] = -hat_transl
                            j_l2p_.append(j_l2p)
                    else:
                        offset = offsets[i]
                        for point, offset_ in enumerate(offset):
                            for sign in [-1, 1]:  # Place at -i*interval and i*interval
                                
                                pos = [offset_[0], offset_[1], offset_[2]+sign * j * links_interval[i]]
                                #self.set_geom_xpos(f"{link_names[i]}_point{num}", pos)
                                #self.set_geom_quat(f"{link_names[i]}_point{num}", links_quat[i])
                                body.add_site(name=f"{self.link_names[i]}_point{num}",
                                        size=size,
                                        pos= pos,
                                        quat= links_quat[i],
                                        rgba= [1,0,0,1])
                                num += 1
                                total_body_sensors_num += 1

                                j_l2p = np.zeros((3,6))
                                pos_l2p = pos
                                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                                hat_transl = self.skew_symmetric(transl)
                                j_l2p[:,0:3] = np.eye(3)
                                j_l2p[:,3:6] = -hat_transl
                                j_l2p_.append(j_l2p)
                    if j == syme-1:
                        for point, vertex in  enumerate(vertex_[i]):
                            body.add_site(name=f"{self.link_names[i]}_point{num}",
                                                size= size,
                                                pos= vertex,
                                                quat= links_quat[i],
                                                rgba= [1,0,0,1])
                            num += 1
                            total_body_sensors_num += 1

                            j_l2p = np.zeros((3,6))
                            pos_l2p = vertex
                            transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                            hat_transl = self.skew_symmetric(transl)
                            j_l2p[:,0:3] = np.eye(3)
                            j_l2p[:,3:6] = -hat_transl
                            j_l2p_.append(j_l2p)
            self.sensors_on_links.append(num-1)
            self.totoal_j_l2p.append(j_l2p_)

        for u in range(len(self.hand_names)):
            body = self.spec.find_body(self.hand_names[u])
            num = 1
            offset = hand_offests[u]
            for point, offset_ in enumerate(offset):
                body.add_site(name=f"{self.hand_names[u]}_point{num}",
                                      size= size,
                                      pos= offset_,
                                      quat= hand_quats[u],
                                      rgba= [1,0,0,1])
                num += 1
                total_hand_sensors_num += 1
                j_l2p = np.zeros((3,6))
                pos_l2p = offset_
                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                hat_transl = self.skew_symmetric(transl)
                j_l2p[:,0:3] = np.eye(3)
                j_l2p[:,3:6] = -hat_transl
                self.hand_j_l2p.append(j_l2p)

        self.total_sensors = total_body_sensors_num
        self.total_sensors_hand = total_hand_sensors_num
        #self.total_sensors += len(hand_offests[0])
        print("total_sensors: ", self.total_sensors + self.total_sensors_hand)
        #self.total_j_l2p = np.array(self.totoal_j_l2p)
        self.hand_j_l2p = np.array(self.hand_j_l2p)
        #print("total_j_l2p: ", self.total_j_l2p.shape)
        #print("hand_j_l2p: ", self.hand_j_l2p.shape)
        self.model = self.spec.compile()
        #print(self.spec.to_xml())
        self.data = mujoco.MjData(self.model)
        
    '''
    # ------------------------------------------read from model and data ----------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    '''
    def get_ee_position(self):
        # gripper
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        return grip_pos
        
    def get_joints_position(self):
        joints = np.zeros((self.rotation_joints.shape[0], 3))
        # joints number is 7
        # but only joint 2,3,5 are at the end of the whole separated arm
        for i in range(self.rotation_joints.shape[0]):
            j = self.rotation_joints[i]
            #joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{j}')
            #joint_body_id = self.model.jnt_bodyid[joint_id]
            #joint_position = self.data.xpos[joint_body_id]

            '''the link position indicated in the xml file is actually the joint position 
            according to https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/common/franka_arm.xacro
            '''
            joint_position = self.get_body_xpos(f'link{j}')
            #joint_position = self._utils.get_body_xpos(self.model, self.data, f'link{j}')
            joints[i] = np.array([joint_position[0], joint_position[1], joint_position[2]])
        
        #print("joints: ", joints)
        return joints
    
    def get_joints_end_position(self):
        # order: joint1, joint3, joint5, end effector
        #start_positions = np.zeros((len(self.rotation_joints)+1,3))
        
        end_effector_position = self.get_ee_position()
        joints_position = self.get_joints_position()
        #print("joints_position: ", joints_position.shape)
        #print("end_effector_position: ", end_effector_position)
        #print("start_positions: ", start_positions.shape)

        #for i in range(0, start_positions.shape[0]-1):
        #    start_positions[i] = np.array([joints_position[i][0], joints_position[i][1], joints_position[i][2]])
        #start_positions[-1] = np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2]])
        start_positions = np.vstack((joints_position, end_effector_position))
        return start_positions

    def get_sensors_end_position(self):
        sensors_pos = self.get_sensor_position()
        ee_pos = self.get_ee_position()
        sensors_pos = np.vstack((sensors_pos, ee_pos))
        return sensors_pos

    def get_joints_sensors_end_position(self):
        joints_pos = self.get_joints_position()
        sensors_pos = self.get_sensors_end_position()
        joints_sensors_pos = np.vstack((joints_pos, sensors_pos))
        #print("start positions shape: ", joints_sensors_pos.shape)
        return joints_sensors_pos

    def get_sensor_position(self):
        sensors_ee_pos = np.zeros((self.total_sensors + self.total_sensors_hand, 3))
        #sensors_on_link = self.total_sensors//len(self.link_names)

        num = 0
        for _, name in enumerate(self.link_names):
            sensors_on_link = self.sensors_on_links[_]
            for i in range(sensors_on_link):
                name_ = name + f'_point{i+1}'
                sensor_pos = self._utils.get_site_xpos(self.model, self.data, name_)
                #sensor_pos = self.get_site_xpos(name_)
                sensors_ee_pos[num] = np.array([sensor_pos[0], sensor_pos[1], sensor_pos[2]])
                num += 1

        for _, name in enumerate(self.hand_names):
            for i in range(self.total_sensors_hand):
                name_ = name + f'_point{i+1}'
                sensor_pos = self._utils.get_site_xpos(self.model, self.data, name_)
                #sensor_pos = self.get_site_xpos(name_)
                sensors_ee_pos[num] = np.array([sensor_pos[0], sensor_pos[1], sensor_pos[2]])
                num += 1
        return sensors_ee_pos

    def get_goal_position(self):
        goal_pos = self._utils.get_site_xpos(self.model, self.data, "target_pos")
        return goal_pos
    
#---------------------------body-------------------------------------------------
    def get_body_xpos(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        #return self.model.body_xpos[body_id]
        return self.data.xpos[body_id]

    def set_body_xpos(self, name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        #self.model.body_xpos[body_id] = pos
        self.data.xpos[body_id] = pos

    def get_body_pos(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_pos[body_id]
        #return self.data.body_xpos[body_id]

    def set_body_pos(self, name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_pos[body_id] = pos
        #self.data.body_xpos[body_id] = pos

    def get_body_xquat(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        #return self.model.body_quat[body_id]
        return self.data.xquat[body_id]
    
    def set_body_xquat(self, name, quat):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        #self.model.body_quat[body_id] = quat
        self.data.xquat[body_id] = quat

    def set_body_size(self, name, size):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_size[body_id] = size
        #self.data.body_size[body_id] = size

    def get_body_size(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_size[body_id]
        #return self.data.body_size[body_id]
    
#---------------------------geom----------------------------------------------------------
    def get_geom_size(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_size[geom_id]
        #return self.data.geom_size[geom_id]

    def set_geom_size(self, name, size):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_size[geom_id] = size
        #self.data.geom_size[geom_id] = size
        #self.data.size[geom_id] = size
        #adjust rbound (radius of bounding sphere for which collisions are not checked)
        #self.model.geom_rbound[geom_id] = np.sqrt(np.sum(np.square(self.model.geom_size[geom_id])))

    def get_geom_xpos(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_pos[geom_id]
        #return self.data.geom_xpos[geom_id]
    
    def set_geom_xpos(self, name, pos):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_pos[geom_id] = pos
        #self.data.geom_xpos[geom_id] = pos

    def get_geom_quat(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_quat[geom_id]
        #return self.data.geom_quat[geom_id]
    
    def set_geom_quat(self, name, quat):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_quat[geom_id] = quat
        #self.data.geom_quat[geom_id] = quat

#---------------------------site---------------------------------------------
    def get_site_size(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_size[site_id]

    def set_site_size(self, name, size):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_size[site_id] = size

    def get_site_pos(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_pos[site_id]
    
    def get_site_xpos(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[site_id]

    def set_site_pos(self, name, pos):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_pos[site_id] = pos

    def get_site_quat(self,name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_quat[site_id]
    
    def get_site_xquat(self,name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_xquat[site_id]
    
    def set_site_quat(self, name, quat):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_quat[site_id] = quat
