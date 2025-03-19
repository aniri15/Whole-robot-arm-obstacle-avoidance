#from mujoco_py import load_model_from_xml
import mujoco
import mujoco.viewer
from gymnasium_robotics.utils import mujoco_utils
from typing import Optional
import mediapy as media
import time
import numpy as np
import matplotlib.pyplot as plt
from extended_roam_examples.human_obstacle.franka_human_avoider_ import MayaviAnimator
import os
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
#import pybullet
from scipy.spatial.transform import Rotation as R
import time

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.4,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([1.5, 0, 0.75]),
}
DEFAULT_SIZE = 480

class End_Effector_Joints_Env_():
    def __init__(
            self,
            scene_path,
            goal=None,
            control_mode='position',
            obj_range=0.1,
            target_range=0.1,
            num_obst=1,
            obj_goal_dist_threshold=0.03,
            obj_gripper_dist_threshold=0.02,
            max_vel=0.1,
            obj_lost_reward=-0.2,
            collision_reward=-1.,
            scenario=None,
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
        self.it_max = 100             #horizon

        # update step size
        self.step_size_position = 0.01
        self.step_size_ik = 0.1 #0.5
        self.delta_time = 0.01      # 100hz

        # initialize for ik controller
        self.tol = 0.05   # unit: m, tolerance is 5cm
        self.length_tol = 0.01
        self.damping = 1e-3
        self.jacp = np.zeros((3, self.model.nv)) #translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) #rotational jacobian
        self.trajectory_qpos = []
        self.data_storage = []
        self.model_storage = []
        self.point_storage = []
        self.link_length_storage = []
        self.time_storage = []
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
        }
        self.rotation_joints = np.array([4]) #np.array([2,4,6])# rotation joints at the 2st, 4rd and 6th joints(at the end of the arms)
        self.rotation_joints_weights = np.array([0.5])
        self.ee_weight = 1
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
        self.bulid_goal_object(goal)
        self.build_sensor_points()
        self.collision_visualization()
        self.initial_simulation()
        
    def initial_simulation(self):
        for name, value in self.initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        mujoco.mj_forward(self.model, self.data)
    

    def move_franka_robot_norm_dir(self,init_position, goal):
        it_max = self.it_max
        sampled_points = init_position
        self.animator = MayaviAnimator(it_max=it_max, current_position=sampled_points, 
                                  attractor_position=goal, dynamic_human=self.dynamic_human, obstacle=self.obstacle)    
        self.animator.obstacle_initiation()
        if self.obstacle:
            self.human_obstacle = self.animator.human_obstacle_3d
            self.adjust_obstacle_init()
        next_desired_pos = []
        self.animator.set_up_franka()

        for ii in range(it_max):
            time_start = time.time()
            velocity= self.animator.run_norm(ii=ii)
            print('iteration: ', ii)

            for jj in range(len(velocity)):
                joint_next_desired_pos = sampled_points[jj] + velocity[jj]* self.delta_time  #self.step_size_position
                next_desired_pos.append(joint_next_desired_pos)

            self.joints_resolved_rate_motion_control_norm_dir(next_desired_pos, velocity)

            init_position = self.get_joints_end_position()
            sampled_points = init_position

            self.animator.update_multiple_points_trajectories(sampled_points, ii)
            if self.obstacle:
                self.human_obstacle = self.animator.human_obstacle_3d
                self.adjust_obstacle()

            time_end = time.time()
            print("time_diff: ", time_end - time_start)
            
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
        
    def joint_velocity_control(self, goal, velocity, init_q, body_id, joints_id):
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


    # -----------------------------------assistive functions-----------------------------------

    def change_sensor_color_size(self, link_sensor, sensor_index, color):
        num = sensor_index+1
        link_sensor_name = f"{link_sensor}_point{num}"
        link_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, link_sensor_name)
        print("link_sensor_name: ", link_sensor_name)
        self.adjust_sensors_name.append(link_sensor_name)
        self.model.site_rgba[link_sensor_id] = color
        self.model.site_size[link_sensor_id] = 0.015

    def unchange_sensor_color_size(self):
        color = [1,0,0,1]
        num = self.sensors_index +1
        link_sensor_name = f"{self.link_sensor}_point{num}"
        link_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, link_sensor_name)
        print("link_sensor_name: ", link_sensor_name)
        self.model.site_rgba[link_sensor_id] = color
        self.model.site_size[link_sensor_id] = 0.005

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
                self.collision_name.append((name_body1, name_body2))
                #print("contact", c.geom1, c.geom2)
                #print("contact", c.pos)
                #print("contact", c.frame)
                #print("contact", c.dist)

        #print("collision number", num_contacts)
        return num_contacts

    def replay(self):
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
                    
                    #print("weights", self.weights_storage)
                    print("---------------------one loop done---------------------")
                    time.sleep(0.5)
                    # Rudimentary time keeping, will drift relative to wall clock.
                    #time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    #if time_until_next_step > 0:
                    #    time.sleep(time_until_next_step)

    def check_goal_reach(self, position, goal):
        if np.linalg.norm(position - goal) < self.tol:
            return True
        return False

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

    
    def store_data(self):
        # store the data of the simulation
        self.model_storage.append((self.model.geom_pos.copy(), self.model.geom_quat.copy()))
        self.model_size_color_storage.append((self.model.site_size.copy(), self.model.site_rgba.copy()))
        self.trajectory_qpos.append(self.data.qpos.copy())

    # -----------------------------------build objects-----------------------------------
    def bulid_goal_object(self,goal):
        name = "goal"
        position = goal
        size = [0.02, 0.02, 0.02]
        self.spec.worldbody.add_site(name= name +'_site',
                type=mujoco.mjtGeom.mjGEOM_BOX,
                rgba=[0, 1, 0, 1],
                size= size,
                pos= position)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        #mujoco.mj_forward(self.model, self.data)

    def build_sensor_points(self):
        ''' attach multiple points to the robot arms, in order to use them for obstacle avoidance method later
        '''
        self.link_names = ['link2_sensor_1','link2_sensor_2','link4_sensor_1','link5_sensor_1', 'link5_sensor_2','link5_sensor_3', 'link7_sensor_1']
        self.hand_names = ['link7_sensor_2']
        links_radius = []
        links_length = []
        links_center = []
        links_interval = []
        links_quat = []
        offsets = []
        vertex_ = []
        syme = 1

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
            offset_ = [[axis_x, 0, 0], [0, axis_y, 0], [0, 0, axis_z], [-axis_x, 0, 0], [0, -axis_y, 0], [0, 0, -axis_z],
                      [axis_x, axis_y, axis_z], [axis_x, -axis_y, axis_z], [-axis_x, axis_y, axis_z], [-axis_x, -axis_y, axis_z],
                      [axis_x, axis_y, -axis_z], [axis_x, -axis_y, -axis_z], [-axis_x, axis_y, -axis_z], [-axis_x, -axis_y, -axis_z]]
            hand_offests.append(offset_)
            hand_quats.append(hand_quat)


        self.total_sensors = (1*len(offset)+ (syme-1)*2*len(offset)+2)*len(self.link_names)
        self.total_sensors_hand = len(offset_)*len(self.hand_names)
        #self.total_sensors += len(hand_offests[0])
        print("total_sensors: ", self.total_sensors + self.total_sensors_hand)
        size = [0.005,0.005,0.005]
        self.totoal_j_l2p = []
        self.hand_j_l2p = []

        for i in range(len(self.link_names)):
            body = self.spec.find_body(self.link_names[i])
            j_l2p_ = []
            #body = self.spec.body(link_names[i])
            num = 1
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

                        j_l2p = np.zeros((3,6))
                        pos_l2p = vertex
                        transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                        hat_transl = self.skew_symmetric(transl)
                        j_l2p[:,0:3] = np.eye(3)
                        j_l2p[:,3:6] = -hat_transl
                        j_l2p_.append(j_l2p)
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
                j_l2p = np.zeros((3,6))
                pos_l2p = offset_
                transl = np.array([pos_l2p[0], pos_l2p[1], pos_l2p[2]])
                hat_transl = self.skew_symmetric(transl)
                j_l2p[:,0:3] = np.eye(3)
                j_l2p[:,3:6] = -hat_transl
                self.hand_j_l2p.append(j_l2p)

        self.total_j_l2p = np.array(self.totoal_j_l2p)
        self.hand_j_l2p = np.array(self.hand_j_l2p)
        print("total_j_l2p: ", self.total_j_l2p.shape)
        print("hand_j_l2p: ", self.hand_j_l2p.shape)
        self.model = self.spec.compile()
        #print(self.spec.to_xml())
        self.data = mujoco.MjData(self.model)

    def adjust_obstacle(self):
        # set the initial position, size and orientation of the obstacles
        #print('------------------------------------------------------')
        obstacles = self.human_obstacle
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

        #self.data = mujoco.MjData(self.model)
        #self.data.qpos = self.trajectory_qpos[-1]
        mujoco.mj_forward(self.model, self.data)
        
        #self.model = self.spec.compile()

    def adjust_obstacle_init(self):
        # set the initial position, size and orientation of the obstacles
        i = 0
        obstacles = self.human_obstacle
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
                        rgba=[1, 0, 0, 1],
                        size= size,
                        pos= position)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.initial_simulation()
        self.collision_visualization()
        #mujoco.mj_forward(self.model,self.data)
        #print(self.spec.to_xml())
    
    # ----------------------------------------math functions----------------------------------------
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
    
    def skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    
    # ------------------read from model and data ----------------------------------------
    # ---------------------jacobian------------------------------------------------------
    def get_ee_qpos(self,body_id,ref_vel):
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

    def get_sensors_jac(self):
        sensors_jac = np.zeros((self.total_sensors, 3, self.model.nv))
        i = 0
        for _, link_sensor in enumerate(self.link_names):
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
        #print("num_link_sensor: ", num_link_sensor)
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
    # ---------------------------------positions-----------------------------------------
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

            '''the link position indicated in the xml file is actually the joint position 
            according to https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/common/franka_arm.xacro
            '''
            joint_position = self.get_body_xpos(f'link{j}')
            #joint_position = self._utils.get_body_xpos(self.model, self.data, f'link{j}')
            joints[i] = np.array([joint_position[0], joint_position[1], joint_position[2]])
        
        #print("joints: ", joints)
        return joints
    
    def get_joints_end_position(self):
        end_effector_position = self.get_ee_position()
        joints_position = self.get_joints_position()
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
        print("joints_sensors_pos: ", joints_sensors_pos.shape)
        return joints_sensors_pos

    def get_sensor_position(self):
        sensors_ee_pos = np.zeros((self.total_sensors + self.total_sensors_hand, 3))
        sensors_on_link = self.total_sensors//len(self.link_names)
        num = 0
        for _, name in enumerate(self.link_names):
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

    #def create_obstacle_ellipse(self, name, pos, quat, size):
