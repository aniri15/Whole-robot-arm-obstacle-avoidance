#from mujoco_py import load_model_from_xml
import mujoco
import mujoco.viewer
from gymnasium_robotics.utils import mujoco_utils
from typing import Optional
import mediapy as media
import time
import numpy as np
import matplotlib.pyplot as plt
from whole_robot_arm_multi_obstacles_avoidance.extended_roam_examples.human_obstacle.franka_human_avoider_ import MayaviAnimator
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

class TestSensorsControlEnv_():
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
        #self.bulid_goal_object(goal)
        self.build_sensor_points()
        self.collision_visualization()
        self.initial_simulation()
        
    def initial_simulation(self):
        for name, value in self.initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        mujoco.mj_forward(self.model, self.data)
    
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
    
    def skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])


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


    
    def get_required_sensor_jac(self, link_sensor, link_index, sensor_index):
        link_sensor_id = self.model.body(link_sensor).id
        jac_q2l = np.zeros((6, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac_q2l[:3], jac_q2l[3:], self.data.body(link_sensor_id).xpos, link_sensor_id)
        jac_q2p = self.jacobian_joint_angles2point(self.data.body(link_sensor_id).xpos, link_sensor,jac_q2l, link_index, sensor_index)
        return jac_q2p
    

    def test_sensor_joint_control(self, index, avoid_vel):
        #index = 30
        print("indx: ", index)
        #print("shape of weights_sensors_obstacles: ", np.array(weights_sensors_obstacles).shape)
        #print("maximum value",weights_sensors_obstacles[index])
        #print("index: ", index)
        #breakpoint()
        num_sensors_link = self.total_sensors // len(self.link_names)
        link_idex = index // num_sensors_link
        self.sensors_index = index - (link_idex * num_sensors_link)
        self.link_sensor = self.link_names[link_idex]
        link_sensor_id = self.model.body(self.link_sensor).id
        self.test_joint_pos = self.data.body(link_sensor_id).xpos.copy()
        sensor_name = f"{self.link_sensor}_point{self.sensors_index+1}"
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sensor_name)
        print("sensor_name: ", sensor_name)
        print("link_sensor position: ", self._utils.get_site_xpos(self.model, self.data, sensor_name))
        time_start = self.data.time
        time_end = 0
        time_diff = 0
        self.change_sensor_color_size(self.link_sensor, self.sensors_index, [0,1,0,1])
        while (np.linalg.norm(time_diff)<= self.delta_time):
            jac = self.get_required_sensor_jac(self.link_sensor, link_idex, self.sensors_index)

            n_sensor = jac.shape[1]
            I_sensor = np.identity(n_sensor)
            product_sensor = jac.T @ jac + self.damping * I_sensor

            if np.isclose(np.linalg.det(product_sensor), 0):
                j_inv_sensor = np.linalg.pinv(product_sensor) @ jac.T
            else:
                j_inv_sensor = np.linalg.inv(product_sensor) @ jac.T

            q_vel = j_inv_sensor @ avoid_vel

            self.data.qvel = q_vel
            mujoco.mj_step(self.model, self.data)
            sensor_pos = self._utils.get_site_xpos(self.model, self.data, sensor_name)
            vel = sensor_pos - self.test_joint_pos
            self.vel_storage.append(vel)
            print("after link sensor position: ", sensor_pos)
            print("velocity: ", vel)
            #self.trajectory_qpos.append(self.data.qpos.copy())
            self.store_data()
            time_end = self.data.time
            time_diff = time_end - time_start

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
    
    def jacobian_joint_angles2point(self, point, link, j_q2l, num, sensor_index):
        if link != self.hand_names[0]:
            j_l2p = self.totoal_j_l2p[num][sensor_index]
            j_q2p = j_l2p @ j_q2l
        else:
            j_l2q = self.hand_j_l2p[sensor_index]
            j_q2p = j_l2q @ j_q2l
        return np.array(j_q2p)
    

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
    

    def test_franka_sensors_control(self,index, velocity):
        it_max = self.it_max

        for ii in range(it_max):
            time_start = time.time()
            print('iteration: ', ii)
            self.test_sensor_joint_control(index, velocity)

            time_end = time.time()
            print("time_diff: ", time_end - time_start)
            print("--------------------------------------------")



    def check_goal_reach(self, position, goal):
        if np.linalg.norm(position - goal) < self.tol:
            return True
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


        
    # ------------------read from model and data ----------------------------------------
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


# main function
if __name__ == '__main__':
    # initializations
    folder_path = os.path.dirname(os.path.abspath(__file__))
    scene_path = folder_path + "/franka_emika_panda/scene2.xml"
    print("scene_path: ", scene_path)
    env = TestSensorsControlEnv_(scene_path)

    # test the sensors control
    # specify the index of the sensor and the velocity
    index = 30
    velocity = np.array([0.3,-0.3,0.3])
    env.test_franka_sensors_control(index, velocity)
    print('------------------------------------')
    print("start replay")
    env.replay()
    