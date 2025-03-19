#from mujoco_py import load_model_from_xml
import mujoco
import mujoco.viewer
from gymnasium_robotics.utils import mujoco_utils
from typing import Optional
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt
from whole_robot_arm_multi_obstacles_avoidance.extended_roam_examples.human_obstacle.franka_human_avoider_ import MayaviAnimator
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
#import pybullet
from scipy.spatial.transform import Rotation as R
import time


class MujocoRobotInterface:
    def __init__(self, model, data, damping, **kwargs):
        self.model = model
        self.data = data
        self._mujoco = mujoco
        self._utils = mujoco_utils
        self.damping = damping

        self.rotation_joints = None
        self.link_names = None
        self.hand_names = None
        self.total_sensors = None
        self.total_sensors_hand = None
        self.totoal_j_l2p = None
        self.hand_j_l2p = None


    '''
    # --------------------------------------------- qpos calculation -------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------
    '''

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
        sensors_q_vel_max = []
        sensors_q_vel = []
        #print("weights_sensors: ", np.array(weights_sensors).shape)
        weights_sensors = np.array(weights_sensors)[len(self.rotation_joints):,:]  # remove the weights of the joints
        weights_sensors_obstacles = weights_sensors[:,:-1]   # remove the last column which is the weight of convergence dynamics
        #weights_sensors_obstacles = weights_sensors[:][:-1]
        print("weights_sensors_obstacles: ", weights_sensors_obstacles.shape)
        #print("weights ", weights_sensors_obstacles)
        
        #if any(np.any(arr > 0.8) for arr in weights_sensors_obstacles):
        if np.any(weights_sensors_obstacles > 0.3):
            index = np.argmax(weights_sensors_obstacles)
            index = index // weights_sensors_obstacles.shape[1]
            print("indx: ", index)
            print("shape of weights_sensors_obstacles: ", np.array(weights_sensors_obstacles).shape)
            print("maximum value",weights_sensors_obstacles[index])
            #print("weights ", weights_sensors_obstacles)
            #print("index: ", index)
            #breakpoint()
            if index < self.total_sensors:
                num_sensors_link = self.total_sensors // len(self.link_names)
                link_idex = index // num_sensors_link
                self.sensors_index = index - (link_idex * num_sensors_link)
                self.link_sensor = self.link_names[link_idex]
            else:
                self.sensors_index = index - self.total_sensors
                self.link_sensor = self.hand_names[0]
                link_idex = 0
            jac = self.get_required_sensor_jac(self.link_sensor, link_idex, self.sensors_index)
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
        #print("num_link_sensor: ", num_link_sensor)
        j = []
        for i in range(num_link_sensor):
            # j_l2p = np.zeros((3,6))
            # pos = self.get_site_pos(f"{link}_point{i}")
            # transl = np.array([pos[0], pos[1], pos[2]])
            # hat_transl = self.skew_symmetric(transl)
            # j_l2p[:,0:3] = np.eye(3)
            # j_l2p[:,3:6] = -hat_transl
            j_l2p = self.totoal_j_l2p[num][i]
            j_q2p = j_l2p @ j_q2l
            j.append(j_q2p)
        return np.array(j)
    
    def jacobian_joint_angles2point(self, point, link, j_q2l, num, sensor_index):
        # get the homogeneous transformation matrix of the center of the link to the point
        num_link_sensor = self.total_sensors // len(self.link_names)
        #print("num_link_sensor: ", num_link_sensor)
        j = []
        #for i in range(num_link_sensor):
            # j_l2p = np.zeros((3,6))
            # pos = self.get_site_pos(f"{link}_point{i}")
            # transl = np.array([pos[0], pos[1], pos[2]])
            # hat_transl = self.skew_symmetric(transl)
            # j_l2p[:,0:3] = np.eye(3)
            # j_l2p[:,3:6] = -hat_transl
        if link != self.hand_names[0]:
            j_l2p = self.totoal_j_l2p[num][sensor_index]
            j_q2p = j_l2p @ j_q2l
        else:
            j_l2q = self.hand_j_l2p[sensor_index]
            j_q2p = j_l2q @ j_q2l
        return np.array(j_q2p)


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
