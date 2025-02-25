#from mujoco_py import load_model_from_xml
import mujoco
#import gymnasium as gym
from gymnasium import spaces
#from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils import mujoco_utils
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from typing import Optional
import mediapy as media

import time
import numpy as np
from envs.config import register_custom_envs
from envs.custom_scenarios import scenarios
#from envs.ik_controller import LevenbegMarquardtIK
import matplotlib.pyplot as plt
from franka_gym_test.franka_human_avoider import MayaviAnimator
import os
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
import pybullet

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.4,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([1.5, 0, 0.75]),
}
DEFAULT_SIZE = 480


class LevenbegMarquardtIK:
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, init_q, body_id):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        delta_q_norm = 0
        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
            error_norm = np.linalg.norm(error)
            delta_q_norm1 = np.linalg.norm(delta_q)
            # if the 
            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            delta_q_norm = delta_q_norm1
        return self.data  
    
class FrankaHumanEnv(EzPickle):
    def __init__(
            self,
            n_substeps=20,
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
            **kwargs
    ):
        self._mujoco = mujoco
        self._utils = mujoco_utils
        self.control_mode = control_mode
        #self.human_shape = HumanShape(pos=[0.5, 0.5, 0])  # Position human obstacle
        #self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_test.xml")
        self.fullpath = "/home/aniri/nonlinear_obstacle_avoidance/franka_gym_test/envs2/franka_emika_panda/scene.xml"
        self.FRAMERATE = 60 #(Hz)
        self.height = 300
        self.width = 300
        self.n_frames = 60
        self.frames = []
        
        # Initialize MuJoCo environment with the combined model
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        #self.model = load_model_from_xml(full_model_xml)
        self.render_mode = 'human'
        self.init_qpos = self.data.qpos.copy()
        
        # Todo: two render methods are used, need to be unified
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, DEFAULT_CAMERA_CONFIG
        )
        #Make a new camera, move it to a closer distance.
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 4
        self.step_size_position = 0.1
        self.it_max = 80
        # initialize for ik controller
        self.step_size_ik = 0.5
        self.tol = 0.005
        self.alpha = 0.5
        self.damping = 0.15
        self.jacp = np.zeros((3, self.model.nv)) #translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) #rotational jacobian
        self.trajectory_qpos = []
        # simulate and render
        # #Render and save frames.
        # if len(self.frames) < self.data.time * self.FRAMERATE:
        #     self.renderer.update_scene(self.data)
        #     pixels = self.renderer.render()
        #     self.frames.append(pixels)
        # media.show_video(self.frames, fps=30)
        self.obstacle_number = 7
        
    
    def render(self):
        self.mujoco_renderer.render(self.render_mode)
        #self.renderer.update_scene(self.data)

    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, init_q, body_id):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        error_norm = np.linalg.norm(error)
        delta_q_norm = 0

        self.trajectory_qpos.append(self.data.qpos.copy())
        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size_ik * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)

            self.trajectory_qpos.append(self.data.qpos.copy())
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            #update the scene
            #self.renderer.update_scene(self.data)
            #calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
            #error_norm = np.linalg.norm(error)
            delta_q_norm1 = np.linalg.norm(delta_q)
            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            delta_q_norm = delta_q_norm1
        return self.data  
    
    def use_ik_controller(self, goal):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        # jacp = np.zeros((3, self.model.nv)) #translation jacobian
        # jacr = np.zeros((3, self.model.nv)) #rotational jacobian
        #goal = [0.49, 0.13, 0.59]
        # step_size = 0.5
        # tol = 0.05
        # alpha = 0.5
        # damping = 0.15
        #init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        goal = goal[0]
        print("goal: ", goal)
        print("init_q: ", init_q)
        print("current_pose: ", self.data.body(body_id).xpos)
        #ik = LevenbegMarquardtIK(self.model, self.data, step_size, tol, alpha, jacp, jacr, damping)
        self.data = self.calculate(goal, init_q, body_id) #calculate the qpos
        #result = self.data.qpos.copy()
        #mujoco.mj_forward(self.model, self.data)
        # simulate and render
        #Render and save frames.
        print("time: ", self.data.time)
        #if len(self.frames) < self.data.time * self.FRAMERATE:
        if len(self.frames) < 10000:
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            self.frames.append(pixels)
        #media.show_video(self.frames, fps=30)

    def compare_frames(self, target):
        #Plot results
        print("Results")
        result = self.data.qpos.copy()

        self.data.qpos = self.init_qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        target_plot = self.renderer.render()

        self.data.qpos = result
        mujoco.mj_forward(self.model, self.data)
        #result_point = data.body('wrist_3_link').xpos
        result_point = self.data.body('robot:gripper_link').xpos
        self.renderer.update_scene(self.data, self.camera)
        result_plot = self.renderer.render()

        print("testing point =>", target)
        print("Levenberg-Marquardt result =>", result_point, "\n")

        images = {
            'Testing point': target_plot,
            'Levenberg-Marquardt result': result_plot,
        }

        #media.show_images(images)
        return images
    
    # Function to display images using matplotlib
    def show_images(self,images):
        fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
        if len(images) == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one image
        
        for ax, (title, img) in zip(axes, images.items()):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.show()
    
    def real_time_render(self):
        m = self.model
        d = self.data
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 80:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def real_time_movement_render(self, init_position, goal):
        m = self.model
        d = self.data
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 80:
                step_start = time.time()

                self.move_robot(init_position, goal)
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                #mujoco.mj_step(m, d)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def replay(self):
        # use the mujoco viewer to replay the simulation
        m = self.model
        d = self.data
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 80:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                for i in range(len(self.trajectory_qpos)):
                    d.qpos = self.trajectory_qpos[i]
                    mujoco.mj_step(m, d)

                    # Example modification of a viewer option: toggle contact points every two seconds.
                    #with viewer.lock():
                    #    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                    # Pick up changes to the physics state, apply perturbations, update options from GUI.
                    viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def move_robot(self,init_position, goal):
        self.mujoco_renderer.render(self.render_mode)
        it_max = self.it_max
        animator = MayaviAnimator(it_max=it_max, end_effector_position=init_position, attractor_position=goal)
        # animator.setup(n_grid=1)
        # while True:
        #     env.render()
            
        animator.setup()
        self.human_obstacle = animator.human_obstacle_3d

        animator.dynamic_human = True
        for ii in range(it_max):
            print('--------------------------------------')
            #update trajectory
            velocity = animator.run(ii=ii)
            if not ii % 10:
                print(f"it={ii}")
            print('iteration: ', ii)
            #print("velocity: ", velocity)
            #print("obs: ", obs)
            print("velocity: ", velocity)
            next_desired_pos = init_position + velocity*self.step_size_position
            #print("next_pos: ", next_pos)
            self.use_ik_controller(next_desired_pos)

            # update end effector position
            init_position = self.get_ee_position()
            init_position = np.array([[init_position[0], init_position[1], init_position[2]]])
            animator.update_trajectories(init_position, ii)


        #env.render()


    # ------------------read from the xml file ------------------
    def get_ee_position(self):
        # gripper
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        return grip_pos
        
    def get_goal_position(self):
        goal_pos = self._utils.get_site_xpos(self.model, self.data, "target_pos")
        return goal_pos
    
    def _get_obs(self):
        # robot
        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.model, self.data, self._model_names.joint_names)
        gripper_placement = [robot_qpos[-1]]

        # gripper
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")

        # object
        object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        object_velp = self._utils.get_site_xvelp(self.model, self.data, "object0")
        object_size = self.get_geom_size("object0")

        # object-gripper (we only need this for the reward)
        object_rel_pos = object_pos - grip_pos
        object_gripper_dist = np.linalg.norm(object_rel_pos.ravel())

        # obstacles
        obstacles = []
        for n in range(self.obstacle_number):
            pos = self._utils.get_joint_qpos(self.model, self.data, f'obstacle{n}' + ':joint')[:3]
            vel = self.custom_get_joint_qvel(self.model, self.data, f'obstacle{n}' + ':joint')[:3]
            size = self.get_geom_size(f'obstacle{n}')
            obstacles.append(np.concatenate([pos, vel, size]))
        if not self.scenario:
            # randomize order to avoid overfitting to the first obstacle during curriculum learning
            self.np_random.shuffle(obstacles)
        obst_states = np.concatenate(obstacles)

        # achieved goal, essentially the object position
        achieved_goal = np.squeeze(object_pos.copy())

        # collisions
        collision = self._check_collisions()

        obs = np.concatenate(
            [
                # robot_qpos,
                # robot_qvel,
                gripper_placement,
                grip_pos,
                grip_velp,
                object_pos,
                object_size,
                object_velp,
                obst_states
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "object_gripper_dist": object_gripper_dist.copy(),
            "collision": collision,
        }

    def transform_degree_to_quat(self, degree, direction):
        """Next try is to calculate the values according to the tutorials formula:
        Form:
        quat = [x, y, z, w]
        degree = [x_degree, y_degree, z_degree]
        direction = [x_dir, y_dir, z_dir]
        Example: rotate 45 degrees around x, y, z axis(direction: -x, -y, z)
        x = RotationAxis.x * sin(RotationAngle / 2) => -1 * sin( (1/4 * pi) / 2 ) = -0,382
        y = RotationAxis.y * sin(RotationAngle / 2) => -1 * sin( (1/4 * pi) / 2 ) = -0,382
        z = RotationAxis.z * sin(RotationAngle / 2) => 1 * sin( (1/4 * pi) / 2 ) = 0,382
        w = cos( (1/4 * pi) / 2) = 0,923"""
        quat = np.zeros(4)
        # transform degree to radian
        radian = np.radians(degree)
        # calculate the values
        quat[0] = direction[0] * np.sin(radian[0] / 2)
        quat[1] = direction[1] * np.sin(radian[1] / 2)
        quat[2] = direction[2] * np.sin(radian[2] / 2)
        quat[3] = np.cos(radian[0] / 2)
        return quat
    
    def initial_obstacle(self):
        # set the initial position, size and orientation of the obstacles
        obstacles = self.human_obstacle
        for ii, obs in enumerate(obstacles):
            if isinstance(obs, Ellipse):
                if obs.pose.orientation is not None:
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
                    print('orientation', orientation)
                    direction = [1,1,1]
                    xml_quat = self.transform_degree_to_quat(orientation,direction)
                    name = obs.name
                    self.set_body_xquat(name,xml_quat)

                self.set_body_pos(name, obs.center_position)
                self.set_body_size(name, obs.axes_length * 0.5)
            
            if isinstance(obs, Cuboid):
                name = obs.name
                self.set_body_pos(name, obs.center_position)
                self.set_body_size(name,obs.axes_length * 0.5)

    # --- Utils ---
    def get_body_pos(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_pos[body_id]

    def set_body_pos(self, name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_pos[body_id] = pos

    def get_body_xquat(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_xquat[body_id]
    
    def set_body_xquat(self, name, quat):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_xquat[body_id] = quat

    def get_geom_size(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_size[geom_id]

    def get_obstacle_info(self):
        obstacles = []
        for n in range(self.total_obst):
            obstacles.append({
                'pos': self._utils.get_joint_qpos(self.model, self.data, f'obstacle{n}' + ':joint')[:3],
                'size': self.get_geom_size(f'obstacle{n}')
            })
        return obstacles

    def set_geom_size(self, name, size):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_size[geom_id] = size
        # adjust rbound (radius of bounding sphere for which collisions are not checked)
        self.model.geom_rbound[geom_id] = np.sqrt(np.sum(np.square(self.model.geom_size[geom_id])))
    
    def set_body_size(self,name,size):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_size[body_id] = size

    def get_site_size(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_size[site_id]

    def set_site_size(self, name, size):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_size[site_id] = size

    def get_site_pos(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_pos[site_id]

    def set_site_pos(self, name, pos):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_pos[site_id] = pos

    def set_site_quat(self, name, quat):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_quat[site_id] = quat

    #def create_obstacle_ellipse(self, name, pos, quat, size):


    # Small bug in gymnasium robotics returns wrong values, use local implementation for now.
    @staticmethod
    def custom_get_joint_qvel(model, data, name):
        """Return the joints linear and angular velocities (qvel) of the model."""
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_type = model.jnt_type[joint_id]
        joint_addr = model.jnt_dofadr[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim

        return data.qvel[start_idx:end_idx]

    @property
    def dt(self):
        """Return the timestep of each Gymanisum step."""
        return self.model.opt.timestep * self.n_substeps

# main function
if __name__ == '__main__':
    #test_env()
    env = FrankaHumanEnv()
    #obs = env.get_ee_pos()
    #obs = env.render()
    