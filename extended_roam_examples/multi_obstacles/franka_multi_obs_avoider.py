from __future__ import annotations  # To be removed in future python versions
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
import math

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from vartools.dynamics import ConstantValue
from vartools.dynamics import LinearSystem, CircularStable
from vartools.states import Pose, Twist
from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.multi_body_franka_obs import create_3d_franka_obs
from nonlinear_avoidance.multi_obs_env import create_3d_human, transform_from_multibodyobstacle_to_multiobstacle
from nonlinear_avoidance.multi_obs_env import create_3d_table,create_3d_long_table, create_3d_table_with_box,create_3d_box, create_3d_cross, create_3d_concave_word, create_3d_star_shape
#from nonlinear_avoidance.multi_obs_env import create_3d_franka_obs2
# from nonlinear_avoidance.multi_body_franka_obs import (
#     transform_from_multibodyobstacle_to_multiobstacle,
# )
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingDynamics3D
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingAttractorDynamics3D

from nonlinear_avoidance.nonlinear_rotation_avoider import (
    ConvergenceDynamicsWithoutSingularity,
)





Vector = np.ndarray

global ctrl
global rot_ctrl


    
class ROAM:
    def __init__(
        self, it_max: int = 300, delta_time: float = 0.005, filename: str = "animation", 
        current_position: np.ndarray = np.zeros(3),
        attractor_position: np.ndarray = np.zeros(3),
        dynamic_human = False,
        obstacle = True) -> None:
        self.it_max = it_max
        self.delta_time = delta_time

        self.filename = filename
        self.figuretype = ".png"

        self.save_to_file = True

        self.main_folder = Path("figures")
        self.image_folder = Path("animation")

        # self.leading_zeros = math.ceil(math.log10(self.it_max + 1))
        self.leading_zeros = 4
 
        self.current_position = current_position
        self.attractor_position = attractor_position
        self.dynamic_human = dynamic_human  # the obstacle is dynamic
        self.obstacle = obstacle # the obstacle is existent

    def run_norm(self,ii):    
        self.update_ee_step(ii)
        self.update_norm_dir(ii)
        if self.dynamic_human:
            #self.update_obstacle_pose()
            self.update_human(ii)
        return self.velocities
    
    def run(self,ii):
        velocities = self.update_step(ii)
        if self.dynamic_human:
            self.update_human(ii)
        return velocities

    def obstacle_initiation(self):
        # step1 create container of obstacles
        self.container = MultiObstacleContainer()

        if self.obstacle:
            # step2 create tree of obstacles
            human_obs_root_position = np.array([0.4, -0.2, 0.25])
            table_obs_root_position = np.array([0.5, 0.6, 0.25])

            human_base_position = Pose(position=human_obs_root_position)
            table_base_position = Pose(position=table_obs_root_position)

            self.human_obstacle_3d = create_3d_human(human_obs_root_position)
            self.table_obstacle_3d = create_3d_table(table_obs_root_position)
            
            # step3 transform tree of obstacles to multiobstacle
            transformed_human = transform_from_multibodyobstacle_to_multiobstacle(
                self.human_obstacle_3d
            )
            transformed_table = transform_from_multibodyobstacle_to_multiobstacle(
                self.table_obstacle_3d
            )

            self.container.append(transformed_human)
            self.container.append(transformed_table)


            self.dynamics = [None]*len(self.container)
            for i in range(len(self.container)):
                pose = self.container.get_obstacle_tree(i).get_pose()
                orientation = pose.orientation.as_euler("xyz", degrees=True)
                direction = np.array([0,np.cos(orientation[1]),0])
                if i == 0:
                    self.dynamics[i] = LinearMovement(pose.position, -direction, distance_max=2).evaluate
                #if i == 1:
                #    self.dynamics[i] = LinearMovement(pose.position, direction, distance_max=2).evaluate

    def obstacle_initiation_table_box(self):
        # step1 create container of obstacles
        self.container = MultiObstacleContainer()

        if self.obstacle:
            # step2 create tree of obstacles
            #human_obs_root_position = np.array([0.4, -0.2, 0.25])
            table_obs_root_position = np.array([0.5, 0, 0.25])
            box_root_position = table_obs_root_position + np.array([0.15, 0.15, 0.075])

            #self.human_obstacle_3d = create_3d_human(human_obs_root_position)
            self.table_obstacle_3d = create_3d_table_with_box(table_obs_root_position)
            self.box_obstacle_3d = create_3d_box(box_root_position)

            
            # step3 transform tree of obstacles to multiobstacle
            transformed_table = transform_from_multibodyobstacle_to_multiobstacle(
                self.table_obstacle_3d
            )
            transformed_box = transform_from_multibodyobstacle_to_multiobstacle(
                self.box_obstacle_3d
            )

            self.container.append(transformed_table)
            self.container.append(transformed_box)

            self.dynamics = [None]*len(self.container)
            for i in range(len(self.container)):
                pose = self.container.get_obstacle_tree(i).get_pose()
                orientation = pose.orientation.as_euler("xyz", degrees=True)
                direction = np.array([np.cos(orientation[1]),0,0])
                if i == 1:
                    self.dynamics[i] = LinearMovement(pose.position, -direction, distance_max=1).evaluate
                #if i == 1:
                #    self.dynamics[i] = LinearMovement(pose.position, direction, distance_max=2).evaluate

    def obstacle_initiation_table_multiobs(self):
        # step1 create container of obstacles
        self.container = MultiObstacleContainer()

        if self.obstacle:
            # step2 create tree of obstacles
            #human_obs_root_position = np.array([0.4, -0.2, 0.25])
            table_obs_root_position = np.array([0.5, 0, 0.25])
            cross_obs_root_position = table_obs_root_position + np.array([0,-0.75,0.125])
            concave_word_obs_root_position = table_obs_root_position + np.array([0,-0.55, 0.1])
            star_shape_obs_root_position = table_obs_root_position + np.array([0,-0.35, 0.06])


            #self.human_obstacle_3d = create_3d_human(human_obs_root_position)
            self.table_obstacle_3d = create_3d_long_table(table_obs_root_position)
            self.cross_obstacle_3d = create_3d_cross(cross_obs_root_position)
            self.concave_word_obstacle_3d = create_3d_concave_word(concave_word_obs_root_position)
            self.star_shape_obstacle_3d = create_3d_star_shape(star_shape_obs_root_position)
            
            # step3 transform tree of obstacles to multiobstacle
            transformed_table = transform_from_multibodyobstacle_to_multiobstacle(
                self.table_obstacle_3d
            )
            transformed_cross = transform_from_multibodyobstacle_to_multiobstacle(
                self.cross_obstacle_3d
            )
            transformed_concave_word = transform_from_multibodyobstacle_to_multiobstacle(
                self.concave_word_obstacle_3d
            )
            transformed_star_shape = transform_from_multibodyobstacle_to_multiobstacle(
                self.star_shape_obstacle_3d
            )

            self.container.append(transformed_table)
            self.container.append(transformed_cross)
            self.container.append(transformed_concave_word)
            self.container.append(transformed_star_shape)

            self.dynamics = [None]*len(self.container)
            for i in range(len(self.container)):
                pose = self.container.get_obstacle_tree(i).get_pose()
                orientation = pose.orientation.as_euler("xyz", degrees=True)
                direction = np.array([0,np.cos(orientation[1]),0])
                if i == 0:
                    self.dynamics[i] = LinearMovement(pose.position, -direction, distance_max=2).evaluate
                #if i == 1:
                #    self.dynamics[i] = LinearMovement(pose.position, direction, distance_max=2).evaluate

    def set_up_franka(self):
        dimension = 3

        dynamics = LinearSystem(attractor_position=self.attractor_position, maximum_velocity=1.0)
        
        # Trajectory integration
        start_positions = self.current_position
        print("start_positions number: ", start_positions.shape[0])
        self.n_traj = start_positions.shape[0]
        # self.trajectories shape is (3,301,25), 25 trajectories, 301 time steps and 3 dimensions
        self.trajectories = np.zeros((dimension, self.it_max + 1, self.n_traj))
        self.trajectories[:, 0, :] = start_positions.T
    

        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]

        # step4 create avoider
        self.avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
            obstacle_container=self.container,
            initial_dynamics=dynamics,
            # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
            create_convergence_dynamics=True,
            convergence_radius=0.55 * math.pi,
            smooth_continuation_power=0.7,)

    def setup(self, n_grid=5):
        dimension = 3

        # Trajectory integration
        start_positions = self.current_position
        #print("start_positions: ", start_positions.shape[0])
        self.n_traj = start_positions.shape[0]
        # self.trajectories shape is (3,301,25), 25 trajectories, 301 time steps and 3 dimensions
        self.trajectories = np.zeros((dimension, self.it_max + 1, self.n_traj))
        self.trajectories[:, 0, :] = start_positions.T
    

        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]

        

#-----------------------main steps -------------------------------------------------------------------
        # step1 create container of obstacles
        self.container = MultiObstacleContainer()
        if self.obstacle:
            # step2 create tree of obstacles
            human_obs_root_position = np.array([0.4, -0.2, 0.25])
            human_base_position = Pose(position=human_obs_root_position)
            self.human_obstacle_3d = create_3d_human(human_obs_root_position)

            # step3 transform tree of obstacles to multiobstacle????
            transformed_human = transform_from_multibodyobstacle_to_multiobstacle(
                self.human_obstacle_3d, base_pose=human_base_position
            )

            self.container.append(transformed_human)

        #self.human_obstacle_3d = create_3d_franka_obs2()
        dynamics = LinearSystem(attractor_position=self.attractor_position)
        

        # step4 create avoider
        self.avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
            obstacle_container=self.container,
            initial_dynamics=dynamics,
            # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
            create_convergence_dynamics=True,
            convergence_radius=0.55 * math.pi,
            smooth_continuation_power=0.7,
        )
#----------------------------------------------------------------------------------------------------------
        #self.visualizer = Visualization3D()

        #self.dynamic_human = False

    def update_human(self, ii: int) -> None:
        # amplitude_leg1 = 0.12
        # frequency_leg1 = 0.2
        # idx = self.human_obstacle_3d.get_obstacle_id_from_name("leg1")
        # obstacle = self.human_obstacle_3d[idx]
        # rotation = Rotation.from_euler(
        #     "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        # )
        # obstacle.orientation = obstacle.orientation * rotation

        # amplitude_leg1 = -0.12
        # frequency_leg1 = 0.2
        # idx = self.human_obstacle_3d.get_obstacle_id_from_name("leg2")
        # obstacle = self.human_obstacle_3d[idx]
        # rotation = Rotation.from_euler(
        #     "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        # )
        # obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = -0.08
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("upperarm1")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = -0.12
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("lowerarm1")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = 0.05
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("upperarm2")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "x", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = rotation * obstacle.orientation

        amplitude_leg1 = -0.05
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("lowerarm2")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "x", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        reference_point_updated = obstacle.get_reference_point(in_global_frame=True)

        self.human_obstacle_3d.align_obstacle_tree()

    def update_obstacle_pose(self):
        self.dt_simulation = self.delta_time
        for dynamics, tree in zip(self.dynamics, self.container):
            # Get updated dynamics and apply
            pose = tree.get_pose()
            if dynamics is None:
                continue

            twist = dynamics(pose)

            pose.position = twist.linear * self.dt_simulation + pose.position

            if twist.angular is not None:
                pose.orientation = twist.angular * self.dt_simulation + pose.orientation

            tree.update_pose(pose)
            tree.twist = twist

        # for deformation_rate, tree in zip(self.deformations, self.container):
        #     if deformation_rate is None:
        #         continue

        #     tree.deformation_rate = deformation_rate.evaluate()
        #     tree.update_deformation(self.dt_simulation)

    def update_step(self, ii: int) -> None:
        # from mayavi import mlab
        velocities = np.zeros((self.n_traj, 3))
        self.weight_ee = []
        for it_traj in range(self.n_traj):
            velocities[it_traj] = self.avoider.evaluate_sequence(self.trajectories[:, ii, it_traj])

            if self.obstacle:
                weights_ee = self.avoider.get_final_weights_for_sensors()
                weights_length = len(self.container.get_tree(0))+1
                #print("weights_length: ", weights_length)
                if len(weights_ee) == weights_length:
                    self.weight_ee.append(weights_ee)
                else:
                    reshaped_weights = np.zeros(weights_length)
                    reshaped_weights[:len(weights_ee)] = weights_ee
                    self.weight_ee.append(reshaped_weights)
            else:
                self.weight_ee.append(np.zeros(1))
        return velocities
    
    def update_ee_step(self, ii: int) -> None:
        self.velocities = np.zeros((self.n_traj, 3))
        self.velocities[-1] = self.avoider.evaluate_sequence(self.trajectories[:, ii, -1])
    
    def update_norm_dir(self,ii:int):
        #velocities = np.zeros((self.n_traj, 3))
        self.weights_sensors = []
    
        for it_traj in range(self.n_traj-1):
            weights_length = 0
            #print("it_traj: ", it_traj)
            self.velocities[it_traj] = self.avoider.evaluate_sequence_norm(self.trajectories[:, ii, it_traj])
            
            if self.obstacle:
                weights_sensor = self.avoider.get_final_weights_for_sensors()
                for num in range(len(self.container)):
                    weights_length += len(self.container.get_tree(num))
                weights_length += 1
                #print("weights_length: ", weights_length)
                if len(weights_sensor) == weights_length:
                    self.weights_sensors.append(weights_sensor)
                elif len(weights_sensor) != 1 and len(weights_sensor) < weights_length:
                    reshaped_weights = np.zeros(weights_length)
                    #reshaped_weights[:len(weights_sensor)] = weights_sensor
                    reshaped_weights[weights_length-len(weights_sensor):] = weights_sensor
                    self.weights_sensors.append(reshaped_weights)
                else:
                    reshaped_weights = np.zeros(weights_length)
                    reshaped_weights[:len(weights_sensor)] = weights_sensor
                    self.weights_sensors.append(reshaped_weights)
            else:
                self.weights_sensors.append(np.zeros(1))
        #return velocities
    
    def update_trajectories(self,position,ii:int):
        for it_traj in range(self.n_traj):
            self.trajectories[:, ii + 1, it_traj] = position

    def update_multiple_points_trajectories(self,positions,ii:int):
        self.trajectories[:, ii + 1, :] = positions.T



@dataclass
class LinearMovement:
    start_position: np.ndarray
    direction: np.ndarray
    distance_max: float

    # Internal state - to ensure motion
    step: int = 0
    frequency: float = 0.1

    p_factor: float = 0.8

    def evaluate(self, pose):
        self.step += self.frequency
        # print("linear step", self.step)

        next_position = (
            (1 - np.cos(self.step)) / 2.0 * self.distance_max * self.direction
        ) + self.start_position

        return Twist(linear=self.p_factor * (next_position - pose.position))

@dataclass
class AngularBackForth:
    start_orientation: float
    delta_angle: float

    frequency: float = 0.1
    step: int = 0

    dimension: int = 2

    p_factor: float = 5.0

    def evaluate(self, pose):
        self.step += self.frequency
        # print("angular step", self.step)

        next_angle = (
            np.cos(self.step) - 1.0
        ) * 0.5 * self.delta_angle + self.start_orientation

        return Twist(
            linear=np.zeros(self.dimension),
            angular=self.p_factor * (next_angle - pose.orientation),
        )


@dataclass
class ScalarBackForth:
    frequency: float = 0.1
    step: int = 0

    dimension: int = 2

    p_factor: float = -0.5

    def evaluate(self, pose=None):
        self.step += self.frequency

        return self.p_factor * np.sin(self.step)


@dataclass
class CircularMovement:
    start_position: np.ndarray
    radius: float

    # Internal state - to ensure motion
    step: int = 0
    frequency: float = 0.07

    def evaluate(self, pose):
        self.step += self.frequency

        next_position = (
            self.radius * np.array([-np.cos(self.step), -np.sin(self.step)])
            + self.start_position
        )

        return Twist(linear=(next_position - pose.position))


@dataclass
class ContinuousRotation:
    start_orientation: float

    # Internal state - to ensure motion
    step: int = 0
    frequency: float = -0.3

    dimension: int = 2

    def evaluate(self, pose):
        self.step += self.frequency
        return Twist(linear=np.zeros(self.dimension), angular=self.frequency)