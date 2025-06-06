"""
Multi-Body obstacles to avoid humansg
"""
from __future__ import annotations  # Self typing

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import numpy.typing as npt
from numpy import linalg

from scipy.spatial.transform import Rotation

import networkx as nx

from vartools.state_filters import PositionFilter, SimpleOrientationFilter
from vartools.states import Pose
from vartools.states import ObjectPose


from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import HyperSphere

from nonlinear_avoidance.rigid_body import RigidBody
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def plot_3d_cuboid(ax, cube: Cuboid, color="green"):
    # TODO: include orientation
    axis = cube.axes_length
    orientation = cube.orientation

    pos_ranges = np.array(
        [
            cube.center_position - axis / 2.0,
            cube.center_position + axis / 2.0,
        ]
    ).T
    posx = pos_ranges[0, :]
    posy = pos_ranges[1, :]
    posz = pos_ranges[2, :]

    # Define the vertices of the cube
    for ii in posx:
        for jj in posy:
            ax.plot([ii, ii], [jj, jj], posz, color=color, marker="o")

    for ii in posx:
        for jj in posz:
            ax.plot([ii, ii], posy, [jj, jj], color=color, marker="o")

    for ii in posy:
        for jj in posz:
            ax.plot(posx, [ii, ii], [jj, jj], color=color, marker="o")


def plot_3d_ellipsoid(ax, ellipse: Ellipse):
    # TODO: inclde orientation?

    # your ellispsoid and center in matrix form
    diag_axes = np.diag(ellipse.axes_length)
    # dimension = 3
    # A = np.eye(dimension)
    # for dd in range(dimension):
    #     A[dd, :] =
    A = diag_axes

    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    # radii = 1.0 / np.sqrt(s)
    radii = ellipse.axes_length / 2.0

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot([x[i, j], y[i, j], z[i, j]], rotation) + ellipse.center_position
            )

    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color="b", alpha=0.2)


# class HumanTrackContainer(Obstacle):
"""functions summary for MultiBodyObstacle:
1. get_obstacle_id_from_name
2. get_obstacle_id_from_optitrackid
3. get_name
4. get_component
5. get_parent_idx
6. get_pose
7. get_root
8. get_update_id
9. set_root
10. create_filters
11. print_orientations
12. get_update_id
13. add_component
14. update_using_optitrack
15. align_obstacle_tree
15. update_dynamic_obstacle
16. update_orientation_based_on_position
17. align_position_with_parent
18. update
19. get_gamma
"""
class MultiBodyObstacle:
    dimension = 3

    def __init__(
        self,
        update_frequency: float = 100.0,
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    ):
        # super().__init__(center_position=np.zeros(3))

        self._obstacle_list: list[Obstacle] = []

        self._graph = nx.DiGraph()
        self.robot = robot

        # Pose updater (Optional) can be for example an OptitrackInterface
        self.pose_updater = pose_updater

        self._id_counter = 0

        self.position_filters = []
        self.orientation_filters = []
        self.update_frequency = update_frequency

        self.visualization_handler = visualization_handler

    def __getitem__(self, key) -> Obstacle:
        return self._obstacle_list[key]

    def __setitem__(self, key: int, value: Obstacle) -> None:
        self._obstacle_list[key] = value

    def get_obstacle_id_from_name(self, name: str) -> int:
        return [x for x, y in self._graph.nodes(data=True) if y["name"] == name][0]

    def get_obstacle_id_from_optitrackid(self, opt_id: int) -> int:
        return [
            x for x, y in self._graph.nodes(data=True) if y["optitrack_id"] == opt_id
        ][0]

    def get_name(self, idx: int) -> str:
        return self._graph.nodes()[idx]["name"]

    def get_component(self, idx: int) -> Obstacle:
        return self._obstacle_list[idx]

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    def get_pose(self) -> Pose:
        return self._obstacle_list[self.root_idx].pose

    @property
    def root_id(self) -> int:
        return self.root_idx

    def get_root(self) -> int:
        return self[self.root_id]

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    def set_root(
        self,
        obstacle: Obstacle,
        name: str,
        update_id: Optional[int] = None,
    ):
        self._obstacle_list.append(obstacle)
        self._graph.add_node(
            self._id_counter,
            name=name,
            update_id=update_id,
            references_children=[],
            indeces_children=[],
        )

        self.create_filters(is_updating=(not update_id is None))

        self._id_counter += 1
        self.root_idx = 0

    def create_filters(self, is_updating: bool):
        if is_updating:
            self.position_filters.append(
                PositionFilter(
                    update_frequency=self.update_frequency,
                    initial_position=np.zeros(3),
                )
            )
            self.orientation_filters.append(
                SimpleOrientationFilter(
                    update_frequency=self.update_frequency,
                    initial_orientation=Rotation.from_euler("x", 0),
                )
            )
        else:
            self.position_filters.append(None)
            self.orientation_filters.append(None)

    def print_orientations(self):
        # TODO: helper / debug function should be removed in the future
        for ii, obs in enumerate(self._obstacle_list):
            print(f"ii: {ii} has orientation", obs.pose.orientation)
            print(f"ii: {ii} has orientation", obs.orientation)

    def get_update_id(self, idx_node: int) -> int:
        return self._graph.nodes[idx_node]["update_id"]

    def add_component(
        self,
        obstacle: Obstacle,
        name: str,
        reference_position: npt.ArrayLike,
        parent_name: str,
        parent_reference_position: npt.ArrayLike,
        update_id: Optional[int] = None,
    ):
        reference_position = np.array(reference_position)
        obstacle.set_reference_point(reference_position, in_global_frame=False)
        self._obstacle_list.append(obstacle)
        parent_ind = self.get_obstacle_id_from_name(parent_name)

        self._graph.add_node(
            self._id_counter,
            name=name,
            update_id=update_id,
            local_reference=reference_position,
            indeces_children=[],
            references_children=[],
        )
        self._graph.nodes[parent_ind]["references_children"].append(
            np.array(parent_reference_position)
        )
        self._graph.nodes[parent_ind]["indeces_children"].append(self._id_counter)

        self._graph.add_edge(parent_ind, self._id_counter)

        self.create_filters(is_updating=(not update_id is None))
        self._id_counter += 1

    @property
    def optitrack_indeces(self):
        indeces_tree_opti = [-1] * self.n_components
        for ii in range(self.n_components):
            if self._graph.nodes[ii]["update_id"] is not None:
                indeces_tree_opti[ii] = self._graph.nodes[ii]["update_id"]

        return indeces_tree_opti

    def update_using_optitrack(self, transform_to_robot_frame: bool = True) -> None:
        if self.pose_updater is not None:
            new_object_poses = self.pose_updater.get_messages()
        else:
            new_object_poses = []
        indeces_measures = [oo.obs_id for oo in new_object_poses]
        indeces_optitrack_tree = self.optitrack_indeces

        if self.robot is not None:
            try:
                index_franka_list = indeces_measures.index(self.robot.optitrack_id)

            except ValueError:
                # Element not in list
                pass

            else:
                # So far: no filter for the robot (!)
                franka_object = new_object_poses[index_franka_list]

                self.robot.pose.position = franka_object.position
                self.robot.pose.rotation = franka_object.rotation

        # Update filters and put to poses
        filtered_poses = [None] * self.n_components
        for ii, idx_meas in enumerate(indeces_measures):
            try:
                idx_tree = indeces_optitrack_tree.index(idx_meas)

            except:
                pass
            else:
                self.position_filters[idx_tree].run_once(new_object_poses[ii].position)
                self.orientation_filters[idx_tree].run_once(
                    new_object_poses[ii].rotation
                )

                if self.position_filters[idx_tree] is None:
                    continue

                if transform_to_robot_frame and self.robot is not None:
                    filtered_poses[idx_tree] = ObjectPose(
                        position=self.robot.pose.transform_position_to_relative(
                            self.position_filters[idx_tree].position
                        ),
                        orientation=self.robot.pose.transform_orientation_to_relative(
                            self.orientation_filters[idx_tree].rotation
                        ),
                    )
                else:
                    filtered_poses[idx_tree] = ObjectPose(
                        position=self.position_filters[idx_tree].position,
                        orientation=self.orientation_filters[idx_tree].rotation,
                    )

        # # Transform to robot frame(?)
        # filtered_poses = [None] * self.n_components
        # if transform_to_robot_frame and self.robot is not None:
        #     for ii in range(self.n_components):
        #         if self.position_filters[ii] is None:
        #             continue
        #         print(f"trafo for {ii}")
        #         filtered_poses[ii] = ObjectPose(
        #             position=self.robot.pose.transform_position_to_relative(
        #                 self.position_filters[ii].position
        #             ),
        #             orientation=self.robot.pose.transform_orientation_to_relative(
        #                 self.orientation_filters[ii].rotation
        #             ),
        #         )
        # else:
        #     for ii in range(self.n_components):
        #         if self.position_filters[ii] is None:
        #             continue
        #         filtered_poses[ii] = ObjectPose(
        #             position=self.position_filters[ii].position,
        #             orientation=self.orientation_filters[ii].rotation,
        #         )
        if filtered_poses[self.root_idx] is not None:
            # print("Doing root")
            # self.update_dynamic_obstacle(self.root_idx, filtered_poses[self.root_idx])
            self[self.root_idx].pose = filtered_poses[self.root_idx]
            self[self.root_idx].pose.position[2] = 0.2  # Set root position specific

            # Assumption of rotation only in z (since it's a human in 2D)
            zyx_rot = self[self.root_idx].pose.orientation.as_euler("zyx")
            self[self.root_idx].pose.orientation = Rotation.from_euler("z", zyx_rot[0])

        obs_indeces = list(self._graph.successors(self.root_idx))
        it_node = 0
        while it_node < len(obs_indeces):
            idx_node = obs_indeces[it_node]
            obs_indeces = obs_indeces + list(self._graph.successors(idx_node))

            # print("it_node", it_node)
            # print("orientation", self[idx_node].orientation)
            it_node += 1  # Iterate

            if filtered_poses[idx_node] is None:
                # Static opbstacle - no optitrack exists...
                # We assume orientation was constant?
                idx_parent = list(self._graph.predecessors(idx_node))[0]
                self[idx_node].orientation = self[idx_parent].orientation

                self.align_position_with_parent(idx_node)

            else:
                self[idx_node].pose = filtered_poses[idx_node]
                # self.update_dynamic_obstacle(idx_node, filtered_poses[idx_node])
                self.update_orientation_based_on_position(idx_node)
                self.align_position_with_parent(idx_node)
                # self.align_position_with_parent(idx_node)

                # Reset position filter
                # self.position_filters[idx_node]._position = self[idx_node].pose.position

    def align_obstacle_tree(self):
        obs_indeces = list(self._graph.successors(self.root_idx))
        it_node = 0
        while it_node < len(obs_indeces):
            idx_node = obs_indeces[it_node]
            obs_indeces = obs_indeces + list(self._graph.successors(idx_node))

            it_node += 1  # Iterate

            self.align_position_with_parent(idx_node)

    # def set_orientation(self, idx_obs: int, orientation: float | Rotation) -> None:
    #     self[idx_obs].orientation = orientation
    #     self.align_position_with_parent(idx_node)
    #     # TODO update all the children, too

    def update_dynamic_obstacle(self, idx_obs: int, obs_measure: RigidBody):
        # Update position
        # self.position_filters[idx_obs].run_once(obs_measure.position)
        # self.orientation_filters[idx_obs].run_once(obs_measure.rotation)

        self[idx_obs].pose.position = self.robot.pose.transform_position_to_relative(
            self.position_filters[idx_obs].position
        )
        self[
            idx_obs
        ].pose.orientation = self.robot.pose.transform_orientation_to_relative(
            self.orientation_filters[idx_obs].rotation
        )
        self[
            idx_obs
        ].linear_velocity = self.robot.pose.transform_linear_velocity_to_relative(
            self.position_filters[idx_obs].velocity
        )

    def update_orientation_based_on_position(
        self,
        idx_obs: int,
        # position_trust: float = 1.0
    ) -> None:
        idx_parent = list(self._graph.predecessors(idx_obs))[0]
        idx_local_ref = self._graph.nodes[idx_parent]["indeces_children"].index(idx_obs)
        local_reference_parent = self._graph.nodes[idx_parent]["references_children"][
            idx_local_ref
        ]
        reference_parent = self[idx_parent].pose.transform_position_from_relative(
            local_reference_parent
        )

        axes_direction = self[idx_obs].position - reference_parent
        # axes_direction = np.array([1.0, 0, -1.0])
        if not (axes_norm := np.linalg.norm(axes_direction)):
            # No information from position only
            return
        axes_direction = axes_direction / axes_norm

        rot_vec = np.cross([0.0, 0, 1.0], axes_direction)

        if rotvec_norm := np.linalg.norm(rot_vec):
            rot_vec = rot_vec / rotvec_norm
            theta = np.arcsin(rotvec_norm)
            quat = np.hstack((rot_vec * np.cos(theta / 2.0), [np.sin(theta / 2.0)]))

        else:
            quat = np.array([0, 0, 0, 1.0])

        # breakpoint()

        self[idx_obs].pose.orientation = Rotation.from_quat(quat)

    def align_position_with_parent(self, idx_obs: int):
        """Update obstacle with respect to the movement of the body-parts (limbs)
        under the assumption of FULL-TRUST(!) to the orientation."""
        idx_parent = list(self._graph.predecessors(idx_obs))[0]
        # try:
        reference_obstacle = self[idx_obs].pose.transform_position_from_relative(
            self._graph.nodes[idx_obs]["local_reference"]
        )
        # except:
        #     # breakpoint()
        #     raise Excpetion

        idx_local_ref = self._graph.nodes[idx_parent]["indeces_children"].index(idx_obs)

        local_reference_parent = self._graph.nodes[idx_parent]["references_children"][
            idx_local_ref
        ]
        reference_parent = self[idx_parent].pose.transform_position_from_relative(
            local_reference_parent
        )

        delta_ref = reference_parent - reference_obstacle
        # Full believe in orientation (and parent)
        self[idx_obs].pose.position = self[idx_obs].pose.position + delta_ref

    def update(self):
        if self.pose_updater is not None:
            self.update_using_optitrack()
        else:
            self.align_obstacle_tree()
            print("Do w/out optitrack")

        if self.robot is not None:
            self.robot.publish_robot_transform()

        if self.visualization_handler is not None:
            self.visualization_handler.update(self._obstacle_list)

    def get_gamma(self, position: Vector, in_global_frame: bool = True) -> bool:
        # in_global_frame is not used but kept for compatibility

        # Get minimum gamma
        gammas = np.zeros(self.n_components)

        for ii in range(self.n_components):
            gammas[ii] = self._obstacle_list[ii].get_gamma(
                position, in_global_frame=True
            )

        return np.min(gammas)


def create_2d_human_faulty():
    upper_arm_axes = [0.5, 0.18]
    lower_arm_axes = [0.4, 0.14]
    head_dimension = [0.2, 0.3]

    dimension = 2

    new_human = MultiObstacle(Pose(np.array([0.2, 0.5])))

    distance_scaling = 3

    human_dict = {
        "body": 0,
        "neck": 1,
        "upperarm1": 2,
        "upperarm2": 3,
        "lowerarm1": 4,
        "lowerarm2": 5,
    }

    human_dict["body"] = new_human.set_root(
        Cuboid(
            axes_length=[0.4, 0.7],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="body",
    )
    new_human[-1].set_reference_point(np.array([0, -0.3]), in_global_frame=False)

    human_dict["neck"] = new_human.add_component(
        Cuboid(
            axes_length=[0.12, 0.12],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="neck",
        # parent_name="body",
        parent_ind=human_dict["body"],
        reference_position=[0.0, -0.05],
        # parent_reference_position=[0.0, 0.30],
    )

    human_dict["head"] = new_human.add_component(
        Ellipse(
            axes_length=[0.2, 0.3],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="head",
        # parent_name="neck",
        parent_ind=human_dict["neck"],
        reference_position=[0.0, -0.12],
        # parent_reference_position=[0.0, 0.05],
    )

    human_dict["upperarm1"] = new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="upperarm1",
        # parent_name="body",
        parent_ind=human_dict["body"],
        reference_position=[-0.2, 0],
        # parent_reference_position=[0.15, 0.3],
    )

    human_dict["lowerarm1"] = new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="lowerarm1",
        # parent_name="upperarm1",
        parent_ind=human_dict["upperarm1"],
        reference_position=[-0.18, 0],
        # parent_reference_position=[0.2, 0],
    )

    human_dict["upperarm2"] = new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="upperarm2",
        # parent_name="body",
        parent_ind=human_dict["body"],
        reference_position=[0.2, 0],
        # parent_reference_position=[-0.15, 0.3],
    )

    human_dict["upperarm2"] = new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        # name="lowerarm2",
        # parent_name="upperarm2",
        parent_ind=human_dict["upperarm2"],
        reference_position=[0.18, 0],
        # parent_reference_position=[-0.2, 0],
    )

    # new_human.update()

    # idx_obs = new_human.get_obstacle_id_from_name("lowerarm1")
    idx_obs = human_dict["lowerarm1"]
    new_human[idx_obs].orientation = 70 * np.pi / 180
    # new_human.set_orientation(idx_obs, orientation=)
    new_human.align_position_with_parent(idx_obs)

    # idx_obs = new_human.get_obstacle_id_from_name("lowerarm2")
    idx_obs = human_dict["lowerarm2"]
    new_human[idx_obs].orientation = 45 * np.pi / 180
    # new_human.set_orientation(idx_obs, orientation=)
    new_human.align_position_with_parent(idx_obs)

    return new_human


def transform_from_multibodyobstacle_to_multiobstacle(
    multibody_obstacle, base_pose=None
):
    if base_pose is None:
        base_pose = Pose(np.zeros(multibody_obstacle.dimension))

    new_multi = MultiObstacle(base_pose)
    new_multi.set_root(multibody_obstacle.get_component(multibody_obstacle.root_idx))

    for index in range(multibody_obstacle.n_components):
        ind_parent = multibody_obstacle.get_parent_idx(index)

        if ind_parent is None:
            # Root is already set
            continue

        new_component = multibody_obstacle.get_component(index)
        reference = new_component.get_reference_point(in_global_frame=False)
        new_multi.add_component(
            new_component,
            parent_ind=ind_parent,
            reference_position=new_component.get_reference_point(in_global_frame=False),
        )

    return new_multi


def create_2d_human() -> MultiObstacle:
    new_human_multibody = create_2d_human_with_multibodyobstacle()
    new_human = transform_from_multibodyobstacle_to_multiobstacle(new_human_multibody)
    return new_human


def create_2d_human_with_multibodyobstacle() -> MultiBodyObstacle:
    upper_arm_axes = [0.5, 0.18]
    lower_arm_axes = [0.4, 0.14]
    head_dimension = [0.2, 0.3]

    dimension = 2

    new_human = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    new_human.dimension = dimension

    distance_scaling = 3

    new_human.set_root(
        Cuboid(
            axes_length=[0.4, 0.7],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="body",
    )
    new_human[-1].set_reference_point(np.array([0, -0.3]), in_global_frame=False)

    new_human.add_component(
        Cuboid(
            axes_length=[0.12, 0.12],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="neck",
        parent_name="body",
        reference_position=[0.0, -0.05],
        parent_reference_position=[0.0, 0.30],
    )

    new_human.add_component(
        Ellipse(
            axes_length=[0.2, 0.3],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="head",
        parent_name="neck",
        reference_position=[0.0, -0.12],
        parent_reference_position=[0.0, 0.05],
    )

    new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="upperarm1",
        parent_name="body",
        reference_position=[-0.2, 0],
        parent_reference_position=[0.15, 0.3],
    )

    new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="lowerarm1",
        parent_name="upperarm1",
        reference_position=[-0.18, 0],
        parent_reference_position=[0.2, 0],
    )

    new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="upperarm2",
        parent_name="body",
        reference_position=[0.2, 0],
        parent_reference_position=[-0.15, 0.3],
    )

    new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="lowerarm2",
        parent_name="upperarm2",
        reference_position=[0.18, 0],
        parent_reference_position=[-0.2, 0],
    )

    new_human.update()

    idx_obs = new_human.get_obstacle_id_from_name("lowerarm1")
    new_human[idx_obs].orientation = 70 * np.pi / 180
    # new_human.set_orientation(idx_obs, orientation=)
    new_human.align_position_with_parent(idx_obs)

    idx_obs = new_human.get_obstacle_id_from_name("lowerarm2")
    new_human[idx_obs].orientation = 45 * np.pi / 180
    # new_human.set_orientation(idx_obs, orientation=)
    new_human.align_position_with_parent(idx_obs)

    return new_human


def create_3d_human(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    # root_position = np.array([0.4, -0.2, 0.25])

    # Ellipse
    head_axes_length = np.array([0.2, 0.25, 0.3])
    upper_arm_axes = [0.18, 0.18, 0.5]
    lower_arm_axes = [0.14, 0.14, 0.4]
    head_dimension = [0.2, 0.15, 0.3]
    legs_axes = [1.0, 0.2, 0.25]

    # Cuboid
    body_exes_length = np.array([0.15, 0.4, 0.5])
    neck_axes_length = np.array([0.12, 0.15, 0.2])
    
    # Reference points
    body_reference = np.array([0.0, 0, -0.2])

    neck_reference = np.array([0.0, 0, -0.07])
    parent_of_neck_reference = np.array([0.0, 0, 0.25])

    #head_reference = np.array([0.2, 0.22, 0.3])
    head_reference = np.array([0.0, 0, 0.0])
    parent_of_head_reference = np.array([0.0, 0, 0.07])

    upperarm1_reference = np.array([0, 0, -0.2])
    parent_of_upperarm1_reference = np.array([0.0, 0.18, 0.2])
    orientation_upperarm1 = Rotation.from_euler("xyz", [0, 0.1 * np.pi, 0 * np.pi])

    lowerarm1_reference = np.array([0, 0, -0.18])
    parent_of_lowerarm1_reference = np.array([0.0, 0, 0.2])
    orientation_lowerarm1 = Rotation.from_euler("xyz", [0, 0.2 * np.pi, 0 * np.pi])

    upperarm2_reference = np.array([0.0, 0, 0.2])
    parent_of_upperarm2_reference = np.array([0, -0.19, 0.2])
    orientation_upperarm2 = Rotation.from_euler("xyz", [0, 0.3 * np.pi, 0 * np.pi])

    lowerarm2_reference = np.array([0, 0.0, 0.18])
    parent_of_lowerarm2_reference = np.array([0, 0.0, -0.2])
    orientation_lowerarm2 = Rotation.from_euler("xyz", [0, 0.5 * np.pi, 0 * np.pi])

    # legs_reference = np.array([0.15, 0.24686981, -0.63978714])
    # parent_of_leg1_reference = np.array([0.15, 0, -0.3])
    # parent_of_leg2_reference = np.array([-0.15, 0, -0.3])

    
    margin_absolut = 0.0

    dimension = 3

    new_human = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_human.set_root(
        Cuboid(
            axes_length=body_exes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="body",
    )

    new_human[-1].set_reference_point(body_reference, in_global_frame=False)

    new_human.add_component(
        Cuboid(
            # axes_length=np.array([0.12, 0.2, 0.12]),
            axes_length=neck_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="neck",
        parent_name="body",
        reference_position=neck_reference,
        parent_reference_position=parent_of_neck_reference,
    )

    new_human.add_component(
        Ellipse(
            axes_length=head_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="head",
        parent_name="neck",
        reference_position=head_reference,
        parent_reference_position=parent_of_head_reference,
    )

    new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, -0.3 * np.pi]),
            orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, 0. * np.pi]),
        ),
        name="upperarm1",
        parent_name="body",
        reference_position= upperarm1_reference,
        parent_reference_position=parent_of_upperarm1_reference,
    )

    new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.2 * np.pi, -0.8 * np.pi]),
            orientation = Rotation.from_euler("xyz", [0, 0.2 * np.pi, 0. * np.pi]),
        ),
        name="lowerarm1",
        parent_name="upperarm1",
        reference_position=lowerarm1_reference,
        parent_reference_position=parent_of_lowerarm1_reference,
    )

    new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            orientation=Rotation.from_euler("xyz", [0, 0.3 * np.pi, 0]),
        ),
        name="upperarm2",
        parent_name="body",
        reference_position=upperarm2_reference,
        parent_reference_position= parent_of_upperarm2_reference,
    )

    new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            orientation=Rotation.from_euler("xyz", [0, 0.5 * np.pi, 0]),
        ),
        name="lowerarm2",
        parent_name="upperarm2",
        reference_position=lowerarm2_reference,
        parent_reference_position=parent_of_lowerarm2_reference,
    )

    # Plus legs (!)
    # new_human.add_component(
    #     Ellipse(
    #         axes_length=legs_axes,
    #         center_position=np.zeros(dimension),
    #         distance_scaling=distance_scaling,
    #         margin_absolut=margin_absolut,
    #         # orientation=Rotation.from_euler("zy", [np.pi * 0.5, np.pi * 0.4]),
    #         orientation=Rotation.from_euler("zx", [np.pi * 0.5, np.pi * 1.7]),
    #     ),
    #     name="leg1",
    #     parent_name="body",
    #     reference_position=legs_reference,
    #     parent_reference_position=parent_of_leg1_reference,
    # )

    # new_human.add_component(
    #     Ellipse(
    #         axes_length=legs_axes,
    #         center_position=np.zeros(dimension),
    #         distance_scaling=distance_scaling,
    #         margin_absolut=margin_absolut,
    #         # orientation=Rotation.from_euler("zy", [np.pi * 0.5, np.pi * 0.4]),
    #         orientation=Rotation.from_euler("zx", [np.pi * 0.5, np.pi * 1.3]),
    #     ),
    #     name="leg2",
    #     parent_name="body",
    #     reference_position=legs_reference,
    #     parent_reference_position=parent_of_leg2_reference,
    # )

    # idx_obs = new_human.get_obstacle_id_from_name("lowerarm1")
    # new_human[idx_obs].orientation = 70 * np.pi / 180
    # # new_human.set_orientation(idx_obs, orientation=)
    # new_human.align_position_with_parent(idx_obs)

    # idx_obs = new_human.get_obstacle_id_from_name("lowerarm2")
    # new_human[idx_obs].orientation = 45 * np.pi / 180
    # # new_human.set_orientation(idx_obs, orientation=)
    # new_human.align_position_with_parent(idx_obs)

    new_human.update()
    return new_human

def create_3d_table(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    #root_position = np.array([0.4, 0.6, 0.25])

    # Cuboid
    leg_axes_length = np.array([0.08, 0.08, 0.2])
    table_axes_length = np.array([0.4, 0.5, 0.15])
    box = np.array([0.04, 0.04, 0.1])
    
    # Reference points
    table_reference = np.array([0, 0, 0])

    leg_reference = np.array([0.0, 0, 0.08])
    parent_of_leg1_reference = np.array([-0.16, -0.21, -0.075])

    parent_of_leg2_reference = np.array([0.16, -0.21, -0.075])

    parent_of_leg3_reference = np.array([0.16, 0.21, -0.075])

    parent_of_leg4_reference = np.array([-0.16, 0.21, -0.075])

    box_reference = np.array([0.0, 0, 0.05])
    parent_of_box_reference = np.array([0.0, 0, 0.075])

    
    margin_absolut = 0.0

    dimension = 3

    new_table = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_table.set_root(
        Cuboid(
            axes_length=table_axes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="table",
    )

    new_table[-1].set_reference_point(table_reference, in_global_frame=False)

    new_table.add_component(
        Cuboid(
            # axes_length=np.array([0.12, 0.2, 0.12]),
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="leg1",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg1_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="leg2",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg2_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, -0.3 * np.pi]),
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, 0. * np.pi]),
        ),
        name="leg3",
        parent_name="table",
        reference_position= leg_reference,
        parent_reference_position=parent_of_leg3_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.2 * np.pi, -0.8 * np.pi]),
            #orientation = Rotation.from_euler("xyz", [0, 0.2 * np.pi, 0. * np.pi]),
        ),
        name="leg4",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg4_reference,
    )


    new_table.update()
    return new_table

def create_3d_long_table(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    #root_position = np.array([0.4, 0.6, 0.25])

    # Cuboid
    leg_axes_length = np.array([0.08, 0.08, 0.2])
    table_axes_length = np.array([0.4, 1.5, 0.15])
    box = np.array([0.04, 0.04, 0.1])
    
    # Reference points
    table_reference = np.array([0, 0, 0])

    leg_reference = np.array([0.0, 0, 0.08])
    parent_of_leg1_reference = np.array([-0.16, -0.71, -0.075])

    parent_of_leg2_reference = np.array([0.16, -0.71, -0.075])

    parent_of_leg3_reference = np.array([0.16, 0.71, -0.075])

    parent_of_leg4_reference = np.array([-0.16, 0.71, -0.075])

    box_reference = np.array([0.0, 0, 0.05])
    parent_of_box_reference = np.array([0.0, 0, 0.075])

    
    margin_absolut = 0.0

    dimension = 3

    new_table = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_table.set_root(
        Cuboid(
            axes_length=table_axes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="table",
    )

    new_table[-1].set_reference_point(table_reference, in_global_frame=False)

    new_table.add_component(
        Cuboid(
            # axes_length=np.array([0.12, 0.2, 0.12]),
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="leg1",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg1_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="leg2",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg2_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, -0.3 * np.pi]),
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, 0. * np.pi]),
        ),
        name="leg3",
        parent_name="table",
        reference_position= leg_reference,
        parent_reference_position=parent_of_leg3_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.2 * np.pi, -0.8 * np.pi]),
            #orientation = Rotation.from_euler("xyz", [0, 0.2 * np.pi, 0. * np.pi]),
        ),
        name="leg4",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg4_reference,
    )


    new_table.update()
    return new_table

def create_3d_table_with_box(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    #root_position = np.array([0.4, 0.6, 0.25])

    # Cuboid
    leg_axes_length = np.array([0.08, 0.08, 0.2])
    table_axes_length = np.array([0.4, 0.5, 0.15])
    box = np.array([0.4, 0.1, 0.1])
    
    # Reference points
    table_reference = np.array([0, 0, 0])

    leg_reference = np.array([0.0, 0, 0.08])
    parent_of_leg1_reference = np.array([-0.16, -0.21, -0.075])

    parent_of_leg2_reference = np.array([0.16, -0.21, -0.075])

    parent_of_leg3_reference = np.array([0.16, 0.21, -0.075])

    parent_of_leg4_reference = np.array([-0.16, 0.21, -0.075])

    box_reference = np.array([0.0, 0, 0.045])
    parent_of_box_reference = np.array([0.0, 0, 0.075])

    
    margin_absolut = 0.0

    dimension = 3

    new_table = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_table.set_root(
        Cuboid(
            axes_length=table_axes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="table",
    )

    new_table[-1].set_reference_point(table_reference, in_global_frame=False)

    new_table.add_component(
        Cuboid(
            # axes_length=np.array([0.12, 0.2, 0.12]),
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="leg1",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg1_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="leg2",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg2_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, -0.3 * np.pi]),
            #orientation=Rotation.from_euler("xyz", [0, 0.1 * np.pi, 0. * np.pi]),
        ),
        name="leg3",
        parent_name="table",
        reference_position= leg_reference,
        parent_reference_position=parent_of_leg3_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=leg_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.2 * np.pi, -0.8 * np.pi]),
            #orientation = Rotation.from_euler("xyz", [0, 0.2 * np.pi, 0. * np.pi]),
        ),
        name="leg4",
        parent_name="table",
        reference_position=leg_reference,
        parent_reference_position=parent_of_leg4_reference,
    )

    new_table.add_component(
        Cuboid(
            axes_length=box,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
            #orientation=Rotation.from_euler("xyz", [0, 0.3 * np.pi, 0]),
        ),
        name="box",
        parent_name="table",
        reference_position=box_reference,
        parent_reference_position= parent_of_box_reference,
    )


    new_table.update()
    return new_table

def create_3d_box(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    #root_position = np.array([0.4, 0.6, 0.25])

    # Cuboid
    box_axes_length = np.array([0.3, 0.1, 0.2])
    
    # Reference points
    box_reference = np.array([0.0, 0, 0.015])

    margin_absolut = 0.0

    dimension = 3

    new_box = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_box.set_root(
        Cuboid(
            axes_length=box_axes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="box",
    )

    #new_box[-1].set_reference_point(box_reference, in_global_frame=False)

    new_box.update()
    return new_box

def create_3d_sphere(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    #root_position = np.array([0.4, 0.6, 0.25])

    sphere_radius = 0.05
    sphere_axes_length = np.array([0.1, 0.1, 0.1])
    
    # Reference points
    sphere_reference = np.array([0.0, 0, 0.015])

    margin_absolut = 0.0

    dimension = 3

    new_sphere = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_sphere.set_root(
        Ellipse(
            axes_length=sphere_axes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="sphere",
    )

    #new_box[-1].set_reference_point(box_reference, in_global_frame=False)

    new_sphere.update()
    return new_sphere

def create_3d_cuboid(root_position):
    #root_position = np.array([0.6, 0.0, 0.25])
    #root_position = np.array([0.4, 0.6, 0.25])

    cuboid_axes_length = np.array([0.2, 0.2, 0.5])
    
    # Reference points
    cuboid_reference = np.array([0.0, 0, 0.015])

    margin_absolut = 0.0

    dimension = 3

    new_cuboid = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )
    # new_human = MultiObstacle(Pose(np.array([0.0, 0.0, 0.0])))

    distance_scaling = 3

    new_cuboid.set_root(
        Cuboid(
            axes_length=cuboid_axes_length,
            #center_position=np.zeros(dimension),
            center_position=root_position,
            #center_position=np.array([1, 0.0, 0.25]),
            distance_scaling=distance_scaling,
            margin_absolut=margin_absolut,
        ),
        name="cuboid",
    )

    #new_box[-1].set_reference_point(box_reference, in_global_frame=False)

    new_cuboid.update()
    return new_cuboid

def create_3d_cross(root_position):
    vertical_axes_length = np.array([0.03, 0.03, 0.1])
    horizontal_axes_length = np.array([0.03, 0.1, 0.03])

    vertical_reference = np.array([0.0, 0, -0.04])
    horizontal_reference = np.array([0.0, 0, 0.0])
    parent_of_horizontal_reference = np.array([0.0, 0, 0.0])
    margin_absolut = 0.0

    dimension = 3

    new_cross = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )

    new_cross.set_root(
        Cuboid(
            axes_length=vertical_axes_length,
            center_position=root_position,
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="vertical_box",
    )

    new_cross[-1].set_reference_point(vertical_reference, in_global_frame=False)

    new_cross.add_component(
        Cuboid(
            axes_length=horizontal_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="horizontal_box",
        parent_name="vertical_box",
        reference_position=horizontal_reference,
        parent_reference_position=parent_of_horizontal_reference,
    )

    new_cross.update()
    return new_cross

def create_3d_concave_word(root_position):
    vertical_1_axes_length = np.array([0.03, 0.03, 0.1])
    vertical_2_axes_length = np.array([0.03, 0.03, 0.1])
    horizontal_axes_length = np.array([0.03, 0.05, 0.03])

    horizontal_reference = np.array([0.0, 0, 0.0])

    vertical_1_reference = np.array([0, -0.015, 0.035])
    parent_of_vertical_1_reference = np.array([0, -0.022, 0])

    vertical_2_reference = np.array([0, 0.015, 0.035])
    parent_of_vertical_2_reference = np.array([0, 0.022, 0])

    margin_absolut = 0.0

    dimension = 3

    new_concave_word = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )

    new_concave_word.set_root(
        Cuboid(
            axes_length=horizontal_axes_length,
            center_position=root_position,
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="horizontal_box",
    )

    new_concave_word[-1].set_reference_point(horizontal_reference, in_global_frame=False)
    
    new_concave_word.add_component(
        Cuboid(
            axes_length=vertical_1_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="vertical_1_box",
        parent_name="horizontal_box",
        reference_position=vertical_1_reference,
        parent_reference_position=parent_of_vertical_1_reference,
    )

    new_concave_word.add_component(
        Cuboid(
            axes_length=vertical_2_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="vertical_2_box",
        parent_name="horizontal_box",
        reference_position=vertical_2_reference,
        parent_reference_position=parent_of_vertical_2_reference,
    )

    new_concave_word.update()
    return new_concave_word

def create_3d_star_shape(root_position):
    star_part_axes_length = np.array([0.02, 0.02, 0.04])

    star_part_0_reference = np.array([0, 0, 0.0])

    star_part_1_reference = np.array([0, -0.02, 0])
    parent_of_star_part_1_reference = np.array([0, -0.01, -0.01])

    star_part_2_reference = np.array([0, -0.02, 0])
    parent_of_star_part_2_reference = np.array([0, -0.01, -0.015])
    orientation_star_part_2 = Rotation.from_euler("xyz", [np.pi/3,0, 0])

    star_part_3_reference = np.array([0, -0.02, 0])
    parent_of_star_part_3_reference = np.array([0, 0.01, -0.015])
    orientation_star_part_3 = Rotation.from_euler("xyz", [np.pi*(2/3),0, 0])

    star_part_4_reference = np.array([0, -0.02, 0])
    parent_of_star_part_4_reference = np.array([0, 0.01, -0.01])
    orientation_star_part_4 = Rotation.from_euler("xyz", [np.pi,0, 0])

    margin_absolut = 0.0

    dimension = 3

    new_star_shape = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )

    new_star_shape.set_root(
        Cuboid(
            axes_length=star_part_axes_length,
            center_position=root_position,
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="star_part_0",
    )

    new_star_shape[-1].set_reference_point(star_part_0_reference, in_global_frame=False)

    new_star_shape.add_component(
        Cuboid(
            axes_length=star_part_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
        ),
        name="star_part_1",
        parent_name="star_part_0",
        reference_position=star_part_1_reference,
        parent_reference_position=parent_of_star_part_1_reference,
    )

    new_star_shape.add_component(
        Cuboid(
            axes_length=star_part_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
            orientation=orientation_star_part_2,
        ),
        name="star_part_2",
        parent_name="star_part_0",
        reference_position=star_part_2_reference,
        parent_reference_position=parent_of_star_part_2_reference,
    )

    new_star_shape.add_component(
        Cuboid(
            axes_length=star_part_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
            orientation=orientation_star_part_3,
        ),
        name="star_part_3",
        parent_name="star_part_0",
        reference_position=star_part_3_reference,
        parent_reference_position=parent_of_star_part_3_reference,
    )

    new_star_shape.add_component(
        Cuboid(
            axes_length=star_part_axes_length,
            center_position=np.zeros(dimension),
            distance_scaling=3,
            margin_absolut=margin_absolut,
            orientation=orientation_star_part_4,
        ),
        name="star_part_4",
        parent_name="star_part_0",
        reference_position=star_part_4_reference,
        parent_reference_position=parent_of_star_part_4_reference,
    )

    new_star_shape.update()
    return new_star_shape