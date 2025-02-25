from __future__ import annotations  # To be removed in future python versions
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
import math

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

# (!!!) Somehow cv2 has to be imported after mayavi (?!) 
import cv2

from vartools.states import Pose
from vartools.dynamics import ConstantValue
from vartools.dynamics import LinearSystem
from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.multi_body_franka_obs import create_3d_franka_obs2
from nonlinear_avoidance.multi_body_human import create_3d_human
from nonlinear_avoidance.multi_body_human import (
    transform_from_multibodyobstacle_to_multiobstacle,
)
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingDynamics3D
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingAttractorDynamics3D

from nonlinear_avoidance.nonlinear_rotation_avoider import (
    ConvergenceDynamicsWithoutSingularity,
)

from scripts.three_dimensional.visualizer3d import CubeVisualizer

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Function to create and display a sphere obstacle
def create_sphere(center, radius, color=(1, 0, 0)):
    phi, theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]
    mlab.mesh(x, y, z, color=color)

# Function to create and display a cylinder obstacle
def create_cylinder(center, radius, height, color=(0, 1, 0)):
    z = np.linspace(-height / 2, height / 2, 50) + center[2]
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    mlab.mesh(x_grid, y_grid, z_grid, color=color)

# Function to create and display a cube obstacle
def create_cube(center, size, color=(0, 0, 1)):
    x, y, z = np.mgrid[center[0] - size/2:center[0] + size/2:2j,
                       center[1] - size/2:center[1] + size/2:2j,
                       center[2] - size/2:center[2] + size/2:2j]
    mlab.points3d(x, y, z, mode='cube', color=color, scale_factor=size)

# Create a figure and display multiple obstacles
mlab.figure('3D Obstacles')

# Add obstacles to the scene
# add 7 spheres
create_sphere(center=[0.7, 0.6, 0.42], radius=0.02, color=(1, 0, 0))  # Red sphere
create_sphere(center=[0.75,0.6,0.42], radius=0.02, color=(1, 0, 0))  # Red sphere
create_sphere(center=[0.8, 0.6, 0.42], radius=0.02, color=(1, 0, 0))  # Red sphere
create_sphere(center=[0.3,0.5,0.45], radius=0.02, color=(1, 0, 0))  # Red sphere
create_sphere(center=[0.3,0.5, 0.45], radius=0.02, color=(1, 0, 0))  # Red sphere
create_sphere(center=[0.405,0.4,0.45], radius=0.02, color=(1, 0, 0))  # Red sphere
create_sphere(center=[0.195,0.4,0.45], radius=0.02, color=(1, 0, 0))  # Red sphere


create_cylinder(center=[5, 5, 0], radius=1, height=5, color=(0, 1, 0))  # Green cylinder
create_cube(center=[-5, -5, 0], size=3, color=(0, 0, 1))  # Blue cube

# Function to display a table with Matplotlib
def create_table():
    fig = Figure(figsize=(4, 2))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    data = [
        ["Sphere", "Center (0, 0, 0)", "Radius 2", "Red"],
        ["Cylinder", "Center (5, 5, 0)", "Radius 1, Height 5", "Green"],
        ["Cube", "Center (-5, -5, 0)", "Size 3", "Blue"]
    ]
    columns = ["Shape", "Center", "Dimensions", "Color"]
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
    table.scale(1, 2)
    
    ax.axis('off')
    canvas.draw()

    # Convert the Matplotlib figure to an image and display in Mayavi
    width, height = fig.get_size_inches() * fig.get_dpi()
    #image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    
    #mlab.imshow(image)

# Display the table
create_table()
# show the coordinates
# mlab.text(0.7, 0.6, 0.42, '0.7, 0.6, 0.42', scale=0.1)
# mlab.text(0.75, 0.6, 0.42, '0.75, 0.6, 0.42', scale=0.1)
# mlab.text(0.8, 0.6, 0.42, '0.8, 0.6, 0.42', scale=0.1)
# mlab.text(0.3, 0.5, 0.45, '0.3, 0.5, 0.45', scale=0.1)
# mlab.text(0.3, 0.5, 0.45, '0.3, 0.5, 0.45', scale=0.1)
# mlab.text(0.405, 0.4, 0.45, '0.405, 0.4, 0.45', scale=0.1)
# mlab.text(0.195, 0.4, 0.45, '0.195, 0.4, 0.45', scale=0.1)

# show the axes
mlab.axes()

# Show the scene
mlab.show()


