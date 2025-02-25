

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

#from scripts.three_dimensional.visualizer3d import CubeVisualizer

import gymnasium as gym
import numpy as np
from pynput import keyboard
from envs.custom_scenarios import scenarios

import random
import time

from envs.config import register_custom_envs

from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data
import xml.etree.ElementTree as ET


# Function to parse XML file and extract data
def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        points = []
        for point in root.findall('Point'):
            x = float(point.get('x'))
            y = float(point.get('y'))
            z = float(point.get('z'))
            points.append((x, y, z))
        return points
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

# Load the XML file
file_path = os.path.join(os.path.dirname(__file__), 'test.xml')
print(f"Loading points from {file_path}")
points = parse_xml(file_path)

# Check if points were loaded successfully
if points:
    # Convert points to a suitable format for Mayavi
    x, y, z = zip(*points)

    # Create a Mayavi figure
    mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 600))

    # Visualize the points
    mlab.points3d(x, y, z, mode='sphere', colormap='copper', scale_factor=0.1)

    # Show the visualization
    mlab.show()
else:
    print("No points to visualize.")


