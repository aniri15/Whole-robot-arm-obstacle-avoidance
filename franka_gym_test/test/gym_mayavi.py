import gymnasium as gym

import numpy as np
from pynput import keyboard
from envs.custom_scenarios import scenarios

import random
import time

from envs.config import register_custom_envs




Vector = np.ndarray

global ctrl
global rot_ctrl



class FrankaEnv:
    def __init__(self):
        register_custom_envs()
        control_mode = ''

        self.env = gym.make('FrankaTestEnv', num_obst=3, n_substeps=20, control_mode="ik_controller", scenario=scenarios['lift_maze'])
        self.env.reset()

    
    def run_gym(self):
        obs, info = self.env.reset()
        
        
        for j in range(1):
            self.env.reset()
            for i in range(10):
                # obs, _, _, _, info = env.step(np.random.rand(4)*2-1)
                # change the action dimension here, e.g. zeros(4) for position controller
                obs, _, _, _, info = self.env.step(np.zeros(4))
                self.env.render()
                time.sleep(0.01)
        self.env.close()
        return obs

class Mayavi:
    def __init__(self):
        from mayavi import mlab
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        # Visualization code using mayavi
        # Function to create and display a sphere obstacle
    def create_sphere(center, radius, color=(1, 0, 0)):
        from mayavi import mlab
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        phi, theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]
        x = radius * np.sin(phi) * np.cos(theta) + center[0]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[2]
        mlab.mesh(x, y, z, color=color)

    # Function to create and display a cylinder obstacle
    def create_cylinder(center, radius, height, color=(0, 1, 0)):
        from mayavi import mlab
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        z = np.linspace(-height / 2, height / 2, 50) + center[2]
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + center[0]
        y_grid = radius * np.sin(theta_grid) + center[1]
        mlab.mesh(x_grid, y_grid, z_grid, color=color)

    # Function to create and display a cube obstacle
    def create_cube(center, size, color=(0, 0, 1)):
        from mayavi import mlab
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        x, y, z = np.mgrid[center[0] - size/2:center[0] + size/2:2j,
                        center[1] - size/2:center[1] + size/2:2j,
                        center[2] - size/2:center[2] + size/2:2j]
        mlab.points3d(x, y, z, mode='cube', color=color, scale_factor=size)

    def create(self):
        from mayavi import mlab
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        # Create a figure and display multiple obstacles
        mlab.figure('3D Obstacles')

        

    # Function to display a table with Matplotlib
    def create_table():
        from mayavi import mlab
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
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
        
    def show(self):
        from mayavi import mlab
        #mlab.imshow(image)
        mlab.show()

    

if __name__ == "__main__":
    env = FrankaEnv()
    obs = env.run_gym()
    print('start testing mayavi')
    plt = Mayavi()
    plt.create()
    # Add obstacles to the scene
    plt.create_sphere(center=[0, 0, 0], radius=2, color=(1, 0, 0))  # Red sphere
    plt.create_cylinder(center=[5, 5, 0], radius=1, height=5, color=(0, 1, 0))  # Green cylinder
    plt.create_cube(center=[-5, -5, 0], size=3, color=(0, 0, 1))  # Blue cube
    plt.create_table()
    plt.show()
    print('done')
    
    

