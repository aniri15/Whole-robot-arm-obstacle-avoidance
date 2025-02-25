# draw 3d points in matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Sample data: points in 3D space
points = [
   [0.7, 0.6, 0.42],
    [0.75,0.6,0.42],
    [0.8, 0.6, 0.42],
    [0.3,0.5,0.45],
    [0.3,0.3, 0.45],
    [0.405,0.4,0.45],
    [0.195,0.4,0.45]
]
boxes = [[0.7,       0.6,       0.46     ],
 [1.31,      0.7,       0.43     ],
 [1.3,       0.7913688, 0.49     ],
 [1.4,       0.95,      0.65     ],
 [1.2,       0.5,       0.55     ],
 [1.48,      1.1,       0.65     ],
 [1.1,       0.35,      0.65     ]]

sizes = [[0.02,  0.02,  0.06 ],
 [0.025, 0.2,   0.025],
 [0.015, 0.015, 0.03 ],
 [0.1,   0.02,  0.02 ],
 [0.1,   0.02,  0.02 ],
 [0.02,  0.1,   0.02 ],
 [0.02,  0.1,   0.02 ]]
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to draw a box
def draw_box(ax, position, size):
    x, y, z = position
    dx, dy, dz = size
    box = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ])
    
    # List of vertices that compose the faces of the box
    verts = [
        [box[0], box[1], box[5], box[4]],
        [box[7], box[6], box[2], box[3]],
        [box[0], box[3], box[7], box[4]],
        [box[1], box[2], box[6], box[5]],
        [box[0], box[1], box[2], box[3]],
        [box[4], box[5], box[6], box[7]]
    ]
    
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

# Plot each box
for position, size in zip(boxes, sizes):
    draw_box(ax, position, size)

# Set plot limits and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(0, 1)

# Show the plot
plt.show()


# create_sphere(center=[0.7, 0.6, 0.42], radius=0.02, color=(1, 0, 0))  # Red sphere
# create_sphere(center=[0.75,0.6,0.42], radius=0.02, color=(1, 0, 0))  # Red sphere
# create_sphere(center=[0.8, 0.6, 0.42], radius=0.02, color=(1, 0, 0))  # Red sphere
# create_sphere(center=[0.3,0.5,0.45], radius=0.02, color=(1, 0, 0))  # Red sphere
# create_sphere(center=[0.3,0.5, 0.45], radius=0.02, color=(1, 0, 0))  # Red sphere
# create_sphere(center=[0.405,0.4,0.45], radius=0.02, color=(1, 0, 0))  # Red sphere
# create_sphere(center=[0.195,0.4,0.45], radius=0.02, color=(1, 0, 0))  # Red sphere