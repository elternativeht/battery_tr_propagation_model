import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_aspect("equal")
ax.set_xlim([0, 8])
ax.set_ylim([0, 4])
ax.set_xticks([])
ax.set_yticks([])

# Create a 2D array of cell positions
cells = np.array([[j, 1-i] for i in range(2) for j in range(7)])
cell_order = np.array([1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12, 14, 13]) - 1

# Define a function to update the animation
def update(i):
    # Highlight the cells that have failed up to the i-th step
    failed_cells = cell_order[:i+1]
    ax.scatter(cells[:,0], cells[:,1], c='gray', s=100)
    ax.scatter(cells[failed_cells,0], cells[failed_cells,1], c='red', s=100)

    # Add the time axis
    ax.text(3.5, -0.5, f"Time: {i+1} s", fontsize=14, ha='center')

# Define the order of cell failure as a list input
order_of_failure = cell_order[np.argsort(np.random.rand(len(cell_order)))]

# Define the time gap list (in seconds) between each cell failure
time_gaps = np.random.rand(len(cell_order)) * 5

# Create the animation
anim = FuncAnimation(fig, update, frames=len(order_of_failure), interval=1000)

# Update the time gap between each frame of the animation
for i in range(len(time_gaps)):
    anim.event_source.interval = int(time_gaps[i] * 1000)
    anim._draw_next_frame()

# Save the animation as a GIF
with Image.new('RGBA', (800, 400), (255, 255, 255, 255)) as bg:
    anim.save('animation.gif', writer='pillow', fps=1, savefig_kwargs={'transparent': True})