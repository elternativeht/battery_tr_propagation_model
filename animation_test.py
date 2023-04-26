import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

def create_rect(ax, width, height):
    rect = plt.Rectangle((0, 0), width, height, clip_on=False, transform=ax.transData, edgecolor='none', facecolor='none')
    return rect
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim([0, 7])
ax.set_ylim([0, 2])
ax.set_xticks([])
ax.set_yticks([])

# Create a 2D array of cell positions
cells = np.array([[j + 0.5, i + 0.5] for i in range(2) for j in range(7)])

def wave_mask(ax, cell_position, radius):
    circle = plt.Circle(cell_position, radius, clip_on=False, transform=ax.transData, alpha=0.5, facecolor='blue', edgecolor='none', linewidth=0)
    return circle

def back_wave_mask(ax, cell_position, radius):
    circle = plt.Circle(cell_position, radius, clip_on=False, transform=ax.transData, alpha=0.5, facecolor='white', edgecolor='none', linewidth=0)
    return circle

# Define a function to update the animation
def update(i):
    global elapsed_time
    ax.clear()
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 2])
    ax.set_xticks([])
    ax.set_yticks([])

    # Highlight the cells that have failed up to the i-th step
    failed_cells = order_of_failure[:i + 1]
    ax.scatter(cells[:, 0], cells[:, 1], c='gray', s=100)
    ax.scatter(cells[failed_cells, 0], cells[failed_cells, 1], c='red', s=100)

    # Add expanding waves and contracting back waves for each failed cell
    for cell_index in failed_cells:
        cell_position = cells[cell_index]
        radius_expansion_rate = 0.1
        radius_contraction_rate = 0.05
        time_since_failure = elapsed_time - time_of_failure[cell_index]
        wave_radius = time_since_failure * radius_expansion_rate
        back_wave_radius = wave_radius - radius_contraction_rate * time_since_failure

        if wave_radius > 0:
            ax.add_patch(wave_mask(ax, cell_position, wave_radius))

        if back_wave_radius > 0:
            ax.add_patch(back_wave_mask(ax, cell_position, back_wave_radius))

    # Display elapsed time
    ax.text(0.5, 1.8, f"Elapsed time: {elapsed_time:.2f}s", fontsize=12)
    elapsed_time += 0.5

order_of_failure = [2, 9, 10, 3, 11, 4, 12, 5, 13, 6, 7, 1, 8, 0]
time_of_failure = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
elapsed_time = 0

# Create the animation
anim = FuncAnimation(fig, update, frames=int(max(time_of_failure) + sum(np.diff(time_of_failure))/0.5) * 2, interval=500)

# Display the animation
plt.show()

# Save the animation as a GIF
anim.save('animation.gif', writer='pillow', fps=2, savefig_kwargs={'transparent': False})