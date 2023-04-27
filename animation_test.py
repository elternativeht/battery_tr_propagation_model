import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import math
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Annulus
from matplotlib.path import Path

L = 0.173 * 100 # cm
W = 0.045 * 100 # cm
H = 0.125 * 100 # cm

NORTH_SOUTH_SPACING = 1e-2 * 100
NORTH_SOUTH_GAP = 1.5e-2 * 100
EAST_WEST_SPACING = 1e-2 * 100
EAST_WEST_GAP = 2e-2 * 100
PADDING = 1e-2 * 100
CONTAINER_LENGTH = W * 7 + NORTH_SOUTH_SPACING * 2 + NORTH_SOUTH_GAP * 6
CONTAINER_WIDTH = L * 2 +  EAST_WEST_SPACING * 2 + EAST_WEST_GAP * 1

def flame_radius_front(time_elapsed, alpha=0.5, initial_radius=W/2, initial_flame_speed=2e-3):
    if alpha == 0.5:
        return initial_radius * np.exp(initial_flame_speed * time_elapsed / initial_radius)
    else:
        return initial_radius * np.power(1 + (1-2*alpha)*time_elapsed*initial_flame_speed/initial_radius, 1/(1-2*alpha))

def axe_axis_linestyle_set(axe):
    dash_pattern = (10, 5)
    axe.spines['left'].set_linestyle((0, dash_pattern))
    axe.spines['bottom'].set_linestyle((0, dash_pattern))
    axe.spines['right'].set_linestyle((0, dash_pattern))
    axe.spines['top'].set_linestyle((0, dash_pattern))

def create_rect(ax, width, height,origin=(0,0),edge_color='black',face_color='none'):
    rect = Rectangle(origin,
                         width,
                         height,
                         clip_on=False,
                         transform=ax.transData,
                         alpha = 1.0,
                         edgecolor=edge_color,
                         facecolor=face_color)
    return rect

def create_battery(ax, center_point, width, height, failed=False):
    xinit = center_point[0] - W/2
    yinit = center_point[1] + L/2
    if failed:
        battery_face_color = 'red'
    else:
        battery_face_color = 'none'
    return create_rect(ax, width, -height, (xinit, yinit), face_color=battery_face_color)

def main1():

    rect = Rectangle((-2,-2),4,2, facecolor="none", edgecolor="none")
    circle = Circle((0,0),1)

    ax = plt.axes()
    ax.add_artist(rect)
    ax.add_artist(circle)

    circle.set_clip_path(rect)

    plt.axis('equal')
    plt.axis((-2,2,-2,2))
    plt.show()


def main(debug=True):
    fig, ax = plt.subplots()
    axe_axis_linestyle_set(ax)
    ax.set_aspect("equal")
    ax.set_xlim([-PADDING, CONTAINER_LENGTH + PADDING])
    ax.set_ylim([-(CONTAINER_WIDTH + PADDING), PADDING])
    ax.set_xticks([])
    ax.set_yticks([])
    if not debug:
        ax.set_axis_off()
    rect = create_rect(ax, CONTAINER_LENGTH, -CONTAINER_WIDTH)
    ax.add_artist(rect)

    cells = np.array([[j * (W + NORTH_SOUTH_GAP) + W/2 + NORTH_SOUTH_SPACING, 
                       -(i * (L + EAST_WEST_GAP) + L/2 + EAST_WEST_SPACING)] 
                       for i in range(2) for j in range(7)])
    battery_list = []
    for i in range(cells.shape[0]):
        cur_bat = create_battery(ax, cells[i], W, L)
        ax.add_artist(cur_bat)
        battery_list.append(cur_bat)

    def update_frame(i, time_of_failure, order_failure, debug_mode=debug, time_step=10):
        nonlocal cells
        assert len(time_of_failure) == len(order_failure)
        if False:
            count = len(time_of_failure) - 1
            failed_cell_index = i // 10
            time_step = i % 10
            if failed_cell_index < count:
                cur_time = time_of_failure[failed_cell_index] + time_step / 10 * (time_of_failure[failed_cell_index+1] - time_of_failure[failed_cell_index])
            else:
                cur_time = time_of_failure[failed_cell_index]
        
        total_elapsed_duration = time_of_failure[-1] + 2 * time_step - time_of_failure[0]

        cur_time = time_of_failure[0] + time_step * (i-1)

        end_ind = np.argmax(np.array(time_of_failure)>cur_time)

        ax.clear()
        axe_axis_linestyle_set(ax)
        ax.set_aspect("equal")
        ax.set_xlim([-PADDING, CONTAINER_LENGTH + PADDING])
        ax.set_ylim([-(CONTAINER_WIDTH + PADDING), PADDING])
        ax.set_xticks([])
        ax.set_yticks([])
        if not debug_mode:
            ax.set_axis_off()
        rect = create_rect(ax, CONTAINER_LENGTH, -CONTAINER_WIDTH)
        ax.add_artist(rect)
        ax.text(0.5, 1.8, f"Time elapsed: {cur_time:.1f}s", fontsize=12)

        clip_path = Rectangle((0,0),
                         CONTAINER_LENGTH, -CONTAINER_WIDTH,
                         clip_on=False,alpha = 1.0,
                         edgecolor='none',facecolor='none')
        ax.add_artist(clip_path)

        cur_fail_list = order_failure[:end_ind]
        for index in range(cells.shape[0]):
            fail_flag = True if index in cur_fail_list else False
            cur_bat = create_battery(ax, cells[index], W, L,failed=fail_flag)
            ax.add_artist(cur_bat)
        for fail_order_index,cell_index in enumerate(cur_fail_list):
            cur_elapsed_time =  cur_time - time_of_failure[fail_order_index]
            cur_r = flame_radius_front(cur_elapsed_time,alpha=0.3,initial_radius=W/100/2, initial_flame_speed=2e-3)
            cur_r = cur_r * 100
            cur_annulus = Annulus(cells[cell_index], cur_r, 1, angle=0, linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.5)
            ax.add_artist(cur_annulus)
            print(cell_index)
            print(cur_elapsed_time)
            print(cur_r)
            cur_annulus.set_clip_path(clip_path)
    test_time_failure = [1800, 1900, 2400, 2460]
    test_order_failure = [2, 1, 3,6]
    time_step = 10.0
    total_count = int(np.ceil(test_time_failure[-1] - test_time_failure[0]) / time_step ) + 1
    update_func = lambda index_v:  update_frame(index_v,time_of_failure=test_time_failure,order_failure=test_order_failure)
    anim = FuncAnimation(fig, update_func, frames=total_count, interval=300)

        

    #clip_path = Rectangle((0,0),
    #                     CONTAINER_LENGTH, -CONTAINER_WIDTH,
    #                     clip_on=False,alpha = 1.0,
    #                     edgecolor='none',facecolor='none')
    #ax.add_artist(clip_path)
    #annulus = Annulus(cells[0], 10, 3, angle=0, linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.5)
    #ax.add_artist(annulus)
    #annulus.set_clip_path(clip_path)
    

    plt.show()
    exit(0)

main()