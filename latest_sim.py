import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from auxiliaries_new import Compartment
#from animation_test import axe_axis_linestyle_set, create_rect, create_battery
from matplotlib.patches import Circle, Rectangle, Annulus
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def axe_axis_linestyle_set(axe):
    '''
    Functions to set the dash line pattern for the given axe; 
    used for the animation of the flames

    Parameters: axe: Matplotlib axe class object
    '''
    dash_pattern = (10, 5)
    axe.spines['left'].set_linestyle((0, dash_pattern))
    axe.spines['bottom'].set_linestyle((0, dash_pattern))
    axe.spines['right'].set_linestyle((0, dash_pattern))
    axe.spines['top'].set_linestyle((0, dash_pattern))

def create_rect(ax, width, height,origin=(0,0),edge_color='black',face_color='none'):
    '''
    create a rectangle object in the animation indicating battery cell
    Parameters:
    - ax: Matplotlib axe object
    - width: the battery width
    - height: the battery length
    - origin: a two-element tuple indicating the location where the origin (starting corner) of
    rectangle; default to be (0, 0)
    - edge_color: the color of the battery edge(side); default to be 'black' string
    - face_color: the color of the battery face; default to be 'none' (indicating transparent)

    returns:
    rect: the `Rectangle` object
    '''
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
    '''
    Wrapping function for creating a battery shape object in a plot.

    Parameters:
    - ax: Matplotlib axe object on which the plot is based
    - center_point: two-element tuple; containing the battery center point (x, y)
    - width: `float` variable; battery width
    - height: `float` variable; battery length
    - failed: `boolean`; whether the cell is failed

    Returns:
    a `Rectangle` object created by previous `create_rect()` function.
    '''
    xinit = center_point[0] - W/2
    yinit = center_point[1] + L/2
    if failed:
        battery_face_color = 'red'
    else:
        battery_face_color = 'none'
    return create_rect(ax, width, -height, (xinit, yinit), edge_color='black',face_color=battery_face_color)

# Physical Constants
# NOT TO BE PARAMETERIZED
SIGMA = 5.67e-8 # Stefan-Boltzmann constant
R_CONST = 8.314 # Universal gas constant
T_STD = 273.15
P_STD = 101325.
AIR_MOLE_WEIGHT = 29e-3
# air conductivity read in and interpolation
AIR_THERMAL_CONDUCTIVITY_TEMP, AIR_THERMAL_CONDUCTIVITY_K = \
np.genfromtxt('air_conductivity.csv',delimiter=',',skip_header=1,unpack=True)
AIR_THERMAL_CONDUCTIVITY_TEMP += 273.15
AIR_THERMAL_CONDUCTIVITY_K *= 1e-3
AIR_THERMAL_CONDUCTIVITY_INTERP = interp1d(AIR_THERMAL_CONDUCTIVITY_TEMP,
                                            AIR_THERMAL_CONDUCTIVITY_K,
                                              bounds_error=False, 
                                              fill_value=(AIR_THERMAL_CONDUCTIVITY_K[0],
                                                          AIR_THERMAL_CONDUCTIVITY_K[-1]))
# air regarded as ideal gas
AIR_SPECIFIC_HEAT_CP = 7 / 2 * R_CONST / AIR_MOLE_WEIGHT
# battery particle specific heat treated as the same as air
SOLID_SPECIFIC_HEAT_CP = AIR_SPECIFIC_HEAT_CP

INIT_HEATING_PWR = 600.0 # initial external heater heating power 



# Geometry
L = 0.173  # m
W = 0.045  # m
H = 0.125  # m

PADDING = 1e-2 # plotting padding; the spacing outside of the axis

NORTH_SOUTH_SPACING = 1e-2
NORTH_SOUTH_GAP = 1.5e-2
EAST_WEST_SPACING = 1e-2
EAST_WEST_GAP = 2e-2
CONTAINER_LENGTH = W * 7 + NORTH_SOUTH_SPACING * 2 + NORTH_SOUTH_GAP * 6
CONTAINER_WIDTH = L * 2 +  EAST_WEST_SPACING * 2 + EAST_WEST_GAP * 1

CONTAINER_HEIGHT = 15.0e-2

# TO-DO: note that the geometry data points are already in the `Compartment` class. Geometry data is 
# included here because of the need to be called in functions. Need to eliminate the repeating definition of geometric
# data.

# Interpolate the gas release rate
TIME_VALUES = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
GAS_RELEASE_RATE_LITERS_PER_SEC = np.array([0, 10.0, 12.5, 18.0, 22.5, 25.0, 30.0, 45.0, 
                                            60.0, 65.0, 60.0, 50.0, 32.5, 12.5, 0.0])
GAS_RELEASE_RATE_KG_PER_SEC = GAS_RELEASE_RATE_LITERS_PER_SEC * P_STD / R_CONST / T_STD * AIR_MOLE_WEIGHT * 1e-3
GAS_RELEASE_RATE_INTERP = interp1d(TIME_VALUES, GAS_RELEASE_RATE_KG_PER_SEC, bounds_error=False, fill_value=0.0)

SOLID_RELEASE_RATE_KG_PER_SEC = GAS_RELEASE_RATE_KG_PER_SEC * 2.140
SOLID_RELEASE_RATE_INTERP = interp1d(TIME_VALUES, SOLID_RELEASE_RATE_KG_PER_SEC, bounds_error=False, fill_value=0.0)

# Parameters potentially ready for tuning
HEAT_FLUX_FLAME_RATE = 3.e4 # flame heat transfer flux for polycarbonate container flames [W/(m2)]
ALPHA = 0.5 # power coefficient; assume flame front velocity v_f proportional to (A)^alpha
V_F0 = 2e-3 # flame front initial velocity in meters per second (m/s)
T_B = 110.0 # burning time in seconds
THRESHOLD_TEMP = 180.0 + 273.15 # threshold temperature for battery going into TR
CELL_CP = 1100 # specific heat of cell in Joules/(kg * K)
T_0 = 300.0 # initial temperature of cell
MASS_0 = 2.01 # initial battery cell mass in kg
BATTERY_TR_RELEASE_PER_MASS = 1e6 # Joules/(kg)

def flame_radius_front(time_elapsed, alpha=ALPHA, initial_radius=W/2, initial_flame_speed=V_F0):
    '''
    Function to calculate the flame front radius after propagation

    Parameters:
    - `time_elapsed`: the time elapsed from the cell thermal runaway onset [seconds]
    - `alpha`: the power coefficient [unitless]
    - `initial_radius`: initial flame front radius at the thermal runaway onset [m]
    - `initial_flame_speed`: initial flame front propagation speed [m/s]

    Returns a `float` containing current flame front radius, with circle center the battery cell center
    '''
    if alpha == 0.5:
        return initial_radius * np.exp(initial_flame_speed * time_elapsed / initial_radius)
    else:
        return initial_radius * np.power(1 + (1-2*alpha)*time_elapsed*initial_flame_speed/initial_radius, 1/(1-2*alpha))
    

def dr_back_dt(timed_elapsed, rb, tau, alpha, initial_radius, initial_flame_speed):
    '''
    Right hand side of the ODE function that defines the flame back propagation rate with time.
    ODE function is to be solved numerically below using `solve_ivp` to obtain the flame back
    radius.

    Parameters:
    - `time_elapsed`: the time elapsed from the cell thermal runaway onset [seconds]
    - `rb`: current flame back radius [m]
    - `tau`: burnout time (the time it takes for the flame to quench) [s]
    - `alpha`: the power coefficient [unitless]
    - `initial_radius`: initial flame back radius at the thermal runaway onset [m]
    - `initial_flame_speed`: initial flame back propagation speed [m/s]

    Returns: a `float` containing information of the flame back radius propagation rate [m/s]
    '''
    rf = flame_radius_front(timed_elapsed, alpha, initial_radius, initial_flame_speed)
    return (rf - rb)/tau

# Calculate the interpolation look-up table between elapsed time and instantaneous flame front radius
# time array
TIME_FLAME_ELAPSED = np.arange(0,5001,1)
# flame front radius array
RF_FLAME_M = flame_radius_front(TIME_FLAME_ELAPSED,ALPHA, initial_radius=W, initial_flame_speed=V_F0)
# flame back radius array; solved numerically
RB_FLAME_M = solve_ivp(fun=dr_back_dt, t_span=[0,5000], y0=[0.0], 
                       t_eval=np.arange(0,5001,1),method='BDF',
                       max_step=0.5,
                       #dense_output=True, 
                       args=(T_B, ALPHA, W, V_F0)).y
RB_FLAME_M = np.reshape(RB_FLAME_M, (5001,))

# screen out the unrealistic numbers
index_max = np.where(RB_FLAME_M<0.5)[0][-1]

RB_FLAME_INTERP = interp1d(TIME_FLAME_ELAPSED[:index_max+1],RB_FLAME_M[:index_max+1],
                           bounds_error=False,
                           fill_value=(0.0,0.5))

def flames_effect_ratio(min_dist, max_dist,r_f_front, r_f_back):
    '''
    Function to calculate the percentage of the surface area exposed to flames.
    Flame front traverses the entire cell. The area exposed to the flame increases linearly
    with the flame front traversal distance (r_f_front - min_dist)/(max_dist - min_dist). 
    The area exposed to the flame decreases linearly with the flame back traversal distance
    (0.0, r_f_back - min_dist)/(max_dist - min_dist).

    Parameters:
    -`min_dist`: the minimal distance from center of the cell which initiates the flame to
     the current cell.
    -`max_dist`: the maximal distance from center of the cell which initiates the flame to 
     the current cell
    - `r_f_front`: current flame front radius of the flame of interest
    - `r_b_front`: current flame back radius of the flame of interest

    Returns: the coefficient representing the ratio of the entire cell surface area exposed
    to the flame
    '''
    front_effect = min(1.0, max(0.0, r_f_front - min_dist))/(max_dist - min_dist)
    back_effect = min(1.0, max(0.0, r_f_back - min_dist))/(max_dist - min_dist)
    return front_effect - back_effect

def rate_calculate_fun(t, y, compartment: Compartment):
    '''
    The function calculating the heat transfer rate given current time and states

    Parameters:
    - t: time variables; absolute time since the start of simulation [s]
    - y: a one-dim `np.ndarray` with shape (28,) storing current state

    Returns:
    - calc_heat_power: a two-dim `np.ndarray` array with shape of (5, 14) storing the 
    - mass_rate
    '''
    temp = y[:14]
    mass_rate = np.zeros(14)

    for i in range(14):
        if temp[i] >= THRESHOLD_TEMP and compartment.status[i]==1:
            compartment.update_compartment(t, i)
            if i in compartment.preheat_list:
                compartment.preheat_list.remove(i)
        if (temp[i]>273.15 + 130.0) and (compartment.preheat_status[i]==0):
            if i > 0:
                compartment.update_preheat(t, i)
    cell_tr_release_rate = np.zeros((14,),dtype=float)
    cell_outflow_enthalpy_rate = np.zeros((14,),dtype=float)
    cell_flame_heat_rate = np.zeros((14, ),dtype=float)
    cell_radiation_rate = np.zeros((14, ),dtype=float)
    cell_conduction_rate = np.zeros((14,),dtype=float)

    if len(compartment.failure_order) == 0:
        cell_conduction_rate[0] = 1/12.0 * 0.8 * MASS_0 * CELL_CP
        cell_conduction_rate[1] = 1/12.0 * 0.2 * MASS_0 * CELL_CP

    else:
        for fail_cell_i in compartment.failure_order:
            failed_time_elapsed = t - compartment.failure_time[fail_cell_i]
            gas_venting_mass_rate = GAS_RELEASE_RATE_INTERP(failed_time_elapsed)
            solid_venting_mass_rate = SOLID_RELEASE_RATE_INTERP(failed_time_elapsed)
            mass_rate[fail_cell_i] = -1 * (gas_venting_mass_rate + solid_venting_mass_rate)
            cell_tr_release_rate[fail_cell_i] = (gas_venting_mass_rate + solid_venting_mass_rate) * \
            BATTERY_TR_RELEASE_PER_MASS
            
            cell_outflow_enthalpy_rate[fail_cell_i] = -1 * \
            (gas_venting_mass_rate * AIR_SPECIFIC_HEAT_CP + \
            solid_venting_mass_rate * SOLID_SPECIFIC_HEAT_CP) * temp[fail_cell_i]
            if fail_cell_i == 0:
                r_flame_front = flame_radius_front(failed_time_elapsed)
                r_flame_front = min(max(0, r_flame_front),compartment.center_to_edge_dist[fail_cell_i])
                r_flame_back = RB_FLAME_INTERP(failed_time_elapsed)
                r_flame_back = min(max(0, r_flame_back),compartment.center_to_edge_dist[fail_cell_i])

                for rec_cell_i in range(14):
                    if rec_cell_i == fail_cell_i:
                        continue
                    cell_flame_heat_rate[rec_cell_i] = max(flames_effect_ratio(
                                        min_dist=compartment.min_dist[rec_cell_i][fail_cell_i],
                                        max_dist=compartment.max_dist[rec_cell_i][fail_cell_i],
                                        r_f_front=r_flame_front, 
                                        r_f_back=r_flame_back) \
                                        * compartment.cell_surface_area * HEAT_FLUX_FLAME_RATE,
                                    cell_flame_heat_rate[rec_cell_i]
                                    )
                
        cell_radiation_rate = compartment.cell_surface_area * SIGMA * \
        (np.matmul(compartment.compt_rad_matrix, np.power(temp,4)))
        cell_radiation_rate -= compartment.cell_surface_area * SIGMA * np.power(temp,4)
        cell_conduction_rate = (np.matmul(compartment.compt_conduction_matrix,temp) -
                                temp * np.sum(compartment.compt_conduction_matrix, axis=1)
                            ) * AIR_THERMAL_CONDUCTIVITY_INTERP(temp)
        
        cell_conduction_rate = np.zeros(14)
    
    calc_heat_power  = np.array([cell_tr_release_rate,
                         cell_outflow_enthalpy_rate,
                         cell_flame_heat_rate,
                         cell_radiation_rate,
                         cell_conduction_rate,
    ])

    return calc_heat_power, mass_rate


def rhs(t, y, bat_compartment: Compartment):
    mass = y[14:28]
    heat_pwr_list, mass_rate = rate_calculate_fun(t, y, bat_compartment)
    temp_rate = np.sum(heat_pwr_list,axis=0) / mass / CELL_CP
    # Combine the state derivatives
    dydt = np.concatenate([temp_rate, mass_rate])
    return dydt


def main(step=1,debug=True):
    battery_compartment = Compartment(dimension=(L,W,H),
                                      spacing=(NORTH_SOUTH_SPACING,EAST_WEST_SPACING),
                                      gap=(NORTH_SOUTH_GAP,EAST_WEST_GAP))
    simulation_time = 3000.0  # seconds
    initial_mass = np.ones(14,) * MASS_0
    initial_temp = np.ones(14,) * T_0
    initial_y = np.concatenate([initial_temp, initial_mass])
    solution = solve_ivp(rhs,
                         t_span=(0,simulation_time),
                         y0 = initial_y,
                         t_eval=np.arange(0,simulation_time+step,step),
                         dense_output=True,
                         max_step=step,
                         args=(battery_compartment,),
                         )
    rebuilding_compartment = Compartment(dimension=(L,W,H),
                                      spacing=(NORTH_SOUTH_SPACING,EAST_WEST_SPACING),
                                      gap=(NORTH_SOUTH_GAP,EAST_WEST_GAP))
    solved_time = solution.t
    pwr_rate_array = []
    mass_rate_array =[]

    for cur_timestep in range(solution.y.shape[1]):
        cur_t = solved_time[cur_timestep]
        cur_y = solution.y[:,cur_timestep]
        cur_pwr_rate, cur_mass_change_rate = rate_calculate_fun(cur_t, cur_y, rebuilding_compartment)
        pwr_rate_array.append(cur_pwr_rate)
        mass_rate_array.append(cur_mass_change_rate)
    pwr_rate_array = np.array(pwr_rate_array)
    mass_rate_array = np.array(mass_rate_array)
    
    fig,axe=plt.subplots(figsize=(8,8))
    for i in range(14):
        axe.plot(solved_time,solution.y[i],label=f'cell{i}')
    axe.legend()
    axe.set_xlim([1800,simulation_time])
    plt.show()
    fig1,axe1=plt.subplots(figsize=(8,8))
    for cell_index in range(14):
        axe1.plot(solved_time,pwr_rate_array[:,0,cell_index],label=f'cell {cell_index+1}')
    plt.show()
    
    
    fig2, axe2 = plt.subplots()
    axe_axis_linestyle_set(axe2)
    axe2.set_aspect("equal")
    axe2.set_xlim([-PADDING, CONTAINER_LENGTH + PADDING])
    axe2.set_ylim([-(CONTAINER_WIDTH + PADDING), PADDING])
    axe2.set_xticks([])
    axe2.set_yticks([])
    if not debug:
        axe2.set_axis_off()
    rect = create_rect(axe2, CONTAINER_LENGTH, -CONTAINER_WIDTH)
    axe2.add_artist(rect)
    #rect_test = create_rect(axe2, width=W,height=-L,origin=(0.033-W/2, -0.096+L/2))
    rect_test = create_battery(axe2, (0.033, -0.096), W, L, failed=False)
    axe2.add_artist(rect_test)

    cells = np.array([[j * (W + NORTH_SOUTH_GAP) + W/2 + NORTH_SOUTH_SPACING, 
                       -(i * (L + EAST_WEST_GAP) + L/2 + EAST_WEST_SPACING)] 
                       for i in range(2) for j in range(7)])
    for i in range(cells.shape[0]):
        cur_bat = create_battery(axe2, cells[i], W, L,failed=False)
        axe2.add_artist(cur_bat)
    
    def update_frame(i, plotting_axe, time_of_failure, order_failure, debug_mode=debug, time_step=10):
        nonlocal cells, rebuilding_compartment
        assert len(time_of_failure) == len(order_failure)
        cur_time = time_of_failure[0] + time_step * (i-1)

        if cur_time < time_of_failure[0]:
            end_ind = 0
        elif cur_time >= time_of_failure[-1]:
            end_ind = len(time_of_failure)
        else:
            end_ind = np.argmax(np.array(time_of_failure)>cur_time)

        plotting_axe.clear()
        axe_axis_linestyle_set(plotting_axe)
        plotting_axe.set_aspect("equal")
        plotting_axe.set_xlim([-PADDING, CONTAINER_LENGTH + PADDING])
        plotting_axe.set_ylim([-(CONTAINER_WIDTH + PADDING), PADDING])
        plotting_axe.set_xticks([])
        plotting_axe.set_yticks([])
        if not debug_mode:
            plotting_axe.set_axis_off()
        rect = create_rect(plotting_axe, CONTAINER_LENGTH, -CONTAINER_WIDTH)
        plotting_axe.add_artist(rect)
        plotting_axe.text(0.5e-2, 1.8e-2, f"Time elapsed: {cur_time:.1f}s", fontsize=12)

        clip_path = Rectangle((0,0),
                         CONTAINER_LENGTH, -CONTAINER_WIDTH,
                         clip_on=False,alpha = 1.0,
                         edgecolor='none',facecolor='none')
        plotting_axe.add_artist(clip_path)

        cur_fail_list = order_failure[:end_ind]
        for index in range(cells.shape[0]):
            fail_flag = True if index in cur_fail_list else False
            cur_bat = create_battery(plotting_axe, cells[index], W, L,failed=fail_flag)
            plotting_axe.add_artist(cur_bat)
        
        for fail_order_index,cell_index in enumerate(cur_fail_list):
            cur_elapsed_time =  cur_time - time_of_failure[fail_order_index]
            cur_rf = flame_radius_front(cur_elapsed_time,alpha=0.3,initial_radius=W/2, initial_flame_speed=2e-3)
            cur_rb = RB_FLAME_INTERP(cur_elapsed_time)
            cur_rf = min(max(W/2, cur_rf),rebuilding_compartment.center_to_edge_dist[cell_index])
            cur_rb = min(max(1e-3, cur_rb),rebuilding_compartment.center_to_edge_dist[cell_index])
            if cur_rf > cur_rb and cur_rf > 0:
                cur_annulus = Annulus(cells[cell_index], cur_rf, cur_rf-cur_rb, angle=0, 
                                  linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.5)
                plotting_axe.add_artist(cur_annulus)
                cur_annulus.set_clip_path(clip_path)

    time_step = 5.0
    total_count = int(np.ceil(battery_compartment.failure_time[-1] - battery_compartment.failure_time[0]) / time_step ) + 5
    update_func = lambda index_v:  update_frame(index_v, plotting_axe=axe2,
                                                time_of_failure=battery_compartment.failure_time,
                                                order_failure=battery_compartment.failure_order,time_step=time_step)
    anim = FuncAnimation(fig2, update_func, frames=total_count, interval=500)
    plt.show()

main()