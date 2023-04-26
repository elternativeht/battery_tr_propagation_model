import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from auxiliaries_new import Compartment
import matplotlib.pyplot as plt
# Constants
SIGMA = 5.67e-8
R_CONST = 8.314
T_STD = 273.15
P_STD = 101325.
HEAT_FLUX_FLAME_RATE = 3.e4 # W/(m2)
AIR_MOLE_WEIGHT = 29e-3
INIT_HEATING_PWR = 600.0
AIR_THERMAL_CONDUCTIVITY_TEMP, AIR_THERMAL_CONDUCTIVITY_K = \
np.genfromtxt('air_conductivity.csv',delimiter=',',skip_header=1,unpack=True)
AIR_THERMAL_CONDUCTIVITY_TEMP += 273.15
AIR_THERMAL_CONDUCTIVITY_K *= 1e-3
AIR_THERMAL_CONDUCTIVITY_INTERP = interp1d(AIR_THERMAL_CONDUCTIVITY_TEMP,
                                            AIR_THERMAL_CONDUCTIVITY_K,
                                              bounds_error=False, 
                                              fill_value=(AIR_THERMAL_CONDUCTIVITY_K[0],
                                                          AIR_THERMAL_CONDUCTIVITY_K[-1]))
AIR_SPECIFIC_HEAT_CP = 7 / 2 * R_CONST / AIR_MOLE_WEIGHT
SOLID_SPECIFIC_HEAT_CP = AIR_SPECIFIC_HEAT_CP

ALPHA = 0.3 # power coefficient; assume flame front velocity v_f proportional to (A)^alpha
V_F0 = 2e-3 # flame front initial velocity in meters per second (m/s)
T_B = 20 # burning time in seconds
THRESHOLD_TEMP = 180.0 + 273.15 # threshold temperature for battery going into TR
CELL_CP = 1100 # specific heat of cell in Joules/(kg * K)
T_0 = 300.0 # initial temperature of cell
MASS_0 = 2.01 
BATTERY_TR_RELEASE_PER_MASS = 1e6 # Joules/(kg)


# Geometry
L = 0.173  # m
W = 0.045  # m
H = 0.125  # m

NORTH_SOUTH_SPACING = 1e-2
NORTH_SOUTH_GAP = 1e-2
EAST_WEST_SPACING = 2e-2
EAST_WEST_GAP = 5e-2
CONTAINER_LENGTH = W * 7 + NORTH_SOUTH_SPACING * 2 + NORTH_SOUTH_GAP * 6
CONTAINER_WIDTH = L * 2 +  EAST_WEST_SPACING * 2 + EAST_WEST_GAP * 1

CONTAINER_HEIGHT = 15.0e-2


# Interpolate the gas release rate
TIME_VALUES = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
GAS_RELEASE_RATE_LITERS_PER_SEC = np.array([0, 10.0, 12.5, 18.0, 22.5, 25.0, 30.0, 45.0, 
                                            60.0, 65.0, 60.0, 50.0, 32.5, 12.5, 0.0])
GAS_RELEASE_RATE_KG_PER_SEC = GAS_RELEASE_RATE_LITERS_PER_SEC * P_STD / R_CONST / T_STD * AIR_MOLE_WEIGHT * 1e-3
GAS_RELEASE_RATE_INTERP = interp1d(TIME_VALUES, GAS_RELEASE_RATE_KG_PER_SEC, bounds_error=False, fill_value=0.0)

SOLID_RELEASE_RATE_KG_PER_SEC = GAS_RELEASE_RATE_KG_PER_SEC * 2.140
SOLID_RELEASE_RATE_INTERP = interp1d(TIME_VALUES, SOLID_RELEASE_RATE_KG_PER_SEC, bounds_error=False, fill_value=0.0)

def flame_radius_front(time_elapsed, alpha=ALPHA, initial_radius=W/2, initial_flame_speed=V_F0):
    if alpha == 0.5:
        return initial_radius * np.exp(initial_flame_speed * time_elapsed / initial_radius)
    else:
        return initial_radius * np.power(1 + (1-2*alpha)*time_elapsed*initial_flame_speed/initial_radius, 1/(1-2*alpha))
    

def dr_back_dt(timed_elapsed, rb, tau, alpha, initial_radius, initial_flame_speed):
    rf = flame_radius_front(timed_elapsed, alpha, initial_radius, initial_flame_speed)
    return (rf - rb)/tau

TIME_FLAME_ELAPSED = np.arange(0,5001,1)

#RF_FLAME_M = flame_radius_front(TIME_FLAME_ELAPSED,ALPHA, initial_radius=W, initial_flame_speed=V_F0)

RB_FLAME_M = solve_ivp(fun=dr_back_dt, t_span=[0,5000], y0=[0.0], 
                       t_eval=np.arange(0,5001,1),method='BDF',
                       max_step=0.5,
                       #dense_output=True, 
                       args=(T_B, ALPHA, W, V_F0)).y
RB_FLAME_M = np.reshape(RB_FLAME_M, (5001,))
index_max = np.where(RB_FLAME_M<0.5)[0][-1]

RB_FLAME_INTERP = interp1d(TIME_FLAME_ELAPSED[:index_max+1],RB_FLAME_M[:index_max+1],
                           bounds_error=False,
                           fill_value=(0.0,0.5))

def flames_effect_ratio(min_dist, max_dist,r_f_front, r_f_back):
    front_effect = min(1.0, max(0.0, r_f_front - min_dist))/(max_dist - min_dist)
    back_effect = min(1.0, max(0.0, r_f_back - min_dist))/(max_dist - min_dist)
    return front_effect - back_effect

def rate_calculate_fun(t, y, compartment: Compartment):
    temp = y[:14]
    mass = y[14:28]

    temp_rate = np.zeros(14)
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
    cell_conduction_rate = np.zeros((14,),dtype=float)

def rhs(t, y, compartment: Compartment):
    temp = y[:14]
    mass = y[14:28]

    temp_rate = np.zeros(14)
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
    cell_conduction_rate = np.zeros((14,),dtype=float)

    self_pre_heat_rate =np.zeros((14,),dtype=float)
    
    if len(compartment.failure_order) == 0:
        temp_rate[0] = 1/12.0
        temp_rate[1] = 0.2/12.0
        #temp_rate = temp_rate / mass / CELL_CP
        dydt = np.concatenate([temp_rate, mass_rate])
        return dydt



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
    if False:
        for preheat_i in compartment.preheat_list:
            if t - compartment.preheat_time[preheat_i] <= 350:
                self_pre_heat_rate[preheat_i] = max(self_pre_heat_rate[preheat_i],300)

    

    
    cell_radiation_rate = compartment.cell_surface_area * SIGMA * \
    (np.matmul(compartment.compt_rad_matrix, np.power(temp,4)))
    cell_radiation_rate -= compartment.cell_surface_area * SIGMA * np.power(temp,4)
    cell_conduction_rate = (np.matmul(compartment.compt_conduction_matrix,temp) -
                            temp * np.sum(compartment.compt_conduction_matrix, axis=1)
                           ) * AIR_THERMAL_CONDUCTIVITY_INTERP(temp)
                                                        
    temp_rate =   cell_tr_release_rate \
                + cell_outflow_enthalpy_rate \
                + cell_flame_heat_rate \
                + cell_radiation_rate \
                + cell_conduction_rate
                #+ self_pre_heat_rate
    temp_rate = temp_rate / mass / CELL_CP
    # Combine the state derivatives
    dydt = np.concatenate([temp_rate, mass_rate])
    return dydt
def main(step=1):
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
                         max_step=1,
                         args=(battery_compartment,),
                         )
    tt = solution.t
    fig,axe=plt.subplots(figsize=(8,8))
    for i in range(14):
        axe.plot(tt,solution.y[i],label=f'cell{i}')
    axe.legend()
    axe.set_xlim([1800,simulation_time])
    plt.show()
    fig,axe=plt.subplots(figsize=(8,8))
    #for i in range(14):
    #    axe.plot(tt,solution.y[i+14],label=f'cell{i}')
    #plt.show()
    print(solution.message)

main()