'''
Simulation script file of battery module failure (thermal runaway) propagation.
'''
import numpy as np
import numpy.ma as ma
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from auxiliaries import CellModule

# All units are in SI-MKG unless explicitly stated; temperature unit is Kelvin

# Constants:

SIGMA = 5.67e-8
R_UNIVERSAL = 8.314
MOLECULAR_WEIGHT_AIR = 29e-3
ABSOLUTE_ZERO_K = 273.15

T_IGN = 180.0 + ABSOLUTE_ZERO_K

# Setup Constants:
CELL_LENGTH = 0.173
CELL_WIDTH = 0.045
CELL_HEIGHT = 0.125

CELL_MEAN_SPECIFIC_HEAT = 1100.0 # J __kg __K

CELL_WIDTH_GAP = 0.03
CELL_WIDTHWISE_CENTER_DIST = CELL_WIDTH_GAP + CELL_WIDTH

CELL_LENGTH_CAP = 0.04
CELL_LENGTHWISE_CENTER_DIST = CELL_LENGTH_CAP + CELL_LENGTH

BASE_TO_CEILING_DIST = 0.35

# Initial values
CELL_INIT_TEMP = 300.0 # Kelvin
CELL_INIT_MASS = 2.01 # kg 
# external heat sources
HEATING_PWR = 500.0 # Watts

H_CONV_HT_COEFF = np.ones(15) * 10 # dummy values

AIR_THERMAL_CONDUCTIVITY_TEMP, AIR_THERMAL_CONDUCTIVITY_K = np.genfromtxt('air_conductivity.csv',
                                                         delimiter=',',skip_header=1,unpack=True)
AIR_THERMAL_CONDUCTIVITY_TEMP += ABSOLUTE_ZERO_K
AIR_THERMAL_CONDUCTIVITY_K *= 1e-3

GAS_VENTING_ELAPSED_TIME, GAS_VENTING_VOL_RATE = np.genfromtxt('gas_venting_rate.csv',
                                                    delimiter=',',skip_header=1,unpack=True)
GAS_VENTING_VOL_RATE *= 1e-3

SPECIFIC_HEAT_CP = 7 / 2 * R_UNIVERSAL / MOLECULAR_WEIGHT_AIR
SPECIFIC_HEAT_ARRAY = np.ones(15)*CELL_MEAN_SPECIFIC_HEAT
SPECIFIC_HEAT_ARRAY[14] = 5/2 * R_UNIVERSAL / MOLECULAR_WEIGHT_AIR

TIME_VENTING = 7.0

REACT_RLSE_ENG_J = 1100 * 1000 * CELL_INIT_MASS
REACT_RLSE_PWR_W = REACT_RLSE_ENG_J / TIME_VENTING

def thermal_conductivity(temp_K):
    '''
    The function calculating the current thermal conductivity of air, given
    the current temperature.

    The points used for interpolation comes from the Engineering Toolbox website:
    https://www.engineeringtoolbox.com/air-properties-viscosity-conductivity-heat-capacity-d_1509.html
    '''
    return  np.interp(x=temp_K,
                      xp=AIR_THERMAL_CONDUCTIVITY_TEMP,
                      fp=AIR_THERMAL_CONDUCTIVITY_K)

def gas_venting_rate(time_s, temp_K):
    '''
    The function used to calcualte the mass gas venting rate (and hence the battery 
    mass loss rate) based on the volumetric gas venting rate data obtained by Erik Archibald.

    Parameters:

   `time_s`: a float showing the elapsed time [s]. It should be between 0 and 7 seconds, but
    violating the limits will not cause severe calculation errors as interpolation function 
    will force out-of-bound bound inputs to obtain 0 results.

    `temp_K`: a `np.ndarray[float]` showing the temperature

    Returns:

    cur_mass_rate: `np.ndarray[float]` displaying the mass change rate (absolute)

    '''
    cur_vol_rate = np.interp(x=time_s, xp=GAS_VENTING_ELAPSED_TIME,fp=GAS_VENTING_VOL_RATE)
    cur_mass_rate = 101325 * cur_vol_rate / temp_K[:14] / 8.314 * 29e-3
    return cur_mass_rate

def tr_function(t, y, init_heating: bool, battery_module: CellModule, vent_time_remaining):
    y_prime = np.zeros(30)

    # first 15 variables: the temperature 1 to 15 (cell 1 to 14 plus air)
    T = y[:15] 
    # last 15 variables: the mass 1 to 15 (cell 1 to 14 plus air)
    M = y[15:30]

    radiation_release = SIGMA * np.power(T, 4) * battery_module.radiation_area \
                        * battery_module.emissivity

    radiation_aborption = np.matmul(battery_module.rad_matrix.transpose(), radiation_release) \
                        * battery_module.absorptivity

    # convection heat aborption from air
    convection_absorption = battery_module.cell_total_area * H_CONV_HT_COEFF * (T[14] - T)
    negative_conv = np.sum(convection_absorption) * (-1)
    convection_absorption[14] = negative_conv

    conduction_absorption = thermal_conductivity(T) * battery_module.cell_height * \
                            battery_module.cell_length * np.matmul(battery_module.cond_matrix, T)

    heating_source = np.zeros(15)
    mass_change_rate = np.zeros(15)
    if init_heating:
        heating_source[0] = HEATING_PWR
    else:
        #cur_ = -1 * gas_venting_rate(TIME_VENTING - vent_time_remaining + t,T)
        temp_mass_array = -1 * gas_venting_rate(TIME_VENTING - vent_time_remaining + t,T)
        for cur_index in range(14):
            if vent_time_remaining[cur_index] > 0.0 and t < vent_time_remaining[cur_index]:
                #heating_source[cur_index] = REACT_RLSE_PWR_W 
                heating_source[cur_index] = REACT_RLSE_PWR_W
                #mass_change_rate[cur_index] = [cur_index]
                mass_change_rate[cur_index] = temp_mass_array[cur_index]
                #heating_source[cur_index] = -1 * mass_change_rate[cur_index] * 1200 * 1000

    signed_enthalpy_change = SPECIFIC_HEAT_CP * mass_change_rate * T
    signed_enthalpy_change[14] = np.sum(SPECIFIC_HEAT_CP* (-1) * mass_change_rate * (T-T[14]))

    y_prime[:15] = (radiation_aborption
                   - radiation_release
                   + convection_absorption
                   + conduction_absorption
                   + heating_source
                   + signed_enthalpy_change
                   )/ (M*SPECIFIC_HEAT_ARRAY)

    y_prime[15:30] = mass_change_rate
    return y_prime

def stop_func(t, y, init_heating: bool, battery_module: CellModule, vent_time_remaining):
    T = y[:14]
    return np.prod(T[battery_module.unfailed_list]<T_IGN)
stop_func.terminal = True
stop_func.direction = -1

def main():
    init_temp_array = np.ones(15) * CELL_INIT_TEMP
    init_mass_array = np.ones(15) * CELL_INIT_MASS

    module_padding = 0.03
    module_length = CELL_WIDTH * 7 + CELL_WIDTH_GAP * 6 + module_padding * 2
    module_width = CELL_LENGTH * 2 + CELL_LENGTH_CAP + module_padding * 2
    air_vol = BASE_TO_CEILING_DIST * module_length * module_width \
              - 14 * CELL_HEIGHT * CELL_HEIGHT * CELL_WIDTH
    init_mass_array[14] = 101325 * air_vol \
                          / R_UNIVERSAL / init_temp_array[14] * MOLECULAR_WEIGHT_AIR

    lco_battery_module = CellModule(cell_dim=(CELL_LENGTH, CELL_WIDTH, CELL_HEIGHT),
                                    cell_dist=(CELL_LENGTH_CAP, CELL_WIDTH_GAP))
    crit_time_stamp = []

    time_range = [0,1500]

    init_y_ = np.ones(30)*CELL_INIT_TEMP
    init_y_[15:30] = init_mass_array

    solution_t = np.array([0])
    solution_y = init_y_.copy()
    solution_y = np.reshape(solution_y,(30,1))

    time_offset = 0.0
    count = 0
    init_heating = True
    vent_time_remaining = np.zeros(14)

    while count <=13:
        print(count)
        cur_solution = solve_ivp(tr_function,
                                 time_range,
                                 init_y_,
                                 events=stop_func,
                                 args=(init_heating,lco_battery_module,vent_time_remaining))
        #print(cur_solution.y[:,-1])
        assert cur_solution.message == 'A termination event occurred.'

        solution_t = np.concatenate((solution_t, cur_solution.t + time_offset))

        solution_y = np.concatenate((solution_y, cur_solution.y), axis=1)
        init_y_ = cur_solution.y_events[0].reshape((30,)).copy()

        cur_elapsed_time = cur_solution.t_events[0][0]
        crit_time_stamp.append(cur_elapsed_time)
        time_offset += cur_elapsed_time

        cur_Y = init_y_.copy()
        mask_array = np.ones(30)
        mask_array[lco_battery_module.unfailed_list] = 0.0
        temp_roi = ma.masked_array(cur_Y, mask=mask_array,fill_value=0.0)
        failing_cell_index = np.argmax(temp_roi)
        assert temp_roi[failing_cell_index]>=T_IGN
        assert (temp_roi[temp_roi>=T_IGN]).count() == 1
        lco_battery_module.update_module(failing_cell_index)

        for index in range(14):
            if vent_time_remaining[index] > 0:
                vent_time_remaining[index] = max(0, vent_time_remaining[index] - cur_elapsed_time)
            else:
                if index == failing_cell_index:
                    vent_time_remaining[index] = TIME_VENTING

        if init_heating:
            init_heating = False 
 
        count +=1

    cur_solution = solve_ivp(tr_function,
                    [0,300],
                    init_y_,
                    events=stop_func,
                    args=(init_heating,lco_battery_module,vent_time_remaining))
    solution_t = np.concatenate((solution_t, cur_solution.t + time_offset))
    solution_y = np.concatenate((solution_y, cur_solution.y), axis=1)

    print(lco_battery_module.failed_list)
    mass_y = solution_y[15:,:]
    mass_profile = np.sum(mass_y,axis=0)
    print(mass_profile.shape)
    fig, axe = plt.subplots(figsize=(10,10))
    axe.plot(solution_t - crit_time_stamp[0],mass_profile, color='black',linewidth=2)
    axe.set(xlim=(-100,900),xlabel='elapsed time since first cell went into thermal away [s]',
            ylabel='total battery module mass [kg]')
    for each_crit_timestamp in np.cumsum(np.array(crit_time_stamp)):
        axe.axvline(x=each_crit_timestamp-crit_time_stamp[0],linestyle='dashed')
    plt.show()
    fig.savefig('./mass_loss.jpg',dpi=300)
    with np.printoptions(precision=1, suppress=True):
        print(np.array(crit_time_stamp))

    fig2, axe2 = plt.subplots(figsize=(10,10))
    for i in range(14):
        axe2.plot(solution_t - crit_time_stamp[0],solution_y[i,:],label=f'cell {i+1}')
    axe2.plot(solution_t - crit_time_stamp[0],solution_y[14,:],label='Air',linestyle='dashed')
    axe2.set(xlabel='elapsed time since first cell went into thermal runaway [s]',
             ylabel='temperature [K]')
    axe2.legend()
    plt.show()
    fig2.savefig('./temperature.jpg',dpi=300)
main()