'''
Simulation script file of battery module failure (thermal runaway) propagation.
'''
import numpy as np
import numpy.ma as ma
import numpy.typing as npt
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

CELL_MEAN_SPECIFIC_HEAT = 950.0 # J __kg __K

CELL_WIDTH_GAP = 0.03
CELL_WIDTHWISE_CENTER_DIST = CELL_WIDTH_GAP + CELL_WIDTH

CELL_LENGTH_CAP = 0.04
CELL_LENGTHWISE_CENTER_DIST = CELL_LENGTH_CAP + CELL_LENGTH

BASE_TO_CEILING_DIST = 0.35

# Initial values
CELL_INIT_TEMP = 300.0 # Kelvin
CELL_INIT_MASS = 2.01 # kg 

CELL_MASS_LOSS = 1.0 
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
TIME_LOCAL_COMBUSTION = 20.0

PWR_LOCAL_COMBUSTION = 5200.0

REACT_RLSE_ENG_J = 1000 * 1000 * 1.0
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

def tr_func_scope(t: float,
                  y: np.ndarray,
                  init_heating: bool,
                  battery_module: CellModule,
                  vent_time_remaining:np.ndarray,
                  local_heat_time_remaining: np.ndarray,
                  cell_ind: int):
    
    T = y[:15] 
    M = y[15:30]

    radiation_release = SIGMA * np.power(T, 4) * battery_module.radiation_area \
                        * battery_module.emissivity
    radiation_aborption_lst = ((battery_module.rad_matrix.transpose())[cell_ind])\
                        * radiation_release * battery_module.absorptivity[cell_ind]

    # convection heat aborption from air
    cur_ind_conv_absp = battery_module.cell_total_area * H_CONV_HT_COEFF * (T[14] - T)
    cur_ind_conv_absp = cur_ind_conv_absp[cell_ind]

    temp_col = battery_module.cond_matrix[cell_ind].copy()
    temp_col[cell_ind] = 0.0
    cur_ind_cond_absp_lst = thermal_conductivity(T[14]) * battery_module.cell_height * \
        battery_module.cell_length * temp_col * (T - T[cell_ind])

    heating_source = np.zeros(15)
    compt_heating_source = np.zeros(15)
    mass_change_rate = np.zeros(15)
    signed_enthalpy_change = np.zeros(15)
    if init_heating:
        heating_source[0] = HEATING_PWR * 0.8
        heating_source[1] = HEATING_PWR * 0.2
    else:
        vent_gas_mass_array = -1 * gas_venting_rate(TIME_VENTING - vent_time_remaining + t,T)
        for cur_index in range(14):
            if vent_time_remaining[cur_index] > 0.0 and t < vent_time_remaining[cur_index]:
                heating_source[cur_index] = REACT_RLSE_PWR_W
                if vent_gas_mass_array[cur_index] < 0:
                    mass_change_rate[cur_index] = - CELL_MASS_LOSS / TIME_VENTING
                    signed_enthalpy_change[cur_index] = SPECIFIC_HEAT_CP * \
                    vent_gas_mass_array[cur_index] * T[cur_index]
        signed_enthalpy_change[14] = np.sum(SPECIFIC_HEAT_CP* (-1) * vent_gas_mass_array * (T-T[14])[:14])

        for cur_index in range(14):    
            if local_heat_time_remaining[cur_index] > 0.0 and t < local_heat_time_remaining[cur_index]:
                compt_heating_source += battery_module.neighbor_matix[cur_index] * PWR_LOCAL_COMBUSTION
    
    cur_ind_heating_source = heating_source[cell_ind]
    cur_ind_compt_heating_source = compt_heating_source[cell_ind]
    cur_ind_mass_change_rate = mass_change_rate[cell_ind]
    cur_ind_signed_enthalpy_change = signed_enthalpy_change[cell_ind]

    return (radiation_release[cell_ind],
           radiation_aborption_lst,
           cur_ind_conv_absp,
           cur_ind_cond_absp_lst,
           cur_ind_heating_source,
           cur_ind_compt_heating_source,
           cur_ind_mass_change_rate,
           cur_ind_signed_enthalpy_change)

def tr_function(t: np.ndarray,
                y: np.ndarray,
                init_heating: bool,
                battery_module: CellModule,
                vent_time_remaining:np.ndarray,
                local_heat_time_remaining: np.ndarray):
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

    conduction_absorption = thermal_conductivity(T[14]) * battery_module.cell_height * \
                            battery_module.cell_length * np.matmul(battery_module.cond_matrix, T)

    heating_source = np.zeros(15)

    compt_heating_source = np.zeros(15)

    mass_change_rate = np.zeros(15)
    signed_enthalpy_change = np.zeros(15)
    if init_heating:
        heating_source[0] = HEATING_PWR * 0.8
        heating_source[1] = HEATING_PWR * 0.2
    else:
        vent_gas_mass_array = -1 * gas_venting_rate(TIME_VENTING - vent_time_remaining + t,T)
        for cur_index in range(14):
            if vent_time_remaining[cur_index] > 0.0 and t < vent_time_remaining[cur_index]:
                heating_source[cur_index] = REACT_RLSE_PWR_W
                if vent_gas_mass_array[cur_index] < 0:
                    mass_change_rate[cur_index] = - CELL_MASS_LOSS / TIME_VENTING
                    #signed_enthalpy_change[cur_index] = SPECIFIC_HEAT_ARRAY[:14][cur_index] * \
                    #(mass_change_rate[cur_index] - vent_gas_mass_array[cur_index]) * T[cur_index] + \
                    #SPECIFIC_HEAT_CP * vent_gas_mass_array[cur_index] * T[cur_index]
                    signed_enthalpy_change[cur_index] = SPECIFIC_HEAT_CP * \
                    vent_gas_mass_array[cur_index] * T[cur_index]
        signed_enthalpy_change[14] = np.sum(SPECIFIC_HEAT_CP* (-1) * vent_gas_mass_array * (T-T[14])[:14])
        for cur_index in range(14):
            if local_heat_time_remaining[cur_index] > 0.0 and t < local_heat_time_remaining[cur_index]:
                compt_heating_source += battery_module.neighbor_matix[cur_index] * PWR_LOCAL_COMBUSTION

    y_prime[:15] = (radiation_aborption
                   - radiation_release
                   + convection_absorption
                   + conduction_absorption
                   + heating_source
                   + compt_heating_source
                   + signed_enthalpy_change
                   )/ (M*SPECIFIC_HEAT_ARRAY)
    y_prime[15:30] = mass_change_rate
    return y_prime

def stop_func(t: np.ndarray,
              y: np.ndarray,
              init_heating: bool,
              battery_module: CellModule,
              vent_time_remaining:np.ndarray,
              local_heat_time_remaining: np.ndarray):
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
    local_heat_time_remaining = np.zeros(14)

    rad_release_dict = {}
    rad_absp_dict = {}
    conv_absp_dict= {}
    cond_absp_dict = {}
    heat_src_dict = {}
    compt_heat_src_dict = {}
    enthalpy_dict = {}

    for index in range(14):
        rad_release_dict[index] = [0.0]
        rad_absp_dict[index] = [np.zeros(15)]
        conv_absp_dict[index] = [0.0]
        cond_absp_dict[index] = [np.zeros(15)]
        heat_src_dict[index] = [0.0]
        compt_heat_src_dict[index] = [0.0]
        enthalpy_dict[index] = [0.0]

    while count <= 14:
        print(count)
        cur_time_range = time_range if count < 14 else [0, 500]
        cur_solution = solve_ivp(tr_function,
                                 cur_time_range,
                                 init_y_,
                                 events=stop_func,
                                 max_step=0.5,
                                 args=(init_heating,lco_battery_module,vent_time_remaining,local_heat_time_remaining))
        for cell_index in range(14):
            for t_step in range(cur_solution.y.shape[1]):
                cur_t = cur_solution.t[t_step]
                cur_y = cur_solution.y[:,t_step]
                cur_scope_ans = tr_func_scope(cur_t, cur_y, init_heating,lco_battery_module,vent_time_remaining,local_heat_time_remaining,cell_index)
                rad_release_dict[cell_index].append(cur_scope_ans[0])
                rad_absp_dict[cell_index].append(cur_scope_ans[1])
                conv_absp_dict[cell_index].append(cur_scope_ans[2])
                cond_absp_dict[cell_index].append(cur_scope_ans[3])
                heat_src_dict[cell_index].append(cur_scope_ans[4])
                compt_heat_src_dict[cell_index].append(cur_scope_ans[5])
                enthalpy_dict[cell_index].append(cur_scope_ans[7])
        solution_t = np.concatenate((solution_t, cur_solution.t + time_offset))
        solution_y = np.concatenate((solution_y, cur_solution.y), axis=1)
        if count < 14:
            assert cur_solution.message == 'A termination event occurred.'
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
            #print((temp_roi[temp_roi>=T_IGN]).count())
            #assert (temp_roi[temp_roi>=T_IGN]).count() == 1
            lco_battery_module.update_module(failing_cell_index)

            for index in range(14):
                if vent_time_remaining[index] > 0:
                    vent_time_remaining[index] = max(0, vent_time_remaining[index] - cur_elapsed_time)
                    local_heat_time_remaining[index] = max(0, local_heat_time_remaining[index] - cur_elapsed_time)
                else:
                    if index == failing_cell_index:
                        vent_time_remaining[index] = TIME_VENTING
                        local_heat_time_remaining[index] = TIME_LOCAL_COMBUSTION if count < 2 else TIME_LOCAL_COMBUSTION * 0.4
        else:
            fail_list = list(map(lambda x:x+1,lco_battery_module.failed_list))
            print(f'The failing order for the cell within the module is  \
                {str(fail_list).strip("[").strip("]").replace(", "," -> ")}')
            mass_y = solution_y[15:,:]
            mass_profile = np.sum(mass_y,axis=0)
            print(f'Total time step number is: {mass_profile.shape[0]}')
        if init_heating:
            init_heating = False
        count +=1

    for cell_index in range(14):
        rad_release_dict[cell_index] = np.array(rad_release_dict[cell_index])
        rad_absp_dict[cell_index] = np.array(rad_absp_dict[cell_index])
        conv_absp_dict[cell_index] = np.array(conv_absp_dict[cell_index])
        cond_absp_dict[cell_index] = np.array(cond_absp_dict[cell_index])
        heat_src_dict[cell_index] = np.array(heat_src_dict[cell_index])
        compt_heat_src_dict[cell_index] = np.array(compt_heat_src_dict[cell_index] )
        enthalpy_dict[cell_index] = np.array(enthalpy_dict[cell_index])
    
    ln_temp_lst = []
    fig_temp, axe_temp = plt.subplots(figsize=(10,10))
    axe_temp.set(xlabel='Time since first cell went into TR [s]',
            ylabel='Temperature [K]')
    for ind in range(14):
        cur_ln = axe_temp.plot(solution_t - crit_time_stamp[0],solution_y[ind,:],label=f'cell {ind+1}')
        ln_temp_lst.append(cur_ln)
    cur_ln = axe_temp.plot(solution_t - crit_time_stamp[0],solution_y[14,:],label='Air temp',linestyle='dashed')
    ln_temp_lst.append(cur_ln)
    axe_temp.legend()
    plt.show()
    fig_temp.savefig('./all_temperature.jpg',dpi=300)

    fig_mass, axe_mass = plt.subplots(figsize=(10,10))
    axe_mass.plot(solution_t - crit_time_stamp[0],mass_profile, color='black',linewidth=2)
    axe_mass.set(#xlim=(-100,500),
                 xlabel='elapsed time since first cell went into thermal away [s]',
                 ylabel='total battery module mass [kg]')
    for each_crit_timestamp in np.cumsum(np.array(crit_time_stamp)):
        axe_mass.axvline(x=each_crit_timestamp-crit_time_stamp[0],linestyle='dashed')
    plt.show()
    fig_mass.savefig('./mass_loss.jpg',dpi=300)
    with np.printoptions(precision=1, suppress=True):
        print_crit_time_stamp = []
        for element in crit_time_stamp:
            print_crit_time_stamp.append("{:.1f}".format(element))
        print(f"Cell-to-cell TR propagation time is:\n{' -> '.join(print_crit_time_stamp)}")
    
    for cell_index in range(14):
        cur_failing_index = lco_battery_module.failed_list.index(cell_index)
        cum_sum_crt_time = np.cumsum(np.array(crit_time_stamp))
        fig_cur_cell, axe_cur_cell = plt.subplots(figsize=(10,10))
        ln1 = axe_cur_cell.plot(solution_t - cum_sum_crt_time[cur_failing_index],solution_y[cell_index,:],label=f'cell {cell_index+1}')
        ln2 = axe_cur_cell.plot(solution_t - cum_sum_crt_time[cur_failing_index],solution_y[14,:],label='Air',linestyle='solid',linewidth=2)
        axe_cur_cell.set(xlabel=f'time since cell {cell_index+1} went into TR [s]',
                          ylabel='Temperature [K]')
        axe_cur_cell.axvline(x=0.0,color='black',linestyle='dotted')
        axe_cur_cell2 = axe_cur_cell.twinx()
        axe_cur_cell2.set(ylabel='Heat Transfer Power [W]')
        axe_cur_cell2.axhline(y=0.0,color='black')

        ln3 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], conv_absp_dict[cell_index],
                        label=f'cell {cell_index+1} conv',linestyle='dashed')
        ln4 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], rad_release_dict[cell_index],
                        label=f'cell {cell_index+1} rad release',linestyle='dashed')
        ln5 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], np.sum(rad_absp_dict[cell_index],axis=1),
                        label=f'cell {cell_index+1} rad absorption',linestyle='dashed')
        ln6 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], np.sum(cond_absp_dict[cell_index],axis=1),
                        label=f'cell {cell_index+1} conduction absorption', linestyle='dashed')
        #ln7 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], heat_src_dict[cell_index],
        #                label=f'cell {cell_index+1} energy change due to heat sources from TR', linestyle='dashed')
        ln8 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], compt_heat_src_dict[cell_index],
                        label=f'cell {cell_index+1} energy change due to compartment heat sources', linestyle='dashed')
        #ln9 = axe_cur_cell2.plot(solution_t - cum_sum_crt_time[cur_failing_index], enthalpy_dict[cell_index],
        #                label=f'cell {cell_index+1} energy change due to mass losss', linestyle='dashed')
        lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6 + ln8 
        #for i in [2,4]:
        #    cur_ln = axe2.plot(solution_t - crit_time_stamp[0], cond_absp_dict[3][:,i],label=f'cell 4 absorption from {i+1} conduction', linestyle='dashed')
        #    lns += cur_ln
        #cur_ln = axe2.plot(solution_t - crit_time_stamp[0], np.sum(rad_absp_dict[3],axis=1),label='cell 4 absorption sum', linestyle='dashed')
        #lns += cur_ln
        #ln3 = axe2.plot(solution_t - crit_time_stamp[0], rad_release_dict[3],
        #                label='cell 4 rad release',linestyle='dashed')
        #lns = ln1 + ln2 + ln3
        #for j in [2,4,9,10,11]:
        #    cur_ln = axe2.plot(solution_t - crit_time_stamp[0], rad_absp_dict[3][:,j],label=f'cell 4 absorption from {j+1} radation', linestyle='dashed')
        #    lns += cur_ln
        #cur_ln2 = axe2.plot(solution_t - crit_time_stamp[0], rad_absp_dict[3][:,14],label='cell 4 absorption from air radation', linestyle='dashed')
        #lns += cur_ln2
        #ln4 = axe2.plot(solution_t - crit_time_stamp[0], cond_absp_dict[])

        labs = [l.get_label() for l in lns]
        axe_cur_cell.legend(lns, labs, loc=0)
        plt.show()
        fig_cur_cell.savefig(f'./{cell_index+1}-cell-temp-pwr-budget.jpg',dpi=300)
main()