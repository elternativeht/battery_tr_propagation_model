
from dataclasses import dataclass,field
import view_factor
import numpy as np
import numpy.ma as ma
from scipy.integrate import solve_ivp

# All units are in SI-MKG unless explicitly stated; temperature unit is Kelvin


# Constants:

SIGMA = 5.67e-8
T_IGN = 180.0 + 273.15

# Setup Constants:

CELL_LENGTH = 0.2 # meters
CELL_WIDTH = 0.05 # meters
CELL_HEIGHT = 0.15 # meters

CELL_SURFACE_AREA = CELL_LENGTH * CELL_WIDTH + CELL_LENGTH*CELL_HEIGHT*2 + CELL_WIDTH*CELL_HEIGHT*2

AIR_AREA = 5678e-4

CELL_MEAN_SPECIFIC_HEAT = 1000.0 # J __kg __K

CELL_WIDTHWISE_CENTER_DIST = 0.07 # meters
CELL_WIDTH_GAP = CELL_WIDTHWISE_CENTER_DIST - CELL_WIDTH
CELL_LENGTHWISE_CENTER_DIST = 0.3 # meters
CELL_LENGTH_CAP = CELL_LENGTHWISE_CENTER_DIST - CELL_LENGTH

CELL_TO_CEILING_DIST = 0.2 # meters

RADIATION_CONST = [0.2824, 0.0318, 0.0070]

RAD_PERCENTAGE_MATRIX = np.zeros((15,15))
for i in range(15):
    if i+1 not in [1,7,8,14,15]:
        RAD_PERCENTAGE_MATRIX[i][i-1] = RAD_PERCENTAGE_MATRIX[i][i+1] = RADIATION_CONST[0]
        if i>=7:
            RAD_PERCENTAGE_MATRIX[i][i-7]  = RADIATION_CONST[1]
            RAD_PERCENTAGE_MATRIX[i][i-8]  = RAD_PERCENTAGE_MATRIX[i][i-6] = RADIATION_CONST[2]
        else:
            RAD_PERCENTAGE_MATRIX[i][i+7]  = RADIATION_CONST[1]
            RAD_PERCENTAGE_MATRIX[i][i+8]  = RAD_PERCENTAGE_MATRIX[i][i+6] = RADIATION_CONST[2]
        RAD_PERCENTAGE_MATRIX[i][14] = 1.0 - RADIATION_CONST[0]*2 - RADIATION_CONST[1] - RADIATION_CONST[2]*2
    else:
        if i == 0:
            RAD_PERCENTAGE_MATRIX[i] = [0.0000, 0.2824, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0318, 0.0070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6788]
        elif i == 6:
            RAD_PERCENTAGE_MATRIX[i] = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2824, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0070, 0.0318, 0.6788]
        elif i == 7:
            RAD_PERCENTAGE_MATRIX[i] = [0.0318, 0.0070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2824, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6788]
        elif i == 13:
            RAD_PERCENTAGE_MATRIX[i] = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0070, 0.0318, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0070, 0.2824, 0.6788]
        else:
            RAD_PERCENTAGE_MATRIX[i] = np.ones(15) * 1/14.0
            RAD_PERCENTAGE_MATRIX[i][-1] = 0.0

# Initial values

cell_init_temp = 300.0 # Kelvin
cell_init_mass = 2.5 # kg 
cell_terminal_temp = 420.0
# external heat sources

heating_pad_power = 500.0 # Watts

@dataclass
class BatteryCell:
    '''class that contains the battery cell properties'''
    #coordinates
    row_index: int
    col_index: int

    # geometries
    length: float
    width: float
    height: float

    #thermodynamic properties
    temperature: float
    mass: float
    specific_heat: float
    emissivity: float

    # thermal runaway mass loss
    total_mass_loss: float
    mass_loss_rate: float
    time_dur_mass_loss: float = field(init=False)

init_mass_array = np.ones(15) * cell_init_mass
init_mass_array[14] = 101325 * 0.16415/287/300.0
init_temp_array = np.ones(15) * cell_init_temp
specific_heat_array = np.ones(15)*CELL_MEAN_SPECIFIC_HEAT
specific_heat_array[14] = 5/2 * 8.314 / 0.029
specific_heat_p = 7/2*8.314/0.029
surface_area_array_rad = np.ones(15) * CELL_SURFACE_AREA
surface_area_array_rad[14] = AIR_AREA
hm_array = np.ones(15)*10.0 # dummy values
emissivity_array = np.ones(15) 
emissivity_array[14] = 0.8

total_gas_release = 1.0
gas_venting_rate = 0.08
time_venting = total_gas_release/gas_venting_rate
exothermic_energy_release = 4000 * 1000
exothermic_reaction_release_pwr = exothermic_energy_release / time_venting

def function0(t, Y, vent_time_remaining, init_heating, unfailed_cell_index_list):
    Yprime = np.zeros(30)

    # first 15 variables: the temperature 1 to 15 (cell 1 to 14 plus air)
    T = Y[:15] 
    # last 15 variables: the mass 1 to 15 (cell 1 to 14 plus air)
    M = Y[15:30]

    # radiation release: emissivity * sigma * radiation area * T^4
    radiation_release = SIGMA*np.power(T, 4)* surface_area_array_rad * emissivity_array
    # RAD_MATRX(i,j): percentage of energy going to j of the total emitted from i
    # A transpose makes the RAD_MATRIX_T(i,j) percentage of energy going to i of the total energy going from j
    # total energy going into i from other radiation sources (assuming absorpivity = emissvity)
    radiation_aborption = np.matmul(RAD_PERCENTAGE_MATRIX.transpose(), radiation_release) * emissivity_array
    # convection heat aborption from air 
    convection_aborption = CELL_SURFACE_AREA*hm_array*(T[14] - T)
    negative_conv = np.sum(convection_aborption) * (-1)
    convection_aborption[14] = negative_conv

    heating_source = np.zeros(15)
    mass_change_rate = np.zeros(15)
    if init_heating: 
        heating_source[0] = heating_pad_power
    else:
        for cur_index in range(14):
            if vent_time_remaining[cur_index]>0.0 and t<vent_time_remaining[cur_index]:
                heating_source[cur_index] = exothermic_reaction_release_pwr 
                mass_change_rate[cur_index] = -1* gas_venting_rate 

    signed_enthalpy_change = specific_heat_p*mass_change_rate*T
    signed_enthalpy_change[14] = np.sum(specific_heat_p*(-1)*mass_change_rate*(T-T[14]))

    Yprime[:15] = (radiation_aborption - radiation_release 
                   + convection_aborption + heating_source
                   + signed_enthalpy_change)/ (M*specific_heat_array)

    Yprime[15:30] = mass_change_rate

    return Yprime


def stop_func(t, Y, vent_time_remaining, init_heating, unfailed_cell_index_list):
    T = Y[:14]
    return np.prod(T[unfailed_cell_index_list]<T_IGN)
stop_func.terminal = True
stop_func.direction = -1

def main():
    
    unfail_cell_list = list(range(14))
    failed_cell_list = []
    crit_time_stamp = []

    trange = [0,1500]
    
    init_Y = np.ones(30)*cell_init_temp
    init_Y[15:30] = init_mass_array

    solution_t = np.array([0])
    solution_y = init_Y.copy()
    solution_y = np.reshape(solution_y,(30,1))

    time_offset = 0.0
    count = 0
    init_heating = True
    vent_time_remaining = np.zeros(14)
    while count <=13:
        print(count)
        cur_solution = solve_ivp(function0,trange,init_Y,events=stop_func,args=(vent_time_remaining,init_heating,unfail_cell_list))
        print(cur_solution.y[:,-1])
        assert cur_solution.message == 'A termination event occurred.'

        solution_t = np.concatenate((solution_t, cur_solution.t + time_offset))

        solution_y = np.concatenate((solution_y, cur_solution.y), axis=1)
        init_Y = cur_solution.y_events[0].reshape((30,)).copy()

        cur_elapsed_time = cur_solution.t_events[0][0]
        crit_time_stamp.append(cur_elapsed_time)
        time_offset += cur_elapsed_time
        
        cur_Y = init_Y.copy()
        mask_array = np.ones(30)
        mask_array[unfail_cell_list] = 0.0
        temp_roi = ma.masked_array(cur_Y, mask=mask_array,fill_value=0.0)
        failing_cell_index = np.argmax(temp_roi)
        assert temp_roi[failing_cell_index]>=T_IGN
        assert (temp_roi[temp_roi>=T_IGN]).count() == 1
        unfail_cell_list.remove(failing_cell_index)
        failed_cell_list.append(failing_cell_index)

        for index in range(14):
            if vent_time_remaining[index] > 0:
                vent_time_remaining[index] = max(0, vent_time_remaining[index] - cur_elapsed_time)
            else:
                if index == failing_cell_index:
                    vent_time_remaining[index] = time_venting
        
        if init_heating:
            init_heating = False           
        
        count +=1
    print(failed_cell_list)
    print(crit_time_stamp)
main()