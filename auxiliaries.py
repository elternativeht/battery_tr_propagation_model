'''
Current module is based on the view factor calculation provided by Isidoro Martinez
in his website: http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf
  
 
'''
import numpy as np
np.set_printoptions(precision=3)

ADJACENT_DICT = {}
for cell_index in range(14):
    cur_location_dict = {}
    if cell_index not in [0, 6, 7, 13]:
        cur_location_dict['row'] = (cell_index - 1, cell_index + 1)
        cur_location_dict['col'] = (cell_index - 7,) if cell_index > 6 else (cell_index + 7,)
        cur_location_dict['diag'] = (cell_index - 8, cell_index - 6) if cell_index > 6 else (cell_index + 6, cell_index + 8)
    else:
        cur_location_dict['row'] = (cell_index + 1,) if cell_index in [0, 7] else (cell_index - 1,)
        cur_location_dict['col'] = (cell_index - 7,) if cell_index > 6 else (cell_index + 7,)
        if cell_index == 0:
            cur_location_dict['diag'] = (8,)
        elif cell_index == 6:
            cur_location_dict['diag'] = (12,)
        elif cell_index == 7:
            cur_location_dict['diag'] = (1,)
        else:
            cur_location_dict['diag'] = (5,)
    ADJACENT_DICT[cell_index] = cur_location_dict

def vectical_calc(W1,W2,H):
    x = W1/H
    y = W2/H
    x1 = np.sqrt(1+x*x)
    y1 = np.sqrt(1+y*y)
    ans1 = np.log(x1*x1*y1*y1/(x1*x1+y1*y1-1))
    ans1 = np.log(x1*x1*y1*y1/(x1*x1+y1*y1-1))
    ans1 += 2*x*(y1*np.arctan(x/y1)-np.arctan(x))
    ans1 += 2*y*(x1*np.arctan(y/x1)-np.arctan(y))
    ans = ans1/(np.pi*x*y)
    return ans

def radvf_calc_2pal(dimension_1: tuple[float], 
                              dimension_2: tuple[float],
                              z_dist: float,
                              dimension_dist: tuple[float]
                              ):
    '''
    The function that calculates the view factor from rectangle 1 to parallel rectangle 2.
    Two rectangle plates are parallel with each other with distance z.
    
    Rectangle 1 is having center located at (x, y, 0); 
    x-direction vertexes located at x = x1 and x2 (x1 < x2);
    y-direction vertexes located at y = y1 and y2 (y1 < y2);
    the whole rectangle 1 lies in the plane z = 0.

    Rectangle 2 is having center located at (psi, eta, z);
    x-direction vertexes located at x = psi1 and psi2 (psi1<psi2);
    y-direction vertexes located at y = eta1 and eta2 (eta1<eta2);
    the whole rectangle 2 lies in the plane z = z.

    Parameters:
    `dimension_1`: a 2-element tuple with plate 1 dimensions [meters].
    `dimension_2`: a 2-element tuple with plate 2 dimensions [meters].
    `z_dist`: plate distance [meters].
    `dimension_dist`: a 2-element tuple with 2-dimension distance between 2 plate centers [meters].

    Returns:

    `F12` unitless view factor of radiation from rectangle 1 to 2.

    Notes:

    It is self-explanatory that F21  =  F12.
    '''

    def b_function(x, y, eta, psi, z):
        u = x - psi
        v = y - eta
        p = np.sqrt(u*u+z*z)
        q = np.sqrt(v*v+z*z)
        return v*p*np.arctan(v/p) + u*q*np.arctan(u/q) - 0.5*z*z*np.log(u*u+v*v+z*z)

    x_list =[-dimension_1[0]/2,dimension_1[0]/2]
    psi_list = [dimension_dist[0]-dimension_2[0]/2,dimension_dist[0]+dimension_2[0]/2]
    y_list = [-dimension_1[1]/2,dimension_1[1]/2]
    eta_list = [dimension_dist[1]-dimension_2[1]/2,dimension_dist[1]+dimension_2[1]/2]

    A1 = dimension_1[0] * dimension_1[1]

    F12 = 0.0

    for i in range(1,3):
        for j in range(1,3):
            for k in range(1,3):
                for l in range(1,3):
                    cur_b = b_function(x_list[i-1],y_list[j-1],eta_list[k-1],psi_list[l-1],z_dist)
                    F12 += (np.power(-1,i+j+k+l) * cur_b) /(2*np.pi*A1)
    return F12

class CellModule(object):
    global ADJACENT_DICT
    def __init__(self, cell_dim: tuple[float], cell_dist: tuple[float]):
        self.cell_length, self.cell_width, self.cell_height = cell_dim
        self.length_gap, self. width_gap = cell_dist

        self.cell_total_area = self.cell_length * self.cell_width + \
            2 * (self.cell_length * self.cell_height + self.cell_width * self.cell_height)

        self.failed_list = []
        self.unfailed_list = list(range(14))

        self.rad_matrix = np.zeros((15,15))

        vf_row = radvf_calc_2pal(dimension_1=(self.cell_length,self.cell_height),
                                 dimension_2=(self.cell_length,self.cell_height),
                                 z_dist=self.width_gap, dimension_dist=(0,0))
        vf_col = radvf_calc_2pal(dimension_1=(self.cell_width,self.cell_height),
                                 dimension_2=(self.cell_width,self.cell_height),
                                 z_dist=self.length_gap, dimension_dist=(0,0))
        vf_diag = radvf_calc_2pal(dimension_1=(self.cell_width,self.cell_height),
                                  dimension_2=(self.cell_width,self.cell_height),
                                  z_dist=self.length_gap, dimension_dist=(self.width_gap + self.cell_width,0))

        self.air_area = 0.0
        for test_id in [1,2]:
            self.air_area += (self.cell_length * self.cell_width
                    + self.cell_length * self.cell_height * (2 - test_id * vf_row)
                    + self.cell_width * self.cell_height * (2 - vf_col - test_id * vf_diag)
                    ) * (4 + (test_id - 1) * 6)

        self.vf_list = [vf_row, vf_col, vf_diag]

        for out_cell_id in range(14):
            cur_sum_area = 0.0
            for keyword, tuple_val in ADJACENT_DICT[out_cell_id].items():
                for in_cell_id in tuple_val:
                    if keyword == 'row':
                       cur_eff_area = self.cell_length * self.cell_height * vf_row
                    elif keyword == 'col':
                        cur_eff_area = self.cell_width * self.cell_height * vf_col
                    elif keyword == 'diag':
                        cur_eff_area = self.cell_width * self.cell_height * vf_diag
                    else:
                        raise ValueError('Key Word Errors from ADJACENT_DICT')
                    cur_sum_area += cur_eff_area
                    self.rad_matrix[out_cell_id][in_cell_id] = cur_eff_area / self.cell_total_area
            self.rad_matrix[out_cell_id][14] = (self.cell_total_area - cur_sum_area)/self.cell_total_area
        self.rad_matrix[14] = np.ones(15) * 1/14.0
        self.rad_matrix[14][14] = 0.0

    def update_module(self, failing_cell_id):
        self.unfailed_list.remove(failing_cell_id)
        self.failed_list.append(failing_cell_id)
        
        for neighboring_cell_id in ADJACENT_DICT[failing_cell_id]['row']:
            total_delta = 0.0
            cur_z_dist = self.width_gap if neighboring_cell_id in self.unfailed_list else self.width_gap/2
            new_z_dist = self.width_gap/2 if neighboring_cell_id in self.unfailed_list else 0.001
            cur_vf_row =  radvf_calc_2pal(dimension_1=(self.cell_length,self.cell_height),
                                dimension_2=(self.cell_length,self.cell_height),
                                z_dist=cur_z_dist, dimension_dist=(0,0))
            new_vf_row  = radvf_calc_2pal(dimension_1=(self.cell_length,self.cell_height),
                                dimension_2=(self.cell_length,self.cell_height),
                                z_dist=new_z_dist, dimension_dist=(0,0))
            cur_delta = self.cell_length * self.cell_height * (new_vf_row - cur_vf_row) \
            / self.cell_total_area

            total_delta += cur_delta
            self.rad_matrix[failing_cell_id][neighboring_cell_id] += cur_delta
            self.rad_matrix[neighboring_cell_id][failing_cell_id] += cur_delta
            self.rad_matrix[neighboring_cell_id][14] -= cur_delta
            
        self.rad_matrix[failing_cell_id][14] -= total_delta

# test_result = radvf_calc_2parallel_rect(dimension_1 = (0.08,0.12), 
#                                        dimension_2 = (0.08,0.12),
#                                        z_dist = 0.12,
#                                        dimension_dist=(0.22,0)
#                                       )
# print(test_result)

def radiation_matrix_calc(cell_dimension: tuple[float], cell_gap: tuple[float]):
    cell_length, cell_width, cell_height = cell_dimension
    length_gap, width_gap = cell_gap

    cell_total_eff_area = cell_length * cell_width + \
                          2 * (cell_length * cell_height + cell_width * cell_height)
    
    vf_rowwise_cell = radvf_calc_2pal(dimension_1=(cell_length,cell_height),
                                                dimension_2=(cell_length,cell_height),
                                                z_dist=width_gap, dimension_dist=(0,0))
    vf_colwise_cell = radvf_calc_2pal(dimension_1=(cell_width,cell_height),
                                                dimension_2=(cell_width,cell_height),
                                                z_dist=length_gap, dimension_dist=(0,0))
    vf_diagonal_cell = radvf_calc_2pal(dimension_1=(cell_width,cell_height),
                                                dimension_2=(cell_width,cell_height),
                                                z_dist=length_gap, dimension_dist=(width_gap + cell_width,0))
    cell_eff_area_rowwise = cell_length * cell_height * vf_rowwise_cell
    cell_eff_area_colwise = cell_width * cell_height * vf_colwise_cell
    cell_eff_area_diagonal = cell_width * cell_height * vf_diagonal_cell

    corner_cell_eff_area_air = cell_total_eff_area - cell_eff_area_rowwise \
                               - cell_eff_area_colwise - cell_eff_area_diagonal
    middle_cell_eff_area_air = cell_total_eff_area - 2 * cell_eff_area_rowwise \
                               - cell_eff_area_colwise - 2 * cell_eff_area_diagonal
    
    corner_cell_ratio_array = np.array([cell_eff_area_rowwise, cell_eff_area_colwise, 
                                        cell_eff_area_diagonal, corner_cell_eff_area_air])
    middle_cell_ratio_array = np.array([cell_eff_area_rowwise, cell_eff_area_colwise, 
                                        cell_eff_area_diagonal, middle_cell_eff_area_air])
    corner_cell_ratio_array = corner_cell_ratio_array/cell_total_eff_area
    middle_cell_ratio_array = middle_cell_ratio_array/cell_total_eff_area
    rad_matrix = np.zeros((15,15))
    for i in range(15):
        if i+1 not in [1,7,8,14,15]:
            rad_matrix[i][i-1] = rad_matrix[i][i+1] = middle_cell_ratio_array[0]
            if i>=7:
                rad_matrix[i][i-7]  = middle_cell_ratio_array[1]
                rad_matrix[i][i-8]  = rad_matrix[i][i-6] = middle_cell_ratio_array[2]
            else:
                rad_matrix[i][i+7]  = middle_cell_ratio_array[1]
                rad_matrix[i][i+8]  = rad_matrix[i][i+6] = middle_cell_ratio_array[2]
            rad_matrix[i][14] = middle_cell_ratio_array[3]
        else:
            if i == 0:
                rad_matrix[i][1] = corner_cell_ratio_array[0]
                rad_matrix[i][7] = corner_cell_ratio_array[1]
                rad_matrix[i][8] = corner_cell_ratio_array[2]
                rad_matrix[i][14] = corner_cell_ratio_array[3]
            elif i == 6:
                rad_matrix[i][5] = corner_cell_ratio_array[0]
                rad_matrix[i][13] = corner_cell_ratio_array[1]
                rad_matrix[i][12] = corner_cell_ratio_array[2]
                rad_matrix[i][14] = corner_cell_ratio_array[3]
            elif i == 7:
                rad_matrix[i][8] = corner_cell_ratio_array[0]
                rad_matrix[i][0] = corner_cell_ratio_array[1]
                rad_matrix[i][1] = corner_cell_ratio_array[2]
                rad_matrix[i][14] = corner_cell_ratio_array[3]
            elif i == 13:
                rad_matrix[i][12] = corner_cell_ratio_array[0]
                rad_matrix[i][6] = corner_cell_ratio_array[1]
                rad_matrix[i][5] = corner_cell_ratio_array[2]
                rad_matrix[i][14] = corner_cell_ratio_array[3]
            else:
                rad_matrix[i] = np.ones(15) * 1/14.0
                rad_matrix[i][-1] = 0.0
    return rad_matrix

def mock_unittest():
    test_module = CellModule(cell_dim=(0.173,0.045,0.125),cell_dist=(0.08,0.03))
    print(test_module.rad_matrix)
    print('------')
    test_module.update_module(failing_cell_id=0)
    print(test_module.rad_matrix)
    print('------')
    test_module.update_module(failing_cell_id=1)
    print(test_module.rad_matrix)
    print('------')
    test_module.update_module(failing_cell_id=2)
    print(test_module.rad_matrix)
def mock_unittest_air():
    test_module = CellModule(cell_dim=(0.173,0.045,0.125),cell_dist=(0.08,0.03))
mock_unittest_air()