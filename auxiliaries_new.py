import numpy as np
np.set_printoptions(precision=3)

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

    The calculation is based on a document of view factor calculation based on 
    [Isidoro Martinez](http://imartinez.etsiae.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf).
    Chapter "Rectangle to rectangle".

    '''

    def b_function(x, y, eta, psi, z):
        '''
        A function trying to calculate B variable in Isidoro Martinez reference of calculating view factor.
        B variable is defined in section "Rectangle to rectangle". 
        '''
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


class Compartment(object):
    '''
    Object abstraction of battery module compartment. 
    
    Class instance contains geometric information for conduction, radiation and flame propagation
    calculations.

    Attributes:
    - length: `float`; cell length [m]
    - width: `float`; cell width [m]
    - height: `float`; cell height [m]

    - ns_spacing: `float`; north-south direction spacing between boundary cells and compartment side face [m]
    - ns_gap: `float`; north-south direction gap between two cells [m]
    - ew_spacing: `float`; east-west direction spacing between boundary cells and compartment side face [m]
    - ew_gap: `float`; east-west direction gap between two cells [m]

    - cell_surface_area: `float`; cell surface area exposed to heat transfer; equal to the total surface area 
      minus the bottom surface area (not exposed to heat transfer) [m2]
    - min_dist: two-dim `np.ndarray`; a symmetric matrix containing the minimum distances of one cell from another cell's center. [m]
    - max_dist: two-dim `np.ndarray`; a symmetric matrix containing the maximum distances of one cell from another cell's center. [m]
    - center_to_edge_dist: one-dim `np.ndarray`; an array containing the maximum distance from current cell
      center to the edge of the compartment
    - neighbor: two-dim `np.ndarray`; a symmetric matrix recording the neighbors of the certain cell

    - status: a one-dim `np.ndarray` containing current cell status; 1 for unfailed and 0 for failed
    - preheat_status: a one-dim `np.ndarray` containing current cell preheating status. 
        - preheating: implemented API trying to simulate additional heat source into cells before thermal runaway
        - value: 1 indicates the corresponding cell has gone into preheating status; 0 otherwise
    - preheat_time: a one-dim `np.ndarray` containing the timestamp of cell starting preheat.
    - preheat_list: a python list containing the order of cells going into preheat
    - failure_time: a one-dim `np.ndarray` containing the timestamp of each cell going into thermal runaway
    - failure_order: a python list containing the order of cells going into thermal runaway
    - compt_rad_matrix: a two-dim `np.ndarray` (matrix) containing radiation heat transfer rate relevant data 
        - symmetric: Matrix[i, j] = Matrix[j, i]
        - the element Matrix[i, j] contains the percentage of total radiating heat transfer rate from cell i going 
        into cell j
        - note: Matrix[i, j] will be updated if i and j are neighbors along the compartment length side
    - compt_cond_matrix: a two-dim `np.ndarray` (matrix) containing the conduction relevant coefficients
        - symmetric: Matrix[i, j] = Matrix[j, i]
        - the element Matrix[i, j] = conductivity * conduction area / distance (assuming to be 1e-3 m)
        - Scenarios: 
            - none of two adjacent cells along the compartment length direction goes into thermal runaway: 
            no conduction
            - one of the two adjacent cells goes into thermal runaway: conduction takes place; distance becomes 1e-3,
            contact area becomes half of the large side area
            - both of the two adjacent cells go into thermal runaway: conduction takes place: distance 1e-3,
            contact area becomes full large side area
    - flame_radius: a one-dim `np.ndarray` containing current flame radius [**NOT USED NOW**]
    
    '''
    def __init__(self,dimension:tuple[float, float, float],spacing: tuple[float, float], gap: tuple[float,float]):
        # Input of the function:
        # dimension: [Length, Width, Height]
        # spacing: [north-south spacing, east-west spacing]
        # gap: [north-south gap, east-west gap]
        self.length, self.width, self.height = dimension
        self.ns_spacing, self.ew_spacing = spacing
        self.ns_gap, self.ew_gap = gap
        # compartment length direction is parallel with the cell width direction
        compt_length = self.width * 7 + self.ns_gap * 6 + self.ns_spacing * 2 
        # compartment width direction is parallel with cell length direction
        compt_width = self.length * 2 + self.ew_gap * 1 + self.ew_spacing * 2
        # excluding the bottom area where no heat transfer is assumed to take place
        self.cell_surface_area = self.length * self.width + self.length * self.height * 2 + self.width * self.height * 2
        # indexing the cell. The northeast corner cell is indexed 0. The southeast cell 6; northwest 7; southwest 13
        # NE########################SE
        # # 0  1  2  3   4   5   6  #
        # #                         #
        # # 7  8  9  10  11  12  13 #
        # NW########################SW
        #
        # The compartment northeast corner was treated as origin. The coordinates of centers of each cell are calculated
        # 
        # calculate each cell's row direction (east-west direction) coordinates
        center_row_ind = self.ew_spacing + self.length/2 + np.array(np.arange(14)>=7,dtype=int) * (self.length + self.ew_gap)
        # calculate each cell's column direction (north-west direction) coordinates
        center_col_ind = self.ns_spacing + self.width/2 + np.array(np.arange(14) % 7, dtype=int) * (self.width + self.ns_gap)
        #initiate the attributes
        self.min_dist = np.zeros((14, 14), dtype=np.float64)
        self.max_dist = np.zeros((14, 14), dtype=np.float64)
        self.center_to_edge_dist = np.zeros((14,),dtype=np.float64)
        self.neighbor = np.zeros((14,14),dtype=int)
        self.status = np.ones((14,),dtype=int) # np array storing the status of battery cell failing; 1: unfailed; 0: failed
        self.preheat_status = np.zeros((14,),dtype=int) # np array storing status of preheating for each cell: 1: yes 0: no
        self.preheat_time = np.zeros((14,)) #np array storing the preheat starting timestamp
        self.preheat_list = [] # list storing preheating starting order
        self.failure_time = np.zeros((14,)) # np array storing the failure timestamp
        self.failure_order = []  # list saving the failure order
        self.compt_rad_matrix = np.zeros((14, 14))
        self.compt_conduction_matrix = np.zeros((14, 14))
    
        self.flame_radius = np.zeros((14,))
        #calculate for each cell the maximum distance from center to the compartment edge
        for i in range(14):
            row_dist = abs(center_row_ind[i] - compt_width) if i < 7 else center_row_ind[i]
            col_dist = abs(center_col_ind[i] - compt_length) if (i % 7)<=3 else center_col_ind[i]
            self.center_to_edge_dist[i] = np.sqrt(row_dist**2 + col_dist**2)
        # calculate the view factor between cell surfaces
        for i in range(14):
            for j in range(i+1, 14):
                if ((i<7) == (j<7)): # if they are at the same row (0-6 or 7-13)
                    if j-i==1: # adjacent 
                        view_factor = radvf_calc_2pal(dimension_1=(self.length,self.height),
                                                dimension_2=(self.length,self.height),
                                                z_dist=self.ns_gap, dimension_dist=(0,0))
                        self.compt_rad_matrix[i][j] = self.compt_rad_matrix[j][i] = view_factor * \
                        self.length * self.height / self.cell_surface_area
                        self.neighbor[i][j] = self.neighbor[j][i] = 1 # neighbor boolean set to 1
                    # at the same row; min_dist and max_dist calculated
                    self.min_dist[i][j] = self.min_dist[j][i] = \
                    np.abs(center_col_ind[i] - center_col_ind[j]) - self.width/2
                    self.max_dist[i][j] = self.max_dist[j][i] = \
                    np.sqrt((np.abs(center_col_ind[i] - center_col_ind[j]) + self.width/2)**2 + (self.length/2)**2)

                elif (i % 7) == (j % 7): # same column (0 and 7; 1 and 8, etc)
                    view_factor = radvf_calc_2pal(dimension_1=(self.width,self.height),
                                            dimension_2=(self.width,self.height),
                                            z_dist=self.ew_gap, dimension_dist=(0,0))
                    self.compt_rad_matrix[i][j] = self.compt_rad_matrix[j][i] = view_factor * \
                    self.width * self.height / self.cell_surface_area
                    
                    self.neighbor[i][j] = self.neighbor[j][i] = 1 # neighbor

                    self.min_dist[i][j] = self.min_dist[j][i] = \
                    np.abs(center_row_ind[i] - center_row_ind[j]) - self.length/2
                    self.max_dist[i][j] = self.max_dist[j][i] = \
                    np.sqrt((np.abs(center_row_ind[i] - center_row_ind[j]) + self.length/2)**2 + (self.width/2)**2)
                else: # neither same column nor same row
                    self.min_dist[i][j] = self.min_dist[j][i] = \
                    np.sqrt((np.abs(center_col_ind[i] - center_col_ind[j]) - self.width/2)**2
                    + (np.abs(center_row_ind[i] - center_row_ind[j]) - self.length/2)**2)
                    self.max_dist[i][j] = self.max_dist[j][i] = \
                    np.sqrt((np.abs(center_col_ind[i] - center_col_ind[j]) + self.width/2)**2
                    + (np.abs(center_row_ind[i] - center_row_ind[j]) + self.length/2)**2)
                    # diagonally neighbor 
                    if abs((i % 7) - (j % 7)) == 1:
                        self.neighbor[i][j] = self.neighbor[j][i] = 1
                        view_factor = radvf_calc_2pal(dimension_1=(self.width,self.height),
                                             dimension_2=(self.width,self.height),
                        z_dist=self.ew_gap, dimension_dist=(self.ns_gap + self.width,0))
                        self.compt_rad_matrix[i][j] = self.compt_rad_matrix[j][i] = view_factor * \
                            self.width * self.height / self.cell_surface_area
    def update_compartment(self, time_,failing_cell):
        '''
        Reflects the change in geometry and hence radiation & conduction parameters due to
        thermal runaway of a cell.

        Input parameters:
        - time_: the current timestamp, i.e. the timestamp when a cell goes into thermal runaway
        - failing_cell: the index of the cell that fails
        '''
        self.status[failing_cell] = 0 # change the status to fail
        self.failure_time[failing_cell] = time_ # change the failing time to current timestamp
        self.failure_order.append(failing_cell)

        # updating geometric and hence radiation and conduction parameters with other cells
        for j in range(14):
            if j == failing_cell:
                continue
            if ((failing_cell<7) == (j<7)) and (abs(j-failing_cell)==1): # same row and neighboring 
                self.compt_rad_matrix[failing_cell][j] = self.compt_rad_matrix[j][failing_cell] = \
                self.length * self.height / self.cell_surface_area 
                if self.status[j] == 1: # the other cell is not failed
                    self.compt_conduction_matrix[failing_cell][j] = self.compt_conduction_matrix[j][failing_cell] = \
                    self.length * self.height / 2 / 1e-3
                else: # the other cell has failed
                    self.compt_conduction_matrix[failing_cell][j] = self.compt_conduction_matrix[j][failing_cell] = \
                    self.length * self.height / 1e-3
    def update_preheat(self, time_, preheat_cell):
        '''
        Update the preheat parameters [**NOT USED**]
        '''
        self.preheat_status[preheat_cell] = 1
        self.preheat_time[preheat_cell] = time_
        self.preheat_list.append(preheat_cell)



        

                






