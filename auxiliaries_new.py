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


class Compartment(object):
    def __init__(self,dimension:tuple[float, float, float],spacing: tuple[float, float], gap: tuple[float,float]):
        # dimension: [Length, Width, Height]
        # spacing: [north-south spacing, east-west spacing]
        # gap: [north-south gap, east-west gap]
        self.length, self.width, self.height = dimension
        self.ns_spacing, self.ew_spacing = spacing
        self.ns_gap, self.ew_gap = gap

        compt_length = self.width * 7 + self.ns_gap * 6 + self.ns_spacing * 2
        compt_width = self.length * 2 + self.ew_gap * 1 + self.ew_spacing * 2
        self.cell_surface_area = self.length * self.width + self.length * self.height * 2 + self.width * self.height * 2
        center_row_ind = self.ew_spacing + self.length/2 + np.array(np.arange(14)>=7,dtype=int) * (self.length + self.ew_gap)

        center_col_ind = self.ns_spacing + self.width/2 + np.array(np.arange(14) % 7, dtype=int) * (self.width + self.ns_gap)

        self.min_dist = np.zeros((14, 14), dtype=np.float64)
        self.max_dist = np.zeros((14, 14), dtype=np.float64)
        self.center_to_edge_dist = np.zeros((14,),dtype=np.float64)
        self.neighbor = np.zeros((14,14),dtype=int)


        self.status = np.ones((14,),dtype=int) # np array storing the order of battery cell failing; 1: unfailed; 0: failed
        self.preheat_status = np.zeros((14,),dtype=int)
        self.preheat_time = np.zeros((14,))
        self.preheat_list = []
        self.failure_time = np.zeros((14,)) # np array storing the failure timestamp
        self.failure_order = []  # list saving the failure order
        self.compt_rad_matrix = np.zeros((14, 14))
        self.compt_conduction_matrix = np.zeros((14, 14))
    
        self.flame_radius = np.zeros((14,))

        for i in range(14):
            row_dist = abs(center_row_ind[i] - compt_width) if i < 7 else center_row_ind[i]
            col_dist = abs(center_col_ind[i] - compt_length) if (i % 7)<=3 else center_col_ind[i]
            self.center_to_edge_dist[i] = np.sqrt(row_dist**2 + col_dist**2)

        for i in range(14):
            for j in range(i+1, 14):
                if ((i<7) == (j<7)):
                    if j-i==1:
                        view_factor = radvf_calc_2pal(dimension_1=(self.length,self.height),
                                                dimension_2=(self.length,self.height),
                                                z_dist=self.ns_gap, dimension_dist=(0,0))
                        self.compt_rad_matrix[i][j] = self.compt_rad_matrix[j][i] = view_factor * \
                        self.length * self.height / self.cell_surface_area
                        self.neighbor[i][j] = self.neighbor[j][i] = 1
            
                    self.min_dist[i][j] = self.min_dist[j][i] = \
                    np.abs(center_col_ind[i] - center_col_ind[j]) - self.width/2
                    self.max_dist[i][j] = self.max_dist[j][i] = \
                    np.sqrt((np.abs(center_col_ind[i] - center_col_ind[j]) + self.width/2)**2 + (self.length/2)**2)

                elif (i % 7) == (j % 7):
                    view_factor = radvf_calc_2pal(dimension_1=(self.width,self.height),
                                            dimension_2=(self.width,self.height),
                                            z_dist=self.ew_gap, dimension_dist=(0,0))
                    self.compt_rad_matrix[i][j] = self.compt_rad_matrix[j][i] = view_factor * \
                    self.width * self.height / self.cell_surface_area
                    
                    self.neighbor[i][j] = self.neighbor[j][i] = 1

                    self.min_dist[i][j] = self.min_dist[j][i] = \
                    np.abs(center_row_ind[i] - center_row_ind[j]) - self.length/2
                    self.max_dist[i][j] = self.max_dist[j][i] = \
                    np.sqrt((np.abs(center_row_ind[i] - center_row_ind[j]) + self.length/2)**2 + (self.width/2)**2)
                else:
                    self.min_dist[i][j] = self.min_dist[j][i] = \
                    np.sqrt((np.abs(center_col_ind[i] - center_col_ind[j]) - self.width/2)**2
                    + (np.abs(center_row_ind[i] - center_row_ind[j]) - self.length/2)**2)
                    self.max_dist[i][j] = self.max_dist[j][i] = \
                    np.sqrt((np.abs(center_col_ind[i] - center_col_ind[j]) + self.width/2)**2
                    + (np.abs(center_row_ind[i] - center_row_ind[j]) + self.length/2)**2)
                    
                    if abs((i % 7) - (j % 7)) == 1:
                        self.neighbor[i][j] = self.neighbor[j][i] = 1
                        view_factor = radvf_calc_2pal(dimension_1=(self.width,self.height),
                                             dimension_2=(self.width,self.height),
                        z_dist=self.ew_gap, dimension_dist=(self.ns_gap + self.width,0))
                        self.compt_rad_matrix[i][j] = self.compt_rad_matrix[j][i] = view_factor * \
                            self.width * self.height / self.cell_surface_area
    def update_compartment(self, time_,failing_cell):
        self.status[failing_cell] = 0
        self.failure_time[failing_cell] = time_
        self.failure_order.append(failing_cell)
        for j in range(14):
            if j == failing_cell:
                continue
            if ((failing_cell<7) == (j<7)) and (abs(j-failing_cell)==1):
                self.compt_rad_matrix[failing_cell][j] = self.compt_rad_matrix[j][failing_cell] = \
                self.length * self.height / self.cell_surface_area 
                if self.status[j] == 1:
                    self.compt_conduction_matrix[failing_cell][j] = self.compt_conduction_matrix[j][failing_cell] = \
                    self.length * self.height / 2 / 1e-3
                else:
                    self.compt_conduction_matrix[failing_cell][j] = self.compt_conduction_matrix[j][failing_cell] = \
                    self.length * self.height / 1e-3
    def update_preheat(self, time_, preheat_cell):
        self.preheat_status[preheat_cell] = 1
        self.preheat_time[preheat_cell] = time_
        self.preheat_list.append(preheat_cell)



        

                






