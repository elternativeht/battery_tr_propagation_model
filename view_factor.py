'''
# Current module is based on the view factor calculation provided by Isidoro Martinez
# in his website: http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf
#  
# 
'''
import numpy as np


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



def radvf_calc_2parallel_rect(dimension_1: tuple[float], 
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

    def B_function(x, y, eta, psi, z):
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

    F12 =0.0

    for i in range(1,3):
        for j in range(1,3):
            for k in range(1,3):
                for l in range(1,3):
                    cur_B = B_function(x_list[i-1],y_list[j-1],eta_list[k-1],psi_list[l-1],z_dist)
                    F12 += (np.power(-1,i+j+k+l) * cur_B) /(2*np.pi*A1)
    return F12

# test_result = radvf_calc_2parallel_rect(dimension_1 = (0.08,0.12), 
#                                        dimension_2 = (0.08,0.12),
#                                        z_dist = 0.12,
#                                        dimension_dist=(0.22,0)
#                                       )
# print(test_result)
