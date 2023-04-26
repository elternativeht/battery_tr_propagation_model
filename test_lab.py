import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class Test_class():
    def __init__(self):
        self.a = 0.2
        self.b = 0.045
        self.c = 0.001
        self.test_list = [50]
        self.flag = [True]

# Helper functions
def flame_radius_front(time_elapsed, alpha, initial_radius, initial_flame_speed):
    if alpha == 0.5:
        return initial_radius * np.exp(initial_flame_speed * time_elapsed / initial_radius)
    else:
        return initial_radius * np.power(1 + (1-2*alpha)*time_elapsed*initial_flame_speed/initial_radius, 1/(1-2*alpha))

def dr_back_dt(timed_elapsed, rb, tau, alpha, initial_radius, initial_flame_speed, dclass: Test_class):
    if rb[0] >= 2 and dclass.flag[0]:
        print('g')
        dclass.test_list[0] = dclass.test_list[0] + timed_elapsed
        print(dclass.test_list[0])
        dclass.flag[0] = False
    if timed_elapsed < dclass.test_list[0]:
        alpha = 0.2
    else:
        alpha = 0.2
    rf = flame_radius_front(timed_elapsed, alpha, initial_radius, initial_flame_speed)
    return (rf - rb)/tau

alpha_cur=0.2
test_class = Test_class()

cur_solution = solve_ivp(dr_back_dt, [0,1000], [0], t_eval=np.arange(0,1001,1),method='RK45',dense_output=True, args=(10.0, alpha_cur, 0.045, 0.001,test_class))



plt.plot(cur_solution.t,flame_radius_front(cur_solution.t,alpha_cur,0.045,0.001)-np.reshape(cur_solution.y,(cur_solution.y.shape[1],)))
#plt.plot(cur_solution.t,np.reshape(cur_solution.y,(cur_solution.y.shape[1],)),label='rb')

plt.show()