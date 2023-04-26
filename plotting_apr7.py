import numpy as np
import matplotlib.pyplot as plt

R0 = 0.001

A0 = 0.25 * np.pi * R0

S0 = np.sqrt(np.pi * A0)

t_range = np.arange(0.0,1000.0,0.1)
Vp0 = 1e-3

alpha_list = [0,0.1,0.2,0.3,0.4,]

A = np.power((Vp0 * S0 * t_range / A0) + 1,2) * np.power(A0,2)


fig,axe = plt.subplots(figsize=(10,10))

for alpha in alpha_list:
    print(alpha)
    A = np.power((Vp0 * S0 * t_range / A0) + 1,1/(0.5-alpha)) * np.power(A0,0.5-alpha)
    axe.plot(t_range,A,label='alpha = ' + str(alpha))
axe.legend()
plt.show()

