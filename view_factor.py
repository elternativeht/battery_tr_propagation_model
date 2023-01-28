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

def B_CC(x,y,eta,xi,z):
    u = x - xi
    v = y - eta
    p = np.sqrt(u*u+z*z)
    q = np.sqrt(v*v+z*z)
    return v*p*np.arctan(v/p) + u*q*np.arctan(u/q) - 0.5*z*z*np.log(u*u+v*v+z*z)

def parallel_calc(X,Y,Z,X_HALF,Y_HALF):
    X_list = [X-X_HALF,X+X_HALF]
    xi_list =[-X_HALF,X_HALF]
    Y_list = [Y-Y_HALF,Y+Y_HALF]
    eta_list = [-Y_HALF,Y_HALF]

    A1 = 2*X_HALF*2*Y_HALF
    sum =0.0
    for i in range(1,3):
        for j in range(1,3):
            for k in range(1,3):
                for l in range(1,3):
                    sum += (np.power(-1,i+j+k+l)*B_CC(X_list[i-1],Y_list[j-1],eta_list[k-1],xi_list[l-1],Z))/(2*np.pi*A1)
    return sum
