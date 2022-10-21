import numpy as np


def IDM(x1,x2,v1,v2):
    dx = x1-x2
    dv = v2-v1
    alpha = 2
    u = 18
    c = 4
    s0 = 2
    L = 4
    T = 1
    b = 3
    s = s0 + v2*T + v2*dv / (2*np.sqrt(alpha*b))
    #a = alpha * (1-(v2/u)**c - (s / (dx-L))**2)
    if abs(dx - L) > 1e-10:
        a = alpha * (1-(v2/u)**c - (s / (dx-L))**2)
    else:
        a = alpha * (1-(v2/u)**c - (s / (1e-6))**2)
    if a > 3:
        a = 3
    elif a < -3:
        a = -3
    return a


def update_state(x,v,a,dt,vmin,vmax):
    if (v == vmax) & (a > 0):
        a = 0
    elif (v == vmin) & (a < 0):
        a = 0
    x = x+v*dt+0.5*a*dt*dt
    v = v+a*dt
    if v > vmax:
        v = vmax
    elif v < vmin:
        v = vmin
    return [x,v]


def time_interval(x1,x2,v1,v2):
    return x1 - x2   


def ttc(x1,x2,v1,v2):
    #if v1 != v2:
    if abs(v1 - v2) < 1e-8:
        return (x1-x2) / (v2-v1)
    else:
        return (x1-x2) / 0.01
        
        
def value_function_IDM(x, dt, STEP):

    omega = 0.2
    v2 = 20
    vmax = 40
    vmin = 0
    Pb = 0.03
    range_ = x[0]
    range_rate = x[1]
    v1 = v2 + range_rate
    car1 = [range_,v1]
    car2 = [0, v2]
    value_list = []
    for i in range(1, STEP+1):
        a1 = 0
        a2 = IDM(car1[0],car2[0],car1[1],car2[1])
        car1 = update_state(car1[0],car1[1],a1,dt,vmin,vmax)
        car2 = update_state(car2[0],car2[1],a2,dt,vmin,vmax)
        value_list.append(time_interval(car1[0],car2[0],car1[1],car2[1]))
    return min(value_list)
