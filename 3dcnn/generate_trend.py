import numpy as np


def generate_trend(length, order, min, max, offset):

    if (order==1):   # linear
        tend = np.linspace(min, max, length)

    elif (order==2): # quadratic
        if (offset==0):
            tend = np.linspace(0, 1, length)
            tend = tend*tend
            tend = tend-min
            tend = max*tend/np.max(tend)

        else:
            tend = tend = np.linspace(-0.5, 0.5, length)
            tend = tend*tend
            tend = tend-min
            tend = 0.5*max*tend/np.max(tend)

    elif (order==3): # cubic
        if (offset==0):
            tend = np.linspace(0, 1, length)
            tend = tend*tend*tend
            tend = tend-min
            tend = max*tend/np.max(tend)

        else:
            tend = tend = np.linspace(-0.5, 0.5, length)
            tend = tend*tend*tend
            tend = tend-min
            tend = 0.5*max*tend/np.max(tend)
    return tend
