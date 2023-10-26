#!/usr/bin/env python3
'''
A set of tools for solving and visualizing the heat equation in 1D.
'''

import numpy as np
import matplotlib.pyplot as plt

solution = np.array([[0., 0., 0., 0., 0.       , 0.       ,
        0., 0., 0., 0.       , 0.       ],
       [0.64     , 0.48     , 0.4      , 0.32     , 0.26     , 0.21     ,
        0.17     , 0.1375   , 0.11125  , 0.09     , 0.0728125],
       [0.96     , 0.8      , 0.64     , 0.52     , 0.42     , 0.34     ,
        0.275    , 0.2225   , 0.18     , 0.145625 , 0.1178125],
       [0.96     , 0.8      , 0.64     , 0.52     , 0.42     , 0.34     ,
        0.275    , 0.2225   , 0.18     , 0.145625 , 0.1178125],
       [0.64     , 0.48     , 0.4      , 0.32     , 0.26     , 0.21     ,
        0.17     , 0.1375   , 0.11125  , 0.09     , 0.0728125],
       [0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
        0.       , 0.       , 0.       , 0.       , 0.       ]])


def sample_init(x):
    '''Simple initial boundary condition function.'''
    return 4*x - 4*x**2


def heat_solve(dt=0.02, dx=0.2, c2=1.0, xmax=1.0, tmax=0.2,
               init=sample_init):
    '''
    Stuff.

    Parameters
    ----------
    dt, dx : float, default=0.02, 0.2
        Time and space step.
    c2 : float, default=1.0
        Thermal diffusivity.
    xmax, tmax : float, default=1.0, 0.2
        Set max values for space and time grids.
    init : scalar or function
        Set initial condition. If a function, should take position
        as an input and return temperature using same units as
        x, temp.


    Returns
    -------
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of time points/y-grid
    temp : numpy 2D array
        Temperature as a function of time and space.

    '''

    # Check stability criterion:
    if (dt > dx**2/(2*c2)):
        raise ValueError('Stability criterion not met: ' +
                         f'dt={dt:6.2f}; dx={dx:06.2f}; c2={c2}')

    # Set constants:
    r = c2 * dt/dx**2

    # Create space and time grids
    x = np.arange(0, xmax+dx, dx)
    t = np.arange(0, tmax+dt, dt)
    # Save number of points.
    M, N = x.size, t.size

    # Create temperature solution array:
    temp = np.zeros([M, N])

    # Set initial and boundary conditions.
    temp[0, :] = 0
    temp[-1, :] = 0

    # Set initial condition.
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init

    # Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1, j] + \
            r*(temp[2:, j]+temp[:-2, j])

    return x, t, temp

