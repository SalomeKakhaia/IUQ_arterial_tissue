#!/usr/bin/env python
# coding: utf-8

import numpy  as  np
import subprocess
import math
from pyDOE import *
from smt.sampling_methods import LHS
from configparser import ConfigParser


def max_coeff(order, initial_coeffs, strain_values):
    '''
    Approximating coefficients of polynomial based on its order 
    '''
    return (initial_coeffs[-1]*strain_values[-1]**6)/(strain_values[-1]**order)

def sample_coefficient_ranges( N, initial_coeffs, strain_values):
    '''
    Coefficient ranges are sampled by latin hypercube sampling between 0 and maximum value of each coefficient, 
    which is approximated based on the order of the coefficient in the polynomial
    
    Args:
        N: number of samples of each coefficients 
        initial_coeffs: initial guess of the coefficient set
        strain_values: strain values
    
    Returns:
        List of ranges of coefficients with the dimensions of (number_of_coefficients, N)

    '''
    coefficient_ranges = []
    for i in range(len(initial_coeffs)):
        if i == len(initial_coeffs) -1:
            approx_coeff = initial_coeffs[-1]*1.2
            coefficient_ranges.append(list(LHS(xlimits = np.array([[0, approx_coeff], [0, approx_coeff]]))(N)[:, 0]))
        else:
            approx_coeff =  max_coeff(i, initial_coeffs, strain_values)
            coefficient_ranges.append(list(LHS(xlimits = np.array([[0, approx_coeff], [0, approx_coeff]]))(N)[:, 0]))
    return coefficient_ranges

                     
def parameter_space_stack(coefficients_ranges):
    '''
    Create parameter space by sampling equal N samples by LHS for each coefficient of equal size N by LHS and 
    stack them in sequence colomn wise 
    '''

    return np.stack([coefficients_ranges[i] for i in range(len(coefficients_ranges))],1)


def sampling_from_ps(parameter_space, N_samples):
    '''
    Randomly sample from the parameter space
    '''
    indices = np.random.choice(parameter_space.shape[0], N_samples) 
   
    return parameter_space[list(set(indices))]


def update_config_file(config, coefficients):
    '''
    Update coefficients in the config file of ISR3D for uniaxial strain test
    Args:
        coefficients: set of coefficients
    '''
    string_coefficients = [str(c) for c in coefficients ] 
    config.set('main', 'strain_force_c1', string_coefficients[0])
    config.set('main', 'strain_force_c2', string_coefficients[1])
    config.set('main', 'strain_force_c3', string_coefficients[2])
    config.set('main', 'strain_force_c4', string_coefficients[3])
    config.set('main', 'strain_force_c5', string_coefficients[4])
    config.set('main', 'strain_force_c6', string_coefficients[5])

    with open('./cxa/stage1.strain_test.cfg', 'w') as f:
        config.write(f)


def uniaxialStrainTest(sampled_coefficients):
    '''
    Run uniaxialStrainTest on ISR3D and collect the model response for sampled coefficients
    
    '''
 
    config = ConfigParser(allow_no_value=True, delimiters=(" "))
    config.read('./cxa/stage1.strain_test.cfg')
    
    for coefficients in sampled_coefficients:
        update_config_file(config, coefficients)
        subprocess.run(["./kernel/absmc/build/uniaxialStrainTest", "./cxa/stage1.strain_test.cfg"], stdout = subprocess.PIPE,universal_newlines = True).stdout



'''
Different approaches for sampling from the parameter space
'''


def parameter_space_stack_row(N, c1_max, c2_max, c3_max, c4_max, c5_max, c6_max):
    '''
    Create parameter space by sampling  arrays of coefficients of equal size N by LHS and  
    stack them in sequence row wise 
    
    Args:
        N - number of samples for each coefficient
        Cn_max - upper bound of coefficients to sample from 
    Returns:
        Parameters space - list of sets of coefficients with dimension N**6
    '''
    C = np.meshgrid(LHS(c1_max,N), LHS(c2_max,N), LHS(c3_max,N), LHS(c4_max,N), LHS(c5_max,N), LHS(c6_max,N))
    
    return np.vstack((C[0].flatten(), C[1].flatten(),C[2].flatten(),
                      C[3].flatten(),C[4].flatten(),C[5].flatten())).T


def parameter_space_different(c1_range,c2_range,c3_range,c4_range,c5_range,c6_range):
    '''
    Create parameter space by sampling  arrays of coefficients of equal size N by LHS and  
    stack them in sequence row wise 
    
    Args:
        cn_range: ranges of coefficients 1D array
        
    Returns:
        parameter space with dimension of product of length of ranges
    '''
    coefficient_space = []
    for c1 in c1_range:
        for c2 in c2_range:
            for c3 in c3_range:
                for c4 in c4_range:
                    for c5 in c5_range:
                        for c6 in c6_range:
                            coefficient_space.append([c1,c2,c3,c4,c5,c6])

    return np.array(coefficient_space)

def parameter_space_random(N, c1_range,c2_range,c3_range,c4_range,c5_range,c6_range):
    '''
    Create parameter space of sets of coefficients by randomly sampling coefficients from the ranges
    
    Args:
        cn_range: ranges of coefficients 1D array
        N: number of samples
    Returns:
        parameter space with dimension of N
    '''
    coefficient_space = []
    
    for i in range(N):
        c1=np.random.choice(c1_range,1)
        c2=np.random.choice(c2_range,1)
        c3=np.random.choice(c3_range,1)
        c4=np.random.choice(c4_range,1)
        c5=np.random.choice(c5_range,1)
        c6=np.random.choice(c6_range,1)
        
        coefficient_space.append([c1,c2,c3,c4,c5,c6])
    
    return np.array(coefficient_space)




