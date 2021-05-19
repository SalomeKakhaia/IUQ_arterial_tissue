#!/usr/bin/env python
# coding: utf-8

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

def saltelli_sampling(bounds, N):
    
    '''
    Sampling sets of parameters by Saltelli's sampling as an input for the Sobol sensitivity analysis.
    
    Args:
        bounds: list
            List where each entry is a list of size 2 : [lower_bound, upper bound]
            
        N : int
            The number of samples to generate
    Retruns:
        Problem and the numpy matrix of the SA model inputs
    
    '''
    # Define the problem
    problem = {
    'num_vars': 6,
    'names': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
    'bounds': bounds
    }

    return problem, saltelli.sample(problem, N)
    
    
def predictions_per_strains(model, parameters_sets, strain_values):
    '''
    Collect the GP predictions for sampled sets of parameters and sort per strain values
    
    Args:
        parameters_sets: numpy matrix 
            Each entry is a set of coefficients.
        strain_values: list
            strain values 
    Returns:
        list of numpy vectors, where each entry is GP predictions for some sampled parameters
        per particular value of strain 
    '''
    gp_predictions = []
    for st in range(len(strain_values)):  
        pred_per_strain = np.zeros([parameters_sets.shape[0]])
        for i, par_set in enumerate(parameters_sets):
            extended_par_set = np.insert(par_set,0,strain_values[st])
            pred_per_strain[i] = model.predict(np.array([extended_par_set]))[0]
        gp_predictions.append(pred_per_strain)
    return gp_predictions

def sa_per_coefficient(problem, GP_predictions, strain_values, coeff_index, si_order = 'S1'):
    '''
    Calculating sensitivity indices per strain values for a single coefficient
    '''

    s_indices = np.array([sobol.analyze(problem, GP_predictions[i])[si_order][coeff_index] 
                          for i in range(len(strain_values))])
    s_confidence = np.array([sobol.analyze(problem, GP_predictions[i])[si_order + '_conf'][coeff_index] 
                             for i in range(len(strain_values))])

    return s_indices, s_confidence


def sa_visualise(problem, GP_predictions, strain_values, coeff_numb = 6, si_order = 'S1'):
    
    # loop over coefficients
    for ci in range(coeff_numb):
        s_indices, s_confidence  = sa_per_coefficient(problem, GP_predictions, strain_values,
                                                      ci, si_order = si_order)
        
        plt.errorbar(strain_values, s_indices, yerr = s_confidence, fmt='-o', label = "c%s"%(ci+1))
        
    plt.legend()
    plt.xlabel("Strain values")
    plt.ylabel(si_order)
    plt.show()

