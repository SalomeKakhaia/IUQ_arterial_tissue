#!/usr/bin/env python
# coding: utf-8
import numpy as np
import GPy
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from scipy.stats import norm
from pyDOE import *


def sampling_ps_random(dimension, N_samples):
    '''
    Randomly sample from the parameter space 
    '''
    indices = np.random.choice(dimension, N_samples) 
    return list(set(indices))


def sampling_ind_lhs(dimension, N_samples):
    '''
    Sample indices of parameters by latin hypercube sampling
    '''
    indices = (lhs(N_samples, samples = 1) * dimension).squeeze()
    indices = [int(i) for i in indices]
    return list(set(indices))

def create_parameters_space(coefficient_sets, stress_values, strain_values):
    '''
    Create spaces of parameters and stress values.
    
    Returns
    -------
    parameters_space: array_like
        [strain value, c1, ...c6]
        
    stress_space: array_like
        AB model stress response for each set of parameters  
    '''
    parameters_space, stress_space = [], []
    
    for i in range(coefficient_sets.shape[0]):
        for j in range(len(strain_values)):
            parameters_space.append(np.insert(coefficient_sets[i],0,strain_values[j]))
            stress_space.append([stress_values[i][j]])
            
    return np.array(parameters_space), np.array(stress_space)


def create_coefficients_set(coefficient_sets, strain_values):
    ''' 
    Create coefficients set : [strain value, c1, ...c6] 
    '''
    parameters_space = []
    for j in range(len(strain_values)):
        parameters_space.append(np.insert(coefficient_sets,0,strain_values[j])) 
    return np.array(parameters_space)


def train_test_data(coefficients_sets, stress_values, strain_values, ratio):
    '''
    ratio - ratio of training and testing data
    
    '''       
    parameters_space, stress_space = create_parameters_space(coefficients_sets, stress_values, strain_values)
    sampled_indices = sampling_ps_random(parameters_space.shape[0], int(parameters_space.shape[0]*0.8))
    unsampled_indices = [ind for ind in np.arange(parameters_space.shape[0]) if ind not in sampled_indices]

    X_train = parameters_space[sampled_indices]
    y_train = stress_space[sampled_indices]
      
    X_test = parameters_space[unsampled_indices]
    y_test = stress_space[unsampled_indices]
    
    return X_train, y_train, X_test, y_test


def gp_regression_model(X_train, y_train):
    '''
    ratio - ratio to split train and validation datasets
    
    '''
    ## Define kernel
    ker = GPy.kern.Matern52(input_dim = 7, ARD=True)

    ## Define GP model
    model = GPy.models.GPRegression(X_train, y_train, ker)
    model.constrain_positive('')

    ## Optimize 
    model.optimize(messages=True, max_f_eval = 3000)
    
    return model




def visualise_gp_comparision(gp_model, coefficient_set, strain_values, real_stress_values,
                               AB_initial = False):
    '''
    Test in experimental data 
    '''

    initial_coefficient_set = create_coefficients_set(coefficient_set, strain_values)
    simY_initial, simMse_initial = gp_model.predict(initial_coefficient_set)


    dyfit_initial = simMse_initial
    rmse_gp = np.round(np.sqrt(mse(real_stress_values, simY_initial))*100,1)  # Root mean squared error
    rmse_ab = np.round(np.sqrt(mse(real_stress_values, AB_initial))*100  ,1)# Root mean squared error

    plt.plot(strain_values, real_stress_values, color = "red", label ="FE model" )
    plt.plot(strain_values, AB_initial, label = "AB initial, rmse = %s"%rmse_ab+"%")
    plt.plot(strain_values, simY_initial, color = "orange", label = "GP initial, rmse = %s"%rmse_gp+"%")
    plt.fill_between(strain_values, simY_initial.squeeze() - dyfit_initial.squeeze(), simY_initial.squeeze() + dyfit_initial.squeeze(),
                     color='orange', label= 'GP prediction variance',alpha=0.25)

    plt.xlabel("Strain")
    plt.ylabel("Shear stress [Mpa]")
    plt.legend()
    plt.show()
    
    
def visualise_resiudal(strain_values, testX, testY, simY):
    '''
    Plotting difference between true stres values and simulated GP stress prediction per strain values 
    '''
    total_array = []
    for strain in strain_values:    
        indices = np.where(testX[:, 0] == strain)
        total_array.append((testY[indices] - simY[indices]).squeeze())
        
    plt.boxplot(total_array)
    plt.axhline(y=0, color='r', linewidth=1, linestyle='--')

    xt = list(np.arange(len(strain_values)))
    yt = [str(i) for i in strain_values]

    xt  = np.array(xt)[list(np.arange(1, len(xt), 2))]
    yt  = np.array(yt)[list(np.arange(1, len(yt), 2))]

    plt.xticks(xt,yt)
    plt.xlabel("Strain")
    plt.ylabel("Residual [MPa]")
    plt.show()

