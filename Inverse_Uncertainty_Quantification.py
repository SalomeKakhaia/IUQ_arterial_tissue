#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az
import scipy.optimize
from scipy import stats
from pymc3 import Model, Normal, Slice, sample, traceplot
from pymc3.distributions import Interpolated
from sklearn.metrics import mean_squared_error as mse
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")


with open('data/GPy_model_media_18.pkl', 'rb') as file:
    GP_model = pickle.load(file)

def create_parameters_set(strain_values, coefficient_sets):
    
    ''' 
    Create new parameters set from calibration coefficients' sets and strain values: 
    
    Args:
      coefficient_sets: list of calibration coefficients' sets - [c1, ...c6]
    
    Returns:
        list of parameter sets with elements - [strain value, c1, ...c6] 

    '''
    
    parameter_space = []
    for i in range(len(strain_values)):
        parameter_space.append(np.insert(coefficient_sets, 0, strain_values[i])) 
    return np.array(parameter_space)


def model_prediction( strain_values,theta1, theta2, theta3, theta4 , theta5, theta6):
    
    ''' 
    GP model prediction for the specific set of parameters  
    
    Args:
        theta1,...theta6: calibration coefficients
    
    Returns:
        mean of GP model prediction for the coefficients
    '''
    
    coefficient_set = np.array([theta1, theta2, theta3, theta4, theta5, theta6])    
    test_par_set = create_parameters_set(strain_values, coefficient_set)  
    simY, _ = GP_model.predict(test_par_set)
    
    return simY.squeeze()

def sample_from_posterior(parameter_name, parameter):
    '''
    Estimation of the PDF of a random variable by Gaussian Kernel density estimation method;
    A linear interpolation of pdf is evaluated on evenly distributed points from interval 
    between 0 and maximum value of posterior theta +- considerable margins.
    
    Args:
        parameter: extracted chain values for the parameter from the trace 
        parameter_name: name of the parameter
    
    Returns:
        Interpolated posterior distribution
    '''
    
    margin = np.max(theta) - np.min(theta)
    x = np.linspace(np.min(theta), np.max(theta), 200)
    y = stats.gaussian_kde(theta)(x)
    
    # Extend the domain (so that lower bound never gets negative) and use linear approximation of density on it
    
    x = np.concatenate([[max(x[0] - 3 * margin,0)], x, [x[-1] + 3 * margin]])
    y = np.concatenate([[0], y, [0]])
    
    return Interpolated(param_name, x, y)


## LogLikelihood and gradient of the LogLikelihood functions

def log_likelihood(theta, calibration_data, strain_values):
    
    ''' 
    Likelihood defined as a joint Gaussian distribution, whose mean is calibration calibration_data 
    given GP model response for the prior uncertain parameters and the variance is sigma, 
    where sigma is modelling error variance.
    

    '''
    (theta1, theta2, theta3, theta4, theta5, theta6, sigma) = theta[0]
    
    y_pred = model_prediction(strain_values, theta1, theta2, theta3, theta4 , theta5, theta6)
    
    logp = -len(calibration_data) * np.log(np.sqrt(2.0 * np.pi) * sigma) - np.sum((calibration_data - y_pred) ** 2.0) / (2.0 * sigma ** 2.0)
        
    return logp



def der_log_likelihood(theta, calibration_data, strain_values):
    '''
    Finite-difference approximation of the gradient of a likelihood function.
    '''
    def lnlike(values):
        return log_likelihood(values, calibration_data, strain_values)

    eps = np.sqrt(np.finfo(float).eps)
    
    grads = scipy.optimize.approx_fprime(theta[0], lnlike, eps * np.ones(len(theta)))
    
    return grads

##  Wrapping loglikelihood and gradient of loglikelihood as theano objects

class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, calibration_data, strain_values):
        self.calibration_data = calibration_data
        self.strain_values = strain_values
        self.loglike_grad = LoglikeGrad(self.calibration_data, self.strain_values)

    def perform(self, node, inputs, outputs):
        logp = log_likelihood(inputs, self.calibration_data, self.strain_values)
        outputs[0][0] = np.array(logp)

    def grad(self, inputs, grad_outputs):
        (theta,) = inputs
        grads = self.loglike_grad(theta)
        return [grad_outputs[0] * grads]

class LoglikeGrad(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, calibration_data, strain_values):
        self.der_likelihood = der_log_likelihood
        self.calibration_data = calibration_data
        self.strain_values = strain_values

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        grads = self.der_likelihood(inputs, self.calibration_data, self.t)
        outputs[0][0] = grads

def initial_inference(likelihood, lower_bounds, upper_bounds, sigma):
    '''
    Initial bayesian inference, when posterior samples are obtained from the prior distributions
    
    Returns:
        Sampled results contained in the trace variable (a pymc3 results object
    '''
    with pm.Model() as model:
        # Sample from prior distributions of IUQ parameters
        theta1 = pm.Uniform("theta_1", lower = lower_bounds[0], upper = upper_bounds[0])
        theta2 = pm.Uniform("theta_2", lower = lower_bounds[1], upper = upper_bounds[1])
        theta3 = pm.Uniform("theta_3", lower = lower_bounds[2], upper = upper_bounds[2])
        theta4 = pm.Uniform("theta_4", lower = lower_bounds[3], upper = upper_bounds[3])
        theta5 = pm.Uniform("theta_5", lower = lower_bounds[4], upper = upper_bounds[4])
        theta6 = pm.Uniform("theta_6", lower = lower_bounds[5], upper = upper_bounds[5])
        sigma  = pm.HalfNormal("sigma",sigma = sigma, testval=0.01)

        # Convert parameters to a tensor vector
        theta = tt.as_tensor_variable([theta1, theta2, theta3, theta4, theta5, theta6, sigma])

        # Create model likelihood with corresponding name
        pm.Potential("Likelihood", likelihood(theta))

        # Sample posterior information
        trace = pm.sample_smc(5, parallel=True)
        
    return trace


def bayesian_calibration(initial_trace, N_iterations, loglike):
    '''
    Bayesian inference where posterior distribution 
    Return traces saved after each calibration step
    '''
    traces = [initial_trace]
    
    for i in np.arange(1,N_iterations,1):
        
        with Model() as model:
            tick = time.time()
            # Priors are posteriors from previous iteration
            theta1 = sample_from_posterior("theta_1", traces[-1]["theta_1"])
            theta2 = sample_from_posterior("theta_2", traces[-1]["theta_2"])
            theta3 = sample_from_posterior("theta_3", traces[-1]["theta_3"])
            theta4 = sample_from_posterior("theta_4", traces[-1]["theta_4"])
            theta5 = sample_from_posterior("theta_5", traces[-1]["theta_5"])
            theta6 = sample_from_posterior("theta_6", traces[-1]["theta_6"]) 
            sigma  = pm.Normal("sigma", mu = np.mean(traces[-1]["sigma"]), sigma = np.std(traces[-1]["sigma"]))

            # convert parameters to a tensor vector
            theta = tt.as_tensor_variable([theta1, theta2, theta3, theta4, theta5, theta6, sigma])
            pm.Potential("like", loglike(theta))
            trace = pm.sample_smc(500, parallel=True)
            traces.append(trace) 
            
    return traces


# In[6]:


def visualise_calibration_comparision(model, traces, AB_calibrated, strain_values, real_stress_values):
    
    calibrated_coeffs = np.array([np.mean(traces[-1]["theta_1"]),
                                  np.mean(traces[-1]["theta_2"]),
                                  np.mean(traces[-1]["theta_3"]),
                                  np.mean(traces[-1]["theta_4"]),
                                  np.mean(traces[-1]["theta_5"]),
                                  np.mean(traces[-1]["theta_6"])])

    ## Create parameters set consisting of certain set of coefficients and all strain values
    calibrated_set = create_parameters_set(strain_values, calibrated_coeffs)
    ## GP predicition 
    simY_calibrated, simMse_calibrated = GP_model.predict(calibrated_set)
    ## Mean squared errors of AB model response and GP model prediction
    rmse_cal = np.round(np.sqrt(mse(real_stress_values, simY_calibrated))*100,2)
    rmse_AB =np.round( np.sqrt(mse(real_stress_values, AB_calibrated))*100,2 )


    plt.plot(strain_values, real_stress_values, label ="FE data", color = "red" )
    plt.plot(strain_values, AB_calibrated, ":", label ="AB model prediction, rmse = %s"%rmse_AB+"%", color = "blue" )
    plt.plot(strain_values,  simY_calibrated, ":", linewidth = "4", label = "GP prediction, rmse = %s"%rmse_cal+"%")
    plt.xlabel("Strain")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    plt.show()
    
    
    
def visualise_pdf_1D(traces):
    '''
    Plot 1d marginalised posterior distributions of parameters through calibration process
    '''
    cmap = mpl.cm.winter
    j=0
    for param in ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5",  "theta_6",   "sigma"]:
        j+=1
        plt.figure(figsize=(6, 2))
        for update_i, trace in enumerate(traces):
            samples = trace[param]
            smin, smax = np.min(samples), np.max(samples)
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)
            if update_i ==0:
                plt.plot(x, y, color = cmap(1 - update_i / len(traces)), label = "Initial PDF")
            if update_i == len(traces)-1:
                plt.plot(x, y, color = cmap(1 - update_i / len(traces)), label = "Converged PDF")      
            else:
                plt.plot(x, y, color = cmap(1 - update_i / len(traces)))
        plt.ylabel("Frequency")
        if j <7:
            plt.title(r'$\theta_%s$'%j)
        else:
            plt.title(param)
        plt.legend()
        plt.show()


# In[ ]:




