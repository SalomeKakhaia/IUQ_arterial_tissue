#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math 
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("seaborn-darkgrid")


##  The reduced polynomial (RP) form of energy potential for an uncompressible material 
def RP_energy_potential(C0, order, strain): 
    '''
    Calculates strain energy potential for given strain values
    
    Args:
        C0: list of material model parameters describing shear behaviour of arterial wall (with length = order);
        order: order of the polynomial;
        strain: single strain value;
    Returns:
        stress value for the strain value given.
    '''
    J = 1
    lmbd_U = 1 + strain
    lmbd_1 = lmbd_U 
    lmbd_2 = lmbd_3 = lmbd_U**(-1/2)
    I_1 = (J**(-1/3))*(lmbd_1 + lmbd_2 + lmbd_3)
    U = 0
    for i in range(1, order+1, 1):
        U += C0[i-1]*( I_1 - 3 )**i
    return U


## Nominal stress-strain relation for the reduced polynomial strain energy potential
## for uniaxial strain tests  

def RP_stress_strain(C0, order, strain):
    '''
    Creates stress-strain relation for reduced polynomial energy potential
    
    Args:
        C0: list of material model parameters describing shear behaviour of arterial wall (with length = order);
        order: order of the polynomial;
        strain: single strain value;
    Returns:
        strain energy value for the strain value given.
    '''
    J = 1
    lmbd_U = 1 + strain
    lmbd_1 = lmbd_U 
    lmbd_2 = lmbd_3 = lmbd_U**(-1/2)
    I_1 = (J**(-2/3))*(lmbd_1**2 + lmbd_2**2 + lmbd_3**2)
    
    U = 0
    for i in range(1,order+1,1):
        U += i*C0[i-1]*( I_1 - 3 )**(i-1)
    
    return 2*(lmbd_U-lmbd_U**(-2))*U


def stress_strain_list(C0, order, strain_values):
    '''
    Stress strain relation for list of strain values:
    
    Args:
        C0: list of material model parameters describing shear behaviour of arterial wall (with length = order);
        order: order of the polynomial;
        strain_values: list of strain values;
    Returns:
        List of stress values per strain values according to RP_stress_strain function.
    '''
    stress_values = []
    for strain in strain_values:
        stress = RP_stress_strain(C0, order, strain)
        stress_values.append(stress)
    return stress_values


def stress_strain_curve_single(C0, order, strain_values, stress_data, compare = False):
    '''
    Visualizing stress-strain curves for single layer of arterial tissue
    '''

    stress_values = stress_strain_list(C0, order, strain_values)
    if compare:
        plt.plot(strain_values, stress_values)
    plt.plot(strain_values, stress_data,color = "b")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.show()


def stress_strain_curve_together(C0s, order, strain_values, labels):
    '''
    Visualizing stress-strain curves for three layers of arterial tissue
    ''' 
    N = 35
    for i in range(3):
        stress_values = stress_strain_list(C0s[i], order, strain_values[i][:N])
        plt.plot(strain_values[i][:N], stress_values, label = labels[i])
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    plt.show()

