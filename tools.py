import numpy as np
from scipy.optimize import curve_fit

def exponential(t, r, P_init):
    return P_init * np.exp(r * t)

def logistic(t, K, r, P_init):
    return K / (1 + ((K - P_init)/P_init) * np.exp(-r * t))

def get_exponential_params(x, y):    
    r_guess = 0.017
    P_init_guess = y.iloc[0]
    optimal_params, _ = curve_fit(exponential, x, y, p0=[r_guess, P_init_guess])
    r, P_init = optimal_params
    return r, P_init

def get_logistic_params(x, y):
    K_guess = max(y) * 1.2
    r_guess = 0.1
    P_init_guess = y.iloc[0]
    optimal_params, _ = curve_fit(logistic, x, y, p0=[K_guess, r_guess, P_init_guess])
    K, r, P_init = optimal_params
    return K, r, P_init