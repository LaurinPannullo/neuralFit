import numpy as np

def BWfun(Aa,Gamma,Mass,Omega):
    num_peaks = len(Aa)

    sum_peak = np.zeros_like(Omega)

    for i in range(num_peaks):
        num = 4.0*Aa[i]*Gamma[i]*Omega;
        den = (Mass[i]**2 + Gamma[i]**2 - Omega**2)**2 + 4.0*(Gamma[i]**2)*(Omega**2)
        sum_peak += num/den

    return sum_peak

def BWfunOverOmega(Aa,Gamma,Mass,Omega):
    num_peaks = len(Aa)

    sum_peak = np.zeros_like(Omega)

    for i in range(num_peaks):
        num = 4.0*Aa[i]*Gamma[i];
        den = (Mass[i]**2 + Gamma[i]**2 - Omega**2)**2 + 4.0*(Gamma[i]**2)*(Omega**2)
        sum_peak += num/den

    return sum_peak