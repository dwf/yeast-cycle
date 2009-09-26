import numpy as np
from scipy.misc import factorial

def radial_poly(rho, n, m):
    upper = (n - np.abs(m)) / 2
    s = np.arange(upper + 1)
    consts = -1 ** s
    consts *= factorial(n - s)
    consts /= factorial(s)
    consts /= factorial((n + np.abs(m))/2 - s)
    consts /= factorial((n - np.abs(m))/2 - s)
    return (consts * rho[..., np.newaxis]**(n - 2 * s)).sum(axis=-1)

def zernike(rho, theta, n, m):
    r_nm_rho = radial_poly(rho, n, m)
    r_nm_rho[rho > 1] = np.nan
    if m > 0:
        func = np.sin
    else:
        func = np.cos
    return r_nm_rho * func(m * theta)

def polargrid(n):
    x, y = np.ogrid[-1.2:1.2:n*1j, -1.2:1.2:n*1j]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan(y / x)
    theta[(n / 2.):] += np.pi
    return rho, theta
