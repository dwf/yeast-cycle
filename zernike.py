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

def zernike_exp(rho, theta, n, m):
    r_nm_rho = radial_poly(rho, n, m)
    r_nm_rho[rho > 1] = np.nan
    return r_nm_rho * np.exp(1.j * m * theta)

def polargrid(n):
    x, y = np.ogrid[-1.2:1.2:n*1j, -1.2:1.2:n*1j]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan(y / x)
    theta[(n / 2.):] += np.pi
    return rho, theta

def n_m_upto(n, m=None):
    for n_index in xrange(1, n + 1):
        top_m = m if m is not None and n_index == n else n_index
        for m_index in range(0, top_m + 1):
            if (n_index - m_index) % 2 == 0:
                yield (n_index, m_index)

def main():
    import matplotlib.pyplot as plt
    x, y = np.mgrid[-1:1:200j, -1:1:200j]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan(y / x)
    theta[(theta.shape[0] / 2):] += np.pi
    nm_pairs = list(n_m_upto(4, 2))
    plt.ioff()
    cur_max = -np.inf
    cur_min = np.inf
    main_axes = plt.gca()
    for index, (n, m) in enumerate(nm_pairs):
        image = zernike_exp(rho, theta, n, m)
        cur_max = max(cur_max, np.max(np.real(image[~np.isnan(image)])),
                      np.max(np.imag(image[~np.isnan(image)])))
        cur_min = min(cur_min, np.min(np.real(image[~np.isnan(image)])),
                      np.min(np.imag(image[~np.isnan(image)])))
        plt.subplot(2, len(nm_pairs), index + 1)
        plt.imshow(np.real(image), cmap=plt.cm.gray)
        plt.title('$\\mathrm{real}(V_{%d,%d}(\\rho, \\theta))$' % (n, m))
        plt.axis('off')
        plt.subplot(2, len(nm_pairs), len(nm_pairs) + index + 1)
        plt.imshow(np.imag(image), cmap=plt.cm.gray)
        plt.title('$\\mathrm{imag}(V_{%d,%d}(\\rho, \\theta))$' % (n, m))
        plt.axis('off')
    for index in range(2 * len(nm_pairs)):
        plt.subplot(2, len(nm_pairs), index + 1)
        plt.clim(cur_min, cur_max)
    print "cur_min =", cur_min
    print "cur_max =", cur_max
    plt.show()
    plt.ion()

if __name__ == "__main__":
    main()
