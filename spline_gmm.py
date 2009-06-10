from mog_oo import GaussianMixture, PLOTTING_AVAILABLE
from tools import internal_knots, coef2knots, plot_from_spline, unmix
import matplotlib.pyplot as pyplot
import numpy as np
from pca import PrincipalComponents

class SplineModelGaussianMixture(GaussianMixture):
    """docstring for SplineModelGaussianMixture"""
    def __init__(self, ncomponent, data, *args, **keywords):
        super(SplineModelGaussianMixture, self).__init__(ncomponent, data,
            norm=True, *args, **keywords)
        
    def display(self, precision=None, logalpha=None, name=None,figures=(1,2)):
        """Display covariances and alphas with matplotlib."""
        super(SplineModelGaussianMixture, self).display(precision, logalpha, 
            name, figures[0])
        figurenum = figures[1]
        if PLOTTING_AVAILABLE:
            print "Okay plotting"
            pyplot.ioff()
            pyplot.figure(figurenum)
            pyplot.clf()
            rows = cols = np.ceil(np.sqrt(self._ncomponent()))
            if rows * cols == self._ncomponent():
                rows = rows + 1
            ncomp, ndim = self._means.shape
            perspline = (ndim - 1) / 2
            for clust in xrange(self._ncomponent()):    
                t = internal_knots(coef2knots(perspline))
                t = np.concatenate((np.zeros(4),t,np.ones(4)))
                pyplot.subplot(rows, cols, clust+1)
                spline1, spline2 = unmix(self._means[clust, :], 
                    self._normmean, self._normstd)
                plot_from_spline(spline1)
                plot_from_spline(spline2)
            pyplot.show()
            pyplot.ion()
            
class SplineModelPCAGaussianMixture(GaussianMixture):
    """docstring for SplineModelGaussianMixture"""
    def __init__(self, ncomponent, data, ncmp, *args, **keywords):
        self._pc = PrincipalComponents(data)
        self._direct()
        data = self.project(data, ncmp)
        super(SplineModelGaussianMixture, self).__init__(ncomponent, data,
            norm=False, *args, **keywords)
        

    def display(self, precision=None, logalpha=None, name=None,figures=(1,2)):
        """Display covariances and alphas with matplotlib."""
        super(SplineModelGaussianMixture, self).display(precision, logalpha, 
            name, figures[0])
        figurenum = figures[1]
        if PLOTTING_AVAILABLE:
            print "Okay plotting"
            pyplot.ioff()
            pyplot.figure(figurenum)
            pyplot.clf()
            rows = cols = np.ceil(np.sqrt(self._ncomponent()))
            if rows * cols == self._ncomponent():
                rows = rows + 1
            ncomp, ndim = self._means.shape
            perspline = (ndim - 1) / 2
            for clust in xrange(self._ncomponent()):    
                t = internal_knots(coef2knots(perspline))
                t = np.concatenate((np.zeros(4),t,np.ones(4)))
                pyplot.subplot(rows, cols, clust+1)
                spline1, spline2 = unmix(self._means[clust, :], 
                    self._normmean, self._normstd)
                plot_from_spline(spline1)
                plot_from_spline(spline2)
            pyplot.show()
            pyplot.ion()