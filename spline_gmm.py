from mog_oo import GaussianMixture, PLOTTING_AVAILABLE
from tools import internal_knots, coef2knots, plot_from_spline, unmix
import matplotlib.pyplot as pyplot
import numpy as np
import pdb
from pca import PrincipalComponents

class SplineModelGaussianMixture(GaussianMixture):
    """docstring for SplineModelGaussianMixture"""
    def __init__(self, ncomponent, data, *args, **keywords):
        super(SplineModelGaussianMixture, self).__init__(ncomponent, data,
            norm=True, *args, **keywords)
        
    def display(self, precision=None, logalpha=None, name=None, extra=1, k=3,
        figures=(1,2)):
        """Display covariances and alphas with matplotlib."""
        pyplot.close('all')
        super(SplineModelGaussianMixture, self).display(precision, logalpha, 
            name, figures[0])
        figurenum = figures[1]
        print figurenum
        if PLOTTING_AVAILABLE:
            print "Okay plotting"
            pyplot.ioff()
            pyplot.figure(figurenum)
            pyplot.clf()
            rows = cols = np.ceil(np.sqrt(self._ncomponent()))
            if rows * cols == self._ncomponent():
                rows = rows + 1
            ncomp, ndim = self._means.shape
            perspline = (ndim - extra) / 2
            #pdb.set_trace()
            for clust in xrange(self._ncomponent()):    
                t = internal_knots(coef2knots(perspline))
                t = np.concatenate((np.zeros(4),t,np.ones(4)))
                pyplot.subplot(rows, cols, clust + 1)
                pyplot.title('Mean curves for component %d' % (clust + 1),
                    fontsize='medium')
                pyplot.xlabel('Position along medial axis', 
                    fontsize='x-small')
                pyplot.ylabel('Fraction of principal axis width',
                    fontsize='x-small')
                spline1, spline2 = unmix(self._means[clust, :], 
                    self._normmean, self._normstd, k=k, extra=extra)
                plot_from_spline(spline1)
                plot_from_spline(spline2)
                pyplot.ylim(-0.2, 1.01)
            pyplot.show()
            pyplot.ion()
            
class SplineModelPCAGaussianMixture(GaussianMixture):
    """docstring for SplineModelGaussianMixture"""
    def __init__(self, ncomponent, data, npc=None, *args, **keywords):
        self._pc = PrincipalComponents(data)
        self._pc._direct()
        if npc is None:
            npc = len(self._pc._eigval)
        data = self._pc.project(npc)
        super(SplineModelPCAGaussianMixture, self).__init__(ncomponent, data,
            norm=False, *args, **keywords)
    
    def display(self, precision=None, logalpha=None, name=None, extra=1,
        k=3, figures=(1,2)):
        """Display covariances and alphas with matplotlib."""
        pyplot.close('all')
        super(SplineModelPCAGaussianMixture, self).display(precision,
            logalpha, name, figures[0])
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
            perspline = (ndim - extra) / 2
            for clust in xrange(self._ncomponent()):    
                t = internal_knots(coef2knots(perspline))
                t = np.concatenate((np.zeros(4),t,np.ones(4)))
                pyplot.subplot(rows, cols, clust+1)
                means = self._pc.reconstruct(self._means[clust, :]).squeeze()
                spline1, spline2 = unmix(means,k=k,extra=extra)
                plot_from_spline(spline1)
                plot_from_spline(spline2)
                pyplot.ylim(-0.2, 1.01)
            pyplot.show()
            pyplot.ion()