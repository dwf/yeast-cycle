from mog_oo import GaussianMixture, GarbageModelGaussianMixture, DiagonalGaussianMixture, PLOTTING_AVAILABLE
from tools import internal_knots, coef2knots, plot_from_spline, unmix
import matplotlib.pyplot as pyplot
import numpy as np
import pdb
from pca import PrincipalComponents

class SplineModelGaussianMixture(GaussianMixture):
    """docstring for SplineModelGaussianMixture"""
        
    def display(self, covariance=None, logalpha=None, name=None, extra=1, k=3,
        figures=(1,2)):
        """Display covariances and alphas with matplotlib."""
        pyplot.close('all')
        super(SplineModelGaussianMixture, self).display(covariance, logalpha, 
            name, figures[0])
        figurenum = figures[1]
        print figurenum
        if PLOTTING_AVAILABLE:
            pyplot.ioff()
            pyplot.figure(figurenum)
            pyplot.clf()
            rows = np.floor(np.sqrt(self._ncomponent()))
            cols = np.ceil(np.sqrt(self._ncomponent()))
            while rows * cols < self._ncomponent():
                rows += 1
            ncomp, ndim = self._means.shape
            perspline = (ndim - extra) / 2
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

                pyplot.vlines(spline1[0], -0.2, 1.01, color='0.8',
                              linestyles='dotted') 
                if len(self._covariances.shape) > 2:
                    eva, evec = np.linalg.eig(self._covariances[clust])
                    pc = evec[:, np.argmax(eva)]
                    uponestd = self._means[clust, :] + \
                               pc * 3*np.sqrt(np.max(eva))
                    downonestd = self._means[clust, :] - \
                               pc * 3*np.sqrt(np.max(eva))
                    spline1u, spline2u = unmix(uponestd, self._normmean,
                                            self._normstd, k=k, extra=extra)
                    
                    spline1d, spline2d = unmix(downonestd, self._normmean,
                                           self._normstd, k=k, extra=extra)
                    plot_from_spline(spline1u, color='0.3')
                    plot_from_spline(spline1d, color='0.3')
                    plot_from_spline(spline2u, color='0.3')
                    plot_from_spline(spline2d, color='0.3')
                
                pyplot.ylim(-0.2, 1.01)
                plot_from_spline(spline1)
                plot_from_spline(spline2)
                
            pyplot.show()
            pyplot.ion()

class SplineGarbageModelGaussianMixture(GarbageModelGaussianMixture, 
                                        SplineModelGaussianMixture):
    pass

class SplineDiagonalGaussianMixture(DiagonalGaussianMixture,
                                    SplineModelGaussianMixture):
    pass


class SplineModelPCAGaussianMixture(GaussianMixture):
    """docstring for SplineModelGaussianMixture"""
    def __init__(self, npc=None, *args, **keywords):
        self._pc = PrincipalComponents(data)
        self._pc._direct()
        if npc is None:
            npc = len(self._pc._eigval)
        data = self._pc.project(npc)
        super(SplineModelPCAGaussianMixture, self).__init__(ncomponent, data,
            norm=False, *args, **keywords)
    
    def display(self, covariance=None, logalpha=None, name=None, extra=1,
        k=3, figures=(1,2)):
        """Display covariances and alphas with matplotlib."""
        pyplot.close('all')
        super(SplineModelPCAGaussianMixture, self).display(covariance,
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
