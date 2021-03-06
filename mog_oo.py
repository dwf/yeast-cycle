"""Okay. This is a docstring."""
import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import pdb
import sys
from functools import wraps

# Constants useful for the module to use
_FLOAT_T = np.float64
_MIN_PROB = np.finfo(_FLOAT_T).tiny
_MIN_LOGPROB = np.log(_MIN_PROB)
_LOG2PI = np.log(2 * np.pi)
_DEBUG = False


try:
    import matplotlib.pyplot as pyplot
    PLOTTING_AVAILABLE = True

except ImportError:
    PLOTTING_AVAILABLE = False

def logsumexp(logx, axis=-1):
    """
    Compute log(sum(exp(x))) in a numerically stable way.  
    Use second argument to specify along which dimensions the logsumexp
    shall be computed. If -1 (which is also the default), logsumexp is 
    computed along the last dimension. 
    
    By Roland Memisevic, distributed under Python License.
    """
    if len(logx.shape) < 2:
        xmax = logx.max()
        return xmax + np.log(np.sum(np.exp(logx-xmax)))
    else:
        if axis != -1:
            logx = logx.transpose(range(axis) + range(axis+1, 
                len(logx.shape)) + [axis])
        lastdim = len(logx.shape)-1
        xmax = logx.max(lastdim)
        return xmax + np.log(np.sum(np.exp( logx-xmax[..., np.newaxis]),
            lastdim))

def _print_now(s):
    print s
    sys.stdout.flush()


def plot_progress(lhistory):
    """Plot the progress with matplotlib (if available)."""
    if PLOTTING_AVAILABLE:
        pyplot.figure(2)
        pyplot.ioff()
        pyplot.clf()
        pyplot.plot(xrange(len(lhistory)), lhistory)
        pyplot.title('Log likelihood vs. iterations')
        pyplot.ylabel('Log likelihood')
        pyplot.xlabel('Iterations')
        pyplot.show()
        pyplot.ion()


class GaussianMixture(object):
    """Basic Gaussian mixture model."""
    
    
    def save(self, file):
        """Save to a .npz file."""
        things_to_save = {
            'classname': self.__class__.__name__,
            'covariances': self._covariances,
            'means': self._means,
            'logalpha': self._logalpha, 
            'update': self._update,
            'history': np.asarray(self._history),
            'ndim': np.asarray(self._ndim),
            'normmean': np.asarray(self._normmean),
            'normstd': np.asarray(self._normstd),
            'resp': np.asarray(self._resp)
        }
        np.savez(file, **things_to_save)
    
    @classmethod
    def from_saved(cls, *args, **kwargs):
        """Load a serialized .npz GaussianMixture class."""
        archive = np.load(*args, **kwargs)
        obj = cls.__new__(cls)
        if cls.__name__ != archive['classname']:
            raise TypeError('needs to be loaded with %s class' %
                            archive['classname'])
        
        for arrname in archive.files:
            value = archive[arrname]
            if arrname == 'classname':
                continue
            if arrname == 'history':
                value = value.tolist()
            elif arrname == 'ndim':
                value = np.asscalar(value)
            elif value.size == 1 and value == np.array(None):
                value = None
            setattr(obj, '_%s' % arrname, value)
        return obj
    
    def __init__(self, ncomponent, ndim, norm=None, pcounts=0, update=None):
        """
        Constructor. Add real documentation.
        """
        self._ndim = ndim
        self._history = []
        
        if norm is not None:
            self._normmean = norm.mean(axis=0)
            self._normstd = norm.std(axis=0)
            
        else:
            self._normmean = None
            self._normstd = None
        
        # Handle the 'update' array, which tells us whether or not we 
        # update a given component's parameters by M-step estimation.
        if update is None:
            self._update = np.ones((ncomponent,), dtype=bool)
        elif len(update) != ncomponent:
            raise ValueError("len(update) must be equal to ncomponent")
        else:
            self._update = np.asarray(update, dtype=bool)
        
        # Make space for our model parameters
        means = np.empty((ncomponent, ndim))
        covariances = self._allocate_covariances(ncomponent, ndim)
        memberships = np.empty((ncomponent))
        resp = None
        
        # Initialize them
        self._logalpha = memberships
        self._covariances = covariances
        self._means = means
        self._resp = resp
        
        # For saving last iteration parameters for debugging
        if _DEBUG:
            self._oldmeans, self._oldcov, self._oldlogalpha = \
                None, None, None
            self._oldresp = None
    
    def _allocate_covariances(self, ncomponent, ndim):
        """For easy overriding in the diagonal case."""
        return np.empty((ncomponent, ndim, ndim))
    
    def _ndim(self):
        """Return the number of dimensions in the training set."""
        return self._ndim
    
    def _ncomponent(self):
        """Return the number of components in the mixture."""
        return self._logalpha.shape[0]
    
    
    def _numparams(self):
        """Return the number of free parameters in the model."""
        covparams = self._ndim * (self._ndim + 1) / 2.
        meanparams = self._ndim
        return self._ncomponent() * (covparams + meanparams) + \
            (self._ncomponent() - 1)
    
     
    def _updating_all(self):
        """
        Quick check to see whether we're updating every mixture component
        so that we can vectorize certain M-step operations more.
        """
        return np.all(self._update)
    
    
    def aic(self, data):
        """
        Akaike information criterion for the current model fit.
        
        NOTE: Does not account for pseudocounts.
        """
        return 2 * self._numparams() - 2 * self.loglikelihood(data)
    
    
    def bic(self, data):
        """
        Bayesian information criterion for the current model fit.
        
        NOTE: Does not account for pseudocounts.
        """
        params = self._numparams()
        return -2 * self.loglikelihood(data) +  params * np.log(data.shape[0])
    
    
    def _random_e_step(self, data):
        """
        Randomly do a hard assignment to a particular component of
        the mixture.
        """
        fakepostidx = random.random_integers(self._ncomponent(),
            size=data.shape[0]) - 1
        fakeposterior = np.zeros((self._ncomponent(), data.shape[0]))
        fakeposterior[fakepostidx, xrange(data.shape[0])] = 1
        self._resp = fakeposterior
    
    
    def log_probs(self, clust, data):
        """
        Log probability of each training data point under a particular
        mixture component.
        """
        centered = data - self._means[clust, ...][np.newaxis, ...]
        ndim = self._ndim
        logphat = linalg.solve(self._covariances[clust, ...], centered.T)
        logphat *= centered.T
        logphat *= -0.5
        logphat = logphat.sum(axis=0)

        # TODO: calculate logdet in a way less prone to overflow.
        lognormalizer = ndim / 2.0 * _LOG2PI + 0.5 * \
                np.log(linalg.det(self._covariances[clust, ...]))
        
        return logphat - lognormalizer
    
    
    def _logjoint(self, data):
        """
        Function that computes the log joint probability of each training
        example.
        """
        logalpha = self._logalpha
        numpts = data.shape[0]
        condprobs = np.nan * np.zeros((self._ncomponent(), numpts))
        for clust in xrange(self._ncomponent()):
            condprobs[clust, :] = self.log_probs(clust, data)
        condprobs += logalpha[:, np.newaxis]
        lik = np.logaddexp.reduce(condprobs, axis=0) # Reduces along first dim
        return condprobs, lik
    
    
    def e_step(self, data):
        """
        Compute expected value of hidden variables under the posterior 
        as specified by the current model parameters.
        """
        
        if self._resp.shape[1] < data.shape[0]:
            self._resp = np.empty((self._ncomponent, data.shape[0]))
        
        logjoint, lik = self._logjoint(data)
        logresp = logjoint - lik
        logresp[logresp < _MIN_LOGPROB] = _MIN_LOGPROB
        np.exp(logresp, self._resp[:, :data.shape[0]])
        return lik.sum()
    
    
    def loglikelihood(self, data):
        """
        Log likelihood of the training data.
        """
        lik = self._logjoint(data)[1]
        return lik.sum()
    
    
    def m_step(self, data, pcounts=0, pcmean=None):
        """
        Maximize the model parameters with respect to the expected
        complete log likelihood.
        """        
        if self._resp is None:
            self._random_e_step(data)

        self._m_step_update_means(data, pcounts, pcmean)
        
        self._m_step_update_covariances(data, pcounts)
        
        self._m_step_update_logalpha(data)
    
    
    def _m_step_update_means(self, data, pcounts=0, pcmean=None):
        """Do the M-step update for the means of each component."""
        resp = self._resp[:, :data.shape[0]]
        sumresp = resp.sum(axis=1)
        if np.any(sumresp == 0):
            _print_now("WARNING: A cluster got assigned 0 responsibility.")
            sumresp[sumresp == 0] = 1.
        num = np.dot(resp, data)
        den = sumresp[:, np.newaxis]
        
        if pcounts > 0:
            num += pcounts * pcmean[np.newaxis, :]
            den += pcounts
        
        # This should avoid copying while doing the division
        means = num
        means /= den
        
        if self._updating_all():
            self._means[...] = means
        else:
            self._means[self._update, ...] = means[self._update, ...]
    

    def _m_step_update_covariances(self, data, pcounts=0):
        """Do the M-step update for the covariances."""
        resp = self._resp[:, :data.shape[0]]
        sumresp = resp.sum(axis=1)
        means = self._means
        meansub = data[np.newaxis, ...] - means[:, np.newaxis, ...]
        meansub2 = np.array(meansub)
        meansub *= resp[..., np.newaxis]
        for clust in xrange(self._ncomponent()):
            if self._update[clust]:
                xmmu = meansub[clust, ...]
                xmmu2 = meansub2[clust, ...]
                #  TODO: Optimize pseudocount addition
                newsigma = (np.dot(xmmu.T, xmmu2) + \
                    pcounts * \
                    np.eye(self._ndim)) / \
                    (sumresp[clust] + pcounts)
                
                self._covariances[clust, ...] = newsigma
    

    def _m_step_update_logalpha(self, data):
        """Do the M-step update for the log prior."""
        sumresp = self._resp[:, :data.shape[0]].sum(axis=1)
        self._logalpha = np.log(sumresp) - np.log(sumresp.sum())
        
        # Any infinite quantities
        self._logalpha[np.isinf(self._logalpha)] = _MIN_LOGPROB
        
        # Renormalize
        alpha = np.exp(self._logalpha)
        self._logalpha = np.log(alpha) - np.log(np.sum(alpha))
    

    def EM(self, data, thresh=1e-6, pcounts=0, plotiter=50, hardlimit=2000):
        """Do expectation-maximization to fit the model parameters."""
        
        if len(self._history) == 0:
            lprev = 0
            lcurr = -np.inf
        else:
            lprev = -np.inf
            lcurr = self._history[-1]
        
        count = len(self._history)
        
        if pcounts > 0:
            pcmean = data.mean(axis=0)
        else:
            pcmean = None
        
        while np.abs(lprev - lcurr) > thresh and count < hardlimit:
            if plotiter != None and count > 0 and count % plotiter == 0:
                plot_progress(self._history)
            
            if _DEBUG:
                self._oldresp = self._resp[:]
                self._oldlogalpha = self._logalpha[:]
                self._oldmeans = self._means[:]
                self._oldcov = self._covariances.copy()
            
            # Update mixture parameters
            self.m_step(data, pcounts, pcmean)
            
            # Save previous likelihood
            lprev = lcurr
            
            # Generate posterior over memberships, and current
            # (expected) complete log likelihood
            lcurr = self.e_step(data)
            
            if lcurr < lprev:
                _print_now("Likelihood went down!")
            
            self._history.append(lcurr)
            
            count += 1
            _print_now("%5d: L = %10.5f" % (count, lcurr))
            
            # Plot progress of the optimization
            if plotiter != None and count > 0 and count % plotiter == 0:
                plot_progress(self._history)
    
    def display(self, covariances=None, logalpha=None, name=None, figurenum=1):
        """Display covariances and alphas with matplotlib."""
        if PLOTTING_AVAILABLE:
            try:
                if covariances == None:
                    covariances = self._covariances
            
                if logalpha == None:
                    logalpha = self._logalpha
            
                if name == None:
                    name = "mixture"
                
                # Save state of interactive mode and turn it off.
                istate = pyplot.isinteractive()
                pyplot.ioff()
            
                # Create figure instance.
                pyplot.figure(figurenum)
                pyplot.clf()
            
                # Calculate necessary subplot dimensions for a roughly square 
                rows = np.floor(np.sqrt(self._ncomponent()))
                cols = np.ceil(np.sqrt(self._ncomponent()))
            
                # Add one more row if we have a perfect square, for the 
                # preferences bar chart
                while rows * cols <= self._ncomponent():
                    rows = rows + 1
            
                for clust in xrange(self._ncomponent()):    
                    pyplot.subplot(rows, cols, clust + 1)
                    self._display_covariance_matrix(clust)
                    title = 'Covariance for %s component %d' % (name, clust + 1)
                    pyplot.title(title, fontsize='small')
                    pyplot.colorbar()
            
                # Plot a bar graph of the mixing proportions in an extra subplot
                pyplot.subplot(rows, cols, self._ncomponent() + 1)
            
                # Do the bar graph and make it look reasonable
                pyplot.bar(
                    bottom=np.zeros(self._ncomponent()), 
                    left=np.arange(1, self._ncomponent() + 1) - 0.5,
                    height=np.exp(self._logalpha),
                    width=1
                )
            
                # xticks for each bar in the bar graph.
                pyplot.xticks(np.arange(1, self._ncomponent() + 1))
                pyplot.title('Mixing proportions')
            
                # Display figure.
                pyplot.show()
            finally:
                # Restore previous interactivity state.
                pyplot.interactive(istate)
    
    def _display_covariance_matrix(self, clust):
        pyplot.matshow(self._covariances[clust, ...], fignum=False)
    

class DiagonalGaussianMixture(GaussianMixture):
    """
    A diagonal Gaussian mixture (i.e. all components) are restricted
    to have diagonal covariance matrices).
    """
    def _allocate_covariances(self, ncomponent, ndim):
        """
        Allocate space for the model covariances. In the diagonal case, 
        this requires only ndim elements.
        """
        return np.empty((ncomponent, ndim))
    
    def _m_step_update_covariances(self, data, pcounts):
        # TODO: reduce code duplication
        resp = self._resp[:, :data.shape[0]]
        sumresp = resp.sum(axis=1)
        means = self._means
        meansub = data[np.newaxis, ...] - means[:, np.newaxis, ...]
        meansub2 = np.array(meansub)
        meansub *= resp[..., np.newaxis]
        for clust in xrange(self._ncomponent()):
            if self._update[clust]:
                xmmu = meansub[clust, ...]
                xmmu2 = meansub2[clust, ...]
                #  TODO: Optimize pseudocount addition
                newsigma = (np.dot(xmmu.T, xmmu2) + \
                    pcounts * \
                    np.eye(self._ndim)) / \
                    (sumresp[clust] + pcounts)
                
                self._covariances[clust, ...] = np.diag(newsigma)

    def log_probs(self, clust, data):
        """
        Log probability of each training data point under a particular
        mixture component.
        """
        logphat = data - self._means[clust, ...][np.newaxis, ...]
        logphat **= 2.
        logphat /= self._covariances[clust, ...][np.newaxis, :]
        
        logphat *= -0.5
        logphat = logphat.sum(axis=1)
        
        # Determinant is the product of the diagonal covariance matrix
        # but we want log determinant anyway, so take the log and sum.
        ndim = self._ndim
        lognormalizer = ndim / 2.0 * _LOG2PI + 0.5 * \
                np.log(self._covariances[clust, ...]).sum()

        return logphat - lognormalizer
        
    def _display_covariance_matrix(self, clust):
        pyplot.matshow(np.diag(self._covariances[clust, ...]), fignum=False)

