"""Okay. This is a docstring."""
import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import pdb

MIN_LOGPROB = np.finfo(np.float64).min
MIN_POSTERIOR = 1.0e-8

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
    if len(logx.shape) < 2:  #only one possible dimension to sum over?
        xmax = logx.max()
        return xmax + np.log(np.sum(np.exp(logx-xmax)))
    else:
        if axis != -1:
            logx = logx.transpose(range(axis) + range(axis+1, 
                len(logx.shape)) + [axis])
        lastdim = len(logx.shape)-1
        xmax = logx.max(lastdim)
        return xmax + np.log(np.sum(np.exp(logx-xmax[..., np.newaxis]),
            lastdim))


def plot_progress(lhistory):
    """Plot the progress with matplotlib (if available)."""
    if PLOTTING_AVAILABLE:
        pyplot.figure(2)
        pyplot.ioff()
        pyplot.clf()
        pyplot.plot(np.arange(len(lhistory)), lhistory)
        pyplot.title('Log likelihood vs. iterations')
        pyplot.ylabel('Log likelihood')
        pyplot.xlabel('Iterations')
        pyplot.show()
        pyplot.ion()


class GaussianMixture:
    """Basic Gaussian mixture model."""
    def __init__(self, ncomponent, data, diagonal=False, update=None, 
        pseudocounts=0, prior=1, cache=True):
        """Constructor. FIXME: Add real documentation."""
        self._ncomponent = ncomponent
        self._data = data
        self._ntrain = data.shape[0]
        self._ndim = data.shape[1]
        self._diagonal = diagonal
        self._cache = cache
        self._pseudocounts = pseudocounts
        self._prior = prior
        
        if update == None:
            self._update = np.ones((self._ncomponent, ), dtype=bool)
        elif len(update) != ncomponent:
            raise ValueError("'update' argument must be K elements long")
        else:
            self._update = np.asarray(update, dtype=bool)
    
        # Make space for our model parameters
        means = np.empty((self._ncomponent, self._ndim))
        precision = np.empty((self._ndim, self._ndim, self._ncomponent))
        memberships = np.empty((self._ncomponent))
    
        # Initialize them
        self._logalpha = memberships
        self._precision = precision
        self._means = means
        self._responsibilities = np.empty((self._ncomponent, self._ntrain))
    
        # Count number of parameters in the model for aic/bic
        if self._diagonal:
            covparams = self._ndim
        else:
            covparams = self._ndim * (self._ndim + 1) / 2.
        meanparams = self._ndim
        self._numparams = self._ncomponent * (covparams + meanparams + \
            (self._ncomponent - 1))
    
        # Saving last iteration parameters for debugging
        self._oldmeans, self._oldprecision, self._oldlogalpha = \
            None, None, None
        self._oldresponsibilities = None
        
        # Do a stochastic hard-assignment E-step
        self._random_e_step()
    
    
    def _updating_all(self):
        """
        Quick check to see whether we're updating every mixture component
        so that we can vectorize certain M-step operations more.
        """
        return np.all(self._update)
    
    
    def aic(self):
        """Akaike information criterion for the current model fit."""
        return 2 * self._numparams - 2 * self.loglikelihood()
    
    
    def bic(self):
        """Bayesian information criterion for the current model fit."""
        params = self._numparams
        ntrain = self._ntrain
        return -2 * self.loglikelihood() +  params * np.log(ntrain)
    
    
    def _random_e_step(self):
        """
        Randomly do a hard assignment to a particular component of
        the mixture.
        """
        fakepostidx = random.random_integers(self._ncomponent,
            size=self._ntrain) - 1
        fakeposterior = np.zeros((self._ncomponent, self._ntrain))
        fakeposterior[fakepostidx, xrange(self._ntrain)] = 1
        self._responsibilities = fakeposterior
    
    
    def log_probs(self, clust):
        """
        Log probability of each training data point under a particular
        mixture component.
        """
        centered = self._data - self._means[clust, :][np.newaxis, :]
        ndim = self._ndim
        #pdb.set_trace()
        logphat = -0.5 * (centered * np.dot(centered,
            self._precision[:, :, clust])).sum(axis=1)
        lognormalizer = ndim/2.0 * np.log(2*np.pi) - 0.5 * \
            np.log(linalg.det(self._precision[:, :, clust]))
        return logphat - lognormalizer
    
    def _logjoint(self):
        """
        Function that computes the log joint probability of each training
        example.
        """
        logalpha = self._logalpha
        condprobs = np.nan * np.zeros((self._ncomponent, self._ntrain))
        # calculate logjoint
        for clust in xrange(self._ncomponent):
            condprobs[clust, :] = self.log_probs(clust)
        condprobs += logalpha[:, np.newaxis]
        lik = logsumexp(condprobs, axis=0)
        return condprobs, lik
    
    
    def e_step(self):
        """
        Compute expected value of hidden variables under the posterior 
        as specified by the current model parameters.
        """
        logjoint, lik = self._logjoint()
        self._responsibilities = np.exp(logjoint - lik)
    
    def loglikelihood(self):
        """
        Log likelihood of the training data.
        """
        lik = self._logjoint()[1]
        return lik.sum()
    
    
    def m_step(self):
        """
        Maximize the model parameters with respect to the expected
        complete log likelihood.
        """        
        resp = self._responsibilities
        sumresp = resp.sum(axis=1)
        if np.any(sumresp == 0):
            print "WARNING WARNING WARNING"
            pdb.set_trace()
            sumresp[sumresp == 0] = 1.
        
        data = self._data
        
        means = np.dot(resp, data) / sumresp[:, np.newaxis]
        
        ncomponent = resp.shape[0]
        
        meansub = data[:, :, np.newaxis] - means.T[np.newaxis, :, :]
        meansub2 = np.array(meansub)
        meansub *= resp.T[:, np.newaxis, :]
        
        
        for clust in xrange(ncomponent):
            if self._update[clust]:
                xmmu = meansub[:, :, clust]
                xmmu2 = meansub2[:, :, clust]
                newsigma = (np.dot(xmmu.T, xmmu2) + \
                    self._pseudocounts * self._prior * np.eye(self._ndim)) / \
                    (sumresp[clust] + self._pseudocounts)
                if self._diagonal:
                    diag = np.diag(newsigma)
                    newsigma[:, :] = 0.
                    newsigma[xrange(self._ndim), xrange(self._ndim)] = diag
                try:
                    newprecision = linalg.inv(newsigma)
                    self._precision[:, :, clust] = newprecision
                except linalg.LinAlgError:
                    print "Failed to invert %d, cond=%f" % (clust,
                        linalg.cond(newsigma))
                    pyplot.figure(2)
                    pyplot.matshow(newsigma)
        if self._updating_all():
            self._means = means
        else:
            self._means[self._update, :] = means[self._update, :]
        
        self._logalpha = np.log(sumresp) - np.log(sumresp.sum())
        self._logalpha[np.isinf(self._logalpha)] = MIN_LOGPROB
        # Renormalize
        alpha = np.exp(self._logalpha)
        self._logalpha = np.log(alpha) - np.log(np.sum(alpha))
    
    
    def EM(self, thresh=1e-10, plotiter=50, hardlimit=2000):
        """Do expectation-maximization to fit the model parameters."""
        self.m_step()
        lhistory = []
        lprev = -np.inf
        lcurr = self.loglikelihood()
        count = 0
        while np.abs(lprev - lcurr) > thresh and count < hardlimit:
            if plotiter != None and count > 0 and count % plotiter == 0:
                plot_progress(lhistory)
            
            self._oldresponsibilities = self._responsibilities[:]
            self._oldlogalpha = self._logalpha[:]
            self._oldmeans = self._means[:]
            self._oldprecision = self._precision.copy()
            
            print self._update
            
            # Generate posterior over memberships
            self.e_step()
            
            
            # Update mixture parameters
            self.m_step()
            
            print ((self._precision -
                self._oldprecision)**2).sum(axis=0).sum(axis=0)
            
            lprev = lcurr
            lcurr = self.loglikelihood()
            
            if len(lhistory) > 0 and lcurr < lprev:
                print "Likelihood went down!"
                pdb.set_trace()
            
            count += 1
            print "%5d: L = %10.5f" % (count, lcurr)
            self.display()
            if lcurr > 0:
                pass
                #pdb.set_trace()
            lhistory.append(lcurr)
        return lhistory
    
    
    def save(self, filename):
        """Save to a .npz file."""
        np.savez(filename, ncomponent=np.array(self._ncomponent), 
            ntrain=np.array(self._ntrain),
            ndim=np.array(self._ndim), precision=self._precision, 
            means=self._means, logalpha=self._logalpha, 
            diagonal=np.array(self._diagonal),
            aic=np.array(self.aic()), bic=np.array(self.bic()),
            update=self._update)
    
    def display(self):
        """Display covariances and alphas with matplotlib."""
        if PLOTTING_AVAILABLE:
            pyplot.ioff()
            pyplot.figure(1)
            pyplot.clf()
            rows = cols = np.ceil(np.sqrt(self._ncomponent))
            for clust in range(self._ncomponent):    
                pyplot.subplot(rows, cols, clust+1)
                pyplot.matshow(self._precision[:, :, clust], fignum=False)
                pyplot.title('Precision for component %d' % clust,
                    fontsize='small')
                pyplot.colorbar()
            pyplot.subplot(rows, cols, self._ncomponent+1)
            pyplot.bar(bottom=np.zeros(self._ncomponent), left=np.arange(1, 
                self._ncomponent+1)-0.5, height=np.exp(self._logalpha),
                    width=1)
            pyplot.show()
            pyplot.ion()
    

class GaussianMixtureWithGarbageModel(GaussianMixture):
    """
    A Gaussian mixture model that has one extra component with parameters
    equal to the mean and covariance of the data and is fixed.
    """
    def __init__(self, ncomponent, data, *args, **keywords):
        """Initialize the model."""
        update = np.ones((ncomponent+1, ), dtype=bool)
        update[0] = False
        if 'update' in keywords:
            if len(keywords['update']) != ncomponent:
                raise ValueError("'update' argument must be K elements long")
            update[1:] = keywords['update']
        keywords['update'] = update
        GaussianMixture.__init__(self, ncomponent + 1, data, *args, **keywords)
        self._train_data_precision = linalg.inv(np.cov(data, rowvar=False))
        self._precision[:, :, 0] = self._train_data_precision
        self._means[0, :] = np.mean(data, axis=0)
    
