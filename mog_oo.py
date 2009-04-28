"""Okay. This is a docstring."""
import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import pdb

FINFO = np.finfo(np.float64)
MIN_PROB = FINFO.tiny
MIN_LOGPROB = np.log(MIN_PROB)

LOOKBACK = 5
DEBUG_LIKELIHOOD_DECREASE = False


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
        return xmax + np.log(np.sum(np.exp( logx-xmax[..., np.newaxis]),
            lastdim))


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


class GaussianMixture:
    """Basic Gaussian mixture model."""
    def __init__(self, ncomponent, data, update=None, pseudocounts=0):
        """
        Constructor. 
        Add real documentation.
        """
        # We use these a lot below, so cache them for readability
        ntrain, ndim = data.shape
        
        self._data = data
        self._pseudocounts = pseudocounts
        
        # Handle the 'update' array, which tells us whether or not we 
        # update a given component's parameters by M-step estimation.
        if update == None:
            self._update = np.ones((ncomponent, ), dtype=bool)
        elif len(update) != ncomponent:
            raise ValueError("len(update) must be equal to ncomponent")
        else:
            self._update = np.asarray(update, dtype=bool)
        
        # Make space for our model parameters
        means = np.empty((ncomponent, ndim))
        precision = np.empty((ndim, ndim, ncomponent))
        memberships = np.empty((ncomponent))
        resp = np.empty((ncomponent, ntrain))
        
        # Initialize them
        self._logalpha = memberships
        self._precision = precision
        self._means = means
        self._resp = resp
        
        # Store the training data mean for pseudocount shrinkage
        self._trainmean = np.mean(self._data)
        
        # For saving last iteration parameters for debugging
        self._oldmeans, self._oldprecision, self._oldlogalpha = \
            None, None, None
        self._oldresponsibilities = None
        
        # Do a stochastic hard-assignment E-step
        self._random_e_step()    
    
    
    def _ntrain(self):
        """Return the number of examples in the training set."""
        return self._data.shape[0]
    
    
    def _ndim(self):
        """Return the number of dimensions in the training set."""
        return self._data.shape[1]
    
    
    def _ncomponent(self):
        """Return the number of components in the mixture."""
        return self._logalpha.shape[0]
    
    
    def _numparams(self):
        """Return the number of free parameters in the model."""
        covparams = self._ndim() * (self._ndim() + 1) / 2.
        meanparams = self._ndim()
        return self._ncomponent() * (covparams + meanparams + \
            (self._ncomponent() - 1))
    
     
    def _updating_all(self):
        """
        Quick check to see whether we're updating every mixture component
        so that we can vectorize certain M-step operations more.
        """
        return np.all(self._update)
    
    
    def aic(self):
        """Akaike information criterion for the current model fit."""
        return 2 * self._numparams() - 2 * self.loglikelihood()
    
    
    def bic(self):
        """Bayesian information criterion for the current model fit."""
        params = self._numparams()
        ntrain = self._ntrain()
        return -2 * self.loglikelihood() +  params * np.log(ntrain)
    
    
    def _random_e_step(self):
        """
        Randomly do a hard assignment to a particular component of
        the mixture.
        """
        fakepostidx = random.random_integers(self._ncomponent(),
            size=self._ntrain()) - 1
        fakeposterior = np.zeros((self._ncomponent(), self._ntrain()))
        fakeposterior[fakepostidx, xrange(self._ntrain())] = 1
        self._resp = fakeposterior
    
    
    def log_probs(self, clust):
        """
        Log probability of each training data point under a particular
        mixture component.
        """
        centered = self._data - self._means[clust, :][np.newaxis, :]
        ndim = self._ndim()
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
        condprobs = np.nan * np.zeros((self._ncomponent(), self._ntrain()))
        for clust in xrange(self._ncomponent()):
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
        logresp = logjoint - lik
        logresp[logresp < MIN_LOGPROB] = MIN_LOGPROB
        self._resp = np.exp(logresp)
    
    
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
        
        self._m_step_update_means()
        
        self._m_step_update_precisions()
        
        self._m_step_update_logalpha()
    
    
    def _m_step_update_means(self):
        """Do the M-step update for the means of each component."""
        resp = self._resp
        data = self._data
        sumresp = resp.sum(axis=1)
        if np.any(sumresp == 0):
            print "WARNING: A cluster got assigned 0 responsibility."
            sumresp[sumresp == 0] = 1.
        data = self._data
        means = np.dot(resp, data) / sumresp[:, np.newaxis]
        if self._updating_all():
            self._means[:, :] = means
        else:
            self._means[self._update, :] = means[self._update, :]
    
    
    def _m_step_update_precisions(self):
        """Do the M-step update for the precisions (inverse covariances)."""
        resp = self._resp
        sumresp = resp.sum(axis=1)
        means = self._means
        meansub = self._data[:, :, np.newaxis] - means.T[np.newaxis, :, :]
        meansub2 = np.array(meansub)
        meansub *= self._resp.T[:, np.newaxis, :]
        for clust in xrange(self._ncomponent()):
            if self._update[clust]:
                xmmu = meansub[:, :, clust]
                xmmu2 = meansub2[:, :, clust]
                #  TODO: Optimize pseudocount addition
                newsigma = (np.dot(xmmu.T, xmmu2) + \
                    self._pseudocounts * \
                    np.eye(self._ndim())) / \
                    (sumresp[clust] + self._pseudocounts)
                try:
                    newprecision = linalg.inv(newsigma)
                    if np.isnan(np.log(linalg.det(newprecision))) or \
                    np.isinf(np.log(linalg.det(newprecision))):
                        print "NEW PRECISION DETERMINANT:" + \
                            str(linalg.det(newprecision))
                    else:
                        self._precision[:, :, clust] = newprecision
                except linalg.LinAlgError:
                    print "Failed to invert %d, cond=%f" % (clust,
                        linalg.cond(newsigma))
                    pyplot.figure(2)
                    pyplot.matshow(newsigma)
    
    
    def _m_step_update_logalpha(self):
        """Do the M-step update for the log prior."""
        sumresp = self._resp.sum(axis=1)
        self._logalpha = np.log(sumresp) - np.log(sumresp.sum())
        
        # Any infinite quantities
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
            
            self._oldresponsibilities = self._resp[:]
            self._oldlogalpha = self._logalpha[:]
            self._oldmeans = self._means[:]
            self._oldprecision = self._precision.copy()
            
            # Generate posterior over memberships
            self.e_step()
            
            # Update mixture parameters
            self.m_step()
            
            lprev = lcurr
            lcurr = self.loglikelihood()
            
            if lcurr < lprev:
                print "Likelihood went down!"
                if DEBUG_LIKELIHOOD_DECREASE:
                    self.display(name="OLD", figurenum=3,
                        precision=self._oldprecision, 
                        logalpha=self._oldlogalpha)
                    self.display(name="CURRENT", figurenum=4)
                    pdb.set_trace()
            
            count += 1
            print "%5d: L = %10.5f" % (count, lcurr)
            lhistory.append(lcurr)
        return lhistory
    
    
    def save(self, filename):
        """Save to a .npz file."""
        np.savez(filename, ncomponent=np.array(self._ncomponent()), 
            ntrain=np.array(self._ntrain()),
            ndim=np.array(self._ndim()), precision=self._precision, 
            means=self._means, logalpha=self._logalpha, 
            #diagonal=np.array(self._diagonal),
            aic=np.array(self.aic()), bic=np.array(self.bic()),
            update=self._update)
    
    
    def display(self, precision=None, logalpha=None, name=None, figurenum=1):
        """Display covariances and alphas with matplotlib."""
        if PLOTTING_AVAILABLE:
            if precision == None:
                precision = self._precision
            if logalpha == None:
                precision = self._logalpha
            if name == None:
                name = "mixture"
            
            pyplot.ioff()
            pyplot.figure(figurenum)
            pyplot.clf()
            rows = cols = np.ceil(np.sqrt(self._ncomponent()))
            if rows * cols == self._ncomponent():
                rows = rows + 1
            
            for clust in xrange(self._ncomponent()):    
                pyplot.subplot(rows, cols, clust+1)
                pyplot.matshow(self._precision[:, :, clust], fignum=False)
                pyplot.title('Precision for %s component %d' % (name, clust),
                    fontsize='small')
                pyplot.colorbar()
            
            pyplot.subplot(rows, cols, self._ncomponent()+1)
            pyplot.bar(bottom=np.zeros(self._ncomponent()), left=xrange(1, 
                self._ncomponent()+1)-0.5, height=np.exp(self._logalpha),
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
        update = np.ones((ncomponent+1,), dtype=bool)
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
    


class DiagonalGaussianMixture(GaussianMixture):
    """
    A Gaussian mixture model that has one extra component with parameters
    equal to the mean and covariance of the data and is fixed.
    """
    def m_step(self):
        """M-step for a diagonal Gaussian  model."""
        GaussianMixture.m_step(self)
        #for clust in xrange(self._ncomponent()):
    
