import numpy as np
import numpy.linalg as linalg
import numpy.random as random
#import ipdb

print "pants"
MIN_LOGPROB = np.finfo(np.float64).min
MIN_POSTERIOR = 1.0e-8


try:
    import matplotlib.pyplot as pyplot

except ImportError:
    pyplot = None

def logsumexp(x, axis=-1):
    """
    Compute log(sum(exp(x))) in a numerically stable way.  
    Use second argument to specify along which dimensions the logsumexp
    shall be computed. If -1 (which is also the default), logsumexp is 
    computed along the last dimension. 
    
    By Roland Memisevic, distributed under Python License.
    """
    if len(x.shape) < 2:  #only one possible dimension to sum over?
        xmax = x.max()
        return xmax + np.log(np.sum(np.exp(x-xmax)))
    else:
        if axis != -1:
            x = x.transpose(range(axis) + range(axis+1, len(x.shape)) + [axis])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + np.log(np.sum(np.exp(x-xmax[...,np.newaxis]),lastdim))

class GaussianMixture:
    def __init__(self, K, data, diagonal=False, update=None, 
        pseudocounts=0.01, prior=0.2, cache=True):
        """Constructor."""
        self._K = K
        self._data = data
        self._N = data.shape[0]
        self._D = data.shape[1]
        self._diagonal = diagonal
        self._cache = cache
        self._pseudocounts = 0.01
        self._prior = 0.2
        self._logj_cache = None
        self._lik_cache = None
        if update == None:
            self._update = np.ones((self._K,),dtype=bool)
        elif len(update) != K:
            raise ValueError("'update' argument must be K elements long")
        else:
            self._update = np.asarray(update, dtype=bool)
        self._initialize()
        
    def _updating_all(self):
        return np.all(self._update)
    
    def _initialize(self):
        """
        Allocate space for the model parameters & initialize, perform
        a random E step.
        """
        K, D, N = self._K, self._D, self._N
        
        # Make space for our model parameters
        means = np.empty((K,D))
        precision = np.empty((D,D,K))
        memberships = np.empty((K))
        
        # Initialize them
        self._logalpha = memberships
        self._precision = precision
        self._mu = means
        self._responsibilities = np.empty((K,N))

        # TODO: Implement stale-checking/updating
        self._stale = True

        # Count number of parameters in the model for AIC/BIC
        if self._diagonal:
            covparams = D
        else:
            covparams = D * (D + 1) / 2.
        meanparams = D
        self._numparams = K * (covparams + meanparams + (K - 1))

        # Do a stochastic hard-assignment E-step
        self._randomEstep()
    
    
    def AIC(self):
        """Akaike information criterion for the current model fit."""
        return 2 * self._numparams - 2 * self.loglikelihood()
    
    
    def BIC(self):
        """Bayesian information criterion for the current model fit."""
        P = self._numparams
        N = self._N
        return -2 * self.loglikelihood() +  P * np.log(N)
    
    
    def _randomEstep(self):
        """
        Randomly do a hard assignment to a particular component of
        the mixture.
        """
        fakepostidx = random.random_integers(self._K,size=self._N) - 1
        fakeposterior = np.zeros((self._K,self._N))
        fakeposterior[fakepostidx,xrange(self._N)] = 1
        self._responsibilities = fakeposterior
    
    
    def log_probs(self, cluster):
        """
        Log probability of each training data point under a particular
        mixture component.
        """
        centered = self._data - self._mu[cluster,:][np.newaxis,:]
        D = self._D
        logphat = -0.5 * (centered * np.dot(centered,
            self._precision[:,:,cluster])).sum(axis=1)
        lognormalizer = -D/2.0 * np.log(2*np.pi) + 0.5 * \
            np.log(linalg.det(self._precision[:,:,cluster]))
        return logphat - lognormalizer
    
    def _logjoint(self):
        """
        Function that computes the log joint probability of each training
        example.
        """
        logalpha = self._logalpha
        precision = self._precision
        mu = self._mu
        K = logalpha.shape[0]
        N = self._data.shape[0]
        p = np.nan * np.zeros((K,N))
        # calculate logjoint
        for cluster in xrange(logalpha.shape[0]):
            p[cluster,:] = self.log_probs(cluster)
        p += logalpha[:,np.newaxis]
        B = 0
        lik = logsumexp(p,axis=0)
        self._logj_cache = p
        self._lik_cache = p
        self._stale = False
        return p, lik
    
    
    def Estep(self):
        """
        Compute expected value of hidden variables under the posterior 
        as specified by the current model parameters.
        """
        data = self._data
        logjoint,lik = self._logjoint()
        #else:
        #logjoint,lik = dists
        self._responsibilities = np.exp(logjoint - lik)
    
    def loglikelihood(self):
        """
        Log likelihood of the training data.
        """
        logjoint,lik = self._logjoint()
        return lik.sum()
    
    
    def Mstep(self, pseudocounts=0.01, prior=0.1):
        """
        Maximize the model parameters with respect to the expected
        complete log likelihood.
        """
        q = self._responsibilities
        sumq = q.sum(axis=1)
        sumq[sumq == 0] = 1.
        data = self._data
        mu = np.dot(q, data) / sumq[:, np.newaxis]
        if np.any(np.isnan(mu)):
            ipdb.set_trace()
        K = q.shape[0]
        meansub = data[:,:,np.newaxis] - mu.T[np.newaxis,:,:]
        meansub2 = np.array(meansub)
        meansub *= q.T[:,np.newaxis,:]
        for cluster in xrange(K):
            if self._updating_all() or self._update[cluster]:
                xmmu = meansub[:,:,cluster] * q[cluster,:][:,np.newaxis]
                xmmu2 = meansub2[:,:,cluster]
                newsigma = np.dot(xmmu.T, xmmu2) / sumq[cluster]
                if self._diagonal:
                    diag = np.diag(newsigma)
                    newsigma[:,:] = 0.
                    newsigma[xrange(self._D),xrange(self._D)] = diag
                if pseudocounts != None:
                    self._regularize(newsigma, pseudocounts, self._prior)
                try:
                    self._precision[:,:,cluster] = linalg.inv(newsigma)
                except linalg.LinAlgError:
                    print "Failed to invert, cond=%f" % cond(newsigma)
                    pass
        if self._updating_all():
            self._mu = mu
        else:
            self._mu[self._update,:] = mu[self._update,:]
        self._logalpha = np.log(sumq) - np.log(sumq.sum())
        self._logalpha[np.isinf(self._logalpha)] = MIN_LOGPROB
    
    
    def _regularize(self, cov, weight, prior):
        denom = (1. + weight)
        cov *= 1. / denom
        cov.flat[::(self._D+1)] += prior * weight / denom
    
    def plot_progress(self,Ls):
        """Plot the progress with matplotlib (if available)."""
        if pyplot:
            pyplot.ioff()
            pyplot.clf()
            pyplot.plot(np.arange(len(Ls)),Ls)
            pyplot.title('Log likelihood vs. iterations')
            pyplot.ylabel('Log likelihood')
            pyplot.xlabel('Iterations')
            pyplot.show()
            pyplot.ion()
    
    def EM(self, thresh=1e-10, plotiter=50, reset=False, hardlimit=2000):
        """Do expectation-maximization to fit the model parameters."""
        data = self._data
        K = self._K
        N,D = data.shape
        if plotiter != None:
            try:
                import matplotlib.pyplot as pyplot
            except ImportError:
                print "matplotlib not available, not plotting"
                plotiter = None
        self.Mstep(pseudocounts=self._pseudocounts)
        Ls = []
        L_old = -np.inf
        L = self.loglikelihood()
        count = 0
        while np.abs(L_old - L) > thresh and count < hardlimit:
            if plotiter != None and count > 0 and count % plotiter == 0:
                self.plot_progress(Ls)
        
            # Generate posterior over memberships
            self.Estep()
            
            # Update mixture parameters
            self.Mstep(pseudocounts=self._pseudocounts)
            
            L_old = L
            L = self.loglikelihood()
            
            if len(Ls) > 0 and L < L_old and not np.allclose(L, L_old):
                print "Likelihood went down!"
            
            count += 1
            print "%5d: L = %10.5f" % (count,L)
            Ls.append(L)
        return Ls
    
    def save(self,filename):
        np.savez(filename, K=np.array(self._K),N=np.array(self._N),
            D=np.array(self._D), precision=self._precision, mu=self._mu,
            logalpha=self._logalpha, diagonal=np.array(self._diagonal),
            AIC=np.array(self.AIC()), BIC=np.array(self.BIC()),
            update=self._update)
    

class GaussianMixtureWithGarbageModel(GaussianMixture):
    """
    A Gaussian mixture model that has one extra component with parameters
    equal to the mean and covariance of the data and is fixed.
    """
    def __init__(self, K, data, *args, **kw):
        update = np.ones((K+1,),dtype=bool)
        update[0] = False
        if 'update' in kw:
            if len(kw['update']) != K:
                raise ValueError("'update' argument must be K elements long")
            update[1:] = kw['update']
        kw['update'] = update
        GaussianMixture.__init__(self, K + 1, data, *args, **kw)
        self._train_data_precision = linalg.inv(np.cov(data, rowvar=False))
        self._precision[:,:,0] = self._train_data_precision
        self._mu[0,:] = np.mean(data, axis=0)
    


        
    # def sample(self, nsamp, component=None, shuffle=False):
    #     """Sample from a mixture model."""
    #     if component == None:
    #         nums = random.multinomial(nsamp,np.exp(params['logalpha']))
    #     else:
    #         nums = np.zeros(len(params['logalpha']),dtype=int)
    #         nums[component] = nsamp
    #     D = params['precision'].shape[0]
    #     samples = np.empty((D,nsamp))
    #     cnt = 0
    #     for cmpt in xrange(len(nums)):
    #         mu = params['mu'][:,cmpt]
    #         precision = params['precision'][:,:,cmpt]
    #         s = cnt
    #         t = cnt + nums[cmpt]
    #         #samples[:,s:t] = random.multivariate_normal(mu, precision, (nums[cmpt],)).T
    #         cnt = t
    #     if shuffle:
    #         samples = np.asarray(samples[:,np.random.shuffle(
    #             np.arange(nsamp))])
    #     return samples
