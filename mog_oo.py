import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import ipdb

MIN_LOGPROB = np.finfo(np.float64).min

try:
    import matplotlib.pyplot as pyplot

except ImportError:
    pyplot = None


class GaussianMixture:
    def __init__(self, K, data, diagonal=False, update=None):
        """Constructor."""
        self._K = K
        self._data = data
        self._N = data.shape[0]
        self._D = data.shape[1]
        self._diagonal = diagonal
        if update == None:
            self._update = np.ones((self._K,),dtype=bool)
        elif len(update) != K:
            raise ValueError("'update' argument must be K elements long")
        else:
            self._update = np.asarray(update, dtype=bool)
        self._updating_all = np.all(self._update)
        self._initialize()
        
    
    def _initialize(self):
        """Do the initialization."""
        K, D, N = self._K, self._D, self._N
        
        # Make space for our model parameters
        means = np.empty((K,D))
        covariances = np.empty((D,D,K))
        memberships = np.empty((K))

        # Initialize them
        self._logalpha = memberships
        self._precision = covariances
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
        return 2 * self._numparams - 2 * self._loglikelihood()
    
    
    def BIC(self):
        """Bayesian information criterion for the current model fit."""
        P = self._numparams
        N = self._N
        return -2 * self._loglikelihood() +  P * np.log(N)
    
    
    def _randomEstep(self):
        fakepostidx = random.random_integers(self._K,size=self._N) - 1
        fakeposterior = np.zeros((self._K,self._N))
        fakeposterior[fakepostidx,xrange(self._N)] = 1
        self._responsibilities = fakeposterior
    
    
    def log_probs(self, cluster):
        centered = self._data - self._mu[cluster,:][np.newaxis,:]
        D = self._D
        logphat = -0.5 * (centered * np.dot(centered, self._precision[:,:,cluster])).sum(axis=1)
        lognormalizer = D/2.0 *np.log(2*np.pi) - 0.5 * \
            np.log(linalg.det(self._precision[:,:,cluster]))
        return logphat - lognormalizer
    
    
    def _logjoint(self):
        logalpha = self._logalpha
        precision = self._precision
        mu = self._mu
        K = logalpha.shape[0]
        N = self._data.shape[0]
        p = np.nan * np.zeros((K,N))
        # calculate logjoint
        for cluster in xrange(logalpha.shape[0]):
            p[cluster,:] = self.log_probs(cluster)
        logjoint = logalpha[:,np.newaxis] + p
        B = 0
        lik = np.log(np.exp(logjoint+B).sum(axis=0))-B
        return logjoint, lik
    
    
    def Estep(self):
        data = self._data
        #if self._dists == None:
        logjoint,lik = self._logjoint()
        #else:
        #logjoint,lik = dists
        self._responsibilities = np.exp(logjoint - lik)
    
    
    def loglikelihood(self):
        logjoint,lik = self._logjoint()
        return lik.sum()
    
    
    def Mstep(self):
        q = self._responsibilities
        sumq = q.sum(axis=1)
        data = self._data
        mu = np.dot(q, data) / sumq[:, np.newaxis]
        #ipdb.set_trace()
        K = q.shape[0]
        meansub = data[:,:,np.newaxis] - mu.T[np.newaxis,:,:]
        meansub2 = np.array(meansub)
        meansub *= q.T[:,np.newaxis,:]
        for cluster in xrange(K):
            if self._updating_all or self._update[cluster]:
                xmmu = meansub[:,:,cluster] * q[cluster,:][:,np.newaxis]
                xmmu2 = meansub2[:,:,cluster]
                newsigma = np.dot(xmmu.T, xmmu2) / sumq[cluster]
                if self._diagonal:
                    diag = np.diag(newsigma)
                    newsigma[:,:] = 0.
                    newsigma[xrange(self._D),xrange(self._D)] = diag
                if linalg.cond(newsigma) < 1.0e14:
                    try:
                        self._precision[:,:,cluster] = linalg.inv(newsigma)
                    except linalg.LinAlgError:
                        pass
                    
            else:
                self._precision[:,:,cluster] = self._precision[:,:,cluster]
        
        self._mu = mu
        self._logalpha = np.log(sumq / sumq.sum())
        self._logalpha[np.isinf(self._logalpha)] = MIN_LOGPROB
    
    
    def plot_progress(self,Ls):
        if pyplot:
            pyplot.ioff()
            pyplot.clf()
            pyplot.plot(np.arange(len(Ls)),Ls)
            pyplot.title('Log likelihood vs. iterations')
            pyplot.ylabel('Log likelihood')
            pyplot.xlabel('Iterations')
            pyplot.show()
            pyplot.ion()
    
    
    def EM(self, thresh=1e-10, plotiter=50, reset=False):
        data = self._data
        K = self._K
        N,D = data.shape
        if plotiter != None:
            try:
                import matplotlib.pyplot as pyplot
            except ImportError:
                print "matplotlib not available, not plotting"
                plotiter = None
        self.Mstep()
        Ls = []
        L_old = -np.inf
        L = self.loglikelihood()
        count = 0
        while np.abs(L_old - L) > thresh:
            if plotiter != None and count > 0 and count % plotiter == 0:
                self.plot_progress(Ls)
        
            # Generate posterior over memberships
            q = self.Estep()
        
            # Update mixture parameters
            self.Mstep()
            
            L_old = L
            L = self.loglikelihood()
            
            if len(Ls) > 0 and L < Ls[-1]:
                print "Likelihood went down!"
            
            count += 1
            print "%5d: L = %10.5f" % (count,L)    
            #ipdb.set_trace()


class GaussianMixtureWithGarbageModel(GaussianMixture):
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
        self._mu[0,:] = np.mean(data,axis=0)
        self._randomEstep()
        
    def _randomEstep(self):
        fakepostidx = random.random_integers(self._K - 1,size=self._N)
        fakeposterior = np.zeros((self._K,self._N))
        fakeposterior[fakepostidx,xrange(self._N)] = 1
        self._responsibilities = fakeposterior

    def Mstep(self):
        GaussianMixture.Mstep(self)
        self._mu[0,:] = np.mean(self._data)
        
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
