import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import ipdb

DETERMINANT_LIMIT = 1.0e-1

try:
    import matplotlib.pyplot as pyplot
except ImportError:
    pyplot = None

class GaussianMixture:
    def __init__(self, K, data):
        self._K = K
        self._data = data
        self._N = data.shape[0]
        self._D = data.shape[1]
        self._diagonal = False
        self._initialize()
        
    def _initialize(self):
        K, D, N = self._K, self._D, self._N
        means = np.empty((K,D))
        covariances = np.empty((D,D,K))
        memberships = np.empty((K))
        self._logalpha = memberships
        self._sigma = covariances
        self._mu = means
        self._responsibilities = np.empty((K,N))
        self._stale = True
        self._randomEstep()
    
    def _randomEstep(self):
        fakepostidx = random.random_integers(self._K,size=self._N) - 1
        fakeposterior = np.zeros((self._K,self._N))
        fakeposterior[fakepostidx,xrange(self._N)] = 1
        self._responsibilities = fakeposterior
    
    def log_probs(self, cluster):
        centered = self._data - self._mu[cluster,:][np.newaxis,:]
        D = self._D
        sigmainv = linalg.inv(self._sigma[:,:,cluster])
        logphat = -0.5 * (centered * np.dot(centered, sigmainv)).sum(axis=1)
        lognormalizer = D/2.0 *np.log(2*np.pi) + 0.5 * \
            np.log(linalg.det(self._sigma[:,:,cluster]))
        return logphat - lognormalizer
    
    def _logjoint(self):
        logalpha = self._logalpha
        sigma = self._sigma
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
        sigma = np.empty(self._sigma.shape)
        for cluster in xrange(K):
            xmmu = meansub[:,:,cluster] * q[cluster,:][:,np.newaxis]
            xmmu2 = meansub2[:,:,cluster]
            newsigma = np.dot(xmmu.T, xmmu2)
            if linalg.det(newsigma) > DETERMINANT_LIMIT:
                sigma[:,:,cluster] = newsigma
            else:
                # What should we do? Reset to covariance of the data?
                # Regularize?
                pass
            if self._diagonal:
                sigma[:,:,cluster] = np.diag(np.diag(sigma[:,:,cluster]))
        
        sigma /= sumq[np.newaxis, np.newaxis, :]
        self._sigma = sigma
        self._mu = mu
        self._logalpha = np.log(sumq / sumq.sum())

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
        L = 0
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
            
    def sample(self, nsamp, component=None, shuffle=False):
        """Sample from a mixture model."""
        if component == None:
            nums = random.multinomial(nsamp,np.exp(params['logalpha']))
        else:
            nums = np.zeros(len(params['logalpha']),dtype=int)
            nums[component] = nsamp
        D = params['sigma'].shape[0]
        samples = np.empty((D,nsamp))
        cnt = 0
        for cmpt in xrange(len(nums)):
            mu = params['mu'][:,cmpt]
            sigma = params['sigma'][:,:,cmpt]
            s = cnt
            t = cnt + nums[cmpt]
            #samples[:,s:t] = random.multivariate_normal(mu, sigma, (nums[cmpt],)).T
            cnt = t
        if shuffle:
            samples = np.asarray(samples[:,np.random.shuffle(
                np.arange(nsamp))])
        return samples
