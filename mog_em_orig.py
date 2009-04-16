import numpy as np
import numpy.linalg as linalg
import numpy.random as random

def log_gaussian(x, mu, sigma):
    centered = x - mu[:,np.newaxis]
    D = sigma.shape[0]
    sigmainv = linalg.inv(sigma)
    logphat = -0.5 * (centered * np.dot(sigmainv, centered)).sum(axis=0)
    lognormalizer = D/2.0 *np.log(2*np.pi) + 0.5 * np.log(linalg.det(sigma))
    return logphat - lognormalizer

def _logjoint(params, data):
    logalpha = params['logalpha']
    K = logalpha.shape[0]
    N = data.shape[1]
    sigma = params['sigma']
    mu = params['mu']
    p = np.nan * np.zeros((K,N))
    # calculate logjoint
    for cluster in xrange(np.shape(logalpha)[0]):
        p[cluster,:] = log_gaussian(data, mu[:,cluster], sigma[:,:,cluster])
    logjoint = logalpha[:,np.newaxis] + p
    B = 0
    lik = np.log(np.exp(logjoint+B).sum(axis=0))-B
    return logjoint, lik


def Estep(params, data, dists): # data is D x N
    if dists == None:
        logjoint,lik = _logjoint(params, data)
    else:
        logjoint,lik = dists
    return np.exp(logjoint - lik)


def loglikelihood(params, data):
    logjoint,lik = _logjoint(params, data)
    return lik.sum(), (logjoint, lik)

def Mstep(params, data, q, diagonal=False):
    sumq = q.sum(axis=1)
    mu = np.dot(data,q.T) / sumq[np.newaxis, :]
    K = np.shape(q)[0]
    meansub = data[:,:,np.newaxis] - mu[:,np.newaxis,:]
    meansub2 = np.array(meansub)
    meansub *= q.T[np.newaxis,:,:]
    sigma = np.empty(np.shape(params['sigma']))
    for cluster in xrange(K):
        xmmu = meansub[:,:,cluster] * q[cluster,:][np.newaxis,:]
        xmmu2 = meansub2[:,:,cluster]
        sigma[:,:,cluster] = np.dot(xmmu, xmmu2.T)
        if diagonal:
            sigma[:,:,cluster] = np.diag(np.diag(sigma[:,:,cluster]))
    sigma /= sumq[np.newaxis, np.newaxis, :]
    params['sigma'] = sigma
    params['mu'] = mu
    params['logalpha'] = np.log(sumq / sumq.sum())


def initialize_mog(D,K):
    N = D.shape[1]
    fakepostidx = random.random_integers(K,size=N) - 1
    fakeposterior = np.zeros((K,N))
    fakeposterior[fakepostidx,xrange(N)] = 1
    means = np.empty((D,K))
    covariances = np.empty((D,D,K))
    memberships = np.empty((K))
    params = dict(logalpha=memberships,sigma=covariances,mu=means)
    return fakeposterior, params

def plot_progress(Ls,pyplot):
    pyplot.ioff()
    pyplot.clf()
    pyplot.plot(np.arange(len(Ls)),Ls)
    pyplot.title('Log likelihood vs. iterations')
    pyplot.ylabel('Log likelihood')
    pyplot.xlabel('Iterations')
    pyplot.show()
    pyplot.ion()

def EM(data,K=None,params=None,thresh=1e-8,plotiter=50,diagonal=False):
    if plotiter != None:
        try:
            import matplotlib.pyplot as pyplot
        except ImportError:
            print "matplotlib not available, not plotting"
            plotiter = None
    
    D,N = np.shape(data)
    if K == None and params == None:
        raise ValueError('Need either a mixture size or a parameters dict')
    elif params == None:
        fakeposterior, params = initialize_mog(D,K,N)
        Mstep(params, data, fakeposterior, diagonal=diagonal)
    Ls = []
    L,dists = loglikelihood(params, data)
    count = 0
    print "%5d: L = %10.5f (initialized)" % (count,L)
    
    
    while len(Ls) == 0 or np.abs(L - Ls[-1]) > thresh:
        if plotiter != None and count > 0 and count % plotiter == 0:
            plot_progress(Ls,pyplot)
        
        # Generate posterior over memberships
        q = Estep(params, data, dists)
        
        # Update mixture parameters
        Mstep(params, data, q, diagonal=diagonal)
        
        Ls.append(L)
        L, dists = loglikelihood(params, data)
        if L < Ls[-1]:
            print "Likelihood went down!"
        
        count += 1
        print "%5d: L = %10.5f" % (count,L)
        
    Ls.append(L)
    
    return Ls,params

def sample_mog(nsamp, params, component=None, shuffle=False):
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
        samples[:,s:t] = random.multivariate_normal(mu, sigma, (nums[cmpt],)).T
        cnt = t
    if shuffle:
        samples = np.asarray(samples[:,np.random.shuffle(np.arange(nsamp))])
    return samples
