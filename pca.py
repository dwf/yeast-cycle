"""
Contains a class that implements 3 methods for extracting principal 
components.
"""

import numpy as np
import numpy.linalg as linalg
import numpy.random as random

class PrincipalComponents(object):
    """Implements principal components analysis."""
    
    def __init__(self, data, rowvar=False, numcmpt=None):
        """"""
        self._data = data
        self._rowvar = rowvar
        if self._rowvar:
            self._mean = np.mean(data, axis=1)[:, np.newaxis]
            self._std = np.std(data, axis=1)[:, np.newaxis]
        else:
            self._mean = np.mean(data, axis=0)[np.newaxis, :]
            self._std = np.std(data, axis=0)[np.newaxis, :]
        self._eigvec = None
        self._eigval = None
    
    def _direct(self):
        """
        Direct, naive method for computing principal components. Takes
        the full data covariance matrix and diagonalizes it.
        """
        cov = np.cov(self._data, rowvar=self._rowvar)
        self._eigval, self._eigvec = linalg.eig(cov)
        
        # Sort eigenvectors by decreasing eigenvalue.
        ordering = np.argsort(self._eigval)[::-1]
        self._eigval = np.asarray(self._eigval[ordering])
        self._eigvec = np.asarray(self._eigvec[:, ordering])
    
    def _snapshot(self):
        """
        Implements the snapshot method (Sirovich, 1987)
        for PCA to learn the first n eigenvectors, when the number of 
        data vectors n << p, the number of predictors.  It works on the
        assumption that the eigenvectors to be found are linear 
        combinations of the data vectors. Instead of diagonalizing the 
        outer product matrix (x - mu)(x - mu)^T (which is rank-deficient) 
        the method diagonalizes the inner product (scatter) matrix 
        (X - mu)^T(X - mu) and projects up the resulting eigenvectors.
        
        For details, see:
        
          L. Sirovich. Turbulence and the dynamics of coherent structures.
            Quarterly Applied Mathematics, 45(3):561-590, 1987.
        """
        centered = self._data - self._mean
        if self._rowvar:
            mcov = np.dot(centered.T, centered) / (self.ndata - 1.)
        else:
            mcov = np.dot(centered, centered.T) / (self.ndata - 1.)
        
        # Diagonalize the scatter matrix.
        seigval, seigvec = linalg.eig(mcov)
        self._eigval = seigval
        
        # Sort eigenvectors by decreasing eigenvalue.
        ordering = np.argsort(self._eigval)[::-1]
        self._eigvec = seigvec[:, ordering]
        self._eigval = self._eigval[ordering]
        
        # Project the reduced eigenvector set onto the centered data.
        if self._rowvar:
            self._eigvec = np.dot(centered, self._eigvec)
        else:
            self._eigvec = np.dot(centered.T, self._eigvec)
        
        # Retain only the first n eigenvectors, just in case.
        self._eigvec = self._eigvec[:,:np.min(centered.shape)]
        self._eigval = self._eigval[:np.min(centered.shape)]
        
        # Normalize the up-projected eigenvetors.
        normalizers = np.apply_along_axis(linalg.norm, 0, self._eigvec)
        self._eigvec /= normalizers[np.newaxis, :]
    
    def _em_one_pass(self, centered=None, numcmpt=1, thresh=1e-14, out=None):
        """
        With numcmpt = 1, computes the first principal component
        of the data. Otherwise computes an unnormalized, non-orthogonal
        spanning set for the first numcmpt principal components. Assumes
        rows are variables, columns are data points.
        """
        csize = (self.ndim, numcmpt)
        if out != None:
            assert out.shape == csize
            comp = out
            comp[:] = random.normal(size=csize)
        else:
            comp = random.normal(size=csize)
        
        # Initialize 'old' array to infinity
        comp_old = np.empty(csize) + np.inf
        
        if centered == None:
            # Center the data with respect to the dataset mean
            centered = self._data - self._mean
            
        # Compensate for the shape of the data
        if not self._rowvar:
            centered = centered.T
        
        while linalg.norm(comp_old - comp, np.inf) > thresh:
            pinvc_times_data = np.dot(linalg.pinv(comp), centered)
            comp_old[:] = comp
            comp[:] = np.dot(centered, linalg.pinv(pinvc_times_data))
        
        # Normalize the eigenvectors we obtained.
        comp /= np.apply_along_axis(linalg.norm, 0, comp)[np.newaxis, :]
    
    def _expectation_maximization(self, numcmpt=1, thresh=1e-14):
        """
        Implements the EM algorithm for PCA (Roweis, 1998; Tipping &
        Bishop, 1999).
        
        The algorithm is randomly initialized eigenvector columns C and
        iteratively refines them: the E-step generates point estimates
        in the low-dimensional principal components space while the M-step
        re-estimates the eigenvectors C based on this projection.
        
        Note that if estimating more than one principal component the 
        method will merely find a (non-orthogonal) basis for the space
        spanned by the first few principal components. Here we instead
        estimate them one at a time, each time subtracting off projection
        of the training data onto the previous eigenvector before 
        computing the next. This is not necessarily the most efficient
        approach but it does guarantee that the basis will be identical
        (up to a sign switch) to those produced by the other methods.
        
        For details, see:
        
          Sam Roweis (1998). EM algorithms for PCA and SPCA. Advances 
            in Neural Information Processing Systems 10 (NIPS'97) pp.626-632
        
          Online: http://www.cs.toronto.edu/~roweis/papers/empca.pdf
        
          M.E. Tipping & C.M. Bishop (1999). Probabilistic Principal
            Components Analysis. Journal of the Royal Statistical Society,
            Series B, 61:611-622.
        
          Online: http://research.microsoft.com/en-us/um/people/cmbishop/
            downloads/Bishop-PPCA-JRSS.pdf
        """
        # Center the data with respect to the dataset mean
        centered = self._data - self._mean
        
        # Make space for the eigenvectors.
        self._eigvec = np.empty((self.ndim, numcmpt))
        for index in xrange(numcmpt):
            ncmp = index + 1
            # Do a single pass
            out = self._eigvec[:, index:ncmp]
            self._em_one_pass(centered, 1, thresh, out)
            centered -= self.reconstruct(self.project(ncmp, centered, False))
    
    def project(self, ncmp, data=None, center=True):
        """Project into principal components space, using ncmp components."""
        if self._eigvec == None or self._eigvec.shape[1] < ncmp:
            raise ValueError('Not enough eigenvectors computed')
        if data != None and center == True:
            centered = data - self._mean
        elif data != None:
            centered = data
        else:
            centered = self._data - self._mean
        centered = centered if self._rowvar else centered.T
        result = np.dot(self._eigvec[:, :ncmp].T, centered)
        if not self._rowvar:
            return result.T
        else: 
            return result
        
    def reconstruct(self, projected):
        """Reconstruct from the principal subspace to data space."""
        if not self._rowvar:
            ncmp = projected.shape[1]
            return np.dot(projected, self._eigvec[:,:ncmp].T)
        else:
            ncmp = projected.shape[0]
            return np.dot(self._eigvec[:, :ncmp], projected)
    
    @property
    def ndata(self):
        """The number of examples/data points in the dataset."""
        return self._data.shape[int(self._rowvar)]
    
    @property
    def ndim(self):
        """The number of dimensions in the dataset."""
        return self._data.shape[1 - int(self._rowvar)]
    
