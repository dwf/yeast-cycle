"""
Contains a class that implements 3 methods for extracting principal 
components.
"""

import numpy as np
import numpy.linalg as linalg
import numpy.random as random

class PrincipalComponents(object):
    """Implements principal components analysis."""
    def __init__(self, data, rowvar=False):
        """Construct."""
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
        """Direct method."""
        cov = np.cov(self._data, rowvar=self._rowvar)
        self._eigval, self._eigvec = linalg.eig(cov)
        ordering = np.argsort(self._eigval)[::-1]
        self._eigval = np.asarray(self._eigval[ordering])
        self._eigvec = np.asarray(self._eigvec[:, ordering])
    
    def _snapshot(self):
        """The snapshot method. Finish."""
        centered = self._data - self._mean
        if self._rowvar:
            mcov = np.dot(centered.T, centered) / (self.ndata - 1.)
        else:
            mcov = np.dot(centered, centered.T) / (self.ndata - 1.)
        seigval, seigvec = linalg.eig(mcov)
        self._eigval = seigval
        ordering = np.argsort(self._eigval)[::-1]
        self._eigvec = seigvec[:, ordering]
        self._eigval = self._eigval[ordering]
        if self._rowvar:
            self._eigvec = np.dot(centered, self._eigvec)
        else:
            self._eigvec = np.dot(centered.T, self._eigvec)
        self._eigvec[:,:np.min(centered.shape)]
        normalizers = np.apply_along_axis(linalg.norm, 0, self._eigvec)
        self._eigvec /= normalizers[np.newaxis, :]
        
    def _expectation_maximization(self, numcmpt=1, thresh=1e-15):
        """EM algorithm. Finish."""
        csize = (self._data.shape[int(not self._rowvar)], numcmpt)
        comp = random.normal(size=csize)
        comp_old = np.empty(csize) + np.inf
        centered = self._data - self._mean
        if not self._rowvar:
            centered = centered.T
        while not linalg.norm(comp_old - comp, np.inf) < thresh:
            pinvc_times_data = np.dot(linalg.pinv(comp), centered)
            comp_old[:] = comp
            comp[:] = np.dot(centered, linalg.pinv(pinvc_times_data))
        # Normalize the eigenvectors we obtained.
        comp /= np.apply_along_axis(linalg.norm, 1, comp)[np.newaxis, :]
        self._eigvec = comp
    
    def project(self, ncmp, data=None):
        """Project into principal components space, using ncmp components."""
        if ncmp < 1:
            raise ValueError, "Need at least one component to project"
        elif ncmp > self._eigvec.shape[0]:
            raise ValueError, "Not enough eigenvectors"
        if data == None:
            data = self._data
        centered = data - self._mean
        
    def reconstruct(self, projected):
        """Reconstruct from the principal subspace to data space."""
        ncmp = projected.shape[0]
        projectup = np.dot(self._eigvec[:, :ncmp], projected)
        return projectup + self._mean
    
    @property
    def ndata(self):
        """The number of examples/data points in the dataset."""
        return self._data.shape[int(self._rowvar)]
    
    @property
    def ndim(self):
        """The number of dimensions in the dataset."""
        return self._data.shape[1 - int(self._rowvar)]
    
