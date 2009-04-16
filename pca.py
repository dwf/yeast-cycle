import numpy as np
import numpy.linalg as linalg
import numpy.random as random

class PrincipalComponents:
    def __init__(self, data):
        self._data = data
        self._mean = np.mean(data, axis=1)[:, np.newaxis]
        self._std = np.std(data, axis=1)[:, np.newaxis]
    
    def _direct(self):
        self._cov = np.cov(self._data)
        self._eigval, self._eigvec = linalg.eig(self._cov)
        ordering = np.argsort(self._eigval)[::-1]
        self._eigval = np.asarray(self._eigval[ordering])
        self._eigvec = np.asarray(self._eigvec[:,ordering])
    
    def _snapshot(self):
        centered_data = self._data - self._mean
        ndatapoints = self._data.shape[1]
        R = np.dot(centered_data.T, centered_data) / (ndatapoints - 1)
        seigval, seigvec = linalg.eig(R)
        self._eigval = seigval
        ordering = np.argsort(self._eigval)[::-1]
        self._eigvec = seigvec
        self._eigval = np.asarray(self._eigval[ordering])
        self._eigvec = np.dot(self._data, self._eigvec[:,ordering])
        normalizers = np.apply_along_axis(linalg.norm, 0, self._eigvec)
        self._eigvec /= normalizers[np.newaxis,:]
    
    def _EM(self, numcmpt=1, thresh=1e-12, disp=False):
        C = random.randn(self._data.shape[0], numcmpt)
        C_old = np.nan
        centered_data = self._data - self._mean
        # "while not" = for NaN
        while not linalg.norm(C_old - C)/C.shape[1] < thresh:
            if disp:
                print linalg.norm(C_old - C)
            X = np.dot(np.dot(linalg.inv(np.dot(C.T, C)),C.T),centered_data)
            C_old = C; C = np.dot(np.dot(centered_data, X.T),
                linalg.inv(np.dot(X,X.T)))
        
        self._eigvec = C
        projection = self.project(numcmpt)
        eigenval, eigenvec = linalg.eig(np.cov(projection))
        ordering = np.argsort(eigenval)[::-1]
        eigenval = eigenval[ordering]
        eigenvec = eigenvec[:,ordering]
        projection = projection - np.mean(projection,axis=1)[:,np.newaxis]
        aligned = np.dot(linalg.inv(np.dot(eigenvec.T,eigenvec)),
            np.dot(eigenvec.T,projection))
        xhatxhatTinv = linalg.inv(np.dot(aligned, aligned.T))
        self._eigenvec = np.dot(np.dot(self._data, aligned.T),xhatxhatTinv)
     
    def project(self, ncmp, data=None):
        if ncmp < 1:
            raise ValueError, "Need at least one component to project"
        elif ncmp > self._eigvec.shape[0]:
            raise ValueError, "Not enough eigenvectors"
        if data == None:
            data = self._data
        z = ((data - self._mean))
        ctcinv = linalg.inv(np.dot(self._eigvec.T, self._eigvec))
        return np.dot(ctcinv,np.dot(self._eigvec.T, z))
        
    def reconstruct(self, projected):
        ncmp = projected.shape[0]
        projectup = np.dot(self._eigvec[:,:ncmp], projected)
        return projectup + self._mean
