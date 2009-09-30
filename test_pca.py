from pca import PrincipalComponents
import numpy as np

def check_cols_match_except_sign(m1, m2):
    """
    Check that every column in m1 is approximately equal to the
    same column in m2, or its additive inverse.
    """
    assert m1.shape == m2.shape
    for col1, col2 in zip(m1.T, m2.T):
        assert np.allclose(col1, col2) or np.allclose(col1, -col2)

def check_rows_match_except_sign(m1, m2):
    """
    Check that every row in m1 is approximately equal to the
    same row in m2, or its additive inverse.
    """
    assert m1.shape == m2.shape
    for row1, row2 in zip(m1, m2):
        assert np.allclose(row1, row2) or np.allclose(row1, -row2)

def test_snapshot_vs_direct_pc_rowvar_false():
    """Tests that snapshot and direct agree."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data)
    pc1._direct()
    pc2._snapshot()
    check_cols_match_except_sign(pc1._eigvec[:,:4], pc2._eigvec[:,:4])

def test_em_vs_direct_pc_rowvar_false():
    """Tests that EM and direct agree."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data)
    pc1._direct()
    pc2._expectation_maximization(5)
    check_cols_match_except_sign(pc1._eigvec[:,:5], pc2._eigvec)

def test_em_vs_snapshot_pc_rowvar_false():
    """Tests that EM and snapshot agree."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data)
    pc1._snapshot()
    pc2._expectation_maximization(5)
    check_cols_match_except_sign(pc1._eigvec[:,:5], pc2._eigvec)


def test_em_with_without_rowvar():
    """Tests the EM method, with and without rowvar, works correctly."""
    np.random.seed(1)
    data = np.random.normal(size=(20, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    pc1._expectation_maximization(10)
    pc2._expectation_maximization(10)
    check_cols_match_except_sign(pc1._eigvec, pc2._eigvec)

def test_snapshot_with_without_rowvar():
    """Tests the snapshot method, with and without rowvar, works correctly."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    pc1._snapshot()
    pc2._snapshot()
    check_cols_match_except_sign(pc1._eigvec, pc2._eigvec)

def test_direct_with_without_rowvar():
    """Tests the direct method, with and without rowvar, works correctly."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    pc1._direct()
    pc2._direct()
    check_cols_match_except_sign(pc1._eigvec, pc2._eigvec)

def test_project_with_without_rowvar():
    """Tests the project works with rowvar correctly."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    pc1._direct()
    pc2._direct()
    check_cols_match_except_sign(pc1.project(3), pc2.project(3).T)
    check_cols_match_except_sign(pc1.project(3), pc2.project(3).T)

def test_reconstruct_with_without_rowvar():
    """Tests the reconstruct works with rowvar correctly."""
    np.random.seed(1)
    data = np.random.normal(size=(10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    pc1._direct()
    pc2._direct()
    check_cols_match_except_sign(pc1.reconstruct(pc1.project(3)), 
        pc2.reconstruct(pc2.project(3)).T)

def test_ndata_property():
    """Tests the ndata property works with rowvar correctly."""
    data = np.zeros((10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    assert pc1.ndata == pc2.ndata

def test_ndim_property():
    """Tests the ndim property works with rowvar correctly."""
    data = np.zeros((10, 50))
    pc1 = PrincipalComponents(data)
    pc2 = PrincipalComponents(data.T, rowvar=True)
    assert pc1.ndim == pc2.ndim
