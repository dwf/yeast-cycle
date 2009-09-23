import sys
import pdb
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

from new_tools import DataSet, Plate, ImageSilhouette, ObjectSilhouette
from new_tools import MedialRepresentation

from rotation import EllipseFitError, EllipseAlignmentError


def process(h5file, nknots, area_thresh, mresidual=False, wresidual=False):
    """
    Read in masks from an HDF5 file with the ImageSilhouette
    class and fit splines to them, save the coefficients into a
    structured array.
    """
    features = []
    total = 0
    failed = 0
    for node in h5file.walkNodes(classname='Leaf'):
        imagesil = ImageSilhouette(node=node)
        for objnum, obj in enumerate(imagesil):
            if np.prod(obj.image.shape) < area_thresh:
                # print >> sys.stderr, "Skipping %s, %d" % \
                #                     (getattr(node, '_v_pathname'), objnum)
                #
                continue
            knots = np.mgrid[0:1:((nknots + 2)*1j)][1:-1]
            try:
                medialrepr = obj.aligned_version.medial_repr
            except Exception, exc:
                print >>sys.stderr, "Image '%s', object %d, %s: %s" % \
                    (imagesil.node, objnum, exc.__class__.__name__, str(exc))
                failed += 1
            dependent_variable = np.mgrid[0:1:(medialrepr.length * 1j)]
            
            try:
                m_spline = LSQUnivariateSpline(dependent_variable,
                    medialrepr.medial_axis, knots)
                w_spline = LSQUnivariateSpline(dependent_variable,
                    medialrepr.width_curve, knots)
            
            except Exception, exc:
                print >>sys.stderr, "Image '%s', object %d, %s: %s" % \
                    (imagesil.node, objnum, exc.__class__.__name__, str(exc))
                failed += 1
                continue
            
            tup = (m_spline._data, w_spline._data, medialrepr.length)
            if mresidual:
                tup += (m_spline.get_residual(),)
            if wresidual:
                tup += (w_spline.get_residual(),)
                
            features.append(tup)
            total += 1
    
    types = [('image', str), ('objnum', int)]
    types += [('medial', float, len(tup[0])), ('width', float, len(tup[1]))]
    types += [('length', float)]
    if mresidual:
        types += [('mresidual', float)]
    if wresidual:
        types += [('wresidual', float)]
    print >> sys.stderr, "Done. %d processed, %d failed." % (total, failed)
    return np.array(features, dtype=np.dtype(types))
