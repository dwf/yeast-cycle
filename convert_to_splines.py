"""Tools to spline-ify a dataset."""
import sys
import pdb
from itertools import izip
import numpy as np
from scipy.interpolate import splrep

from new_tools import DataSet, Plate, ImageSilhouette, ObjectSilhouette
from new_tools import MedialRepresentation

from rotation import EllipseFitError, EllipseAlignmentError


def make_spline_dtype(splinetup):
    """
    Make a dtype object that corresponds to a ``(t,c,k)`` pair returned
    by ``scipy.interpolate.splrep``. Needs an actual (t,c,k) pair to get
    widths right.
    """
    return np.dtype([
        ('t', float, len(splinetup[0])),
        ('c', float, len(splinetup[0])),
        ('k', int)
    ])


def splines_from_h5file(h5file, nknots, area_thresh, where='/'):
    """
    Read in masks from an HDF5 file with the ImageSilhouette
    class and fit splines to them, save the coefficients into a
    structured array.
    """
    features = []
    total = 0
    failed = 0
    toosmall = 0
    # We're assuming all of the leaves are binary arrays that we want
    for node in h5file.walkNodes(where=where, classname='Leaf'):
        imagesil = ImageSilhouette(node=node)
        for objnum, obj in enumerate(imagesil):
            # Ignore it if it's too small to be an actual cell.
            if np.prod(obj.image.shape) < area_thresh:
                toosmall += 1
                continue
            # Internal knots only - evenly spaced from 0 to 1 non-inclusive
            knots = np.mgrid[0:1:((nknots + 2)*1j)][1:-1]
            try:
                # Lazily computed
                medialrepr = obj.aligned_version.medial_repr
            except Exception, exc:
                print >>sys.stderr, "Image '%s', object %d, %s: %s" % \
                    (imagesil.node, objnum, exc.__class__.__name__, str(exc))
                failed += 1
            dependent_variable = np.mgrid[0:1:(medialrepr.length * 1j)]
            
            try:
                m_spline, mresid, ier1, msg1 = splrep(dependent_variable,
                    medialrepr.medial_axis, t=knots, full_output=1)
                w_spline, wresid, ier2, msg2 = splrep(dependent_variable,
                    medialrepr.width_curve, t=knots, full_output=1)
            except Exception, exc:
                print >>sys.stderr, "Image '%s', object %d, %s: %s" % \
                    (imagesil.node, objnum, exc.__class__.__name__, str(exc))
                failed += 1
                continue
            
            # Save the image path (in the HDF5 archive) and object number
            tup = (imagesil.node._v_pathname, objnum + 1)
            
            # Save the (t,c,k) triples and the length
            tup += (m_spline, w_spline, medialrepr.length)
            
            # Save the residuals
            tup += (mresid, wresid,)
            
            # Append to our list
            features.append(tup)
            total += 1
    
    # Construct the dtype argument
    types = [('image', 'S100'), ('objnum', int)]   # Overestimate str length
    
    # Splines have a nested dtype constructed by looking at the last one made
    types += [('medial', make_spline_dtype(m_spline))]
    types += [('width', make_spline_dtype(w_spline))]
    
    # Scalars of interest
    types += [('length', float)] # This is an int but floats are convenient
    types += [('mresidual', float)]
    types += [('wresidual', float)]
    
    print >> sys.stderr, "Done. %d processed, %d failed." % (total, failed)
    
    # Construct a structured array
    return np.array(features, dtype=np.dtype(types))

if __name__ == "__main__":
    usage = """
    %s <filename> <nknots> <areathresh> [where]
    
        <nknots> = integer number of equidistant knots
        <areathresh> = discard object smaller than this (bounding box)
        [where] = optional, where in the h5file to start e.g. /plate01
        
    """ % sys.argv[0]
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print usage
        sys.exit(1)
    else:
        try:
            int(sys.argv[2])
            int(sys.argv[3])
        except:
            print usage
            sys.exit(1)
    
    if len(sys.argv) == 4:
        filename, nknots, areathresh = sys.argv[1:]
        where = None
    else:
        filename, nknots, areathresh, where = sys.argv[1:]
    areathresh = int(areathresh)
    nknots = int(nknots)
    
    h5file = tables.openFile(sys.argv[1])
    
    
