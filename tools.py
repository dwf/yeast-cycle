# System imports
import os, os.path, re, sys, csv

# Python Imaging Library
import PIL.Image

# SciPy imports
import numpy as np
import scipy.linalg
import scipy.ndimage as ndimage
import scipy.interpolate as interp

# Matplotlib imports
import matplotlib.image
import matplotlib.pyplot as pyplot
import matplotlib.legend as mlegend
import matplotlib.font_manager as font


import pdb
import mog_em

class ImageEmptyError(ValueError):
    pass

class EllipseFitError(ValueError):
    pass

class ObjectTooSmallError(ValueError):
    pass


_ellipse_cstrt = np.zeros((6,6))
_ellipse_cstrt[2,0] = -2; _ellipse_cstrt[0,2] = -2; _ellipse_cstrt[1,1] =  1
    
def imread_binary(*kw, **args):
    """
    Reads in a binary PNG image with PIL.
    
    For keywords and arguments, see the documentation for PIL.Image.open.
    
    """
    im = PIL.Image.open(*kw, **args)
    return matplotlib.image.pil_to_array(im)[:,:,0:3].sum(axis=2) > 0

def fit_ellipse(x,y):
    """
    Directly fit an ellipse to some scattered 2-D data.
    
    This function fits an ellipse to some scattered 2-D data using the method
    of Fitzgibbon, Pilu and Fisher (1999), minimizing algebraic distance 
    subject to 4ac^2 - b = 1.
    
    Keyword arguments:
    x -- numpy.array of x coordinates (1-dimensional)
    y -- numpy.array of y coordinates (1-dimensional)
    
    Returns:
    Array of coefficients of the ellipse in the order (x^2, xy, y^2, x, y, 1)
    
    This code was directly adapted from MATLAB(tm) code provided in-text by
    the original authors of this method.
    """
    D = np.concatenate(((x * x)[:,np.newaxis],(x * y)[:,np.newaxis], 
        (y * y)[:,np.newaxis], x[:,np.newaxis], y[:,np.newaxis], 
        np.ones((len(y),1))), axis=1)
    S = np.dot(D.T, D)
    global _ellipse_cstrt
    C = _ellipse_cstrt
    geval, gevec = scipy.linalg.eig(S, C)
    NegC = np.where((geval < 0) & (~np.isinf(geval)))
    return np.real(gevec[:,NegC]).squeeze()

def radians_to_degrees(angle):
    """Encapsulate turning radians into degrees for code clarity."""
    return angle * 180.0 / np.pi

def medial_axis_representation(obj):
    """Return the (mean-subtracted) medial points and width."""
    width = np.zeros(obj.shape[0])
    medial = np.zeros(obj.shape[0])
    for rowidx in xrange(obj.shape[0]):
        row = obj[rowidx]
        inked = np.where(row)[0]
        if len(inked) == 0:
            # Just take the mean up to this point
            medial[rowidx] = np.mean(medial[:rowidx])
            # And width 0 seems appropriate
            width[rowidx] = 0
            continue    
        first = np.min(inked)
        last = np.max(inked)
        medial[rowidx] = (first + last)/2.0
        width[rowidx] = last - first + 1
    
    medial -= np.mean(medial)
    medial /= np.float64(len(medial))
    width /= np.float64(len(medial))
    return medial, width

def internal_knots(nknots):
    return np.mgrid[0:1:(nknots+2)*1j][1:-1]

def generate_spline(data, nknots, order=3):
    #print nknots
    # Internal knots - without [-1:1] all values are 0, wtf
    knots = internal_knots(nknots)
    # Dependent variable 
    x = np.mgrid[0:1:(len(data)*1j)]
    tck,fp,ier,msg = interp.splrep(x, data, k=order, task=-1, \
        t=knots, full_output=1)
    #print tck[0]
    #print "Sum of squared residuals: %e" % fp
    return tck

def spline_features(obj,axisknots=5,widthknots=5,order=3,plot=False):
    axis_splines = []
    width_splines = []
    medial_lengths = []
    count = 0
    if plot:
        pyplot.ioff()
        pyplot.figure(1)
        pyplot.clf()
        pyplot.subplot(211)
        pyplot.imshow(obj.T)
        pyplot.axis('off')
        pyplot.title('Aligned object #%d' % count, fontsize='small')
        
    med, width = medial_axis_representation(obj)
    if len(med) <= order or len(med) <= axisknots:
        exc = ObjectTooSmallError()
        exc.obj = obj
        exc.number = count
        
    if plot:
        pyplot.subplot(212)
        pyplot.plot(np.mgrid[0:1:(len(med)*1j)],med,label='Medial axis')
        pyplot.plot(np.mgrid[0:1:(len(width)*1j)],width,label='Width')
        pyplot.legend(prop=font.FontProperties(size='x-small'))
        pyplot.title('Medial axis representation for object #%d'%count, \
            fontsize='small')
        pyplot.show()
        pyplot.ion()
    
    dep_var = np.mgrid[0:1:500j]
    med_tck = generate_spline(med, nknots=axisknots, order=order)
    
    if plot:
        medsplinevalues = interp.splev(dep_var, med_tck)
        pyplot.plot(dep_var, medsplinevalues,
            label='Medial axis spline fit')
        
    width_tck = generate_spline(width, nknots=widthknots,order=order)
    
    assert np.allclose(med_tck[0],width_tck[0])
    
    if plot:
        widthsplinevalues = interp.splev(dep_var, width_tck)
        pyplot.ioff()
        pyplot.plot(dep_var, widthsplinevalues, 
            label='Width curve spline fit')
        pyplot.legend(prop=font.FontProperties(size='x-small'))
        pyplot.show()
        pyplot.title('Spline fits for object #%d' % count, \
            fontsize='small')
        pyplot.ion()
        raw_input()
    # Why up to -4? Because these are always zero, for some reason,
    # for our purposes.
    width_spline = width_tck[1][:-4]
    axis_spline = med_tck[1][:-4]
    medial_length = len(med)
    count += 1
    return np.concatenate((width_spline, axis_spline, [medial_length]))


def aligned_objects_from_im(sil, locations, ids):
    """
    Generator that processes each object in a binary threshold image.
    Yields an image of the object rotated to align to the best fit ellipse,
    and an additional 90 degrees if width > height, so that the longest
    so that the vertical medial axis is as long as possible.
    """
    # Reverse rows
    sil = sil[::-1]
    
    sil = ndimage.binary_fill_holes(sil)
    
    labels, numfound = ndimage.label(sil)
    objects = ndimage.find_objects(labels)
    #found = np.zeros((numfound+1,), dtype=bool)
    db_obj_accounted_for = []
    found_objects = {}
    features = []
    
    for ii in xrange(len(locations)):
        # Rows and columns are y and x, respectively, so we reverse x and y
        # after rounding to nearest
        gridpos = tuple([int(round(a)) for a in locations[ii]][::-1])
        #gridpos = (labels.shape[0] - gridpos[0], gridpos[1])
        labelnumber = labels[gridpos]
        #pdb.set_trace()
        found[labelnumber] = True
        #pyplot.plot([gridpos[1]],[gridpos[0]],'o')
        if labelnumber == 0:
            row,col = gridpos
            neighb = labels[(row-3):(row+3), (col-3):(col+3)]
            if np.all(neighb == 0):
                print >>sys.stderr, "[e] Label # missing im #%d, obj #%d" % \
                    tuple(ids[ii])
                pyplot.clf()
                pyplot.imshow(sil)
                pyplot.plot([gridpos[1]],[gridpos[0]],'yx')
                missing.append(ii)
            else:
                u, s = zip(*[(u, np.sum(neighb == u)) for u in \
                    np.unique(neighb)])
                labelnumber = u[np.argsort(s)[-1]]
        im = labels[objects[labelnumber-1]] == labelnumber
        #pdb.set_trace()
        coeffs = fit_ellipse(*(np.where(im)))
        if coeffs.size != 6:
            print >>sys.stderr, "[e] Ellipse fit im #%d, obj #%d" % \
                tuple(ids[ii])
            missing.append(ii)
            continue
        else:
            a,b,c,d,e,f = coeffs
        preangle = b / (a - c)
        if not np.isinf(preangle):
            angle = radians_to_degrees(-0.5 * np.arctan(preangle))
            rotated = ndimage.rotate(np.float64(im),angle)
            bounds = ndimage.find_objects(rotated > 0)[0]
            height, width = np.shape(rotated[bounds])
            if width > height:
                angle -= 90.0
                rotated = ndimage.rotate(im, angle)
                bounds = ndimage.find_objects(rotated > 0)[0]
            key = "obj_"+str(ids[ii,0])+str(ids[ii,1])
            found_objects[key] = obj = rotated[bounds]
            features.append(spline_features(obj))[np.newaxis,:]
            db_obj_accounted_for.append(ii)
    return np.concatenate(features,axis=0), ids[db_obj_accounted_for]

def load_and_process(path, locs, ids, prefix="_home_moffatopera_",
    suffix='_binary.png',  plot=False):
    """asdfsadgsahfd"""
    
    if hasattr(locs,'files'):
        iterable_locs = locs.files
    else:
        iterable_locs = locs
    
    
    #import progressbar as pbar
    #widg = widgets=[pbar.RotatingMarker(), ' ', pbar.Percentage(),' ', \
    #    pbar.Bar(), ' ', pbar.ETA()]
    #pb = pbar.ProgressBar(maxval=len(iterable_locs), widgets=widg).start()
    #count = 0
    

    
    for image in iterable_locs:
        imroot = image.split('.')[0]
        im = imread_binary(os.path.join(path,prefix+imroot+suffix))
        im_locs = locs[image]
        im_ids = ids[image]
        objects = aligned_objects_from_im(im,im_locs,im_ids)
        count += 1
        #pb.update(count)
    #pb.finish()
    # data = []
    
    # import progressbar as pbar
    # widg = widgets=[pbar.RotatingMarker(), ' ', pbar.Percentage(),' ', \
    #     pbar.Bar(), ' ', pbar.ETA()]
    # pb = pbar.ProgressBar(maxval=len(files), widgets=widg).start()
    # 
    # count = 0
    # for fn, image in load_silhouettes(files):
    #     try:
    #         widths, axes, lengths = process_all(image, plot=plot)
    #     except ImageEmptyError, e:
    #         #print >> sys.stderr, "%s contained nothing" % fn
    #         #print >> sys.stderr, str(type(e)), str(e)
    #         continue
    #     #print np.shape(widths)
    #     #print np.shape(axes)
    #     #print np.shape(lengths)
    #     bigmat = np.concatenate((widths,axes,lengths[np.newaxis,:]),axis=0)
    #     data.append(bigmat)
    #     count += 1
    #     pb.update(count)
    # pb.finish()
    # return np.concatenate(data,axis=1)


def coef2knots(x):
    return x - 4

def unmix(data, meandata, stddata, k=3):
    D = len(meandata)
    perspline = (D - 1)/2
    t = internal_knots(coef2knots(perspline))
    t = np.concatenate((np.zeros(4),t,np.ones(4)))
    data = data * stddata # intentionally not inplace
    data += meandata
    return (t,np.concatenate((data[:perspline],np.zeros((4)))),k), \
        (t,np.concatenate((data[perspline:(2*perspline)],np.zeros(4))),k)


def unmix_and_sample(params,meandata,stddata,k=3,component=None):
    samp = mog_em.sample_mog(1,params,component=component).squeeze()
    return unmix(samp, meandata, stddata, k)


def plot_from_spline(tck, samples=500, *args, **kwds):
    dep_var = np.mgrid[0:1:(samples * 1j)]
    splval = interp.splev(dep_var, tck)
    pyplot.plot(dep_var,splval,*args,**kwds)

def sample_plots(params,meandata,stddata,subplotsize,nsamp):
    pyplot.ioff()
    pyplot.clf()
    for i in xrange(len(params['logalpha'])):
        pyplot.subplot(*tuple(subplotsize + (i+1,)))
        means = unmix(params['mu'][:,i],meandata,stddata)
        plot_from_spline(means[0],500)
        plot_from_spline(means[1],500)
        pyplot.title("Cluster %d" % (i+1,))
        for k in xrange(nsamp):
            tck1,tck2 = unmix_and_sample(params,meandata,stddata,component=i)
            plot_from_spline(tck1,500,'--')
            plot_from_spline(tck2,500,'-.')
    pyplot.show()
    pyplot.ion()


def cells_in_database_per_imagefile(datafiles, headerfile=None):
    """
    Reads in CSV files that contain object records.

    Returns a dictionary where the keys are the image filename (with 
    / replaced by _) and the values are NumPy arrays containing coordinate
    lists, as well as another dictionary giving ImageNumber/ObjectNumbers.
    """
    if hasattr(datafiles, 'read'):
        datafiles = [datafiles]
    cells = {}
    if headerfile:
        headers = [h[1:-1] for h in headerfile.read().split(',')]
    else:
        headers = None
    for datafile in datafiles:
        reader = csv.DictReader(datafile, fieldnames=headers,
            delimiter=',',quotechar='"')
        for row in reader:
    	    key = row['Image_PathName_GFP'] + '/' + \
    	        row['Image_FileName_GFP']
    	    imgnum = row['ImageNumber']
    	    objnum = row['ObjectNumber']
    	    location = [np.float64(row['cells_Location_Center_X']),
    	        np.float64(row['cells_Location_Center_Y'])]
    	    cells.setdefault(key, []).append(((imgnum,objnum),location))
    	print "Finished with %s" % datafile.name
    allids = {}
    for image in cells.keys():
        ids, locations = zip(*cells[image])
        ids = np.array(ids,dtype=int)
        locations = np.array(locations)
        del cells[image]
        image = image.replace('/','_')
        cells[image] = locations
        allids[image] = ids
    return cells, allids
