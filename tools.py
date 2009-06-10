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
import wx



class ImageEmptyError(ValueError):
    pass

class EllipseFitError(ValueError):
    pass

class ObjectTooSmallError(ValueError):
    pass


def imread_binary(*kw, **args):
    """
    Reads in a binary PNG image with PIL.
    
    For keywords and arguments, see the documentation for PIL.Image.open.
    
    """
    im = PIL.Image.open(*kw, **args)
    return matplotlib.image.pil_to_array(im)[:,:,0] == 255


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

def spline_features(obj,axisknots=3,widthknots=3,order=4,plot=False,fig=None):
    axis_splines = []
    width_splines = []
    medial_lengths = []
    count = 0
    if plot:
        axes = fig.axes[0]
        axes.clear()
        axes.axis('off')
        axes.matshow((obj.T),cmap=matplotlib.cm.bone)
    
    med, width = medial_axis_representation(obj)
    if len(med) <= order or len(med) <= axisknots:
       	raise ObjectTooSmallError()
    if plot:
        axes = fig.axes[1]
        axes.clear()
        axes.plot(np.mgrid[0:1:(len(med)*1j)],med,label='Medial axis')
        axes.plot(np.mgrid[0:1:(len(width)*1j)],width,label='Width')
        axes.legend(prop=font.FontProperties(size='x-small'))
        if hasattr(fig.canvas, 'draw'):
            wx.CallAfter(fig.canvas.draw)
        
    dep_var = np.mgrid[0:1:500j]
    med_tck = generate_spline(med, nknots=axisknots, order=order)
    
    if plot:
        medsplinevalues = interp.splev(dep_var, med_tck)
        axes.plot(dep_var, medsplinevalues,
            label='Medial axis spline fit')
        
    width_tck = generate_spline(width, nknots=widthknots,order=order)
    
    assert np.allclose(med_tck[0],width_tck[0])
    
    if plot:
        widthsplinevalues = interp.splev(dep_var, width_tck)
        axes.plot(dep_var, widthsplinevalues, 
            label='Width curve spline fit')
        reconstruct_values = interp.splev(dep_var, (width_tck[0], np.concatenate((width_tck[1][:-4],np.zeros(4))), width_tck[2]))
        axes.plot(dep_var, reconstruct_values, label='Reconstruction')
        axes.legend(prop=font.FontProperties(size='x-small'))
    # Why up to -4? Because these are always zero, for some reason,
    # for our purposes.
    width_spline = width_tck[1][:-4]
    axis_spline = med_tck[1][:-4]
    medial_length = len(med)
    count += 1
    return np.concatenate((width_spline, axis_spline, [medial_length]))


def aligned_objects_from_im(sil, locations, ids, fn, plot=False):
    """
    Generator that processes each object in a binary threshold image.
    Yields an image of the object rotated to align to the best fit ellipse,
    and an additional 90 degrees if width > height, so that the longest
    so that the vertical medial axis is as long as possible.
    """
    # Reverse rows
    sil = sil[::-1]
    
    #sil = ndimage.binary_fill_holes(sil)
    
    # labels is an array indicating what pixels belong to which object
    labels, numfound = ndimage.label(sil)
    
    # List of slices indexing the different objects.
    objects = ndimage.find_objects(labels)
    
    db_obj_accounted_for = []
    found_objects = {}
    features = []


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
    
    allobjects = []
    allids = []
    for image in iterable_locs:
        print type(ids)
        imroot = image.split('.')[0]
        im = imread_binary(os.path.join(path,prefix+imroot+suffix))
        im_locs = locs[image]
        im_ids = ids[image]
        print prefix+imroot+suffix
        if plot:
            aligned_objects_from_im(im,im_locs,im_ids,image,plot=plot)
            print "Hit enter for next image."
        else:
            objects, newids = aligned_objects_from_im(im,im_locs,im_ids,image,
                plot=plot)
        #count += 1
        if not plot:
            allobjects.append(objects)
            allids.append(newids)
        #pb.update(count)
    if not plot:
        allobjects = np.concatenate(allobjects,axis=0)
        allids = np.concatenate(allids,axis=0)
        return allobjects, allids

def internal_knots(nknots):
    return np.mgrid[0:1:(nknots+2)*1j][1:-1]


def plot_from_spline(tck, samples=500, *args, **kwds):
    dep_var = np.mgrid[0:1:(samples * 1j)]
    splval = interp.splev(dep_var, tck)
    pyplot.plot(dep_var,splval,*args,**kwds)


def coef2knots(x):
    return x - 4


def unmix(data, meandata=None, stddata=None, k=4):
    D = len(data)
    perspline = (D - 1)/2
    t = internal_knots(coef2knots(perspline))
    t = np.concatenate((np.zeros(4),t,np.ones(4)))
    if stddata is not None:
        data = data * stddata # intentionally not inplace
    if meandata is not None:
        data += meandata
    print t
    return (t,np.concatenate((data[:perspline],np.zeros((4)))),k), \
        (t,np.concatenate((data[perspline:(2*perspline)],np.zeros(4))),k)

