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
from enthought.traits.api import HasTraits, Range, Instance, Int
from enthought.traits.ui.api import View, Item, RangeEditor
from embedded_figure import MPLFigureEditor
from matplotlib.figure import Figure


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


_ellipse_cstrt = np.zeros((6,6))
_ellipse_cstrt[2,0] = -2; _ellipse_cstrt[0,2] = -2; _ellipse_cstrt[1,1] =  1


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
        #pyplot.title('Spline fits for object' % count, \
        #    fontsize='small')
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
            
    def process_item(ii, append=False, figure=None):
        # Get a tuple of rounded-to-nearest pixel locations
        gridpos = tuple([int(round(a)) for a in locations[ii]][::-1])
        
        # Get the number from the label array at that grid position
        labelnumber = labels[gridpos]
        
        # If it's a background pixel, check the neighbourhood.
        if labelnumber == 0:
            row,col = gridpos
            neighb = labels[(row-3):(row+3), (col-3):(col+3)]
            # If everybody's 0 in the -3+3 neighbourhood, report it and move on
            if np.all(neighb == 0):
                print >>sys.stderr, "[e] Label # missing im #%d, obj #%d" % \
                    tuple(ids[ii])
            
            # Otherwise, take the most frequently occurring label in that 
            # neighbourhood
            else:
                u, s = zip(*[(u, np.sum(neighb == u)) for u in \
                    np.unique(neighb)])
                labelnumber = u[np.argsort(s)[-1]]
        
        # Grab this part of the silhouette
        im = labels[objects[labelnumber-1]] == labelnumber
        
        # Fit an ellipse using every nonzero pixel location in the image.
        coeffs = fit_ellipse(*(np.where(im)))
        
        if coeffs.size != 6:
            print >>sys.stderr, "[e] Ellipse fit im #%d, obj #%d" % \
                tuple(ids[ii])
            return
        else:
            a,b,c,d,e,f = coeffs
        preangle = b / (a - c)
        
        if not np.isinf(preangle):
            angle = radians_to_degrees(-0.5 * np.arctan(preangle))
            old_angle = angle
            # Order = 0 prevents interpolation from being done and screwing 
            # with our object boundaries.
            rotated = ndimage.rotate(im, angle, order=0)
            height, width = rotated[ndimage.find_objects(rotated)[0]].shape
        else:
            angle = 0.
            old_angle = 0.
        
        if width > height:
            angle -= 90.0
            rotated = ndimage.rotate(im, angle, order=0)
        
        # Correct so that in budding cells, the "major" hump is always
        # on the first.          
        if np.argmax(rotated.sum(axis=1)) > rotated.shape[0] // 2:
            angle -= 180.0
            rotated = ndimage.rotate(im, angle, order=0)
        
        # Do a find_objects on the resultant array after rotation in 
        # order to _just_ get the object and not any of the extra 
        # space that's been added.
        try:
            bounds = ndimage.find_objects(rotated)[0]
        except IndexError, e:
            pass
        
        key = "obj_" + str(ids[ii,0])+str(ids[ii,1])
        found_objects[key] = obj = rotated[bounds]
        try:
            these_features = spline_features(obj,plot=plot,
                    fig=figure)
            if append:
                features.append(these_Features[:,np.newaxis])
                db_obj_accounted_for.append(ii)
        except ObjectTooSmallError, e:
            pass
    
    if plot:
        class InteractiveViewer(HasTraits):
            figure = Instance(Figure, ())
            viewed_item = Int(0, min=0, max=(len(locations) - 1))
            view = View(Item('figure', editor=MPLFigureEditor(),
                             show_label=False), 
                              Item('viewed_item',
                             editor=RangeEditor(high=len(locations)-1,
                             low=0),show_label=False),
                             width=700,
                             height=600,
                             resizable=True, title=fn)
            def __init__(self):
                super(InteractiveViewer, self).__init__()
                axes1 = self.figure.add_subplot(211)
                axes2 = self.figure.add_subplot(212)
                
            
            def _viewed_item_changed(self):
                process_item(self.viewed_item, 
                    append=False, figure=self.figure)
        InteractiveViewer().configure_traits()
    #else:
    #    for ii in xrange(len(locations)):
    #        process_item(ii, append=True)
    
    #return np.concatenate(features,axis=0), ids[db_obj_accounted_for]


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
    print len(tck[0])
    print len(tck[1])
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

if __name__ == "__main__":
    import sys
    if sys.argv[1] == "plot":
        locs = np.load('/Users/dwf/locations.npz')
        ids = np.load('/Users/dwf/ids.npz')
        load_and_process('/Users/dwf/data/binary', locs, ids,plot=True)
