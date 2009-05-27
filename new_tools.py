from itertools import izip, islice

# For our matplotlib figure
import matplotlib
from matplotlib.figure import Figure

# For image operations and file I/O, ndimage and pytables
import tables
import numpy as np
import scipy.ndimage as ndimage

# Traits-related imports
from enthought.traits.api import HasTraits, Instance, Int, Array, List, Str

# View components
from enthought.traits.ui.api import View, Item

# Editor components
from enthought.traits.ui.api import RangeEditor, InstanceEditor
from embedded_figure import MPLFigureEditor

from tools import fit_ellipse, spline_features, radians_to_degrees
from tools import ObjectTooSmallError

class ObjectSilhouette(HasTraits):
    figure = Instance(Figure, ())    
    image = Array(dtype=bool) 
    
    def __init__(self, image):
        """Document me."""
        super(ObjectSilhouette, self).__init__()
        self.image = image
    
    def process_item(self, fig):
        im = self.image
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
        
        try:
            these_features = spline_features(rotated[bounds],plot=True,
                    fig=fig)
        except ObjectTooSmallError, e:
            pass
    


class ImageSilhouette(HasTraits):
    label_image = Array()
    object_slices = List(slice, ())
    
    def __init__(self, node):
        super(ImageSilhouette, self).__init__()
        
        # Label the binary array from the HDF5 file
        self.label_image, number = ndimage.label(node.read())
        
        # Get slices that index the array
        self.object_slices = ndimage.find_objects(self.label_image)
    
    def __len__(self):
        return len(self.object_slices)
    
    def __getitem__(self, key):
        if type(key) is slice:
            indices = islice(xrange(len(self.object_slices)), *key.indices())
            return [self[nkey] for nkey in indices]
        else:
            im = self.label_image[self.object_slices[key]] == (key + 1)
            return ObjectSilhouette(im)
    
    def __contains__(self):
        raise TypeError("Containment checking not supported")
    

class Plate(HasTraits):
    node = Instance(tables.Group)
    images = List(Str)
    
    def __init__(self, node):
        super(Plate, self).__init__()
        self.node = node
        for imnode in node._f_walkNodes('Leaf'):
            self.images.append(imnode._v_pathname)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        if type(key) is slice:
            indices = islice(xrange(len(self.images)), *key.indices())
            return [self[nkey] for nkey in indices]
        else:
            im = self.label_image[self.images[key]] == (key + 1)
            return ObjectSilhouette(im)

    def __contains__(self):
        raise TypeError("Containment checking not supported")






class DataSet(HasTraits):
    h5file = Instance(tables.File)
    plates = List(Str)
    
    def __init__(self, h5file):
        super(DataSet, self).__init__()
        for platenode in h5file.root:
            self.plates.append(platenode._v_pathname)
    


class DataSetBrowser(HasTraits):
    figure = Instance(Figure, ())
    
    dataset = Instance(DataSet, ())
    current_plate = Instance(Plate, ())
    current_image = Instance(ImageSilhouette, ())
    current_object = Instance(ObjectSilhouette, ())
    
    plate_index = Int(0)
    image_index = Int(0)
    object_index = Int(0)
    
    num_plates = Property
    num_images = Property
    num_objects = Property
    
    view = View(Item('figure', editor=MPLFigureEditor(),
                    show_label=False), 
            Item('object_index',editor=RangeEditor(low=1,mode='slider')),
            Item('image_index',editor=RangeEditor(low=1,mode='slider')),
            Item('plate_index',editor=RangeEditor(low=1,mode='slider')),
            width=700,
            height=600,
            resizable=True)
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_plate = self.dataset.plates[self.plate_index]
        axes1 = self.figure.add_subplot(211)
        axes2 = self.figure.add_subplot(212)
    
    def _plate_index_changed(self):
        self.current_plate = self.dataset.plates[self.plate_index - 1]    
    
    def _image_index_changed(self):
        self.current_image = self.current_plate.images[self.image_index - 1]
    
    def _object_index_changed(self):
        # Act on ImageSilhouette
        self.current_object = self.current_image[self.object_index]
    
    def _get_num_plates(self):
        return len(self.dataset.plates)
    
    def _get_num_images(self):
        return len(self.current_plate.images)
    
    def _get_num_objects(self):
        return len(self.current_image.objects)
    
    