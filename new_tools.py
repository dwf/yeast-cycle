"""Traits-based tools for image preprocessing."""
from itertools import islice

# For our matplotlib figure
from matplotlib.figure import Figure

# For image operations and file I/O, ndimage and pytables
import tables
import numpy as np
import scipy.ndimage as ndimage

# Traits-related imports
from enthought.traits.api import Int, Str
from enthought.traits.api import List, Tuple, Array
from enthought.traits.api import HasTraits, Instance, Property

# Chaco
from enthought.chaco.api import ArrayPlotData, Plot, bone, jet
from enthought.enable.component_editor import ComponentEditor

# View components
from enthought.traits.ui.api import View, Item, Group, VGroup

# Editor components
from enthought.traits.ui.api import RangeEditor
from embedded_figure import MPLFigureEditor

from tools import fit_ellipse, spline_features, radians_to_degrees
from tools import ObjectTooSmallError

class ObjectSilhouette(HasTraits):
    """Class representing a single cell silhouette in an image."""
    figure = Instance(Figure, ())    
    image = Array(dtype=bool) 
    
    def __init__(self, image):
        """Document me."""
        super(ObjectSilhouette, self).__init__()
        self.image = image
    
    def process_item(self, fig):
        image = self.image
        # Fit an ellipse using every nonzero pixel location in the image.
        coeffs = fit_ellipse(*(np.where(image)))
        if coeffs.size != 6:
            print "[e] Ellipse fit im #%d, obj #%d"
            return
        else:
            coeff_a, coeff_b, coeff_c = coeffs[:3]
        preangle = coeff_b / (coeff_a - coeff_c)
        if not np.isinf(preangle):
            angle = radians_to_degrees(-0.5 * np.arctan(preangle))
            # Order = 0 prevents interpolation from being done and screwing 
            # with our object boundaries.
            rotated = ndimage.rotate(image, angle, order=0)
            height, width = rotated[ndimage.find_objects(rotated)[0]].shape
        else:
            angle = 0.
            
        if width > height:
            angle -= 90.0
            rotated = ndimage.rotate(image, angle, order=0)
        # Correct so that in budding cells, the "major" hump is always
        # on the first.          
        if np.argmax(rotated.sum(axis=1)) > rotated.shape[0] // 2:
            angle -= 180.0
            rotated = ndimage.rotate(image, angle, order=0)
        
        # Do a find_objects on the resultant array after rotation in 
        # order to _just_ get the object and not any of the extra 
        # space that's been added.
        try:
            bounds = ndimage.find_objects(rotated)[0]
        except IndexError:
            pass
        
        try:
            spline_features(rotated[bounds], plot=True,
                    fig=fig)
        except ObjectTooSmallError:
            pass
    

class ImageSilhouette(HasTraits):
    """Class representing a silhouette image of segmented cells."""
    label_image = Array()
    object_slices = List(Tuple(slice, slice), ())
    
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
            indices = islice(xrange(len(self)), *key.indices(len(self)))
            return [self[nkey] for nkey in indices]
        else:
            image = self.label_image[self.object_slices[key]] == (key + 1)
            return ObjectSilhouette(image)
    
    def __contains__(self):
        raise TypeError("Containment checking not supported")
    

class Plate(HasTraits):
    """Class representing a single plate of imaged wells."""
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
            indices = islice(xrange(len(self)), 
                *key.indices(len(self.images)))
            return [self[nkey] for nkey in indices]
        else:
            image = self.node._v_file.getNode(self.images[key])
            return ImageSilhouette(image)

    def __contains__(self):
        raise TypeError("Containment checking not supported")
    

class DataSet(HasTraits):
    """
    A class encapsulating a dataset of binary masks of segmented images 
    stored in an HDF5 file.
    """
    h5file = Instance(tables.File)
    plates = List(Str)
    
    def __init__(self, h5file):
        """
        Construct a DataSet object from the given PyTables file handle.
        """
        super(DataSet, self).__init__()
        for platenode in h5file.root:
            self.plates.append(platenode._v_pathname)
        self.h5file = h5file
    
    def __len__(self):
        return len(self.plates)

    def __getitem__(self, key):
        if type(key) is slice:
            indices = islice(xrange(len(self)), *key.indices(len(self)))
            return [self[nkey] for nkey in indices]
        else:
            node = self.h5file.getNode(self.plates[key])
            return Plate(node)
    
    def __contains__(self):
        raise TypeError("Containment checking not supported")
    

class DataSetBrowser(HasTraits):
    """
    A class that allows browsing of a DataSet object with sliders
    to navigate through plates, images within plates, and objects 
    within images.
    """
    # Chaco plot
    plot = Instance(Plot)
    
    # matplotlib Figure instance
    figure = Instance(Figure, ())
    
    # DataSet being viewed
    dataset = Instance(DataSet)
    
    # Plate object currently being examined
    current_plate = Instance(Plate)

    # ImageSilhouette object currently being examined
    current_image = Instance(ImageSilhouette)

    # ObjectSilhouette object currently being examined
    current_object = Instance(ObjectSilhouette)

    # Index traits that control the selected plate/image/object
    plate_index = Int(1, editor=RangeEditor(low=1, 
        high_name='num_plates', mode='slider'))
    image_index = Int(1, editor=RangeEditor(low=1, 
        high_name='num_images', mode='slider'))
    object_index = Int(1, editor=RangeEditor(low=1, 
        high_name='num_objects', mode='slider'))

    # Number of plates, images, and objects in the current context
    num_plates = Property(Int, depends_on='dataset')
    num_images = Property(Int, depends_on='current_plate')
    num_objects = Property(Int, depends_on='current_image')
    
    view = View(
            VGroup(
                Item('figure', editor=MPLFigureEditor(), show_label=False), 
                Group(Item('object_index'),
                    Item('image_index'),
                    Item('plate_index'))
            ),
            height=600,
            width=700,
            resizable=True)

    def __init__(self, dataset, **metadata):
        """Construct a DataSetBrowser from the specified DataSet object."""
        super(DataSetBrowser, self).__init__()
        self.dataset = dataset
        self.current_plate = self.dataset[self.plate_index - 1]
        self.current_image = self.current_plate[self.image_index - 1]
        self.current_object = self.current_image[self.object_index - 1]
        self.figure = Figure()
        self.figure.add_subplot(211)
        self.figure.add_subplot(212)
        self._object_index_changed()
        
        plotdata = ArrayPlotData(imagedata=self.current_object.image)
        plot = Plot(plotdata)
        xbounds = np.arange(self.current_object.image.shape[1])
        ybounds = np.arange(self.current_object.image.shape[0])
        plot.img_plot("imagedata", xbounds=xbounds, ybounds=ybounds,colormap=bone)
        self.plot = plot
        
    ######################### Private interface ##########################    

    def _plate_index_changed(self):
        """Handle the plate index changing."""
        try:
            self.current_plate = self.dataset[self.plate_index - 1]
        except IndexError:
            self.current_plate = None
        self.image_index = 1
        self._image_index_changed()

    def _image_index_changed(self):
        """Handle the image index slider changing."""
        try:
            self.current_image = self.current_plate[self.image_index - 1]
        except IndexError:
            self.current_image = None
        self.object_index = 1
        self._object_index_changed()

    def _object_index_changed(self):
        """Handle object index slider changing."""
        try:
            self.current_object = self.current_image[self.object_index - 1]
            self.current_object.process_item(self.figure)
        except IndexError:
            self.current_object = None
    
    def _get_num_plates(self):
        """Return the number of plates in the currently viewed dataset."""
        return len(self.dataset)

    def _get_num_images(self):
        """Return the number of images in the currently viewed plate."""
        return len(self.current_plate)

    def _get_num_objects(self):
        """Return the number of objects in the currently viewed image."""
        return len(self.current_image)
