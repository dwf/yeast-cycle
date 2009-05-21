# For our matplotlib figure
import matplotlib
from matplotlib.figure import Figure

# For image operations and file I/O, ndimage and pytables
import scipy.ndimage as ndimage
import tables
import numpy as np

# Traits-related imports
from enthought.traits.api import HasTraits, Range, Instance, Int, Array, List
from enthought.traits.ui.api import View, Item, RangeEditor, InstanceEditor
from embedded_figure import MPLFigureEditor

from tools import fit_ellipse, spline_features, radians_to_degrees
from tools import ObjectTooSmallError

class ObjectSilhouette(HasTraits):
    figure = Instance(Figure, ())    
    image = Array(dtype=bool) 
    view = View(Item('figure', editor=MPLFigureEditor(),
                     show_label=False), 
                resizable=True)

    def __init__(self, image):
        """Document me."""
        # Set up matplotlib stuff.
        super(ObjectSilhouette, self).__init__()
        axes1 = self.figure.add_subplot(211)
        axes2 = self.figure.add_subplot(212)

        # Retain a reference to the dataset.
        self.image = image
        
        # Show the image.
        axes1.matshow(image,cmap=matplotlib.cm.bone)
        
        self.process_item()
    
    def process_item(self):
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
                    fig=self.figure)
        except ObjectTooSmallError, e:
            pass
    

class ImageSilhouette(HasTraits):
    label_image = Array()
    current_object = Instance(ObjectSilhouette)
    current_index = Int(editor=RangeEditor(low=0,mode='slider'))
    objects = List()
    
    view = View(
        Item('current_object', 
            editor=InstanceEditor(),
            show_label=False,
            style='custom',
            width=700,
            height=500),
        Item('current_index',
            show_label=False))
    
    def __init__(self, node):
        super(ImageSilhouette, self).__init__()
        self.label_image, number = ndimage.label(node.read())
        print "Number of objects found: %d" % number
        self.objects = ndimage.find_objects(self.label_image)
        current_index = 0
        index_editor = self.traits()['current_index'].editor
        index_editor.high = number - 1
        self._current_index_changed()
        #self.traits()['current_object_index'].editor = \
        #    RangeEditor(low=0, high=number-1)
    
    def _current_index_changed(self):
        idx = self.current_index
        
        try:
            objsil = self.label_image[self.objects[idx]] == (idx + 1)
            self.current_object = ObjectSilhouette(objsil)
        except IndexError:
            print IndexError, idx, len(self.objects)


class Plate(HasTraits):
    node = Instance(tables.Group)
    images = List()
    current_index = Int(editor=RangeEditor(low=0,mode='slider'))
    current_object = Instance(ImageSilhouette)

    view = View(
        Item('current_object', 
            editor=InstanceEditor(),
            show_label=False,
            style='custom'),
        Item('current_index',
            show_label=False))

    def __init__(self, node):
        super(Plate, self).__init__()
        self.node = node
        for imnode in node._f_walkNodes('Leaf'):
            self.images.append(imnode._v_pathname)

        index_editor = self.traits()['current_index'].editor
        print len(self.images)
        index_editor.high = len(self.images) - 1
        print index_editor.high
        self._current_index_changed()

        self.current_index = 0

    def _current_index_changed(self):
        idx = self.current_index
        imnode = self.node._v_file.getNode(self.images[idx])
        self.current_object = ImageSilhouette(imnode)

class DataSet(HasTraits):
    h5file = Instance(tables.File)
    plates = List()
    current_index = Int(editor=RangeEditor(low=0,mode='slider'))
    current_object = Instance(Plate)
    
    view = View(
        Item('current_object', 
            editor=InstanceEditor(),
            show_label=False,
            style='custom'),
        Item('current_index',
            show_label=False),
        title='Dataset Viewer')

    def __init__(self, h5file):
        super(DataSet, self).__init__()
        for platenode in h5file.root:
            self.plates.append(platenode)
        editor = self.traits()['current_index'].editor
        editor.high = len(self.plates) - 1
        self._current_index_changed()
        
    def _current_index_changed(self):
        idx = self.current_index
        self.current_object = Plate(self.plates[idx])
    
