"""Traits-based tools for image preprocessing."""
from itertools import islice
import traceback
import sys
import re

# For image operations and file I/O, ndimage and pytables
import tables
import numpy as np
import scipy.ndimage as ndimage

# Traits-related imports
from enthought.traits.api import Int, Float, Str, Bool, Range
from enthought.traits.api import List, Tuple, Array
from enthought.traits.api import HasTraits, Instance, Property

# Chaco
from enthought.chaco.api import ArrayPlotData, Plot, bone
from enthought.enable.component_editor import ComponentEditor

# View components
from enthought.traits.ui.api import View, Item, Group, VGroup, HGroup

# Editor components
from enthought.traits.ui.api import RangeEditor

from rotation import fit_ellipse, align_image_to_ellipse
from rotation import EllipseFitError, EllipseAlignmentError

from scipy.interpolate import LSQUnivariateSpline

DEBUG = False

class MedialRepresentation(HasTraits):
    """docstring for MedialRepresentation"""
    _silhouette = Instance("ObjectSilhouette")
    silhouette = Property(depends_on="_silhouette")
    length = Property(Float, depends_on="silhouette")
    medial_axis = Property(Array, depends_on="silhouette")
    width_curve = Property(Array, depends_on="silhouette")
    
    def __init__(self, silhouette):
        """Construct a MedialRepresentation from an ObjectSilhouette."""
        super(MedialRepresentation, self).__init__()
        if not silhouette.is_aligned:
            raise ValueError("Must be aligned version")
        self._silhouette = silhouette
        self._width_curve = None
        self._medial_axis = None
    
    def _get_silhouette(self):
        """Return the silhouette object."""
        return getattr(self, "_silhouette", None)
        
    def _get_length(self):
        """
        Return the length of the medial axis (major axis of the 
        silhouette).
        """
        return self.silhouette.image.shape[0]
    
    def _compute(self):
        """
        Do the computation necessary to fill the medial axis and width 
        curve arrays.
        """
        self._medial_axis = np.zeros(self.length)
        self._width_curve = np.zeros(self.length)
        for index, row in enumerate(self.silhouette.image):
            inked = np.where(row)[0]
            if len(inked) == 0:
                self._medial_axis[index] = self._medial_axis[:index].mean()
                self._width_curve[index] = 0.
            else:
                first = inked.min()
                last = inked.max()
                self._medial_axis[index] = (first + last) / 2.0
                self._width_curve[index] = last - first + 1
        
        self._medial_axis -= self._medial_axis.mean()
        self._medial_axis /= self.length
        self._width_curve /= self.length
    
    def _get_medial_axis(self):
        """Document me"""
        if getattr(self, "_medial_axis", None) is None:
            self._compute()
        return self._medial_axis
    
    def _get_width_curve(self):
        """Document me"""
        if getattr(self, "_width_curve", None) is None:
            self._compute()
        return self._width_curve
    

class ObjectSilhouette(HasTraits):
    """Class representing a single cell silhouette in an image."""
    image = Array(dtype=bool) 
    is_aligned = Bool()
    _aligned_version = Instance("ObjectSilhouette")
    _medial_repr = Instance("MedialRepresentation")
    
    # This could be a Delegate, except that we'd like to trigger creation
    # on get
    aligned_version = Property(depends_on='_aligned_version')
    medial_repr = Property(depends_on='_medial_repr')
    
    def __init__(self, image, angle=None, is_aligned=False):
        """Document me."""
        super(ObjectSilhouette, self).__init__()
        self.image = image
        self.angle = angle
        self.is_aligned = is_aligned
        
    ######################### Private interface ##########################
    def _get_aligned_version(self):
        """
        Return an aligned version of this ObjectSilhouette
        """
        # If we _are_ the aligned version
        if self.is_aligned:
            return self
        
        # In case we have it cached
        if not getattr(self, '_aligned_version', None):
            image = self.image
            try:
                coeffs = fit_ellipse(*(np.where(image)))
            except EllipseFitError:
                traceback.print_exc()
                return None
            try:
                rotated, angle = align_image_to_ellipse(coeffs, image)
            except EllipseAlignmentError:
                traceback.print_exc()
                return None
            self._aligned_version = ObjectSilhouette(rotated, angle, 
                is_aligned=True)
        return self._aligned_version
    
    def _get_medial_repr(self):
        """
        Return the medial representation object, constructing one if
        necessary.
        """
        if self.is_aligned:
            if not getattr(self, "_medial_repr", None):
                self._medial_repr = MedialRepresentation(self)
            return self._medial_repr
        else:
            raise ValueError("Object %s is not aligned version" % repr(self))
    

class ImageSilhouette(HasTraits):
    """Class representing a silhouette image of segmented cells."""
    label_image = Array()
    object_slices = List(Tuple(slice, slice), ())
        
    def __init__(self, node):
        """
        Construct an ImageSilhouette object from a PyTables node containing
        a binary mask array.
        """
        super(ImageSilhouette, self).__init__()
        
        # Label the binary array from the HDF5 file
        self.label_image, number = ndimage.label(node.read())
        
        if DEBUG:
            print "%d objects segmented." % number
        
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
        raise TypeError("Containment checking not supported: %s" % str(self))
    

class Plate(HasTraits):
    """Class representing a single plate of imaged wells."""
    node = Instance(tables.Group)
    h5file = Instance(tables.File)
    images = List(Str)
    def __init__(self, h5file, node):
        """
        Construct a Plate object from a PyTables File reference
        and a node from that file representing the plate.
        """
        super(Plate, self).__init__()
        self.node = node
        self.h5file = h5file
        for groupnode in h5file.walkNodes(node, 'Group'):
            if re.match(r'site\d', getattr(groupnode, '_v_name')):
                self.images.append(getattr(groupnode, '_v_pathname'))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        if type(key) is slice:
            indices = islice(xrange(len(self)), 
                *key.indices(len(self.images)))
            return [self[nkey] for nkey in indices]
        else:
            image = self.h5file.getNode(self.images[key] + '/mask')
            return ImageSilhouette(image)

    def __contains__(self):
        raise TypeError("Containment checking not supported: %s" % str(self))
    


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
            self.plates.append(getattr(platenode, '_v_pathname'))
        self.h5file = h5file
    
    def __len__(self):
        return len(self.plates)

    def __getitem__(self, key):
        if type(key) is slice:
            indices = islice(xrange(len(self)), *key.indices(len(self)))
            return [self[nkey] for nkey in indices]
        else:
            node = self.h5file.getNode(self.plates[key])
            return Plate(self.h5file, node)
    
    def __contains__(self):
        raise TypeError("Containment checking not supported: %s" % str(self))
    

class DataSetBrowser(HasTraits):
    """
    A class that allows browsing of a DataSet object with sliders
    to navigate through plates, images within plates, and objects 
    within images.
    """
    
    view = View(
            VGroup(
                HGroup(
                    Item('sil_plot', editor=ComponentEditor(size=(200, 200)), 
                        show_label=False),
                    Item('rotated_plot',
                        editor=ComponentEditor(size=(200, 200)),
                        show_label=False)
                ),
                Item('splines_plot', 
                    editor=ComponentEditor(size=(250, 250)),
                    show_label=False),

                Group(Item('object_index', editor=RangeEditor(low=1, 
                    high_name='num_objects', mode='slider')),
                    Item('image_index', editor=RangeEditor(low=1, 
                        high_name='num_images', mode='slider')),
                    Item('plate_index', editor=RangeEditor(low=1, 
                        high_name='num_plates', mode='slider')),
                ),
                HGroup(
                    Item('num_internal_knots', 
                        label='Number of internal spline knots'),
                    Item('legend_visible', label='Legend visible?')
                )
            ),
            height=700,
            width=800,
            resizable=True)
    
    # Chaco plot
    gfp_plot = Instance(Plot)
    sil_plot = Instance(Plot)
    rotated_plot = Instance(Plot)
    splines_plot = Instance(Plot)
    legend_visible = Property(Bool)
        
    # DataSet being viewed
    dataset = Instance(DataSet)
    
    # Plate object currently being examined
    current_plate = Instance(Plate)
    
    # ImageSilhouette object currently being examined
    current_image = Instance(ImageSilhouette)

    # ObjectSilhouette object currently being examined
    current_object = Instance(ObjectSilhouette)

    # Index traits that control the selected plate/image/object
    plate_index = Int(1)
    image_index = Int(1)
    object_index = Int(1)

    # Number of plates, images, and objects in the current context
    num_plates = Property(Int, depends_on='dataset')
    num_images = Property(Int, depends_on='current_plate')
    num_objects = Property(Int, depends_on='current_image')
    num_internal_knots = Range(1, 20, 3)
    
    def __init__(self, dataset, **metadata):
        """Construct a DataSetBrowser from the specified DataSet object."""
        super(DataSetBrowser, self).__init__(**metadata)
        self.dataset = dataset
        self.current_plate = self.dataset[self.plate_index - 1]
        self.current_image = self.current_plate[self.image_index - 1]
        self.current_object = self.current_image[self.object_index - 1]
        self.sil_plot = Plot()
        self._object_index_changed()
    
        
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
            
            # Display 
            sil = self.current_object.image
            self._update_img_plot('sil_plot', sil, 'Extracted mask')
            
            # .T to get major axis horizontal
            rotated = self.current_object.aligned_version.image.T 
            self._update_img_plot('rotated_plot', rotated, 'Aligned mask')
            
            self._update_spline_plot()
         
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
    
    def _update_img_plot(self, plot_name, image, title):
        """Update an image plot."""
        plotdata = ArrayPlotData(imagedata=image)
        xbounds = (0, image.shape[1] - 1)
        ybounds = (0, image.shape[0] - 1)
        
        plot = Plot(plotdata)
        plot.aspect_ratio = float(xbounds[1]) / float(ybounds[1])
        plot.img_plot("imagedata", colormap=bone, xbounds=xbounds,
            ybounds=ybounds)
        plot.title = title
        
        setattr(self, plot_name, plot)
        getattr(self, plot_name).request_redraw()
    
    def _update_spline_plot(self):
        """Update the spline plot."""
        knots = np.mgrid[0:1:((self.num_internal_knots + 2)*1j)][1:-1]
        medial_repr = self.current_object.aligned_version.medial_repr
        dependent_variable = np.mgrid[0:1:(medial_repr.length * 1j)]
        m_spline = LSQUnivariateSpline(dependent_variable,
            medial_repr.medial_axis, knots)
        w_spline = LSQUnivariateSpline(dependent_variable,
            medial_repr.width_curve, knots)
        # sample at double the frequency
        spl_dep_var = np.mgrid[0:1:(medial_repr.length * 2j)]
        plot = self.splines_plot
        if plot is None:
            # Render the plot for the first time.
            plotdata = ArrayPlotData(medial_x=dependent_variable,
                medial_y=medial_repr.medial_axis,
                width_x=dependent_variable,
                width_y=medial_repr.width_curve,
                medial_spline_x=spl_dep_var,
                medial_spline_y=m_spline(spl_dep_var),
                width_spline_x=spl_dep_var,
                width_spline_y=w_spline(spl_dep_var)
            )
            plot = Plot(plotdata)
            
            
            # Medial data
            self._medial_data_renderer, = plot.plot(("medial_x", "medial_y"), 
                type="line", color="blue", line_style="dash", 
                name="Original medial axis data")
            
            # Width data 
            self._width_data_renderer, = plot.plot(("width_x", "width_y"),
                type="line", color="blue", name="Original width curve data")
            
            # Medial spline
            self._medial_spline_renderer, = plot.plot(("medial_spline_x",
                "medial_spline_y"), type="line", color="green",
                line_style="dash", name="Medial axis spline")
                        
            # Width spline
            self._width_spline_renderer, = plot.plot(("width_spline_x",
                "width_spline_y"), type="line", color="green", 
                name="Width curve spline")
            
            # Titles for plot & axes
            plot.title = "Extracted splines"
            plot.x_axis.title = "Normalized position on medial axis"
            plot.y_axis.title = "Fraction of medial axis width"
            self.splines_plot = plot
        else:
            def render_update(renderer, index, value):
                renderer.index.set_data(index)
                renderer.value.set_data(value)
            
            # Update the real medial curve
            self._medial_data_renderer.index.set_data(dependent_variable)
            self._medial_data_renderer.value.set_data(medial_repr.medial_axis)
            
            # Update the real width curve
            self._width_data_renderer.index.set_data(dependent_variable)
            self._width_data_renderer.value.set_data(medial_repr.width_curve)

            # Update the fitted medial spline
            self._medial_spline_renderer.index.set_data(spl_dep_var)
            self._medial_spline_renderer.value.set_data(m_spline(spl_dep_var))
            
            # Update the fitted width spline
            self._width_spline_renderer.index.set_data(spl_dep_var)
            self._width_spline_renderer.value.set_data(w_spline(spl_dep_var))
        
        # No matter what, refresh        
        plot.request_redraw()
    
    def _get_legend_visible(self):
        """Hook to provide access to the legend's 'visible' property."""
        return self.splines_plot.legend.visible
    
    def _set_legend_visible(self, visible):
        """Hook to update the plot when we enable/disable the legend."""
        self.splines_plot.legend.visible = visible
        self.splines_plot.request_redraw()
    
    def _num_internal_knots_changed(self):
        """Hook to update the plot when we chane the number of knots."""
        self._update_spline_plot()
    

def main():
    """Initiates the Traits dialog."""
    h5file = tables.openFile(sys.argv[1])
    dataset = DataSet(h5file)
    browser = DataSetBrowser(dataset)
    browser.configure_traits(kind="livemodal")
    del browser
    del dataset
    h5file.close()

if __name__ == "__main__" and len(sys.argv) > 1:
    main()