"""Browser class for the """

# Standard library
import sys

# NumPy and SciPy
import numpy as np
import scipy.ndimage as ndimage

# Traits & TraitsUI
from enthought.traits.api import HasTraits, Instance, Property
from enthought.traits.api import Int, Array, Range
from enthought.traits.ui.api import View, Item, Group, VGroup, HGroup
from enthought.traits.ui.api import RangeEditor

# Chaco and Enable
from enthought.chaco.api import ArrayPlotData, Plot, HPlotContainer, bone
from enthought.chaco.api import VPlotContainer, GridPlotContainer
from enthought.enable.component_editor import ComponentEditor

# SciPy splines
from scipy.interpolate import LSQUnivariateSpline

# PyTables imports
import tables

# local imports
from new_tools import DataSet, Plate, ImageSilhouette
from new_tools import ObjectSilhouette, MedialRepresentation
from util.line_overlay import Line


class DataSetBrowser(HasTraits):
    """
    A class that allows browsing of a DataSet object with sliders
    to navigate through plates, images within plates, and objects 
    within images.
    """
    
    view = View(
            VGroup(
                HGroup(
                    Item('image_plots', 
                         editor=ComponentEditor(size=(50, 50)),
                         show_label=False
                    ),
                ),
                HGroup(
                    Item('plots', editor=ComponentEditor(size=(250, 300)),
                            show_label=False),
                ),
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
                    Item('smoothing', 
                        label='Amount of smoothing applied')
                )
            ),
            height=700,
            width=800,
            resizable=True)
    
    # Chaco plot
    gfp_plot = Instance(Plot)
    sil_plot = Instance(Plot)
    image_plots = Instance(HPlotContainer)
    rotated_plot = Instance(Plot)
    plots = Instance(GridPlotContainer)
    #legends = Instance(VPlotContainer)    
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
    smoothing = Range(0.0, 2.0, 0)
    
    
    def __init__(self, *args, **kwargs):
        """Construct a DataSetBrowser from the specified DataSet object."""
        super(DataSetBrowser, self).__init__(*args, **kwargs)
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
            
            self.image_plots = HPlotContainer(self.sil_plot,
                                              self.rotated_plot,
                                              valign="top", 
                                              bgcolor="transparent")
            
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
        laplacian = ndimage.gaussian_laplace(medial_repr.width_curve, 
            self.smoothing, mode='constant', cval=np.nan)
        m_spline = LSQUnivariateSpline(dependent_variable,
            medial_repr.medial_axis, knots)
        w_spline = LSQUnivariateSpline(dependent_variable,
            medial_repr.width_curve, knots)
        # sample at double the frequency
        spl_dep_var = np.mgrid[0:1:(medial_repr.length * 2j)]
        plots = self.plots
        if plots is None:
            # Render the plot for the first time.
            plotdata = ArrayPlotData(medial_x=dependent_variable,
                medial_y=medial_repr.medial_axis,
                width_x=dependent_variable,
                width_y=medial_repr.width_curve,
                medial_spline_x=spl_dep_var,
                medial_spline_y=m_spline(spl_dep_var),
                width_spline_x=spl_dep_var,
                width_spline_y=w_spline(spl_dep_var),
                laplacian_y=laplacian,
            )
            plot = Plot(plotdata)
            
            
            # Width data 
            self._width_data_renderer, = plot.plot(("width_x", "width_y"),
                type="line", color="blue", name="Original width curve data")
            
            filterdata = ArrayPlotData(
                            x=dependent_variable,
                            laplacian=laplacian
                        )
            filterplot = Plot(filterdata)
            self._laplacian_renderer, = filterplot.plot(("x",
                            "laplacian"), type="line", color="black", 
                            name="Laplacian-of-Gaussian")
            
            # Titles for plot & axes
            plot.title = "Width curves"
            plot.x_axis.title = "Normalized position on medial axis"
            plot.y_axis.title = "Fraction of medial axis width"
                        
            # Legend mangling stuff
            legend = plot.legend
            plot.legend = None
            legend.set(
                    component = None,
                    visible = True,
                    resizable = "",
                    auto_size=True, 
                    bounds = [250, 70],
                    padding_top = plot.padding_top)
            
            filterlegend = filterplot.legend
            filterplot.legend = None
            filterlegend.set(
                    component = None,
                    visible = True,
                    resizable = "",
                    auto_size=True, 
                    bounds = [250, 50],
                    padding_top = filterplot.padding_top)
            
            self.plots = GridPlotContainer(plot, legend, filterplot,
                                        filterlegend, shape=(2,2),
                                        valign="top", bgcolor="transparent")
            
            
        else:

            # Update the real width curve
            self._width_data_renderer.index.set_data(dependent_variable)
            self._width_data_renderer.value.set_data(medial_repr.width_curve)
            
            # Render the Laplacian
            self._laplacian_renderer.index.set_data(dependent_variable)
            self._laplacian_renderer.value.set_data(laplacian)
            
    
    def _num_internal_knots_changed(self):
        """Hook to update the plot when we change the number of knots."""
        self._update_spline_plot()
    
    def _smoothing_changed(self):
        """Hook to update the plot when we change the smoothing parameter."""
        self._update_spline_plot()
        

def deserialize(data):
    """Robert's deserialization code."""
    self = LSQUnivariateSpline.__new__(LSQUnivariateSpline)
    self._data = data
    self._reset_class()
    return self

def main():
    """Initiates the Traits dialog."""
    h5file = tables.openFile(sys.argv[1])
    dataset = DataSet(h5file=h5file)
    browser = DataSetBrowser(dataset=dataset)
    browser.configure_traits()#kind="livemodal")
    del browser
    del dataset
    h5file.close()

if __name__ == "__main__" and len(sys.argv) > 1:
    main()