import wx

import matplotlib
# We want matplotlib to use a wxPython backend
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from enthought.traits.api import Any, Instance, Int, Dict




from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.wx.basic_editor_factory import BasicEditorFactory
# GUI imports
import wx
from enthought.traits.api import HasTraits, Range, Instance, Int, Array
from enthought.traits.ui.api import View, Item, RangeEditor
#from embedded_figure import MPLFigureEditor
from matplotlib.figure import Figure

from dynamic_range import DynamicRange, DynamicTraitRange

import scipy.ndimage as ndimage
import tables

class _MPLFigureEditor(Editor):
    
    scrollable  = True
    
    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()
    
    
    def update_editor(self):
        pass
    
    
    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel
    


class MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor
        




platerange = DynamicTraitRange(0,0)
imagerange = DynamicTraitRange(0,0)
objectrange = DynamicTraitRange(0,0)

class InteractiveViewer(HasTraits):
    figure = Instance(Figure, ())
    
    viewed_plate = DynamicRange(platerange)
    viewed_image = DynamicRange(imagerange)
    viewed_item = DynamicRange(objectrange)
    
    current_image = Array(dtype=bool)
    
    plates = Dict()
    images = Dict()
    
    view = View(Item('figure', editor=MPLFigureEditor(),
                     show_label=False), 
                Item('viewed_plate',
                    show_label=False),
                Item('viewed_image',
                    show_label=False),
                Item('viewed_item',
                    show_label=False),
                width=700,
                height=600,
                resizable=True)
    
    def __init__(self, hdf5file):
        
        # Set up matplotlib stuff.
        
        super(InteractiveViewer, self).__init__()
        axes1 = self.figure.add_subplot(211)
        axes2 = self.figure.add_subplot(212)
        
        # Retain a reference to the dataset.
        
        self.dataset = hdf5file
        
        for index, plate in enumerate(hdf5file.root):
            self.plates[index] = plate._v_name
        
        platerange._high = index
    
    def _viewed_plate_changed(self):
        self.images.clear()
        where = '/' + self.plates[self.viewed_plate]
        
        for index, node in enumerate(self.dataset.walkNodes(where, 'CArray')):
            if node._v_name == "mask":
                self.images[index] = node._v_pathname
        imagerange._high = index
        self.configure_traits()

if __name__ == "__main__":
    h = tables.openFile('/Users/dwf/HOwt.h5','r')
    v = InteractiveViewer(h)
    v.configure_traits(kind="livemodal")
