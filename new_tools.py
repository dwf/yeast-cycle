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
from enthought.traits.api import List, Tuple, Array, Str, Bool, Float
from enthought.traits.api import HasTraits, Instance, Property

from rotation import fit_ellipse, align_image_to_ellipse
from rotation import EllipseFitError, EllipseAlignmentError

DEBUG = False

class MedialRepresentation(HasTraits):
    """docstring for MedialRepresentation"""
    #_silhouette = Instance("ObjectSilhouette")
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
    #_medial_repr = Instance("MedialRepresentation")
    
    # This could be a Delegate, except that we'd like to trigger creation
    # on get
    aligned_version = Property(depends_on='_aligned_version')
    medial_repr = Property(depends_on='_medial_repr')
        
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
            newobj = ObjectSilhouette(image=rotated, is_aligned=True)
            self._aligned_version = newobj
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
    
    def __init__(self, *args, **kwargs):
        """
        Construct an ImageSilhouette object from a PyTables node containing
        a binary mask array.
        """
        super(ImageSilhouette, self).__init__(*args, **kwargs)
        
        # Label the binary array from the HDF5 file
        self.label_image, number = ndimage.label(self.node.read())
        
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
            return ObjectSilhouette(image=image)
    
    def __contains__(self):
        raise TypeError("Containment checking not supported: %s" % str(self))
    

class Plate(HasTraits):
    """Class representing a single plate of imaged wells."""
    node = Instance(tables.Group)
    h5file = Instance(tables.File)
    images = List(Str)
    def __init__(self, *args, **kwargs):
        """
        Construct a Plate object from a PyTables File reference
        and a node from that file representing the plate.
        """
        super(Plate, self).__init__(*args, **kwargs)
        for groupnode in self.h5file.walkNodes(self.node, 'Group'):
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
            return ImageSilhouette(node=image)

    def __contains__(self):
        raise TypeError("Containment checking not supported: %s" % str(self))
    


class DataSet(HasTraits):
    """
    A class encapsulating a dataset of binary masks of segmented images 
    stored in an HDF5 file.
    """
    h5file = Instance(tables.File)
    plates = List(Str)
    
    def __init__(self, *args, **kwargs):
        """
        Construct a DataSet object from the given PyTables file handle.
        """
        super(DataSet, self).__init__(*args, **kwargs)
        for platenode in self.h5file.root:
            self.plates.append(getattr(platenode, '_v_pathname'))
    
    def __len__(self):
        return len(self.plates)

    def __getitem__(self, key):
        if type(key) is slice:
            indices = islice(xrange(len(self)), *key.indices(len(self)))
            return [self[nkey] for nkey in indices]
        else:
            node = self.h5file.getNode(self.plates[key])
            return Plate(h5file=self.h5file, node=node)
    
    def __contains__(self):
        raise TypeError("Containment checking not supported: %s" % str(self))

